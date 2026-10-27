#pragma once

// =============================================================================
// naga/cospan/system.hpp — System: structure and wiring (r04)
// =============================================================================
//
// A System never does work and never runs: it GROUPS leaves and DECLARES wiring. It has only setup() (run_setup below)
// plus the declaration calls; there is no prepare_forward and no forward (internal/spec/revisions/r04-...). The run
// walks the model's tree (model.hpp); everything a System declares is recorded in the model-wide metadata
// (model_meta.hpp) and nothing is replayed.
//
// DECLARATION CALLS (connect, promote, tie_inputs, tie_outputs, init_input, shape_hook, depends_on, set_device,
// set_run_policy / clear_run_policy, the marks) resolve names RELATIVE to this System ("a.y" is this System's name + "." +
// "a.y"; the root adds no dot), check their rules in the call, and record into the metadata. They are legal inside a
// setup() and, from main, after Model::setup() and before finalize (a LOGIC error otherwise). add_subsystem is legal inside
// a setup() only. A name is unique across the whole model; a duplicate is refused at the call.
//
// ADDING A CHILD. add_subsystem builds the child from a descriptor, gives it its full resolved name and the metadata,
// registers it (its tree node, its depth-first ordinal), and runs its setup before it returns: setup is depth-first and
// declaration order is execution order. The descriptor carries the name and the constructor arguments only.

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/component.hpp"
#include "naga/cospan/component_driver.hpp"
#include "naga/cospan/init.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/model_meta.hpp"
#include "naga/cospan/run_policy.hpp"
#include "naga/cospan/shape_hook.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wiring.hpp"

namespace naga::cospan {

// -----------------------------------------------------------------------------
// Descriptors
// -----------------------------------------------------------------------------

/// What add_subsystem builds a child from: the NAME and the CONSTRUCTOR ARGUMENTS of a subsystem of type T, and
/// nothing else. It carries no device: the device is an add_subsystem argument, and constructors never take one.
template <class T, class... Args>
struct SubsystemDesc {
    std::string                       name;
    std::tuple<std::decay_t<Args>...> args;
};

/// What a descriptor can build: a Subsystem (a user-defined System, a fused group) or a component of any of the three types.
template <class T>
concept SubsystemOrComponent = std::derived_from<T, Subsystem> || std::derived_from<T, ComponentBase>;

/// subsystem_desc<T>("name", ctor args...): a descriptor of a Component (or CustomComponent / HostComponent) or a user-defined
/// System.
template <class T, class... Args>
    requires SubsystemOrComponent<T>
SubsystemDesc<T, Args...> subsystem_desc(std::string name, Args&&... args) {
    return SubsystemDesc<T, Args...>{std::move(name), std::tuple<std::decay_t<Args>...>{std::forward<Args>(args)...}};
}

namespace detail {
/// A fresh node object from copies of the descriptor's arguments: a component goes into a SoloDriver (the leaf), anything else
/// is the node itself.
template <class T, class Tuple>
std::unique_ptr<Subsystem> make_subsystem(Tuple& args) {
    return std::apply(
        [](auto&... a) -> std::unique_ptr<Subsystem> {
            if constexpr (std::derived_from<T, ComponentBase>)
                return std::make_unique<SoloDriver>(std::make_unique<T>(a...));
            else
                return std::make_unique<T>(a...);
        },
        args);
}
}  // namespace detail

// -----------------------------------------------------------------------------
// System
// -----------------------------------------------------------------------------

class System : public Subsystem {
public:
    ~System() override = default;

    // ---- HOOK ------------------------------------------------------------------
    /// Structure and wiring (legal here only for add_subsystem). There is no init hook: initial values are declared with
    /// init_input.
    virtual void setup() {}

    // ---- the Subsystem verbs -----------------------------------------------------
    bool is_leaf() const noexcept override { return false; }

    /// Runs setup() with the metadata's setup flag up.
    void run_setup() override {
        detail::ModelMeta::SetupScope scope{meta_};
        setup();
    }

    // ---- adding children (legal inside setup() only) ---------------------------
    /// The child is registered and its run_setup has run BEFORE this returns. Properties: the run policy of the child's node.
    /// A device is the child's placement (the outer call wins when finalize applies the placements from the root).
    template <class T, class... A>
    void add_subsystem(SubsystemDesc<T, A...> desc) {
        add_child(std::move(desc.name), property_list{}, std::nullopt, detail::make_subsystem<T>(desc.args));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, SubsystemDesc<T, A...> desc) {
        add_child(std::move(desc.name), props, std::nullopt, detail::make_subsystem<T>(desc.args));
    }
    template <class T, class... A>
    void add_subsystem(const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        add_child(std::move(desc.name), property_list{}, dev, detail::make_subsystem<T>(desc.args));
    }
    template <class T, class... A>
    void add_subsystem(const property_list& props, const sycl::device& dev, SubsystemDesc<T, A...> desc) {
        add_child(std::move(desc.name), props, dev, detail::make_subsystem<T>(desc.args));
    }

    // ---- structure: connect / promote / tie ------------------------------------------
    /// Feeds `dst` from `src`; the source's variable wins (wiring.hpp, connect). Merging two variables that are both
    /// materialized is an error here ("merging two materialized variables is not defined (yet)").
    void connect(std::string_view src, std::string_view dst) {
        guard("connect");
        const auto s = instance_at(src);
        const auto d = instance_at(dst);
        detail::connect(site(s.get(), d.get()), s, d);
    }

    /// Kind-agnostic and always allowed: a PEER alias in this namespace joining the member's variable. DT must match the
    /// member's value dtype and Rank its rank. The alias takes the member's device now (an outer set_device can override it).
    template <DType DT, int Rank>
    void promote(std::string_view child_path, std::string promoted_name) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        guard("promote");
        const std::string full = check_new_name(promoted_name, "alias");
        const auto        member = instance_at(child_path);
        auto              a      = detail::promote<DT, Rank>(site(), member, promoted_name, this);
        store_alias(a, full);
    }

    /// A NEW variable (target, not source); every member's variable merges into it and the alias wins. The members are
    /// inputs of dtype DT and rank Rank. The tie's shape is a DEFINITION like a port's (cs::shape): the definitions of the
    /// members and of everything connected to it merge (unspecified absorbs specified, two specified values must agree), and
    /// a dimension still unspecified comes from the connected output, the initializer, or a shape_hook on the alias.
    /// `device` (the default device when none is given): where the alias lives and where the tie's initialization runs.
    /// A member may itself be a tie alias (ties of ties): it is then ABSORBED by the outer tie.
    template <DType DT, int Rank>
    void tie_inputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                    std::optional<sycl::device> device = {}) {
        tie_inputs_of<DT, Rank>(std::span<const std::string_view>{members.begin(), members.size()}, std::move(name),
                                std::move(shape), std::move(device));
    }

    /// A NEW variable (source, not target); atomic member dtypes unless allow_plain. The members are outputs of dtype DT and
    /// rank Rank, all on one device (checked at the call and again at finalize); the alias owns the shape (a shape_hook on
    /// the alias sets the dimensions the definition leaves open). The variable is push-suppressed: the members only register
    /// their writes, and ONE push through the outermost alias is placed at finalize, right after the last member's leaf.
    /// A member may itself be a tie alias (ties of ties): it is then ABSORBED by the outer tie.
    template <DType DT, int Rank>
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name, ShapeSpec shape = {},
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic, std::optional<sycl::device> device = {}) {
        tie_outputs_of<DT, Rank>(std::span<const std::string_view>{members.begin(), members.size()}, std::move(name),
                                 std::move(shape), policy, std::move(device));
    }

    // ---- shape hooks -----------------------------------------------------------------
    /// A shape hook is this System's contract for the shape of an input, a tie alias of either kind, or an ordinary untied
    /// output (shape_hook.hpp). One descriptor per dimension (cs::dim::as_is, from_shape, from_value, custom, fixed_at_run),
    /// or for a ragged target ONE of cs::ragged::counts_from / offsets_from. Each hooked dimension must be unspecified in the
    /// definition and is owned by this hook only. A hook and a connection exclude each other on an input. A member of a
    /// tie_outputs is refused: hook the tie alias. The hook is PLACED AT FINALIZE: directly after the last producer of the
    /// sources it names; a hook with no traceable source runs just before the first holder of its target. A source produced
    /// after that first holder is an error at finalize.
    template <HookDescriptor... D>
    void shape_hook(std::string_view target, D... descriptors) {
        static_assert(sizeof...(D) >= 1, "cospan: a shape hook takes one descriptor per dimension");
        guard("shape_hook");
        std::vector<detail::HookInfo> infos;
        (infos.push_back(detail::hook_info(descriptors)), ...);
        detail::ModelMeta& m     = meta();
        const std::uint32_t owner = ++m.next_hook_owner;
        using Step                = detail::ShapeHookStep<D...>;
        auto  step                = std::make_unique<Step>(std::tuple<D...>{std::move(descriptors)...}, std::string{target}, owner,
                                          [this](std::string_view p) { return instance_at(p); });
        Step* raw                 = step.get();
        const HookTarget found    = install_hook(std::string{target}, infos, owner);
        raw->set_target(found.inst);
        // An initializer with a shape defines every dimension: one already declared conflicts with the dimensions this hook owns.
        if (const detail::VarMeta& tv = *found.inst->meta; tv.init && tv.init->shape() && (!tv.has_producer() || tv.cycle))
            tv.check_init_against_definition(*tv.init->shape());
        detail::HookRecord rec;
        rec.target      = detail::ModelMeta::child_name(path(), target);
        rec.target_inst = found.inst;
        for (std::size_t i = 0; i < found.srcs.size(); ++i) {
            if (!found.srcs[i]) continue;
            raw->set_source(i, found.srcs[i]);
            rec.sources.push_back(detail::ModelMeta::child_name(path(), infos[i].path));
            rec.source_insts.push_back(found.srcs[i]);
        }
        rec.step = std::move(step);
        m.hooks.push_back(std::move(rec));
    }

    // ---- initial values ----------------------------------------------------------------
    /// Declares the initial value of a free input's variable: `path` is a child input path or one of this System's promoted
    /// or tied input aliases. The last declaration wins. It is ignored when the input ends up with a source, and naming an
    /// output throws.
    void init_input(std::string_view target, InitSpec init) {
        guard("init_input");
        const auto inst = instance_at(target);
        detail::declare_init(site(), inst, std::move(init), /*restale=*/true);  // on a variable that holds data: initialized again
        meta().mark_pending(inst->meta);
    }

    // ---- ordering ---------------------------------------------------------------------------
    /// `b` depends on `a` (subsystems of this System, leaves or systems): for one run at a time, the events of every leaf in
    /// a's range are handed to every leaf in b's range. `a` must have been added before `b`. A duplicate edge is ignored.
    /// Placed at finalize as one after-operation on the last leaf of a's range.
    void depends_on(std::string_view b, std::string_view a) {
        guard("depends_on");
        detail::ModelMeta& m  = meta();
        const std::string  fb = detail::ModelMeta::child_name(path(), b);
        const std::string  fa = detail::ModelMeta::child_name(path(), a);
        const detail::Node* nb = m.find_node(fb);
        const detail::Node* na = m.find_node(fa);
        if (nb == nullptr) fail("depends_on", "'" + std::string{b} + "' is not a subsystem of " + describe());
        if (na == nullptr) fail("depends_on", "'" + std::string{a} + "' is not a subsystem of " + describe());
        if (na->has_leaves && nb->has_leaves && na->last_ordinal >= nb->ordinal)
            fail("depends_on", "'" + std::string{b} + "' can depend only on a subsystem added before it ('" + std::string{a} + "' was not)");
        for (const detail::DependsRecord& e : m.depends)
            if (e.a == fa && e.b == fb) return;
        m.depends.push_back(detail::DependsRecord{fb, fa});
    }

    // ---- placement and run policies ------------------------------------------------------------
    /// Places this System's own aliases and everything under it on `d`. The OUTER call wins: finalize applies the placements
    /// again from the root, and the outermost one holds.
    void set_device(const sycl::device& d) {
        guard("set_device");
        detail::Node& n = self_node();
        n.device_override = d;
        detail::apply_device_subtree(n, d);
    }
    /// The same for the subsystem at `path` (relative to this System).
    void set_device(std::string_view rel, const sycl::device& d) {
        guard("set_device");
        detail::Node& n = node_at(rel, "set_device");
        n.device_override = d;
        detail::apply_device_subtree(n, d);
    }

    /// Sets the run policy of the node at `path` (relative to this System; a leaf or a system): no property is eternal,
    /// {transient}, {run_n{n}} and {do_once} are the others. The last caller wins; the counter starts over. The node's scope
    /// is not changed (rearm_subsystem puts it back in scope). Legal at any time after the model's setup().
    void set_run_policy(std::string_view rel, const property_list& policy = {}) {
        require_after_setup("set_run_policy");
        node_at(rel, "set_run_policy").policy = detail::make_run_policy(policy, describe() + ": set_run_policy('" + std::string{rel} + "')");
    }
    /// Removes the run policy of the node at `path`: it just runs.
    void clear_run_policy(std::string_view rel) {
        require_after_setup("clear_run_policy");
        node_at(rel, "clear_run_policy").policy.reset();
    }

    /// Attaches a solver to this System: it solves the cycles this System owns (docs/solvers.md). Legal inside this System's
    /// setup() and at any time after the model's setup().
    void set_solver(std::unique_ptr<Solver> s) {
        detail::Node& n = self_node();
        n.solver        = std::shared_ptr<Solver>(std::move(s));
        n.last_solve.reset();
    }

    // ---- optimization marks: the last of a kind on a variable wins ---------------
    /// A free input of fixed shape.
    void add_design_var(std::string_view path, Bounds b = {}, Scaling s = {}) {
        guard("add_design_var");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::input)
            fail("add_design_var", "'" + std::string{path} + "' is an output; a design variable is a free input");
        if (const auto fed = detail::first_of_kind(v, IOKind::output))
            fail("add_design_var", "connected inputs are computed, not designed ('" + std::string{path} + "' is fed by '" +
                                       fed->full_path() + "')");
        if (!all_fixed(v))
            fail("add_design_var", "'" + std::string{path} + "' is dynamically shaped; a dynamic design variable is ill-formed");
        check_sizes(flat_size(v), b, std::nullopt, s, "add_design_var", path);
        store_mark(MarkKind::design_var, path, inst, MarkRecord{std::move(b), std::nullopt, std::move(s)});
    }

    /// An output; `equals` and bounds are exclusive.
    void add_constraint(std::string_view path, Bounds b = {}, std::optional<std::vector<double>> equals = {}, Scaling s = {}) {
        guard("add_constraint");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::output) fail("add_constraint", "'" + std::string{path} + "' is not an output");
        if (equals && (b.lower || b.upper)) fail("add_constraint", "'equals' and bounds are exclusive");
        if (all_fixed(v)) check_sizes(flat_size(v), b, equals, s, "add_constraint", path);
        store_mark(MarkKind::constraint, path, inst, MarkRecord{std::move(b), std::move(equals), std::move(s)});
    }

    /// An output of fixed shape with flat size 1.
    void add_objective(std::string_view path, Scaling s = {}) {
        guard("add_objective");
        const auto inst = instance_at(path);
        const detail::VarMeta& v = *inst->meta;
        if (inst->kind != IOKind::output) fail("add_objective", "'" + std::string{path} + "' is not an output");
        if (!all_fixed(v))
            fail("add_objective", "'" + std::string{path} + "' is not fixed in shape; an objective is one value (flat size 1)");
        const std::size_t n = flat_size(v);
        if (n != 1)
            fail("add_objective", "'" + std::string{path} + "' has flat size " + std::to_string(n) + "; an objective is one value");
        check_sizes(n, Bounds{}, std::nullopt, s, "add_objective", path);
        store_mark(MarkKind::objective, path, inst, MarkRecord{Bounds{}, std::nullopt, std::move(s)});
    }

protected:
    friend class Model;

    // ---- resolution (one map lookup) -------------------------------------------------
    /// The holder a path names, from THIS namespace (a leaf's input or output, a System's alias, a fused group's stage port):
    /// the full name is this System's name + "." + the path. Throws std::invalid_argument naming the path.
    std::shared_ptr<detail::Instance> instance_at(std::string_view rel) {
        const std::shared_ptr<detail::Instance> h = meta().find_holder(detail::ModelMeta::child_name(path(), rel));
        if (!h) fail("resolve", "no input, output or alias '" + std::string{rel} + "'");
        return h;
    }

    /// The bodies of tie_inputs and tie_outputs, over a span of member paths (a fused group's ties_desc has its members in a
    /// vector, not in an initializer list). Everything the public calls do is done here, once.
    template <DType DT, int Rank>
    void tie_inputs_of(std::span<const std::string_view> members, std::string name, ShapeSpec shape = {},
                       std::optional<sycl::device> device = {}) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        guard("tie_inputs");
        const std::string        full   = check_new_name(name, "alias");
        const detail::TieMembers ms     = resolve_members(members);
        const auto               inside = tie_aliases_in(ms);
        auto a = detail::tie_inputs<DT, Rank>(site(), ms, name, this, shape,
                                              std::optional<sycl::device>{device.value_or(detail::default_device())});
        for (const auto& i : inside) i->absorbed = true;
        store_alias(a, full);
    }
    template <DType DT, int Rank>
    void tie_outputs_of(std::span<const std::string_view> members, std::string name, ShapeSpec shape = {},
                        TieOutputsPolicy policy = TieOutputsPolicy::require_atomic, std::optional<sycl::device> device = {}) {
        static_assert(Rank >= 1 && Rank <= 3, "cospan: rank is 1..3");
        guard("tie_outputs");
        const std::string        full   = check_new_name(name, "alias");
        const detail::TieMembers ms     = resolve_members(members);
        const auto               inside = tie_aliases_in(ms);
        auto a = detail::tie_outputs<DT, Rank>(site(), ms, name, this, shape, policy, device.value_or(detail::default_device()));
        for (const auto& i : inside) i->absorbed = true;
        store_alias(a, full);
        detail::TieOutRecord t;
        t.owner = path();
        t.alias = a;
        for (const std::string_view mm : members) t.members.push_back(detail::ModelMeta::child_name(path(), mm));
        meta().tie_outs.push_back(std::move(t));
    }

    /// "the root System" or "System 'path'": the prefix of every System error.
    std::string describe() const {
        return path().empty() ? std::string{"the root System"} : "System '" + path() + "'";
    }

    [[noreturn]] void fail(const char* what, const std::string& why) const {
        throw std::invalid_argument(std::string{"naga::cospan: "} + describe() + ": " + what + ": " + why);
    }

    detail::ModelMeta& meta() const {
        if (meta_ == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + " is not part of a model yet (it is registered by add_subsystem)");
        return *meta_;
    }

    /// A declaration call is legal inside some setup() and, after the model's setup(), until finalize.
    void guard(const char* what) const { meta().require_declaring(what, describe()); }

    void require_after_setup(const char* what) const {
        const detail::ModelMeta& m = meta();
        if (m.state == detail::Lifecycle::created && m.setup_depth == 0)
            throw std::logic_error(std::string{"naga::cospan: "} + describe() + ": " + what + " is illegal before the model's setup()");
    }

    detail::Node& self_node() const {
        detail::Node* n = meta().node_for(this);
        if (n == nullptr) throw std::logic_error("naga::cospan: " + describe() + " has no node in the model");
        return *n;
    }

    /// The node of the subsystem at `rel` (this System's name + "." + rel). Throws std::invalid_argument.
    detail::Node& node_at(std::string_view rel, const char* what) const {
        detail::Node* n = meta().find_node(detail::ModelMeta::child_name(path(), rel));
        if (n == nullptr) fail(what, "no subsystem '" + std::string{rel} + "'");
        return *n;
    }

private:
    enum class MarkKind : std::uint8_t { design_var, constraint, objective };

    // ---- adding children -----------------------------------------------------------------
    /// add_subsystem: legal inside a setup() only.
    void add_child(std::string name, const property_list& props, std::optional<sycl::device> dev, std::unique_ptr<Subsystem> child) {
        meta().require_in_setup("add_subsystem", describe());
        add_child_unchecked(std::move(name), props, std::move(dev), std::move(child));
    }

    /// The registration itself (the Model's own add_subsystem, before its setup(), comes here too).
    void add_child_unchecked(std::string name, const property_list& props, std::optional<sycl::device> dev,
                             std::unique_ptr<Subsystem> child) {
        detail::ModelMeta& m = meta();
        detail::ModelMeta::check_local_name(name, describe(), "subsystem");
        const std::string full = detail::ModelMeta::child_name(path(), name);
        std::shared_ptr<detail::RunPolicy> policy = detail::make_run_policy(props, describe() + ": add_subsystem('" + name + "')");
        System*                            sys    = dynamic_cast<System*>(child.get());
        const detail::NodeKind             kind   = child->is_leaf() ? detail::NodeKind::leaf : detail::NodeKind::system;
        detail::Node&                      node   = m.register_node(&self_node(), full, std::move(child), kind, sys, describe());
        node.policy                               = std::move(policy);
        if (dev) node.device_override = *dev;
        node.sub->run_setup();
        if (dev) detail::apply_device_subtree(node, *dev);
    }

    // ---- names -------------------------------------------------------------------------
    /// A new alias name: non-empty, and its full name is taken by nothing in the model. Returns the full name.
    std::string check_new_name(const std::string& name, const char* what) const {
        detail::ModelMeta::check_local_name(name, describe(), what);
        const std::string full = detail::ModelMeta::child_name(path(), name);
        if (meta().name_taken(full))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + full + "' is already a subsystem or alias name in this model");
        return full;
    }

    void store_alias(const std::shared_ptr<detail::Instance>& a, const std::string& full) {
        alias_order_.push_back(a);
        meta().register_holder(full, a, &self_node(), describe());
    }

    // ---- wiring sites --------------------------------------------------------------------
    /// The ordinal a consumer reads at: the leaf that holds the input, else (an alias) the first leaf holding an input port
    /// of its variable.
    std::optional<std::size_t> consumer_ordinal(const detail::Instance& d) const {
        if (const auto o = meta().leaf_ordinal(d)) return o;
        std::optional<std::size_t> best;
        for (const auto& i : d.meta->live_instances()) {
            if (i->kind != IOKind::input || i->role != detail::Role::port) continue;
            if (const auto o = meta().leaf_ordinal(*i)) best = best ? std::min(*best, *o) : *o;
        }
        return best;
    }

    /// Where a wiring call happens: this System's text, the ordinals of the two ends (the source variable's LAST producer and
    /// the consumer), the model's warning set.
    detail::WiringSite site(const detail::Instance* src = nullptr, const detail::Instance* dst = nullptr) {
        detail::ModelMeta&         m = meta();
        std::optional<std::size_t> so, dord;
        if (src != nullptr)
            if (const detail::Node* p = m.last_producer(*src->meta)) so = p->ordinal;
        if (dst != nullptr) dord = consumer_ordinal(*dst);
        return detail::WiringSite{describe(), so, dord, &m.warned_cycles};
    }

    detail::TieMembers resolve_members(std::span<const std::string_view> members) {
        detail::TieMembers out;
        for (const std::string_view mm : members) out.emplace_back(std::string{mm}, instance_at(mm));
        return out;
    }

    /// The tie aliases already in the members' variables (a member that is itself a tie alias, or a promotion of one): the
    /// new tie ABSORBS them, so the outermost alias is the one that pushes. Read BEFORE the tie merges the variables.
    static std::vector<std::shared_ptr<detail::Instance>> tie_aliases_in(const detail::TieMembers& ms) {
        std::vector<std::shared_ptr<detail::Instance>> out;
        for (const auto& mm : ms)
            for (const auto& i : mm.second->meta->live_instances())
                if (i->role == detail::Role::tie) out.push_back(i);
        return out;
    }

    // ---- shape hooks ---------------------------------------------------------------------
    /// Gives `inst`'s variable the hook's marks: the dimensions it owns become `hook` (an unspecified one; a dimension
    /// already owned by this hook stays), fixed_at_run ones become fixed; an input target is no longer a target.
    void mark_hooked(detail::Instance& inst, const std::vector<detail::HookInfo>& infos, std::uint32_t owner) {
        using How = detail::DimRecord::How;
        using Kind = detail::HookInfo::Kind;
        detail::VarMeta& m   = *inst.meta;
        const auto       own = [&](std::size_t u) {
            detail::DimRecord& r = m.definition[u];
            if (r.how == How::unspecified) {
                r.how        = How::hook;
                r.hook_owner = owner;
            }
        };
        if (infos.size() == 1 && infos[0].is_ragged()) {
            own(0);
            own(1);
        } else {
            for (std::size_t u = 0; u < infos.size(); ++u) {
                const detail::HookInfo& d = infos[u];
                if (d.kind == Kind::as_is) continue;
                if (d.kind == Kind::fixed_at_run) {
                    detail::DimRecord& r = m.definition[u];
                    if (r.how == How::unspecified) {
                        r.how    = How::fixed;
                        r.extent = d.index;
                    }
                } else {
                    own(u);
                }
            }
        }
        if (inst.kind == IOKind::input) {
            inst.can_be_target = false;  // a hook and a connection exclude each other
            inst.hooked        = true;
        }
    }

    /// What install_hook found: the target and, per descriptor, the instance of the source it names (null for the others).
    struct HookTarget {
        std::shared_ptr<detail::Instance>              inst;
        std::vector<std::shared_ptr<detail::Instance>> srcs;
    };

    /// Validates a hook declaration and marks the definition. The target is an input, a tie alias of either kind, or an
    /// ordinary untied output; a member of a tie_outputs is refused (hook the tie alias instead).
    HookTarget install_hook(const std::string& path, const std::vector<detail::HookInfo>& infos, std::uint32_t owner) {
        using How  = detail::DimRecord::How;
        using Kind = detail::HookInfo::Kind;
        const auto inst = instance_at(path);
        const auto bad  = [&](const std::string& why) { fail("shape_hook", "'" + path + "': " + why); };
        if (inst->kind == IOKind::output && !detail::is_tie_output(*inst) &&
            (inst->tied_member || (inst->role == detail::Role::port && inst->meta->tied_output))) {
            std::string alias = "its tie alias";
            for (const auto& i : inst->meta->live_instances())
                if (detail::is_tie_output(*i)) alias = "'" + i->full_path() + "'";
            bad("it is a member of a tie_outputs: its shape belongs to the tie; hook the tie alias instead (" + alias + ")");
        }
        detail::VarMeta& m = *inst->meta;
        // An input that is already hooked was never connected or tied (connect refuses it), so for it the per-dimension
        // ownership check below answers: a second hook on a dimension gets "only one hook owns it", not the connected message.
        if (inst->kind == IOKind::input && !inst->hooked && (!inst->can_be_target || detail::first_of_kind(m, IOKind::output)))
            bad("it is connected or tied; a shape hook and a connection exclude each other");
        const bool ragged_target = m.is_ragged() && m.rank == 2;
        if (ragged_target) {
            // ONE ragged descriptor for both dimensions, or a dense descriptor for N alone (the counts left as_is)
            const bool one_ragged = infos.size() == 1 && infos[0].is_ragged();
            const bool n_only     = infos.size() == 2 && !infos[0].is_ragged() && infos[1].kind == Kind::as_is;
            if (!one_ragged && !n_only)
                bad("a ragged target takes ONE ragged descriptor (cs::ragged::counts_from or offsets_from) for both dimensions, or a dense "
                    "descriptor for N and cs::dim::as_is() for the counts");
        } else {
            if (infos.size() != static_cast<std::size_t>(m.rank))
                bad("it has rank " + std::to_string(m.rank) + " but " + std::to_string(infos.size()) + " descriptors were given");
            for (const detail::HookInfo& d : infos)
                if (d.is_ragged()) bad("a ragged descriptor (counts_from, offsets_from) is for a ragged target");
        }
        const auto need_unspecified = [&](std::size_t u) {
            if (m.definition[u].how != How::unspecified)
                bad("dimension " + std::to_string(u) + " is " + m.declared_rule(static_cast<int>(u)) +
                    "; a hook can only own an unspecified dimension (and only one hook owns it)");
        };
        bool any = false;
        if (ragged_target && infos.size() == 1) {
            need_unspecified(1);
            if (m.definition[0].how == How::hook) bad("dimension 0 is already set by a shape hook");
            any = true;
        } else {
            for (std::size_t u = 0; u < infos.size(); ++u) {
                if (infos[u].kind == Kind::as_is) continue;
                need_unspecified(u);
                any = true;
            }
        }
        if (!any) bad("every descriptor is as_is: nothing to hook");
        HookTarget out;
        out.inst = inst;
        for (const detail::HookInfo& d : infos) {
            if (!d.names_source()) {
                out.srcs.push_back(nullptr);
                continue;
            }
            const auto up = instance_at(d.path);
            if (d.kind != Kind::from_shape && !detail::is_integer_value_dtype(up->meta->dtype))
                bad("'" + d.path + "' drives a size by value and must have an integer dtype");
            if (d.kind == Kind::from_shape && d.index >= static_cast<std::size_t>(up->meta->rank))
                bad("'" + d.path + "' has rank " + std::to_string(up->meta->rank) + ", no dimension " + std::to_string(d.index));
            out.srcs.push_back(up);
        }
        mark_hooked(*inst, infos, owner);
        return out;
    }

    // ---- marks ---------------------------------------------------------------------------
    /// Every dimension is specified (a ragged variable's counts are not a single extent: it is never all fixed).
    static bool all_fixed(const detail::VarMeta& v) {
        return !v.def_counts && std::all_of(v.definition.begin(), v.definition.end(),
                                            [](const detail::DimRecord& r) { return r.how == detail::DimRecord::How::fixed; });
    }
    static std::size_t flat_size(const detail::VarMeta& v) {
        std::size_t n = 1;
        for (const detail::DimRecord& r : v.definition) n *= r.extent;
        return n;
    }

    /// The last mark of a kind on a variable replaces the earlier one. Marks are kept by absolute path in the metadata.
    void store_mark(MarkKind kind, std::string_view rel, const std::shared_ptr<detail::Instance>& inst, MarkRecord rec) {
        detail::ModelMeta& m = meta();
        std::vector<Mark>& v = kind == MarkKind::design_var ? m.design_vars : kind == MarkKind::constraint ? m.constraints : m.objectives;
        Mark d{detail::ModelMeta::child_name(path(), rel), inst, std::move(rec)};
        for (Mark& e : v) {
            const auto o = e.inst.lock();
            if (o && o->meta == inst->meta) {
                e = std::move(d);
                return;
            }
        }
        v.push_back(std::move(d));
    }

    /// Bounds, equals and scaling are per-element or scalar (size 1 broadcasts).
    void check_sizes(std::size_t n, const Bounds& b, const std::optional<std::vector<double>>& eq, const Scaling& s,
                     const char* what, std::string_view path) const {
        auto ok  = [&](const std::vector<double>& v) { return v.size() == 1 || v.size() == n; };
        auto bad = [&](const char* which) {
            fail(what, std::string{"'"} + std::string{path} + "': " + which + " must have 1 or " + std::to_string(n) +
                           " values (the variable's flat size)");
        };
        if (b.lower && !ok(*b.lower)) bad("lower");
        if (b.upper && !ok(*b.upper)) bad("upper");
        if (eq && !ok(*eq)) bad("equals");
        if (!ok(s.ref)) bad("scaling ref");
        if (!ok(s.ref0)) bad("scaling ref0");
    }

    // ---- state ---------------------------------------------------------------------------
    std::vector<std::shared_ptr<detail::Instance>> alias_order_;  ///< this System owns its aliases
};

}  // namespace naga::cospan
