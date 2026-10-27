#pragma once

// =============================================================================
// naga/cospan/model_meta.hpp — the model-wide metadata object (r04)
// =============================================================================
//
// internal/spec/revisions/r04-2026-10-11-flat-model-metadata.md. ONE object per model holds the structure:
//
//   THE TREE         Nodes, owned here. A system node holds its children in declaration order; a leaf (a component or a
//                    fused group) has none. Each node owns its subsystem object. The run walks this tree with an
//                    explicit stack (model.hpp). Every leaf gets a depth-first ordinal when it is registered, and a
//                    system node knows the range of ordinals under it.
//   FLAT MAPS        full name -> node, and full name -> holder (the instances of inputs, outputs, aliases and fused-group
//                    stage ports). A lookup never walks the tree. Names are never parsed: a resolved name is the
//                    parent's name, a ".", and the local name (the root's is "" and adds no dot); dots inside a local
//                    name are allowed. Every full name is unique across the model.
//   RECORDS          what finalize needs: the tied outputs, the shape hooks, the depends_on edges, the marks.
//   STATE            the lifecycle (created / editable / finalized), the cycle-warning set and the pending-
//                    materialization set.
//
// Components and Systems keep owning their holders (shared_ptr); the maps refer to them weakly, so the hot path is
// unchanged. A reset (Model::setup called again) replaces the whole object.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/instance.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/shape_hook.hpp"
#include "naga/cospan/solver.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"

namespace naga::cospan {

class System;
class ComponentBase;  // component.hpp

/// One mark, by absolute path, collected as the marks are declared. The instance is held weakly: a mark on a holder
/// that is gone expires.
struct Mark {
    std::string                     path;
    std::weak_ptr<detail::Instance> inst;
    MarkRecord                      rec;
};

namespace detail {

class RunPolicy;  // run_policy.hpp

/// The one writer of a subsystem's name and metadata pointer.
struct SubsystemAccess {
    static void bind(Subsystem& s, std::string name, ModelMeta* m) {
        s.name_ = std::move(name);
        s.meta_ = m;
    }
};

enum class NodeKind : std::uint8_t { system, leaf };
enum class Lifecycle : std::uint8_t { created, editable, finalized };

/// A step run before or after a node (a shape hook, a tied-output push, a depends_on handoff). `kind` is for diagnostics.
struct Operation {
    const char*                   kind;
    std::function<void(Runtime&)> fn;
};

struct Node {
    std::string                name;      ///< the full resolved name
    std::unique_ptr<Subsystem> sub;       ///< the node owns the object
    System*                    sys = nullptr;  ///< the same object as a System (system nodes only)
    NodeKind                   kind = NodeKind::leaf;
    Node*                      parent = nullptr;
    std::vector<Node*>         children;  ///< declaration order

    /// A leaf: its depth-first ordinal. A system: the ordinal of the first leaf registered under it (valid when
    /// has_leaves), and last_ordinal the ordinal of the last.
    std::size_t ordinal      = 0;
    std::size_t last_ordinal = 0;
    bool        has_leaves   = false;

    bool                        in_scope = true;
    std::shared_ptr<RunPolicy>  policy;
    std::shared_ptr<Solver>     solver;      ///< a system node: solves the cycles it owns (solver.hpp)
    std::optional<SolveResult>  last_solve;  ///< the last solve at this node
    std::vector<Operation>      before, after;
    std::optional<sycl::device> device_override;       ///< set_device, applied at finalize
    std::vector<std::weak_ptr<Instance>> holders;     ///< the instances this node owns (a system's aliases, a leaf's ports)
    std::vector<sycl::event>             extra_events;  ///< this turn's events of the leaf's after-operations (tied pushes)
};

/// One declared tie_outputs, for finalize (its push and its one-device check).
struct TieOutRecord {
    std::string                       owner;    ///< the declaring system's full name
    std::shared_ptr<Instance>         alias;
    std::vector<std::string>          members;  ///< full names
};

/// One shape_hook call, for finalize (its placement).
struct HookRecord {
    std::unique_ptr<HookBase>                    step;
    std::string                                  target;  ///< full name
    std::vector<std::string>                     sources; ///< full names of the instances the descriptors name
    std::weak_ptr<Instance>                      target_inst;
    std::vector<std::weak_ptr<Instance>>         source_insts;
};

/// One depends_on(b, a) call, for finalize.
struct DependsRecord {
    std::string b, a;  ///< full names of nodes
};

struct ModelMeta {
    ModelMeta()                            = default;
    ModelMeta(const ModelMeta&)            = delete;
    ModelMeta& operator=(const ModelMeta&) = delete;

    // ---- state -----------------------------------------------------------------------
    Lifecycle state       = Lifecycle::created;
    int       setup_depth = 0;  ///< > 0 while some setup() is running

    // ---- the tree and the maps -------------------------------------------------------
    std::unordered_map<std::string, std::unique_ptr<Node>>   nodes;
    Node*                                                    root = nullptr;
    std::unordered_map<const Subsystem*, Node*>              node_of_sub;  ///< a node's object, and a fused group's stage objects
    std::unordered_map<std::string, std::weak_ptr<Instance>> holders;
    std::unordered_set<std::string>                          stage_names;  ///< a fused group's stages: objects, not tree nodes
    /// Every bound component (a lone component and a fused group's stage) by full name, filled by ComponentBase::bind. Raw: the
    /// component's driver owns it.
    std::unordered_map<std::string, ComponentBase*>          components;
    std::size_t                                              next_ordinal = 0;

    // ---- records for finalize --------------------------------------------------------
    std::vector<TieOutRecord>  tie_outs;
    std::vector<HookRecord>    hooks;
    std::vector<DependsRecord> depends;
    std::uint32_t              next_hook_owner = 0;

    // ---- marks (by absolute path) ----------------------------------------------------
    std::vector<Mark> design_vars, constraints, objectives;

    // ---- warnings and materialization ------------------------------------------------
    std::set<std::string>                  warned_cycles;
    std::vector<std::weak_ptr<VarMeta>>    pending;      ///< variables to visit at the start of the next run
    std::unordered_set<const VarMeta*>     pending_set;  ///< dedupe

    // ---- names -----------------------------------------------------------------------
    /// The full resolved name of `local` under `parent`: the root ("") adds no dot. Names are never parsed.
    static std::string child_name(const std::string& parent, std::string_view local) {
        if (parent.empty()) return std::string{local};
        std::string out;
        out.reserve(parent.size() + 1 + local.size());
        out += parent;
        out += '.';
        out += local;
        return out;
    }

    static void check_local_name(std::string_view local, const std::string& where, const char* what) {
        if (local.empty())
            throw std::invalid_argument("naga::cospan: " + where + ": the " + what + " name must be non-empty");
    }

    bool name_taken(const std::string& full) const {
        return nodes.contains(full) || holders.contains(full) || stage_names.contains(full) || components.contains(full);
    }

    // ---- registration ----------------------------------------------------------------
    /// Registers `sub` as a child of `parent` (null for the root) under the full name `full`: binds its name and the
    /// metadata, creates its node, appends it to the parent's children and, for a leaf, assigns the next depth-first
    /// ordinal (and widens its ancestors' ranges). Throws std::invalid_argument for a duplicate full name.
    Node& register_node(Node* parent, std::string full, std::unique_ptr<Subsystem> sub, NodeKind kind, System* sys,
                        const std::string& where) {
        if (name_taken(full))
            throw std::invalid_argument("naga::cospan: " + where + ": '" + full + "' is already a subsystem or alias name in this model");
        auto node  = std::make_unique<Node>();
        Node* raw  = node.get();
        raw->name  = full;
        raw->kind  = kind;
        raw->sys   = sys;
        raw->parent = parent;
        SubsystemAccess::bind(*sub, full, this);
        node_of_sub[sub.get()] = raw;
        raw->sub               = std::move(sub);
        if (kind == NodeKind::leaf) {
            raw->ordinal = raw->last_ordinal = next_ordinal++;
            raw->has_leaves                  = true;
        } else {
            raw->ordinal      = next_ordinal;  // the first leaf registered under it, if any
            raw->last_ordinal = next_ordinal;
        }
        if (parent != nullptr) parent->children.push_back(raw);
        nodes.emplace(std::move(full), std::move(node));
        if (kind == NodeKind::leaf)
            for (Node* a = parent; a != nullptr; a = a->parent) {
                if (!a->has_leaves) a->ordinal = raw->ordinal;
                a->has_leaves   = true;
                a->last_ordinal = raw->ordinal;
            }
        if (root == nullptr) root = raw;
        return *raw;
    }

    /// An object owned by a leaf that is not a tree node of its own (a fused group's stage): it answers to the leaf.
    void adopt_sub(const Subsystem* sub, Node& leaf) { node_of_sub[sub] = &leaf; }

    /// Registers `inst` as a holder under `full` (the owner's full name + "." + the instance's local name). Throws
    /// std::invalid_argument for a duplicate.
    void register_holder(const std::string& full, const std::shared_ptr<Instance>& inst, Node* owner, const std::string& where) {
        if (name_taken(full))
            throw std::invalid_argument("naga::cospan: " + where + ": '" + full + "' is already a subsystem or alias name in this model");
        holders.emplace(full, inst);
        if (owner != nullptr) owner->holders.push_back(inst);
    }

    // ---- lookups (one map lookup each) -------------------------------------------------
    Node* find_node(std::string_view full) {
        const auto it = nodes.find(std::string{full});
        return it == nodes.end() ? nullptr : it->second.get();
    }
    const Node* find_node(std::string_view full) const {
        const auto it = nodes.find(std::string{full});
        return it == nodes.end() ? nullptr : it->second.get();
    }
    std::shared_ptr<Instance> find_holder(std::string_view full) const {
        const auto it = holders.find(std::string{full});
        return it == holders.end() ? nullptr : it->second.lock();
    }
    /// The node of the object (a node's own object, or a fused group's stage), else null.
    Node* node_for(const Subsystem* sub) const {
        const auto it = node_of_sub.find(sub);
        return it == node_of_sub.end() ? nullptr : it->second;
    }
    /// The node a holder belongs to (its declarer's node), else null (a host view has none).
    Node* node_of(const Instance& inst) const { return inst.declarer == nullptr ? nullptr : node_for(inst.declarer); }

    // ---- scope -----------------------------------------------------------------------
    /// The node and all its ancestors are in scope.
    static bool effective_scope(const Node& n) {
        for (const Node* p = &n; p != nullptr; p = p->parent)
            if (!p->in_scope) return false;
        return true;
    }
    /// A holder is in scope when its owner node is (a view, which has no node, always is).
    bool holder_in_scope(const Instance& inst) const {
        const Node* n = node_of(inst);
        return n == nullptr || effective_scope(*n);
    }

    // ---- ordinals --------------------------------------------------------------------
    /// The ordinal of the leaf that owns a holder (a system's alias has none).
    std::optional<std::size_t> leaf_ordinal(const Instance& inst) const {
        const Node* n = node_of(inst);
        if (n == nullptr || n->kind != NodeKind::leaf) return std::nullopt;
        return n->ordinal;
    }

    /// The last producer of a variable: the leaf with the highest ordinal that holds an output port of the variable
    /// (tie members included; promotions share the variable). Null when no leaf produces it.
    Node* last_producer(VarMeta& var) {
        Node* best = nullptr;
        for (const auto& i : var.live_instances()) {
            if (i->kind != IOKind::output || i->role != Role::port) continue;
            Node* n = node_of(*i);
            if (n == nullptr || n->kind != NodeKind::leaf) continue;
            if (best == nullptr || n->ordinal > best->ordinal) best = n;
        }
        return best;
    }
    /// The first leaf (lowest ordinal) that holds any port of a variable, input or output. Null when none does.
    Node* first_holder(VarMeta& var) {
        Node* best = nullptr;
        for (const auto& i : var.live_instances()) {
            if (i->role != Role::port) continue;
            Node* n = node_of(*i);
            if (n == nullptr || n->kind != NodeKind::leaf) continue;
            if (best == nullptr || n->ordinal < best->ordinal) best = n;
        }
        return best;
    }

    // ---- pending materialization -------------------------------------------------------
    void mark_pending(const std::shared_ptr<VarMeta>& v) {
        if (!v) return;
        if (pending_set.insert(v.get()).second) pending.push_back(v);
    }

    // ---- the lifecycle guards ----------------------------------------------------------
    /// A declaration call is legal inside some setup(), or after Model::setup and before finalize.
    void require_declaring(const char* what, const std::string& where) const {
        if (state == Lifecycle::finalized)
            throw std::logic_error(std::string{"naga::cospan: "} + where + ": " + what + " is illegal after finalize: the structure is fixed");
        if (state == Lifecycle::created && setup_depth == 0)
            throw std::logic_error(std::string{"naga::cospan: "} + where + ": " + what + " is illegal before the model's setup()");
    }
    /// add_subsystem is legal inside some setup() only (the Model's own add_subsystem, before setup(), does not come here).
    void require_in_setup(const char* what, const std::string& where) const {
        if (setup_depth <= 0)
            throw std::logic_error(std::string{"naga::cospan: "} + where + ": " + what + " is legal inside setup() only");
    }

    /// RAII: a setup() is running.
    struct SetupScope {
        explicit SetupScope(ModelMeta* m) : m_{m} {
            if (m_ != nullptr) ++m_->setup_depth;
        }
        ~SetupScope() {
            if (m_ != nullptr) --m_->setup_depth;
        }
        SetupScope(const SetupScope&)            = delete;
        SetupScope& operator=(const SetupScope&) = delete;
        ModelMeta*  m_;
    };
};

/// Places the holders of `n` and of everything under it on `d` (an author's or a caller's placement; the outermost one wins
/// when finalize applies them again from the root).
inline void apply_device_node(Node& n, const sycl::device& d) {
    for (const std::weak_ptr<Instance>& w : n.holders) {
        const std::shared_ptr<Instance> inst = w.lock();
        if (!inst) continue;
        inst->device = d;
        inst->refresh_container();  // the container of the new placement
        // A tie alias moves its initializer with it. A free input port moves its own (a port that is a member of a tie has the
        // tie's variable, which keeps the tie's device; a promotion changes nothing).
        if (inst->kind == IOKind::input && inst->meta->init_device) {
            if (inst->role == Role::tie) inst->meta->init_device = d;
            else if (inst->role == Role::port && !inst->meta->tied_input) inst->meta->init_device = d;
        }
    }
}
inline void apply_device_subtree(Node& root, const sycl::device& d) {
    std::vector<Node*> stack{&root};
    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        apply_device_node(*n, d);
        for (Node* c : n->children) stack.push_back(c);
    }
}

/// The start-of-run pass over the pending variables (those whose state changed since the last run: finalize fills it with
/// every variable, update_input and init_input after the first run and rearm add theirs). A variable whose shape is already
/// settled is materialized now, on every in-scope placement (outputs included; an unconnected input also initializes); one
/// whose shape is not (a hooked dimension, an output whose producer sets it) is left to the first holder that touches it.
/// The set is empty afterwards.
inline void materialize_pending(ModelMeta& m, Runtime& rt) {
    std::vector<std::weak_ptr<VarMeta>> work;
    work.swap(m.pending);
    m.pending_set.clear();
    for (const std::weak_ptr<VarMeta>& w : work) {
        const std::shared_ptr<VarMeta> v = w.lock();
        if (!v || !v->shape_resolvable()) continue;
        bool any_in_scope = false;
        for (const std::shared_ptr<Instance>& i : v->live_instances())
            if (!i->dearmed) any_in_scope = true;
        if (!any_in_scope) continue;
        v->materialize(rt, nullptr);
    }
}

}  // namespace detail
}  // namespace naga::cospan
