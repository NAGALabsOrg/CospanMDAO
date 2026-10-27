#pragma once

// =============================================================================
// naga/cospan/instance.hpp — one declared input, output or alias
// =============================================================================
//
// Every input, output and alias is an INSTANCE, owned by exactly one component or System through a
// shared_ptr (internal/design-review-2026-10-03.md, section 1). Nothing outside the owner holds an instance
// strongly, and there is no global record of anything: no roots, no shadows, no ledgers. An instance is
// generic and passive. It holds:
//
//   - the variable's shared metadata (VarMeta, held by every instance of the variable);
//   - its own device (section 2: devices are never on the metadata) and storage kind;
//   - a shared pointer to the data (the container) of ITS placement: created empty when the instance is declared
//     (or when its device or kind is set, or it joins another variable) and kept for its whole life;
//   - its connect flags (can it be a source, can it be a target);
//   - for a port, the wrapper its component's kernel reads (the typed leaves below).
//
// When an instance goes away it simply stops holding its data and the metadata: the metadata's weak
// entry for it expires. There is nothing to unregister.
//
// The rules about which variables merge (connect, tie, promote) and who wins live in those calls, a later
// phase; rebind() is the one operation they need from here.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/derivative_error.hpp"
#include "naga/cospan/dispatch.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {
namespace detail {

/// What an instance is: a component's port, a System's promotion, a System's tie alias, or a host view.
enum class Role : uint8_t { port, promotion, tie, host_view };

/// Row-major strides of a 1-padded extents triple. For a rank-R instance the first R strides are exactly
/// rank-R row-major strides.
constexpr std::array<std::size_t, 3> contiguous_strides(const std::array<std::size_t, 3>& e) {
    return {e[1] * e[2], e[2], 1};
}

/// What a wrapper is pointed at: shape and stride, 1-padded.
struct Geometry {
    std::array<std::size_t, 3> shape{1, 1, 1};
    std::array<std::size_t, 3> stride{1, 1, 1};
};

/// The only writer of wrapper fields (friendship; wrappers.hpp): user code never sees a pointer.
struct WrapperAccess {
    template <class W, class P>
    static void write_wrapper(W& w, const Geometry& g, P* data) {
        w.shape_  = make_range<W::rank>(g.shape);
        w.stride_ = make_range<W::rank>(g.stride);
        w.data_   = data;
    }

    template <class W, class P>
    static void write_ragged(W& w, std::size_t n, P* data, const std::uint32_t* offsets,
                             const std::array<std::size_t, 2>& ext, const std::array<const std::uint32_t*, 2>& per) {
        w.n_       = n;
        w.data_    = data;
        w.offsets_ = offsets;
        w.ext_     = ext;
        w.per_     = per;
    }
};

class Instance : public std::enable_shared_from_this<Instance> {
public:
    Instance(std::string n, std::string owner, const Subsystem* decl, IOKind k, Role r, std::vector<DimRecord> d,
             bool can_source, bool can_target)
        : name{std::move(n)}
        , owner_path{std::move(owner)}
        , declarer{decl}
        , kind{k}
        , role{r}
        , dims{std::move(d)}
        , can_be_source{can_source}
        , can_be_target{can_target} {}

    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    virtual ~Instance()                  = default;

    std::string            name;
    /// The full name of the component (or the System's path) that owns this instance: the first part of full_path().
    std::string            owner_path;
    /// The NODE object this instance answers to: for a component's holders the driver (the leaf in the tree), for a System's
    /// aliases the System, for a host view null.
    const Subsystem*       declarer;
    IOKind                 kind;
    Role                   role;
    std::vector<DimRecord> dims;  ///< as declared HERE; the variable's own definition is on the metadata
    /// A ragged port's declared per-index counts (cs::counts / cs::offsets), as declared HERE; the variable's own are
    /// VarMeta::def_counts.
    std::optional<std::vector<std::size_t>> def_counts;
    /// A shape hook targets this input (section 10): it cannot be connected or tied (can_be_target is cleared with it).
    bool hooked = false;
    /// A member of a tie_outputs: it cannot set its output's shape (the tie's System owns it, through a hook on the alias) and
    /// cannot be a hook target. Set by the tie call.
    bool tied_member = false;
    /// A tie alias that became a member of another tie (ties of ties): the outermost alias is the one that pushes and
    /// that owns the shape; an absorbed alias keeps its name and its variable.
    bool absorbed = false;
    /// The holder's node is out of scope (dearmed): it holds no data and no materialization gives it any until it is rearmed.
    bool dearmed = false;

    sycl::device device       = default_device();  ///< section 2: set_device overrides it
    sk::kind     storage_kind = sk::kind::device;

    bool can_be_source;
    bool can_be_target;

    std::shared_ptr<VarMeta>         meta;  ///< shared by every instance of the variable
    std::shared_ptr<DeviceDataBase>  data;  ///< the data of THIS instance's placement: empty until materialized (a holder)

    /// The instance's dotted path from the root System: its owner's full name plus its name, or the name alone at the root.
    std::string full_path() const { return owner_path.empty() ? name : owner_path + "." + name; }

    /// Where this instance's data lives.
    DataKey key() const { return DataKey{device, storage_kind}; }

    /// Takes the container of this instance's current key from its variable (a new empty one if the variable has no live
    /// one for it). Called whenever the key or the variable changes: declaration, set_device, a merge. The variable keeps
    /// the container weakly, so the holders are what keeps it alive.
    void refresh_container() { data = meta ? meta->container_for(key()) : nullptr; }

    /// Joins another variable: takes the metadata, enters its instance list, and takes the container of its key there
    /// (the winner's, or one the winner adopted from the loser: VarMeta::adopt_containers). The merge walks of connect,
    /// tie and promote re-point the instances of the losing metadata with this.
    void rebind(std::shared_ptr<VarMeta> m) {
        meta = std::move(m);
        if (meta) meta->add_instance(weak_from_this());
        refresh_container();
    }

    /// Writes the component's member wrapper from the current shape and data. Aliases have none.
    virtual void point_wrapper() {}

    /// Derivatives: writes the member-type wrapper at `wrapper` over the buffer `data` (a float array), with this variable's
    /// shape. `wrapper` points at an Input<f32,R> / Output<f32,R> of this port's type.
    virtual void point_wrapper_into(void* /*wrapper*/, void* /*data*/) const {
        throw std::logic_error("naga::cospan: '" + full_path() + "' has no member wrapper");
    }

    /// A dense Input port (not RaggedInput): it cannot see per-index extents (wiring.hpp, connect).
    virtual bool is_dense_input_port() const { return false; }
};

// What variable.hpp declared for VarMeta's sake.
inline DataKey             key_of(const Instance& i) { return i.key(); }
inline std::shared_ptr<DeviceDataBase> data_of(const Instance& i) { return i.data; }
inline void                refresh_container(Instance& i) { i.refresh_container(); }
inline bool                is_dearmed(const Instance& i) { return i.dearmed; }
inline IOKind              kind_of(const Instance& i) { return i.kind; }
inline bool                is_output_port(const Instance& i) { return i.kind == IOKind::output && i.role == Role::port; }
inline std::string         path_of(const Instance& i) { return i.full_path(); }
inline const sycl::device& device_of(const Instance& i) { return i.device; }
inline std::shared_ptr<VarMeta> meta_of(const Instance& i) { return i.meta; }
inline bool                is_tie_output(const Instance& i) { return i.kind == IOKind::output && i.role == Role::tie; }

// -----------------------------------------------------------------------------
// Typed leaves: the instances that hold a member wrapper
// -----------------------------------------------------------------------------

template <DType DT, int Rank>
class TypedInstance : public Instance {
public:
    using value_type = cpp_type_t<DT>;
    using Instance::Instance;

protected:
    /// A dense wrapper: the variable's shape with contiguous strides over this instance's data (null while it
    /// has none).
    template <class W>
    void point(W& w) {
        const ResolvedShape& s = meta->shape;
        if (s.ragged())
            throw std::logic_error("naga::cospan: '" + full_path() + "' is ragged; declare it with RaggedInput / "
                                   "RaggedOutput (a dense Input / Output cannot see per-index extents)");
        Geometry g;
        if (s.known) {
            g.shape  = s.extents;
            g.stride = contiguous_strides(s.extents);
        } else {
            g.shape  = {0, 0, 0};
            g.stride = {0, 0, 0};
        }
        auto* base = data ? static_cast<value_type*>(data->raw()) : nullptr;
        WrapperAccess::write_wrapper(w, g, base);
    }

    /// The derivative form of point(): the same geometry over a float buffer; f32 only, dense only.
    template <class W>
    void point_deriv(W& w, void* buf) const {
        if constexpr (DT == DType::f32) {
            const ResolvedShape& s = meta->shape;
            if (s.ragged())
                throw DerivativeError("naga::cospan: ragged variable on a derivative path: '" + full_path() + "'");
            Geometry g;
            if (s.known) {
                g.shape  = s.extents;
                g.stride = contiguous_strides(s.extents);
            } else {
                g.shape  = {0, 0, 0};
                g.stride = {0, 0, 0};
            }
            WrapperAccess::write_wrapper(w, g, static_cast<float*>(buf));
        } else {
            (void)w;
            (void)buf;
            throw DerivativeError("naga::cospan: derivatives need f32: '" + full_path() + "'");
        }
    }

    /// A ragged wrapper: the data, the block count, and this instance's device copies of the index arrays. A DENSE
    /// variable is accepted too (offsets null: block i is row i of the rectangle).
    template <class W>
    void point_ragged(W& w) {
        const ResolvedShape&                s    = meta->shape;
        auto*                               base = data ? static_cast<value_type*>(data->raw()) : nullptr;
        std::array<std::size_t, 2>          ext{1, 1};
        std::array<const std::uint32_t*, 2> per{nullptr, nullptr};
        const std::uint32_t*                offs = nullptr;
        if (s.ragged()) {  // rank 2: dimension 1 is per-index
            const auto* rd = dynamic_cast<const RaggedDeviceData<value_type>*>(data.get());
            if (rd != nullptr && rd->per) per[0] = rd->per->data();
            if (rd != nullptr && rd->offs) offs = rd->offs->data();
        } else {
            for (int d = 1; d < Rank; ++d) ext[static_cast<std::size_t>(d - 1)] = s.extents[static_cast<std::size_t>(d)];
        }
        WrapperAccess::write_ragged(w, s.extents[0], base, offs, ext, per);
    }
};

/// An input: can be a target, cannot be a source.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class InputInstance final : public TypedInstance<DT, Rank> {
public:
    InputInstance(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                  Input<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), std::move(owner_path), declarer, IOKind::input, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), false, true}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }
    void point_wrapper_into(void* wrapper, void* buf) const override {
        this->point_deriv(*static_cast<Input<DT, Rank>*>(wrapper), buf);
    }
    Input<DT, Rank>& member() const { return *member_; }
    bool is_dense_input_port() const override { return true; }

private:
    Input<DT, Rank>* member_;
};

/// An output: can be a source, cannot be a target.
template <DType DT, int Rank>
class OutputInstance final : public TypedInstance<DT, Rank> {
public:
    OutputInstance(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                   Output<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), std::move(owner_path), declarer, IOKind::output, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), true, false}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }
    void point_wrapper_into(void* wrapper, void* buf) const override {
        this->point_deriv(*static_cast<Output<DT, Rank>*>(wrapper), buf);
    }
    Output<DT, Rank>& member() const { return *member_; }

private:
    Output<DT, Rank>* member_;
};

template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class RaggedInputInstance final : public TypedInstance<DT, Rank> {
public:
    RaggedInputInstance(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                        RaggedInput<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), std::move(owner_path), declarer, IOKind::input, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), false, true}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }
    void point_wrapper_into(void*, void*) const override {
        throw DerivativeError("naga::cospan: ragged variable on a derivative path: '" + this->full_path() + "'");
    }

private:
    RaggedInput<DT, Rank>* member_;
};

template <DType DT, int Rank>
class RaggedOutputInstance final : public TypedInstance<DT, Rank> {
public:
    RaggedOutputInstance(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                         RaggedOutput<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), std::move(owner_path), declarer, IOKind::output, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), true, false}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }
    void point_wrapper_into(void*, void*) const override {
        throw DerivativeError("naga::cospan: ragged variable on a derivative path: '" + this->full_path() + "'");
    }

private:
    RaggedOutput<DT, Rank>* member_;
};

/// A System-level alias (a promotion or a tie alias): no member wrapper. Its flags are given.
template <DType DT, int Rank>
class AliasInstance final : public TypedInstance<DT, Rank> {
public:
    AliasInstance(std::string name, std::string owner_path, const Subsystem* declarer, IOKind k, Role role, std::vector<DimRecord> dims,
                  bool can_source, bool can_target)
        : TypedInstance<DT, Rank>{std::move(name), std::move(owner_path), declarer, k, role, std::move(dims), can_source, can_target} {}
};

// -----------------------------------------------------------------------------
// Factories: each makes the instance and, unless told to join a variable, a variable of its own
// -----------------------------------------------------------------------------

/// Gives a new instance its own shared metadata (ragged when the port is) and enters it in that metadata's instance
/// list. The definition is the instance's declared dims, and a ragged port's declared counts.
inline void adopt_new_meta(const std::shared_ptr<Instance>& inst, DType dt, int rank, bool ragged) {
    inst->meta             = make_var_meta(dt, rank, inst->dims, ragged);
    inst->meta->def_counts = inst->def_counts;
    inst->meta->add_instance(inst);
    inst->refresh_container();  // its (still default) placement; refreshed again when the device or kind is set
    // A free input initializes on the device it was declared on (set_device keeps this in step: apply_device_node).
    if (inst->kind == IOKind::input) inst->meta->init_device = inst->device;
}

/// The caller owns the returned instance (a component keeps its ports; a System its aliases).
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
std::shared_ptr<Instance> make_input(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                                     Input<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<InputInstance<DT, Rank>>(std::move(name), std::move(owner_path), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, false);
    return inst;
}

template <DType DT, int Rank>
std::shared_ptr<Instance> make_output(std::string name, std::string owner_path, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                                      Output<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<OutputInstance<DT, Rank>>(std::move(name), std::move(owner_path), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, false);
    return inst;
}

template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
std::shared_ptr<Instance> make_ragged_input(std::string name, std::string owner_path, const Subsystem* declarer,
                                            const std::array<DimRecord, Rank>& dims, RaggedInput<DT, Rank>* member,
                                            std::optional<std::vector<std::size_t>> counts = std::nullopt) {
    std::shared_ptr<Instance> inst = std::make_shared<RaggedInputInstance<DT, Rank>>(std::move(name), std::move(owner_path), declarer, dims, member);
    inst->def_counts               = std::move(counts);
    adopt_new_meta(inst, DT, Rank, true);
    return inst;
}

template <DType DT, int Rank>
std::shared_ptr<Instance> make_ragged_output(std::string name, std::string owner_path, const Subsystem* declarer,
                                             const std::array<DimRecord, Rank>& dims, RaggedOutput<DT, Rank>* member,
                                             std::optional<std::vector<std::size_t>> counts = std::nullopt) {
    std::shared_ptr<Instance> inst = std::make_shared<RaggedOutputInstance<DT, Rank>>(std::move(name), std::move(owner_path), declarer, dims, member);
    inst->def_counts               = std::move(counts);
    adopt_new_meta(inst, DT, Rank, true);
    return inst;
}

/// An alias instance of run-time dtype and rank, joining `meta` (a fresh variable of its own when null).
inline std::shared_ptr<Instance> make_alias_instance(DType dt, int rank, std::string name, std::string owner_path, const Subsystem* declarer,
                                                     IOKind kind, Role role, std::vector<DimRecord> dims, bool can_source,
                                                     bool can_target, std::shared_ptr<VarMeta> meta) {
    std::shared_ptr<Instance> inst;
    dispatch_dtype_rank(dt, rank, [&](auto dtc, auto rc) {
        constexpr DType DT = decltype(dtc)::value;
        constexpr int   R  = decltype(rc)::value;
        inst = std::make_shared<AliasInstance<DT, R>>(std::move(name), std::move(owner_path), declarer, kind, role, std::move(dims), can_source,
                                                      can_target);
    });
    if (meta) {
        inst->meta = std::move(meta);
        inst->meta->add_instance(inst);
        inst->refresh_container();
    } else {
        adopt_new_meta(inst, dt, rank, false);
    }
    return inst;
}

/// The writable twin handed to cs::init::custom: the same geometry as a member wrapper, over the variable's
/// data on one device, with the variable's shape.
template <DType DT, int Rank>
Initializer<DT, Rank> make_initializer(VarMeta& var, DeviceDataBase& data) {
    Initializer<DT, Rank> init;
    const ResolvedShape&  s = var.shape;
    Geometry              g;
    if (s.known) {
        g.shape  = s.extents;
        g.stride = contiguous_strides(s.extents);
    } else {
        g.shape  = {0, 0, 0};
        g.stride = {0, 0, 0};
    }
    WrapperAccess::write_wrapper(init, g, static_cast<cpp_type_t<DT>*>(data.raw()));
    return init;
}

}  // namespace detail
}  // namespace naga::cospan
