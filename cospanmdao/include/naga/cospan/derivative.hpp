#pragma once

// =============================================================================
// naga/cospan/derivative.hpp — derivatives, phase 1: the result types, the execution list, activity, the session
// =============================================================================
//
// A component may define static compute_jvp / compute_vjp next to compute_forward (component.hpp). Model::compute_totals
// (model.hpp) chains them over the model's execution list: forward mode seeds a tangent on the free inputs and sweeps the
// components in run order (JVP); reverse mode seeds an adjoint on the outputs and sweeps in reverse order (VJP).
//
// The DerivSession owns the buffers: one zeroed f32 array per ACTIVE variable (reachable from the free inputs asked about AND
// reaching the outputs asked about), and per device a `zero` array (read in place of an inactive variable) and a `sink` array
// (written in place of one). A derivative kernel reads with get / operator[] and writes only with add.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/component.hpp"
#include "naga/cospan/component_driver.hpp"
#include "naga/cospan/derivative_error.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/model_meta.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/variable.hpp"

namespace naga::cospan {

enum class DerivMode { forward, reverse, automatic };

/// One block of a Jacobian: d(of)/d(wrt), row-major, rows = elements of `of`, cols = elements of `wrt`.
struct JacBlock {
    std::size_t         rows = 0, cols = 0;
    std::vector<double> data;
};

class Jacobian {
public:
    /// The block for these two paths exactly as they were given; std::out_of_range if there is none.
    const JacBlock& block(std::string_view of, std::string_view wrt) const {
        const auto it = blocks.find(std::pair<std::string, std::string>{std::string{of}, std::string{wrt}});
        if (it == blocks.end())
            throw std::out_of_range("naga::cospan: Jacobian has no block for of='" + std::string{of} + "' wrt='" + std::string{wrt} + "'");
        return it->second;
    }
    std::map<std::pair<std::string, std::string>, JacBlock> blocks;
};

/// What the last sweep did (tests and diagnostics).
struct DerivStats {
    std::vector<std::string> active_vars;  ///< display paths, sorted
    std::vector<std::string> launched;     ///< component paths of the last sweep, in launch order
    int                      n_jvp = 0, n_vjp = 0;
};

struct CheckOptions {
    double        h_rel      = 1e-2;
    double        rtol       = 2e-2;
    double        atol       = 1e-3;
    int           directions = 3;
    std::uint64_t seed       = 12345;
};

struct PartialsReport {
    bool   has_jvp = false, has_vjp = false;
    double fd_max_rel_err = 0, dot_rel_err = 0;
    bool   fd_pass = false, dot_pass = false;
};

namespace detail {

/// The components of the model in run order: the same depth-first walk as Model::walk, over the nodes in scope. A run policy
/// anywhere in scope throws (phase 1).
inline std::vector<ComponentBase*> build_execution_list(Node& start) {
    std::vector<ComponentBase*> out;
    std::vector<Node*>          stack{&start};
    while (!stack.empty()) {
        Node* n = stack.back();
        stack.pop_back();
        if (!n->in_scope) continue;  // the node and its subtree
        if (n->policy)
            throw DerivativeError("naga::cospan: derivatives do not support run policies yet ('" + n->name + "')");
        if (n->kind == NodeKind::leaf) {
            auto* d = dynamic_cast<ComponentDriver*>(n->sub.get());
            if (d == nullptr) throw std::logic_error("naga::cospan: derivatives: leaf '" + n->name + "' is not a component driver");
            for (ComponentBase* c : d->parts()) out.push_back(c);
            continue;
        }
        for (auto it = n->children.rbegin(); it != n->children.rend(); ++it) stack.push_back(*it);  // declaration order
    }
    return out;
}

/// The whole model.
inline std::vector<ComponentBase*> build_execution_list(ModelMeta& m) { return build_execution_list(*m.root); }

/// The distinct variables a component reads (in) and writes (out), in port order.
struct VarSets {
    std::vector<const VarMeta*> in, out;
};

inline VarSets vars_of(const ComponentBase& c) {
    VarSets v;
    for (const std::shared_ptr<Instance>& inst : ComponentAccess::ports(c)) {
        std::vector<const VarMeta*>& vec = inst->kind == IOKind::input ? v.in : v.out;
        const VarMeta*               m   = inst->meta.get();
        if (std::find(vec.begin(), vec.end(), m) == vec.end()) vec.push_back(m);
    }
    return v;
}

/// Active variables: F (reachable forward from the seeds) intersected with B (reaching the responses backward).
inline std::unordered_set<const VarMeta*> activity(const std::vector<ComponentBase*>&        list,
                                                   const std::unordered_set<const VarMeta*>& seeds,
                                                   const std::unordered_set<const VarMeta*>& responses) {
    std::vector<VarSets> sets;
    sets.reserve(list.size());
    for (const ComponentBase* c : list) sets.push_back(vars_of(*c));

    const auto closure = [&](std::unordered_set<const VarMeta*> reach, bool forward) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const VarSets& s : sets) {
                const std::vector<const VarMeta*>& from = forward ? s.in : s.out;
                const std::vector<const VarMeta*>& to   = forward ? s.out : s.in;
                bool                               hit  = false;
                for (const VarMeta* v : from)
                    if (reach.contains(v)) hit = true;
                if (!hit) continue;
                for (const VarMeta* v : to)
                    if (reach.insert(v).second) changed = true;
            }
        }
        return reach;
    };
    const std::unordered_set<const VarMeta*> f = closure(seeds, true);
    const std::unordered_set<const VarMeta*> b = closure(responses, false);
    std::unordered_set<const VarMeta*>       out;
    for (const VarMeta* v : f)
        if (b.contains(v)) out.insert(v);
    return out;
}

/// The input port instances of the cycle variables Z that belong to a component at or before the variable's producer in
/// `list`: they read the PREVIOUS value of the variable (the one the producer has not yet replaced).
inline std::unordered_set<const Instance*> readers_before(const std::vector<ComponentBase*>&        list,
                                                          const std::unordered_set<const VarMeta*>& Z) {
    std::unordered_set<const Instance*> out;
    for (const VarMeta* z : Z) {
        std::size_t producer = list.size();
        int         count    = 0;
        for (std::size_t i = 0; i < list.size(); ++i) {
            const VarSets v = vars_of(*list[i]);
            if (std::find(v.out.begin(), v.out.end(), z) != v.out.end()) {
                producer = i;
                ++count;
            }
        }
        if (count != 1)
            throw DerivativeError("naga::cospan: cycle variable '" + z->display_path() + "' has " + std::to_string(count) +
                                  " producers on the derivative path");
        for (std::size_t i = 0; i <= producer; ++i)
            for (const std::shared_ptr<Instance>& inst : ComponentAccess::ports(*list[i]))
                if (inst->kind == IOKind::input && inst->meta.get() == z) out.insert(inst.get());
    }
    return out;
}

/// The one gating and capability check of a component that will launch.
inline void check_launchable(const ComponentBase& c, bool forward) {
    if (!c.is_kernel_stage())
        throw DerivativeError("naga::cospan: " + c.path() + ": custom-op and host components have no derivatives");
    if (forward && !c.has_jvp()) throw DerivativeError("naga::cospan: '" + c.path() + "' needs compute_jvp");
    if (!forward && !c.has_vjp()) throw DerivativeError("naga::cospan: '" + c.path() + "' needs compute_vjp");
}

/// The components that launch: at least one active input variable and at least one active output variable. In list order
/// for forward, reversed for reverse.
inline std::vector<ComponentBase*> select(const std::vector<ComponentBase*>& list, const std::unordered_set<const VarMeta*>& active,
                                          bool forward) {
    std::vector<ComponentBase*> out;
    for (ComponentBase* c : list) {
        const VarSets v = vars_of(*c);
        const auto    any_active = [&](const std::vector<const VarMeta*>& vs) {
            return std::any_of(vs.begin(), vs.end(), [&](const VarMeta* m) { return active.contains(m); });
        };
        if (!any_active(v.in) || !any_active(v.out)) continue;
        check_launchable(*c, forward);
        out.push_back(c);
    }
    if (!forward) std::reverse(out.begin(), out.end());
    return out;
}

/// The derivative buffers and the sweep protocol. Owned by the Model's Impl; reset with the model.
class DerivSession final : public DerivBinding {
public:
    struct Buf {
        sycl::device                         dev;
        std::size_t                          n = 0;
        std::optional<sk::usm_vector<float>> v;
    };
    struct Scratch {
        std::size_t                          n = 0;
        std::optional<sk::usm_vector<float>> zero, sink;
    };

    std::unordered_map<const VarMeta*, Buf>        bufs;
    std::unordered_map<sycl::device, Scratch>      scratch;
    std::unordered_set<const VarMeta*>             active;
    DerivStats                                     stats;

    // Cycles (docs/solvers.md): `cut` are the cycle variables of the coupled solve, `prev` their "previous value" buffers (what a
    // reader placed before the producer sees), `reads_prev` those readers' input ports.
    std::unordered_map<const VarMeta*, Buf>        prev;
    std::unordered_set<const VarMeta*>             cut;
    std::unordered_set<const Instance*>            reads_prev;

    /// The buffer a derivative kernel's wrapper for this port is pointed at.
    void* pointer(const Instance& inst, DerivRole role) const override {
        const VarMeta* m = inst.meta.get();
        if (active.contains(m) && reads_prev.contains(&inst)) return const_cast<float*>(prev.at(m).v->data());
        if (active.contains(m)) return const_cast<float*>(bufs.at(m).v->data());
        const Scratch& s = scratch.at(inst.device);
        if (role == DerivRole::tangent_in || role == DerivRole::adjoint_in) return const_cast<float*>(s.zero->data());
        return const_cast<float*>(s.sink->data());
    }

    /// Makes the buffers the launched components need: one per active variable, and the per-device zero and sink arrays.
    void prepare(Runtime& rt, const std::vector<ComponentBase*>& launched) {
        for (const ComponentBase* c : launched) {
            for (const std::shared_ptr<Instance>& inst : ComponentAccess::ports(*c)) {
                const VarMeta* m     = inst->meta.get();
                const std::size_t numel = m->shape.elements();
                const sycl::device& dev = inst->device;
                const auto ensure = [&](std::unordered_map<const VarMeta*, Buf>& map) {
                    auto it = map.find(m);
                    if (it != map.end() && !(it->second.dev == dev))
                        throw DerivativeError("naga::cospan: variable spans devices: '" + m->display_path() + "'");
                    if (it == map.end() || it->second.n != numel) {
                        Buf b;
                        b.dev = dev;
                        b.n   = numel;
                        b.v.emplace(numel, rt.allocator_for<float>(dev, sk::kind::device));
                        map.insert_or_assign(m, std::move(b));
                    }
                };
                if (active.contains(m)) {
                    ensure(bufs);
                    if (cut.contains(m)) ensure(prev);
                }
                Scratch& sc = scratch[dev];
                if (sc.n < numel || !sc.zero) {
                    const std::size_t n = std::max(sc.n, numel);
                    sc.n                = n;
                    sc.zero.emplace(n, rt.allocator_for<float>(dev, sk::kind::device));
                    sc.sink.emplace(n, rt.allocator_for<float>(dev, sk::kind::device));
                    sycl::event e = sc.zero->fill(0.0f);
                    e.wait();
                }
            }
        }
    }

    /// One sweep: zero every active buffer that is not seeded, copy in the seeds, launch the components in the given order (each
    /// after all the earlier ones), wait on the last.
    void sweep(Runtime& rt, const std::vector<ComponentBase*>& launched, bool forward,
               const std::unordered_map<const VarMeta*, std::vector<float>>& seeds,
               const std::unordered_map<const VarMeta*, std::vector<float>>& prev_seeds = {}) {
        stats.launched.clear();
        stats.n_jvp = 0;
        stats.n_vjp = 0;

        std::vector<sycl::event> pre;
        for (auto& [m, b] : bufs) {
            if (!active.contains(m)) continue;
            const auto sd = seeds.find(m);
            if (sd == seeds.end()) {
                pre.push_back(b.v->fill(0.0f));
            } else {
                if (sd->second.size() != b.n) throw std::logic_error("naga::cospan: derivative seed has the wrong size for '" + m->display_path() + "'");
                pre.push_back(b.v->from_host(std::span<const float>{sd->second}));
            }
        }
        for (auto& [m, b] : prev) {
            if (!active.contains(m) || !cut.contains(m)) continue;
            const auto sd = prev_seeds.find(m);
            if (sd == prev_seeds.end()) {
                pre.push_back(b.v->fill(0.0f));
            } else {
                if (sd->second.size() != b.n) throw std::logic_error("naga::cospan: derivative seed has the wrong size for '" + m->display_path() + "'");
                pre.push_back(b.v->from_host(std::span<const float>{sd->second}));
            }
        }
        sycl::event::wait(pre);

        std::vector<sycl::event> evs;
        for (ComponentBase* c : launched) {
            const sycl::event ev = forward ? c->launch_jvp(rt, *this, evs) : c->launch_vjp(rt, *this, evs);
            evs.push_back(ev);
            stats.launched.push_back(c->path());
            if (forward) ++stats.n_jvp;
            else ++stats.n_vjp;
        }
        sycl::event::wait(evs);
    }

    /// The "previous value" buffer of an active cut variable on the host; zeros if there is none.
    std::vector<float> read_prev(const VarMeta* v) {
        const auto it = prev.find(v);
        if (active.contains(v) && it != prev.end() && it->second.v) {
            auto hv = it->second.v->host_view();
            return std::vector<float>(hv.begin(), hv.end());
        }
        return std::vector<float>(v->shape.elements(), 0.0f);
    }

    /// The buffer of an active variable on the host; zeros for an inactive one.
    std::vector<float> read(const VarMeta* v) {
        const auto it = bufs.find(v);
        if (active.contains(v) && it != bufs.end() && it->second.v) {
            auto hv = it->second.v->host_view();
            return std::vector<float>(hv.begin(), hv.end());
        }
        return std::vector<float>(v->shape.elements(), 0.0f);
    }
};

/// The values of a float variable's container on the host (after the run has completed).
inline std::vector<float> read_host(const Instance& i) {
    auto& d  = dynamic_cast<DeviceData<float>&>(*i.data);
    auto  hv = d.v().host_view();
    return std::vector<float>(hv.begin(), hv.end());
}

}  // namespace detail
}  // namespace naga::cospan
