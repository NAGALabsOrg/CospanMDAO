#pragma once

// =============================================================================
// naga/cospan/component_driver.hpp — the leaf of the tree that owns components (r05)
// =============================================================================
//
// A component is not a tree node (component.hpp): a DRIVER is. The driver is a Subsystem leaf; it owns its component(s), gives
// each its full name and itself (bind) before declare() runs, and runs the component's forward steps. The holders of a
// component's ports answer to the driver's node.
//
//   SoloDriver   one component. A lone component `c` is the leaf `c`; its ports are `c.x`.
//   FusedDriver  a fused group's driver (fused_component.hpp): many stages, one launch.

#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/component.hpp"
#include "naga/cospan/model_meta.hpp"
#include "naga/cospan/subsystem.hpp"

namespace naga::cospan::detail {

/// A leaf that owns components. Its events and depends_on inputs are the components'.
class ComponentDriver : public Subsystem {
public:
    ~ComponentDriver() override = default;

    bool is_leaf() const noexcept override { return true; }

    /// The components' launch events, in order.
    std::vector<sycl::event> events() const override {
        std::vector<sycl::event> out;
        for (const ComponentBase* c : parts_) {
            const std::vector<sycl::event> e = c->events();
            out.insert(out.end(), e.begin(), e.end());
        }
        return out;
    }

    /// depends_on: every component's next submission adds them for one forward.
    void add_dependencies(std::span<const sycl::event> deps) override {
        for (ComponentBase* c : parts_) c->add_dependencies(deps);
    }

    /// The components this driver owns, in order (derivatives walk them).
    const std::vector<ComponentBase*>& parts() const { return parts_; }

protected:
    ComponentDriver() = default;

    std::vector<ComponentBase*> parts_;  ///< the components this driver owns (raw: the derived class holds them)
};

/// The driver of one component.
class SoloDriver final : public ComponentDriver {
public:
    explicit SoloDriver(std::unique_ptr<ComponentBase> c) : comp_{std::move(c)} { parts_.push_back(comp_.get()); }

    /// Binds the component (the metadata, this leaf's own full name, this driver) and runs its declare().
    void run_setup() override {
        ComponentAccess::bind(*comp_, meta_, path(), this);
        ComponentAccess::run_declare(*comp_);
    }

    /// ONE forward: the pre steps, the submission, the post steps.
    void forward(Runtime& rt) override {
        comp_->step_pre(rt);
        comp_->step_submit(rt);
        comp_->step_post(rt);
    }

private:
    std::unique_ptr<ComponentBase> comp_;
};

}  // namespace naga::cospan::detail
