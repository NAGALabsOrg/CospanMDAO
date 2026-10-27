#pragma once

// The phase-1 Sellar parts (D1, D2, Obj) and their host-reference helpers, shared by test_derivatives.cpp and test_solvers.cpp.

#include <cmath>
#include <cstddef>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
using cs::DType;

namespace tder {

// ---- the Sellar parts (index-range, rank 1, over one index) -----------------------------------------------------------------
class D1 : public cs::Component<D1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> z, x, y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y1;
    };
    static void compute_forward(sycl::id<1>, const Inputs& in, const Outputs& out) {
        const std::size_t k = 0;
        out.y1[k]           = in.z[k] * in.z[k] + in.z[std::size_t{1}] + in.x[k] - 0.2f * in.y2[k];
    }
    static void compute_jvp(sycl::id<1>, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        const std::size_t k = 0;
        dy.y1.add(k, 2.0f * x.z[k] * dx.z[k] + dx.z[std::size_t{1}] + dx.x[k] - 0.2f * dx.y2[k]);
    }
    static void compute_vjp(sycl::id<1>, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const std::size_t k = 0;
        const float       g = ybar.y1.get(k);
        xbar.z.add(k, 2.0f * x.z[k] * g);
        xbar.z.add(std::size_t{1}, g);
        xbar.x.add(k, g);
        xbar.y2.add(k, -0.2f * g);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.z, "z", 2);
        declare_input(inputs.x, "x", 1);
        declare_input(inputs.y2, "y2", 1);
        declare_output(outputs.y1, "y1", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{1})); }
};

class D2 : public cs::Component<D2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> z, y1;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y2;
    };
    static void compute_forward(sycl::id<1>, const Inputs& in, const Outputs& out) {
        const std::size_t k = 0;
        out.y2[k]           = sycl::sqrt(in.y1[k]) + in.z[k] + in.z[std::size_t{1}];
    }
    static void compute_jvp(sycl::id<1>, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        const std::size_t k = 0;
        dy.y2.add(k, 0.5f / sycl::sqrt(x.y1[k]) * dx.y1[k] + dx.z[k] + dx.z[std::size_t{1}]);
    }
    static void compute_vjp(sycl::id<1>, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const std::size_t k = 0;
        const float       g = ybar.y2.get(k);
        xbar.y1.add(k, 0.5f / sycl::sqrt(x.y1[k]) * g);
        xbar.z.add(k, g);
        xbar.z.add(std::size_t{1}, g);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.z, "z", 2);
        declare_input(inputs.y1, "y1", 1);
        declare_output(outputs.y2, "y2", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{1})); }
};

class Obj : public cs::Component<Obj> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x, z, y1, y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> f;
    };
    static void compute_forward(sycl::id<1>, const Inputs& in, const Outputs& out) {
        const std::size_t k = 0;
        out.f[k]            = in.x[k] * in.x[k] + in.z[std::size_t{1}] + in.y1[k] + sycl::exp(-in.y2[k]);
    }
    static void compute_jvp(sycl::id<1>, const Inputs& x, const Inputs& dx, const Outputs& df) {
        const std::size_t k = 0;
        df.f.add(k, 2.0f * x.x[k] * dx.x[k] + dx.z[std::size_t{1}] + dx.y1[k] - sycl::exp(-x.y2[k]) * dx.y2[k]);
    }
    static void compute_vjp(sycl::id<1>, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const std::size_t k = 0;
        const float       g = ybar.f.get(k);
        xbar.x.add(k, 2.0f * x.x[k] * g);
        xbar.z.add(std::size_t{1}, g);
        xbar.y1.add(k, g);
        xbar.y2.add(k, -sycl::exp(-x.y2[k]) * g);
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", 1);
        declare_input(inputs.z, "z", 2);
        declare_input(inputs.y1, "y1", 1);
        declare_input(inputs.y2, "y2", 1);
        declare_output(outputs.f, "f", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{1})); }
};

/// The Sellar constraint c = 3.16 - y1 (<= 0).
class Con1 : public cs::Component<Con1> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y1;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> c;
    };
    static void compute_forward(sycl::id<1>, const Inputs& in, const Outputs& out) { out.c[std::size_t{0}] = 3.16f - in.y1[std::size_t{0}]; }
    static void compute_jvp(sycl::id<1>, const Inputs&, const Inputs& dx, const Outputs& dc) { dc.c.add(std::size_t{0}, -dx.y1[std::size_t{0}]); }
    static void compute_vjp(sycl::id<1>, const Inputs&, const Outputs& ybar, const Inputs& xbar) {
        xbar.y1.add(std::size_t{0}, -ybar.c.get(std::size_t{0}));
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y1, "y1", 1);
        declare_output(outputs.c, "c", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{1})); }
};

/// The Sellar constraint c = y2 - 24 (<= 0).
class Con2 : public cs::Component<Con2> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> y2;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> c;
    };
    static void compute_forward(sycl::id<1>, const Inputs& in, const Outputs& out) { out.c[std::size_t{0}] = in.y2[std::size_t{0}] - 24.0f; }
    static void compute_jvp(sycl::id<1>, const Inputs&, const Inputs& dx, const Outputs& dc) { dc.c.add(std::size_t{0}, dx.y2[std::size_t{0}]); }
    static void compute_vjp(sycl::id<1>, const Inputs&, const Outputs& ybar, const Inputs& xbar) {
        xbar.y2.add(std::size_t{0}, ybar.c.get(std::size_t{0}));
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.y2, "y2", 1);
        declare_output(outputs.c, "c", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{1})); }
};


/// The Sellar objective at (z0, z1, x) with y2 held at 1.5 in d1: y1 = z0^2 + z1 + x - 0.3, then d2 and the objective.
inline double sellar_f(double z0, double z1, double x) {
    const double y1 = z0 * z0 + z1 + x - 0.3;
    const double y2 = std::sqrt(y1) + z0 + z1;
    return x * x + z1 + y1 + std::exp(-y2);
}

/// The block's value `got` against the host central difference `want`: |a - b| <= 1e-3 |b| + 1e-4.
inline bool sellar_close(double got, double want) { return std::fabs(got - want) <= 1e-3 * std::fabs(want) + 1e-4; }

}  // namespace tder
