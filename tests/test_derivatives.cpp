// =============================================================================
// tests/test_derivatives.cpp — derivatives, phase 1: compute_jvp / compute_vjp, activity, totals, check_partials
// =============================================================================
//
// Every component here is f32 and takes its sizes in its constructor. A derivative kernel reads with operator[] / get and
// writes only with add. The components live in namespace tder.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "derivative_parts.hpp"
#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace tder {

using V = std::vector<std::string>;

/// `f` must throw an E whose message contains `needle`.
template <class E, class F>
void expect_throws_as(F&& f, const std::string& needle) {
    std::string what  = "<no exception>";
    bool        right = false;
    try {
        f();
    } catch (const E& e) {
        what  = e.what();
        right = true;
    } catch (const std::exception& e) {
        what = std::string{"the wrong exception type: "} + e.what();
    }
    INFO("thrown: " << what);
    CHECK(right);
    CHECK_THAT(what, Catch::Matchers::ContainsSubstring(needle));
}

/// x[i] = 0.5 + 0.01 i.
std::vector<float> ramp(std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = 0.5f + 0.01f * static_cast<float>(i);
    return v;
}

// ---- Square(n): y = a x^2 -------------------------------------------------------------------------------------------
class Square : public cs::Component<Square> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x, a;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        out.y[i] = in.a[std::size_t{0}] * in.x[i] * in.x[i];
    }
    static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        dy.y.add(i, 2.0f * x.a[std::size_t{0}] * x.x[i] * dx.x[i] + x.x[i] * x.x[i] * dx.a[std::size_t{0}]);
    }
    static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const float g = ybar.y.get(i);
        xbar.x.add(i, 2.0f * x.a[std::size_t{0}] * x.x[i] * g);
        xbar.a.add(std::size_t{0}, x.x[i] * x.x[i] * g);
    }

    Inputs  inputs;
    Outputs outputs;

    explicit Square(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(n_));
        init_input("x", cs::init::from_host(ramp(n_)));
        declare_input(inputs.a, "a", 1);
        init_input("a", cs::init::from_host(std::vector<float>{1.3f}));
        declare_output(outputs.y, "y", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

// ---- Stencil3(n): y[i] = x[i-1] - 2 x[i] + x[i+1] (zero outside) --------------------------------------------------------
class Stencil3 : public cs::Component<Stencil3> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const std::size_t k = i[0], n = in.x.shape()[0];
        out.y[i] = (k > 0 ? in.x[k - 1] : 0.0f) - 2.0f * in.x[k] + (k + 1 < n ? in.x[k + 1] : 0.0f);
    }
    static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        const std::size_t k = i[0], n = x.x.shape()[0];
        dy.y.add(k, (k > 0 ? dx.x[k - 1] : 0.0f) - 2.0f * dx.x[k] + (k + 1 < n ? dx.x[k + 1] : 0.0f));
    }
    static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const std::size_t k = i[0], n = x.x.shape()[0];
        const float       g = ybar.y.get(k);
        xbar.x.add(k, -2.0f * g);
        if (k > 0) xbar.x.add(k - 1, g);
        if (k + 1 < n) xbar.x.add(k + 1, g);
    }

    Inputs  inputs;
    Outputs outputs;

    explicit Stencil3(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(n_));
        init_input("x", cs::init::from_host(ramp(n_)));
        declare_output(outputs.y, "y", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

// ---- SumReduce: y[0] = sum of x (64 elements, one work-group of 64, local memory) -----------------------------------------
class SumReduce : public cs::Component<SumReduce> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    using Mem = cs::LocalMem<cs::LocalArray<DType::f32, 1>>;
    static constexpr std::size_t kN = 64;

    static void compute_forward(cs::GroupItem<1> it, const Inputs& in, const Outputs& out, Mem lm) {
        const std::size_t l = it.get_local_linear_id();
        auto [t]            = lm.get(it);
        t[l]                = in.x[l];
        cs::group_barrier(it.get_group());
        for (std::size_t s = kN / 2; s > 0; s /= 2) {
            if (l < s) t[l] += t[l + s];
            cs::group_barrier(it.get_group());
        }
        if (l == 0) out.y[std::size_t{0}] = t[0];
    }
    static void compute_jvp(cs::GroupItem<1> it, const Inputs&, const Inputs& dx, const Outputs& dy, Mem lm) {
        const std::size_t l = it.get_local_linear_id();
        auto [t]            = lm.get(it);
        t[l]                = dx.x[l];
        cs::group_barrier(it.get_group());
        for (std::size_t s = kN / 2; s > 0; s /= 2) {
            if (l < s) t[l] += t[l + s];
            cs::group_barrier(it.get_group());
        }
        if (l == 0) dy.y.add(std::size_t{0}, t[0]);
    }
    static void compute_vjp(cs::GroupItem<1> it, const Inputs&, const Outputs& ybar, const Inputs& xbar, Mem) {
        const std::size_t l = it.get_local_linear_id();
        xbar.x.add(l, ybar.y.get(std::size_t{0}));
    }

    Inputs  inputs;
    Outputs outputs;

    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(kN));
        init_input("x", cs::init::from_host(ramp(kN)));
        declare_output(outputs.y, "y", 1);
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override {
        return create_launch_config(cs::range_desc(sycl::range<1>{kN}), cs::wg_desc(sycl::range<1>{kN}), cs::local_desc(sycl::range<1>{kN}));
    }
};

// ---- Rank2Scale(r, c): y[i,j] = x[i,j] w[j] --------------------------------------------------------------------------------
class Rank2Scale : public cs::Component<Rank2Scale> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> x;
        cs::Input<DType::f32, 1> w;
    };
    struct Outputs {
        cs::Output<DType::f32, 2> y;
    };
    static void compute_forward(sycl::id<2> id, const Inputs& in, const Outputs& out) { out.y[id] = in.x[id] * in.w[id[1]]; }
    static void compute_jvp(sycl::id<2> id, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        dy.y.add(id, dx.x[id] * x.w[id[1]] + x.x[id] * dx.w[id[1]]);
    }
    static void compute_vjp(sycl::id<2> id, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const float g = ybar.y.get(id);
        xbar.x.add(id, g * x.w[id[1]]);
        xbar.w.add(id[1], x.x[id] * g);
    }

    Inputs  inputs;
    Outputs outputs;

    Rank2Scale(std::size_t r, std::size_t c) : r_{r}, c_{c} {}
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(r_), static_cast<int>(c_));
        init_input("x", cs::init::from_host(ramp(r_ * c_)));
        declare_input(inputs.w, "w", static_cast<int>(c_));
        std::vector<float> w(c_);
        for (std::size_t j = 0; j < c_; ++j) w[j] = 1.0f + 0.1f * static_cast<float>(j);
        init_input("w", cs::init::from_host(w));
        declare_output(outputs.y, "y", static_cast<int>(r_), static_cast<int>(c_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<2>{r_, c_})); }

private:
    std::size_t r_, c_;
};

// ---- Add(n): w = u + v ------------------------------------------------------------------------------------------------------
class Add : public cs::Component<Add> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> u, v;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> w;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.w[i] = in.u[i] + in.v[i]; }
    static void compute_jvp(sycl::id<1> i, const Inputs&, const Inputs& dx, const Outputs& dy) { dy.w.add(i, dx.u[i] + dx.v[i]); }
    static void compute_vjp(sycl::id<1> i, const Inputs&, const Outputs& ybar, const Inputs& xbar) {
        const float g = ybar.w.get(i);
        xbar.u.add(i, g);
        xbar.v.add(i, g);
    }

    Inputs  inputs;
    Outputs outputs;

    explicit Add(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.u, "u", static_cast<int>(n_));
        init_input("u", cs::init::from_host(ramp(n_)));
        declare_input(inputs.v, "v", static_cast<int>(n_));
        init_input("v", cs::init::from_host(ramp(n_)));
        declare_output(outputs.w, "w", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

// ---- NoDeriv(n): y = x + 1, with no derivative kernels ----------------------------------------------------------------------
class NoDeriv : public cs::Component<NoDeriv> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> x;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> y;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.y[i] = in.x[i] + 1.0f; }

    Inputs  inputs;
    Outputs outputs;

    explicit NoDeriv(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.x, "x", static_cast<int>(n_));
        init_input("x", cs::init::from_host(ramp(n_)));
        declare_output(outputs.y, "y", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

// ---- the models -------------------------------------------------------------------------------------------------------------
using Build = std::function<void(cs::System&)>;

class Top : public cs::System {
public:
    explicit Top(Build b) : build_{std::move(b)} {}
    void setup() override { build_(*this); }

private:
    Build build_;
};

cs::Model make(Build b) { return cs::create_model<Top>(default_config(), std::move(b)); }

/// c1 -> c3.u, c2 -> c3.v.
class Branches : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Square>("c1", std::size_t{16}));
        add_subsystem(cs::subsystem_desc<Square>("c2", std::size_t{16}));
        add_subsystem(cs::subsystem_desc<Add>("c3", std::size_t{16}));
        connect("c1.y", "c3.u");
        connect("c2.y", "c3.v");
    }
};

/// c1, nd, c3 (n = 8). `feed_nd`: c1.y feeds nd.x; otherwise c1.y feeds c3.u and nd stands alone.
class WithNoDeriv : public cs::System {
public:
    explicit WithNoDeriv(bool feed_nd) : feed_nd_{feed_nd} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<Square>("c1", std::size_t{8}));
        add_subsystem(cs::subsystem_desc<NoDeriv>("nd", std::size_t{8}));
        add_subsystem(cs::subsystem_desc<Add>("c3", std::size_t{8}));
        connect("c1.y", feed_nd_ ? "nd.x" : "c3.u");
    }

private:
    bool feed_nd_;
};

/// c1 -> c3.u, c3.v free; unfused, and the same two stages as one fused group "g".
class Chain : public cs::System {
public:
    explicit Chain(bool fused) : fused_{fused} {}
    void setup() override {
        if (fused_) {
            add_subsystem(cs::fuse_components("g", cs::connects_desc({{"c1.y", "c3.u"}}),
                                              cs::segment(cs::subsystem_desc<Square>("c1", std::size_t{16}), cs::subsystem_desc<Add>("c3", std::size_t{16}))));
        } else {
            add_subsystem(cs::subsystem_desc<Square>("c1", std::size_t{16}));
            add_subsystem(cs::subsystem_desc<Add>("c3", std::size_t{16}));
            connect("c1.y", "c3.u");
        }
    }

private:
    bool fused_;
};

/// Sellar with the cycle cut at d1.y2 (a free input, 1.5); `close_cycle` connects d2.y2 to d1.y2 as well.
class Sellar : public cs::System {
public:
    explicit Sellar(bool close_cycle) : close_{close_cycle} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<D1>("d1"));
        add_subsystem(cs::subsystem_desc<D2>("d2"));
        add_subsystem(cs::subsystem_desc<Obj>("obj"));
        tie_inputs<DType::f32, 1>({"d1.z", "d2.z", "obj.z"}, "z");
        tie_inputs<DType::f32, 1>({"d1.x", "obj.x"}, "x");
        connect("d1.y1", "d2.y1");
        connect("d1.y1", "obj.y1");
        connect("d2.y2", "obj.y2");
        if (close_) connect("d2.y2", "d1.y2");
        init_input("z", cs::init::from_host(std::vector<float>{5.0f, 2.0f}));
        init_input("x", cs::init::from_host(std::vector<float>{1.0f}));
        init_input("d1.y2", cs::init::from_host(std::vector<float>{1.5f}));
    }

private:
    bool close_;
};


/// setup(), one run, and a wait: the state compute_totals and check_partials need.
void run_ready(cs::Model& m) {
    m.setup();
    m.run();
    m.wait();
}

}  // namespace tder

// =================================================================================================================================

TEST_CASE("derivatives: check_partials passes for Square, Stencil3, SumReduce, Rank2Scale", "[derivatives]") {
    const auto check = [](tder::Build b, const char* label) {
        cs::Model model = tder::make(std::move(b));
        model.setup();
        model.run();
        const cs::PartialsReport r = model.check_partials("c");
        INFO(label << ": fd_max_rel_err " << r.fd_max_rel_err << ", dot_rel_err " << r.dot_rel_err);
        CHECK(r.has_jvp);
        CHECK(r.has_vjp);
        CHECK(r.fd_pass);
        CHECK(r.dot_pass);
    };
    SECTION("Square") {
        check([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tder::Square>("c", std::size_t{1000})); }, "Square");
    }
    SECTION("Stencil3") {
        check([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tder::Stencil3>("c", std::size_t{1000})); }, "Stencil3");
    }
    SECTION("SumReduce") {
        check([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tder::SumReduce>("c")); }, "SumReduce");
    }
    SECTION("Rank2Scale") {
        check([](cs::System& s) { s.add_subsystem(cs::subsystem_desc<tder::Rank2Scale>("c", std::size_t{16}, std::size_t{8})); }, "Rank2Scale");
    }
}

TEST_CASE("derivatives: activity skips an unrelated branch", "[derivatives]") {
    cs::Model model = cs::create_model<tder::Branches>(default_config());
    tder::run_ready(model);

    // d(c3.w)/d(c1.x) = diag(2 a x_i): the other branch (c2) and the free inputs are inactive.
    const auto want = [](std::size_t i, std::size_t j) {
        return i == j ? 2.0 * 1.3 * static_cast<double>(tder::ramp(16)[i]) : 0.0;
    };
    const auto block_ok = [&](const cs::Jacobian& jac) {
        const cs::JacBlock& b = jac.block("c3.w", "c1.x");
        REQUIRE(b.rows == 16);
        REQUIRE(b.cols == 16);
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 16; ++j) CHECK(std::fabs(b.data[i * 16 + j] - want(i, j)) <= 1e-5);
    };

    SECTION("forward") {
        const cs::Jacobian jac = model.compute_totals(tder::V{"c3.w"}, tder::V{"c1.x"}, cs::DerivMode::forward);
        const cs::DerivStats& st = model.last_deriv_stats();
        CHECK(st.launched == tder::V{"c1", "c3"});
        CHECK(st.active_vars.size() == 3);
        block_ok(jac);
    }
    SECTION("reverse") {
        const cs::Jacobian jac = model.compute_totals(tder::V{"c3.w"}, tder::V{"c1.x"}, cs::DerivMode::reverse);
        const cs::DerivStats& st = model.last_deriv_stats();
        CHECK(st.launched == tder::V{"c3", "c1"});
        CHECK(st.active_vars.size() == 3);
        block_ok(jac);
    }
}

TEST_CASE("derivatives: a component without derivative kernels", "[derivatives]") {
    SECTION("off the active path: not launched, no error") {
        cs::Model model = cs::create_model<tder::WithNoDeriv>(default_config(), false);
        tder::run_ready(model);
        CHECK_NOTHROW(model.compute_totals(tder::V{"c3.w"}, tder::V{"c1.x"}));
    }
    SECTION("on the active path: DerivativeError naming it") {
        cs::Model model = cs::create_model<tder::WithNoDeriv>(default_config(), true);
        tder::run_ready(model);
        tder::expect_throws_as<cs::DerivativeError>([&] { (void)model.compute_totals(tder::V{"nd.y"}, tder::V{"c1.x"}); }, "nd");
    }
}

TEST_CASE("derivatives: fused group gives the same totals", "[derivatives]") {
    cs::Model plain = cs::create_model<tder::Chain>(default_config(), false);
    cs::Model fused = cs::create_model<tder::Chain>(default_config(), true);
    tder::run_ready(plain);
    tder::run_ready(fused);
    for (const cs::DerivMode mode : {cs::DerivMode::forward, cs::DerivMode::reverse}) {
        const cs::Jacobian a = plain.compute_totals(tder::V{"c3.w"}, tder::V{"c1.x"}, mode);
        const cs::Jacobian b = fused.compute_totals(tder::V{"g.c3.w"}, tder::V{"g.c1.x"}, mode);
        const cs::JacBlock& ba = a.block("c3.w", "c1.x");
        const cs::JacBlock& bb = b.block("g.c3.w", "g.c1.x");
        REQUIRE(ba.data.size() == bb.data.size());
        double worst = 0;
        for (std::size_t k = 0; k < ba.data.size(); ++k) worst = std::max(worst, std::fabs(ba.data[k] - bb.data[k]));
        INFO("mode " << (mode == cs::DerivMode::forward ? "forward" : "reverse") << ": largest difference " << worst);
        CHECK(worst <= 1e-5);
    }
}

TEST_CASE("derivatives: Sellar with the cycle cut matches finite differences", "[derivatives]") {
    cs::Model model = cs::create_model<tder::Sellar>(default_config(), false);
    tder::run_ready(model);

    constexpr double h  = 1e-4;
    const double     z0 = 5.0, z1 = 2.0, x = 1.0;
    const double     dz0 = (tder::sellar_f(z0 + h, z1, x) - tder::sellar_f(z0 - h, z1, x)) / (2 * h);
    const double     dz1 = (tder::sellar_f(z0, z1 + h, x) - tder::sellar_f(z0, z1 - h, x)) / (2 * h);
    const double     dx  = (tder::sellar_f(z0, z1, x + h) - tder::sellar_f(z0, z1, x - h)) / (2 * h);

    for (const cs::DerivMode mode : {cs::DerivMode::forward, cs::DerivMode::reverse}) {
        const cs::Jacobian  jac = model.compute_totals(tder::V{"obj.f"}, tder::V{"z", "x"}, mode);
        const cs::JacBlock& bz  = jac.block("obj.f", "z");
        const cs::JacBlock& bx  = jac.block("obj.f", "x");
        REQUIRE(bz.data.size() == 2);
        REQUIRE(bx.data.size() == 1);
        INFO((mode == cs::DerivMode::forward ? "forward" : "reverse") << ": dz0 " << bz.data[0] << " vs " << dz0 << ", dz1 " << bz.data[1]
                                                                       << " vs " << dz1 << ", dx " << bx.data[0] << " vs " << dx);
        CHECK(tder::sellar_close(bz.data[0], dz0));
        CHECK(tder::sellar_close(bz.data[1], dz1));
        CHECK(tder::sellar_close(bx.data[0], dx));
    }
}

TEST_CASE("derivatives: compute_totals before run throws std::logic_error", "[derivatives]") {
    cs::Model model = cs::create_model<tder::Branches>(default_config());
    model.setup();
    CHECK_THROWS_AS(model.compute_totals(tder::V{"c3.w"}, tder::V{"c1.x"}), std::logic_error);
}