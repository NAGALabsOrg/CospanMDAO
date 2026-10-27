// =============================================================================
// tests/test_optimizers.cpp — derivatives, phase 3: the reduced-space IPOPT, on host problems and on the Sellar model
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

#include "derivative_parts.hpp"
#include "support.hpp"

namespace cs = naga::cospan;
using cs::DType;
using cst::default_config;

namespace topt {

constexpr double inf = std::numeric_limits<double>::infinity();

/// An NlpProblem from lambdas; `valid` false makes every evaluation fail.
struct Gen : cs::NlpProblem {
    std::size_t         n = 0, m = 0;
    std::vector<double> xl, xu, gl, gu, x0;
    std::function<double(std::span<const double>)>                  f;
    std::function<void(std::span<const double>, std::span<double>)> gradf, g, jac;
    std::function<bool(std::span<const double>)>                    valid = [](std::span<const double>) { return true; };

    std::size_t n_vars() const override { return n; }
    std::size_t n_cons() const override { return m; }
    void        bounds(std::span<double> a, std::span<double> b, std::span<double> c, std::span<double> d) const override {
        std::copy(xl.begin(), xl.end(), a.begin());
        std::copy(xu.begin(), xu.end(), b.begin());
        std::copy(gl.begin(), gl.end(), c.begin());
        std::copy(gu.begin(), gu.end(), d.begin());
    }
    void initial_point(std::span<double> x) const override { std::copy(x0.begin(), x0.end(), x.begin()); }
    bool eval_f(std::span<const double> x, double& o) override {
        if (!valid(x)) return false;
        o = f(x);
        return true;
    }
    bool eval_g(std::span<const double> x, std::span<double> o) override {
        if (!valid(x)) return false;
        if (m > 0) g(x, o);
        return true;
    }
    bool eval_grad_f(std::span<const double> x, std::span<double> o) override {
        gradf(x, o);
        return true;
    }
    bool eval_jac_g(std::span<const double> x, std::span<double> o) override {
        if (m > 0) jac(x, o);
        return true;
    }
};

bool solved(cs::OptStatus s) { return s == cs::OptStatus::solved || s == cs::OptStatus::solved_to_acceptable; }

/// xorshift64* as in check_partials; a uniform draw in [-1, 1).
struct Rng {
    std::uint64_t st;
    std::uint64_t next() {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        return st * 0x2545F4914F6CDD1Dull;
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53 * 2.0 - 1.0; }
};

/// The Sellar MDA with the cycle closed, the two constraints and the marks (design variables z, x; objective obj.f).
class SellarOpt : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<tder::D1>("d1"));
        add_subsystem(cs::subsystem_desc<tder::D2>("d2"));
        add_subsystem(cs::subsystem_desc<tder::Obj>("obj"));
        add_subsystem(cs::subsystem_desc<tder::Con1>("con1"));
        add_subsystem(cs::subsystem_desc<tder::Con2>("con2"));
        tie_inputs<DType::f32, 1>({"d1.z", "d2.z", "obj.z"}, "z");
        tie_inputs<DType::f32, 1>({"d1.x", "obj.x"}, "x");
        connect("d1.y1", "d2.y1");
        connect("d1.y1", "obj.y1");
        connect("d2.y2", "obj.y2");
        connect("d2.y2", "d1.y2");
        connect("d1.y1", "con1.y1");
        connect("d2.y2", "con2.y2");
        init_input("z", cs::init::from_host(std::vector<float>{5.0f, 2.0f}));
        init_input("x", cs::init::from_host(std::vector<float>{1.0f}));
        init_input("d1.y2", cs::init::from_host(std::vector<float>{1.5f}));
        add_design_var("z", cs::Bounds{std::vector<double>{-10, 0}, std::vector<double>{10, 10}});
        add_design_var("x", cs::Bounds{std::vector<double>{0}, std::vector<double>{10}});
        add_objective("obj.f");
        add_constraint("con1.c", cs::Bounds{std::nullopt, std::vector<double>{0}});
        add_constraint("con2.c", cs::Bounds{std::nullopt, std::vector<double>{0}});
    }
};

}  // namespace topt

// =================================================================================================================================

TEST_CASE("ipopt: HS071", "[optimizers]") {
    topt::Gen p;
    p.n  = 4;
    p.m  = 2;
    p.xl = {1, 1, 1, 1};
    p.xu = {5, 5, 5, 5};
    p.gl = {25, 40};
    p.gu = {topt::inf, 40};
    p.x0 = {1, 5, 5, 1};
    p.f  = [](auto x) { return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]; };
    p.gradf = [](auto x, auto o) {
        o[0] = x[3] * (2 * x[0] + x[1] + x[2]);
        o[1] = x[0] * x[3];
        o[2] = x[0] * x[3] + 1;
        o[3] = x[0] * (x[0] + x[1] + x[2]);
    };
    p.g = [](auto x, auto o) {
        o[0] = x[0] * x[1] * x[2] * x[3];
        o[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    };
    p.jac = [](auto x, auto o) {
        o[0] = x[1] * x[2] * x[3];
        o[1] = x[0] * x[2] * x[3];
        o[2] = x[0] * x[1] * x[3];
        o[3] = x[0] * x[1] * x[2];
        for (int i = 0; i < 4; ++i) o[4 + i] = 2 * x[i];
    };
    cs::IpoptOptions o;
    o.tol            = 1e-7;
    o.acceptable_tol = 1e-6;
    o.max_iter       = 300;
    cs::IpoptOptimizer opt{o};
    const cs::OptimizeResult r = opt.optimize(p);
    INFO("status " << static_cast<int>(r.status) << ", iterations " << r.iterations << ", f " << r.f);
    CHECK(topt::solved(r.status));
    CHECK(std::fabs(r.f - 17.0140173) <= 1e-4);
    const double want[4] = {1.0, 4.7429994, 3.8211503, 1.3794082};
    for (int i = 0; i < 4; ++i) CHECK(std::fabs(r.x[static_cast<std::size_t>(i)] - want[i]) <= 1e-3);
}

TEST_CASE("ipopt: bound-only", "[optimizers]") {
    topt::Gen p;
    p.n  = 3;
    p.xl = {0, 0, 0};
    p.xu = {1, 1, 1};
    p.x0 = {0.5, 0.5, 0.5};
    p.f  = [](auto x) {
        double s = 0;
        for (const double v : x) s += (v - 2) * (v - 2);
        return s;
    };
    p.gradf = [](auto x, auto o) {
        for (std::size_t i = 0; i < 3; ++i) o[i] = 2 * (x[i] - 2);
    };
    cs::IpoptOptimizer opt;
    const cs::OptimizeResult r = opt.optimize(p);
    CHECK(r.status == cs::OptStatus::solved);
    for (const double v : r.x) CHECK(std::fabs(v - 1.0) <= 1e-4);
}

TEST_CASE("ipopt: equality", "[optimizers]") {
    topt::Gen p;
    p.n  = 2;
    p.m  = 1;
    p.xl = {-topt::inf, -topt::inf};
    p.xu = {topt::inf, topt::inf};
    p.gl = {1};
    p.gu = {1};
    p.x0 = {0, 0};
    p.f  = [](auto x) { return x[0] * x[0] + x[1] * x[1]; };
    p.gradf = [](auto x, auto o) {
        o[0] = 2 * x[0];
        o[1] = 2 * x[1];
    };
    p.g   = [](auto x, auto o) { o[0] = x[0] + x[1]; };
    p.jac = [](auto, auto o) {
        o[0] = 1;
        o[1] = 1;
    };
    cs::IpoptOptimizer opt;
    const cs::OptimizeResult r = opt.optimize(p);
    CHECK(topt::solved(r.status));
    CHECK(std::fabs(r.x[0] - 0.5) <= 1e-4);
    CHECK(std::fabs(r.x[1] - 0.5) <= 1e-4);
}

TEST_CASE("ipopt: infeasible start", "[optimizers]") {
    topt::Gen p;
    p.n  = 1;
    p.m  = 1;
    p.xl = {-topt::inf};
    p.xu = {topt::inf};
    p.gl = {1};
    p.gu = {topt::inf};
    p.x0 = {0};
    p.f  = [](auto x) { return (x[0] - 0.2) * (x[0] - 0.2); };
    p.gradf = [](auto x, auto o) { o[0] = 2 * (x[0] - 0.2); };
    p.g     = [](auto x, auto o) { o[0] = x[0]; };
    p.jac   = [](auto, auto o) { o[0] = 1; };
    cs::IpoptOptimizer opt;
    const cs::OptimizeResult r = opt.optimize(p);
    CHECK(topt::solved(r.status));
    CHECK(std::fabs(r.x[0] - 1.0) <= 1e-4);
    CHECK(r.constr_viol <= 1e-6);
}

TEST_CASE("ipopt: evaluation failure is backtracked", "[optimizers]") {
    topt::Gen p;
    p.n  = 1;
    p.xl = {0};
    p.xu = {10};
    p.x0 = {0};
    p.f  = [](auto x) { return (x[0] - 4) * (x[0] - 4); };
    p.gradf = [](auto x, auto o) { o[0] = 2 * (x[0] - 4); };
    p.valid = [](auto x) { return x[0] <= 6; };  // eval_f and eval_g fail for x > 6
    cs::IpoptOptimizer opt;
    const cs::OptimizeResult r = opt.optimize(p);
    CHECK(r.status == cs::OptStatus::solved);
    CHECK(std::fabs(r.x[0] - 4.0) <= 1e-4);
}

TEST_CASE("ipopt: the L-BFGS matrix satisfies the secant condition", "[optimizers]") {
    topt::Rng rng{7};
    const double q[4] = {1, 2, 3, 4};
    std::vector<std::pair<std::vector<double>, std::vector<double>>> pairs;
    for (int k = 0; k < 3; ++k) {
        std::vector<double> s(4), y(4);
        for (std::size_t i = 0; i < 4; ++i) {
            s[i] = rng.uniform();
            y[i] = q[i] * s[i];
        }
        pairs.emplace_back(s, y);
    }
    const std::vector<double> B = cs::detail::lbfgs_matrix(4, pairs);
    const auto& [sn, yn]        = pairs.back();
    double err = 0, ynorm = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        double v = 0;
        for (std::size_t j = 0; j < 4; ++j) v += B[i * 4 + j] * sn[j];
        err   = std::max(err, std::fabs(v - yn[i]));
        ynorm = std::max(ynorm, std::fabs(yn[i]));
    }
    CHECK(err <= 1e-10 * ynorm);
    double asym = 0;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) asym = std::max(asym, std::fabs(B[i * 4 + j] - B[j * 4 + i]));
    CHECK(asym <= 1e-12);
}

TEST_CASE("optimizers: Sellar with IPOPT", "[optimizers]") {
    cs::Model model = cs::create_model<topt::SellarOpt>(default_config());
    model.setup();
    model.set_solver(std::make_unique<cs::NewtonKrylov>());
    cs::IpoptOptimizer o;
    const cs::OptimizeResult r = model.optimize(o);
    INFO("status " << static_cast<int>(r.status) << ", iterations " << r.iterations << ", n_f " << r.n_f << ", n_grad " << r.n_grad);
    INFO("f " << r.f << ", z " << r.x[0] << " " << r.x[1] << ", x " << r.x[2] << ", viol " << r.constr_viol);
    std::cout << "[optimizers] Sellar IPOPT: status " << static_cast<int>(r.status) << ", iterations " << r.iterations << ", n_f " << r.n_f
              << ", n_grad " << r.n_grad << ", f " << r.f << ", z " << r.x[0] << " " << r.x[1] << ", x " << r.x[2] << "\n";
    CHECK(topt::solved(r.status));
    CHECK(std::fabs(r.f - 3.18339) <= 2e-3);
    CHECK(std::fabs(r.x[0] - 1.9776) <= 1e-2);
    CHECK(std::fabs(r.x[1]) <= 1e-2);
    CHECK(std::fabs(r.x[2]) <= 1e-2);
    CHECK(r.constr_viol <= 1e-4);
    const float f_model = cst::to_vector(cst::synced_output<DType::f32, 1>(model, "obj.f")).at(0);
    CHECK(std::fabs(static_cast<double>(f_model) - r.f) <= 1e-5);
}
