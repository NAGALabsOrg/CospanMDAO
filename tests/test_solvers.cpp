// =============================================================================
// tests/test_solvers.cpp — derivatives, phase 2: GMRES, NewtonKrylov, BlockGaussSeidel, solved cycles, totals through cycles
// =============================================================================
//
// Mock problems test the solvers alone; the models test them on cycles of components (Sellar, and a dense coupled loop).
// Components and helpers live in namespace tsol.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>
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

namespace tsol {

using V = std::vector<std::string>;

// ---- mock problems -----------------------------------------------------------------------------------------------------------

/// A = tridiag(-1, 2, -1) applied to v.
void apply_tridiag(std::span<const double> v, std::span<double> out) {
    const std::size_t n = v.size();
    for (std::size_t i = 0; i < n; ++i) out[i] = 2.0 * v[i] - (i > 0 ? v[i - 1] : 0.0) - (i + 1 < n ? v[i + 1] : 0.0);
}

/// r(z) = atan(z) (n = 1); jvp = v / (1 + z^2) at the last evaluated z.
class AtanProblem final : public cs::NonlinearProblem {
public:
    explicit AtanProblem(double z0) : z_{z0} {}
    std::size_t size() const override { return 1; }
    void        get_state(std::span<double> z) override { z[0] = z_; }
    void        evaluate(std::span<const double> z, std::span<double> r) override {
        z_   = z[0];
        r[0] = std::atan(z_);
    }
    void jvp(std::span<const double> v, std::span<double> out) override { out[0] = v[0] / (1.0 + z_ * z_); }
    void vjp(std::span<const double> v, std::span<double> out) override { out[0] = v[0] / (1.0 + z_ * z_); }
    double z() const { return z_; }

private:
    double z_;
};

/// r(z) = A z - b with A = tridiag(-1, 2, -1), b = ones.
class LinearProblem final : public cs::NonlinearProblem {
public:
    explicit LinearProblem(std::size_t n) : z_(n, 0.0) {}
    std::size_t size() const override { return z_.size(); }
    void        get_state(std::span<double> z) override { std::copy(z_.begin(), z_.end(), z.begin()); }
    void        evaluate(std::span<const double> z, std::span<double> r) override {
        z_.assign(z.begin(), z.end());
        apply_tridiag(z, r);
        for (double& e : r) e -= 1.0;
    }
    void jvp(std::span<const double> v, std::span<double> out) override { apply_tridiag(v, out); }
    void vjp(std::span<const double> v, std::span<double> out) override { apply_tridiag(v, out); }

private:
    std::vector<double> z_;
};

/// r(z) = z - 0.5 cos(z), elementwise, n = 4: G(z) = 0.5 cos(z), a contraction.
class CosProblem final : public cs::NonlinearProblem {
public:
    CosProblem() : z_(4, 0.0) {}
    std::size_t size() const override { return z_.size(); }
    void        get_state(std::span<double> z) override { std::copy(z_.begin(), z_.end(), z.begin()); }
    void        evaluate(std::span<const double> z, std::span<double> r) override {
        z_.assign(z.begin(), z.end());
        for (std::size_t i = 0; i < z_.size(); ++i) r[i] = z_[i] - 0.5 * std::cos(z_[i]);
    }
    void jvp(std::span<const double> v, std::span<double> out) override {
        for (std::size_t i = 0; i < v.size(); ++i) out[i] = v[i] * (1.0 + 0.5 * std::sin(z_[i]));
    }
    void vjp(std::span<const double> v, std::span<double> out) override { jvp(v, out); }
    const std::vector<double>& z() const { return z_; }

private:
    std::vector<double> z_;
};

// ---- the Sellar cycle -------------------------------------------------------------------------------------------------------------

/// The phase-1 Sellar parts with the cycle closed: d1.y2 is fed by d2.y2 (d1 is registered before d2), initialized to 1.5.
class SellarLoop : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<tder::D1>("d1"));
        add_subsystem(cs::subsystem_desc<tder::D2>("d2"));
        add_subsystem(cs::subsystem_desc<tder::Obj>("obj"));
        tie_inputs<DType::f32, 1>({"d1.z", "d2.z", "obj.z"}, "z");
        tie_inputs<DType::f32, 1>({"d1.x", "obj.x"}, "x");
        connect("d1.y1", "d2.y1");
        connect("d1.y1", "obj.y1");
        connect("d2.y2", "obj.y2");
        connect("d2.y2", "d1.y2");
        init_input("z", cs::init::from_host(std::vector<float>{5.0f, 2.0f}));
        init_input("x", cs::init::from_host(std::vector<float>{1.0f}));
        init_input("d1.y2", cs::init::from_host(std::vector<float>{1.5f}));
    }
};

/// The host-double fixed point of the Sellar cycle at (z0, z1, x): y2 iterated 200 times through y1 = z0^2 + z1 + x - 0.2 y2,
/// y2 = sqrt(y1) + z0 + z1. Returns {y1, y2, f}.
struct SellarRef {
    double y1, y2, f;
};
SellarRef sellar_fixed_point(double z0, double z1, double x) {
    double y2 = 1.5, y1 = 0;
    for (int k = 0; k < 200; ++k) {
        y1 = z0 * z0 + z1 + x - 0.2 * y2;
        y2 = std::sqrt(y1) + z0 + z1;
    }
    return {y1, y2, x * x + z1 + y1 + std::exp(-y2)};
}

bool rel_close(double got, double want, double rel) { return std::fabs(got - want) <= rel * std::fabs(want); }

float scalar_of(cs::Model& model, const char* path) { return cst::to_vector(cst::synced_output<DType::f32, 1>(model, path)).at(0); }

// ---- the dense coupled cycle -----------------------------------------------------------------------------------------------------

/// w = A z (A is n x n).
class Lin : public cs::Component<Lin> {
public:
    struct Inputs {
        cs::Input<DType::f32, 2> A;
        cs::Input<DType::f32, 1> z;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> w;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) {
        const std::size_t k = i[0], n = in.z.shape()[0];
        float             s = 0.0f;
        for (std::size_t j = 0; j < n; ++j) s += in.A[sycl::id<2>{k, j}] * in.z[j];
        out.w[i] = s;
    }
    static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dy) {
        const std::size_t k = i[0], n = x.z.shape()[0];
        float             s = 0.0f;
        for (std::size_t j = 0; j < n; ++j) s += dx.A[sycl::id<2>{k, j}] * x.z[j] + x.A[sycl::id<2>{k, j}] * dx.z[j];
        dy.w.add(k, s);
    }
    static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const std::size_t k = i[0], n = x.z.shape()[0];
        const float       g = ybar.w.get(k);
        for (std::size_t j = 0; j < n; ++j) {
            xbar.z.add(j, x.A[sycl::id<2>{k, j}] * g);
            xbar.A.add(sycl::id<2>{k, j}, x.z[j] * g);
        }
    }

    Inputs  inputs;
    Outputs outputs;

    explicit Lin(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.A, "A", static_cast<int>(n_), static_cast<int>(n_));
        declare_input(inputs.z, "z", static_cast<int>(n_));
        declare_output(outputs.w, "w", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

/// z = tanh(w) + b.
class Nl : public cs::Component<Nl> {
public:
    struct Inputs {
        cs::Input<DType::f32, 1> w, b;
    };
    struct Outputs {
        cs::Output<DType::f32, 1> z;
    };
    static void compute_forward(sycl::id<1> i, const Inputs& in, const Outputs& out) { out.z[i] = sycl::tanh(in.w[i]) + in.b[i]; }
    static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dz) {
        const float t = sycl::tanh(x.w[i]);
        dz.z.add(i, (1.0f - t * t) * dx.w[i] + dx.b[i]);
    }
    static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
        const float t = sycl::tanh(x.w[i]);
        const float g = ybar.z.get(i);
        xbar.w.add(i, (1.0f - t * t) * g);
        xbar.b.add(i, g);
    }

    Inputs  inputs;
    Outputs outputs;

    explicit Nl(std::size_t n) : n_{n} {}
    void declare() override {
        declare_input(inputs.w, "w", static_cast<int>(n_));
        declare_input(inputs.b, "b", static_cast<int>(n_));
        declare_output(outputs.z, "z", static_cast<int>(n_));
    }
    std::unique_ptr<cs::LaunchDescription> define_forward() override { return create_launch_config(cs::range_desc(sycl::range<1>{n_})); }

private:
    std::size_t n_;
};

constexpr std::size_t kN = 32;

std::vector<float> matrix_A() {
    std::vector<float> a(kN * kN);
    for (std::size_t i = 0; i < kN; ++i)
        for (std::size_t j = 0; j < kN; ++j)
            a[i * kN + j] = 0.9f * std::sin(static_cast<float>(7 * i + 3 * j + 1)) / static_cast<float>(kN);
    return a;
}
std::vector<float> vector_b() {
    std::vector<float> b(kN);
    for (std::size_t i = 0; i < kN; ++i) b[i] = 0.1f * std::cos(static_cast<float>(i));
    return b;
}

/// lin -> nl.w, nl -> lin.z: a cycle (lin is registered before nl). `self_solve`: the System attaches a NewtonKrylov to itself.
template <bool SelfSolve>
class LoopT : public cs::System {
public:
    void setup() override {
        add_subsystem(cs::subsystem_desc<Lin>("lin", kN));
        add_subsystem(cs::subsystem_desc<Nl>("nl", kN));
        connect("lin.w", "nl.w");
        connect("nl.z", "lin.z");
        init_input("lin.z", cs::init::from_host(std::vector<float>(kN, 0.0f)));
        init_input("lin.A", cs::init::from_host(matrix_A()));
        init_input("nl.b", cs::init::from_host(vector_b()));
        if constexpr (SelfSolve) set_solver(std::make_unique<cs::NewtonKrylov>());
    }
};
using Loop     = LoopT<false>;
using LoopSelf = LoopT<true>;

/// The top System holding one subsystem `loop` of type L.
template <class L>
class LoopTop : public cs::System {
public:
    void setup() override { add_subsystem(cs::subsystem_desc<L>("loop")); }
};

struct LoopRef {
    std::vector<double> z;                // the fixed point
    std::vector<double> dz_db;            // n x n, row-major: (I - D A)^-1
};

/// Host double reference: z <- tanh(A z) + b, 500 times; then dz/db = (I - D A)^-1 by Gaussian elimination with partial pivoting.
LoopRef loop_reference() {
    const std::vector<float> Af = matrix_A(), bf = vector_b();
    const std::size_t        n  = kN;
    std::vector<double>      z(n, 0.0), Az(n);
    const auto               mul = [&](const std::vector<double>& x, std::vector<double>& y) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0;
            for (std::size_t j = 0; j < n; ++j) s += static_cast<double>(Af[i * n + j]) * x[j];
            y[i] = s;
        }
    };
    for (int k = 0; k < 500; ++k) {
        mul(z, Az);
        for (std::size_t i = 0; i < n; ++i) z[i] = std::tanh(Az[i]) + static_cast<double>(bf[i]);
    }
    mul(z, Az);
    // M = I - D A, then invert M by elimination on [M | I].
    std::vector<double> M(n * 2 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = std::tanh(Az[i]), d = 1.0 - t * t;
        for (std::size_t j = 0; j < n; ++j) M[i * 2 * n + j] = (i == j ? 1.0 : 0.0) - d * static_cast<double>(Af[i * n + j]);
        M[i * 2 * n + n + i] = 1.0;
    }
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < n; ++r)
            if (std::fabs(M[r * 2 * n + c]) > std::fabs(M[piv * 2 * n + c])) piv = r;
        if (piv != c)
            for (std::size_t j = 0; j < 2 * n; ++j) std::swap(M[c * 2 * n + j], M[piv * 2 * n + j]);
        const double p = M[c * 2 * n + c];
        for (std::size_t j = 0; j < 2 * n; ++j) M[c * 2 * n + j] /= p;
        for (std::size_t r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = M[r * 2 * n + c];
            if (f == 0.0) continue;
            for (std::size_t j = 0; j < 2 * n; ++j) M[r * 2 * n + j] -= f * M[c * 2 * n + j];
        }
    }
    LoopRef ref;
    ref.z = z;
    ref.dz_db.resize(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) ref.dz_db[i * n + j] = M[i * 2 * n + n + j];
    return ref;
}

std::vector<float> loop_z(cs::Model& model) { return cst::to_vector(cst::synced_output<DType::f32, 1>(model, "loop.nl.z")); }

}  // namespace tsol

// =================================================================================================================================

TEST_CASE("gmres: SPD tridiagonal", "[solvers]") {
    const std::size_t   n = 50;
    std::vector<double> b(n, 1.0), x(n, 0.0), r(n);
    cs::GmresOptions    o;
    o.restart  = 10;
    o.max_iter = 2000;  // GMRES(10) on this matrix needs about 1250 iterations to reach 1e-10
    o.rtol     = 1e-10;
    o.atol     = 0.0;
    const cs::GmresResult g = cs::gmres(n, tsol::apply_tridiag, b, x, o);
    tsol::apply_tridiag(x, r);
    double err = 0, nb = 0;
    for (std::size_t i = 0; i < n; ++i) {
        err += (b[i] - r[i]) * (b[i] - r[i]);
        nb += b[i] * b[i];
    }
    INFO("iterations " << g.iterations << ", residual " << g.residual_norm);
    CHECK(g.converged);
    CHECK(std::sqrt(err) <= 1e-9 * std::sqrt(nb));
}

TEST_CASE("newton: scalar atan with line search", "[solvers]") {
    SECTION("default options") {
        tsol::AtanProblem p{2.0};
        cs::NewtonKrylov  nk;
        const cs::SolveResult r = nk.solve(p);
        INFO("iterations " << r.iterations << ", message " << r.message);
        CHECK(r.converged);
        CHECK(r.iterations <= 15);
        CHECK(std::fabs(p.z()) < 1e-5);
    }
    SECTION("max_ls = 0 does not converge") {
        tsol::AtanProblem p{2.0};
        cs::NewtonOptions o;
        o.max_ls = 0;
        cs::NewtonKrylov nk{o};
        const cs::SolveResult r = nk.solve(p);
        CHECK_FALSE(r.converged);
    }
}

TEST_CASE("newton: a linear system converges in one iteration", "[solvers]") {
    tsol::LinearProblem p{50};
    cs::NewtonOptions   o;
    o.atol    = 1e-8;
    o.rtol    = 1e-8;
    o.eta_0   = 1e-10;
    o.eta_max = 1e-10;
    cs::NewtonKrylov nk{o};
    const cs::SolveResult r = nk.solve(p);
    INFO("iterations " << r.iterations << ", message " << r.message << ", r_norm " << r.r_norm);
    CHECK(r.converged);
    CHECK(r.iterations == 1);
}

TEST_CASE("newton: forcing terms", "[solvers]") {
    tsol::AtanProblem p{2.0};
    cs::NewtonKrylov  nk;
    const cs::SolveResult r = nk.solve(p);
    REQUIRE_FALSE(r.forcing_history.empty());
    CHECK(r.forcing_history[0] == 0.5);
    for (const double eta : r.forcing_history) {
        CHECK(eta > 0.0);
        CHECK(eta <= 0.9);
    }
}

TEST_CASE("bgs: converges on a contraction", "[solvers]") {
    tsol::CosProblem p;
    cs::BlockGaussSeidel bgs;
    const cs::SolveResult r = bgs.solve(p);
    INFO("iterations " << r.iterations);
    CHECK(r.converged);
    for (const double z : p.z()) CHECK(std::fabs(z - 0.5 * std::cos(z)) < 1e-5);
}

TEST_CASE("solvers: Sellar with Newton at the root", "[solvers]") {
    cs::Model model = cs::create_model<tsol::SellarLoop>(default_config());
    model.setup();
    model.set_solver(std::make_unique<cs::NewtonKrylov>());
    model.run();
    model.wait();
    const tsol::SellarRef ref = tsol::sellar_fixed_point(5.0, 2.0, 1.0);
    const cs::SolveResult& r  = model.solve_result("");
    std::cout << "[solvers] Sellar Newton: iterations " << r.iterations << ", evals " << r.n_evals << ", jvp " << r.n_jvp << "\n";
    CHECK(tsol::rel_close(tsol::scalar_of(model, "d1.y1"), ref.y1, 1e-3));
    CHECK(tsol::rel_close(tsol::scalar_of(model, "d2.y2"), ref.y2, 1e-3));
    CHECK(r.converged);
    CHECK(r.iterations <= 10);
}

TEST_CASE("solvers: Sellar with BlockGaussSeidel", "[solvers]") {
    cs::Model model = cs::create_model<tsol::SellarLoop>(default_config());
    model.setup();
    model.set_solver(std::make_unique<cs::BlockGaussSeidel>());
    model.run();
    model.wait();
    const tsol::SellarRef ref = tsol::sellar_fixed_point(5.0, 2.0, 1.0);
    const cs::SolveResult& r  = model.solve_result("");
    std::cout << "[solvers] Sellar BGS: iterations " << r.iterations << "\n";
    CHECK(tsol::rel_close(tsol::scalar_of(model, "d1.y1"), ref.y1, 1e-3));
    CHECK(tsol::rel_close(tsol::scalar_of(model, "d2.y2"), ref.y2, 1e-3));
    CHECK(r.converged);
    CHECK(r.iterations <= 50);
}

TEST_CASE("solvers: Sellar totals through the cycle", "[solvers]") {
    cs::Model model = cs::create_model<tsol::SellarLoop>(default_config());
    model.setup();
    model.set_solver(std::make_unique<cs::NewtonKrylov>());
    model.run();
    model.wait();

    constexpr double h = 1e-4;
    const auto       f = [](double z0, double z1, double x) { return tsol::sellar_fixed_point(z0, z1, x).f; };
    const double     z0 = 5.0, z1 = 2.0, x = 1.0;
    const double     dz0 = (f(z0 + h, z1, x) - f(z0 - h, z1, x)) / (2 * h);
    const double     dz1 = (f(z0, z1 + h, x) - f(z0, z1 - h, x)) / (2 * h);
    const double     dx  = (f(z0, z1, x + h) - f(z0, z1, x - h)) / (2 * h);

    for (const cs::DerivMode mode : {cs::DerivMode::forward, cs::DerivMode::reverse}) {
        const cs::Jacobian  jac = model.compute_totals(tsol::V{"obj.f"}, tsol::V{"z", "x"}, mode);
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

TEST_CASE("solvers: dense cycle with Newton and BGS", "[solvers]") {
    const tsol::LoopRef ref = tsol::loop_reference();
    const auto          check = [&](cs::Model& model, const char* label) {
        const cs::SolveResult& r = model.solve_result("loop");
        std::cout << "[solvers] dense loop " << label << ": iterations " << r.iterations << ", converged " << r.converged << "\n";
        CHECK(r.converged);
        const std::vector<float> z = tsol::loop_z(model);
        REQUIRE(z.size() == tsol::kN);
        for (std::size_t i = 0; i < tsol::kN; ++i) CHECK(std::fabs(static_cast<double>(z[i]) - ref.z[i]) <= 1e-4);
    };
    SECTION("NewtonKrylov") {
        cs::Model model = cs::create_model<tsol::LoopTop<tsol::Loop>>(default_config());
        model.setup();
        model.set_solver("loop", std::make_unique<cs::NewtonKrylov>());
        model.run();
        model.wait();
        check(model, "Newton");
    }
    SECTION("BlockGaussSeidel") {
        cs::Model model = cs::create_model<tsol::LoopTop<tsol::Loop>>(default_config());
        model.setup();
        model.set_solver("loop", std::make_unique<cs::BlockGaussSeidel>(cs::BgsOptions{200, 1e-6, 1e-6}));
        model.run();
        model.wait();
        check(model, "BGS");
    }
}

TEST_CASE("solvers: dense cycle totals", "[solvers]") {
    const tsol::LoopRef ref = tsol::loop_reference();
    double              ref_max = 0;
    for (const double v : ref.dz_db) ref_max = std::max(ref_max, std::fabs(v));

    cs::Model model = cs::create_model<tsol::LoopTop<tsol::Loop>>(default_config());
    model.setup();
    model.set_solver("loop", std::make_unique<cs::NewtonKrylov>());
    model.run();
    model.wait();
    for (const cs::DerivMode mode : {cs::DerivMode::forward, cs::DerivMode::reverse}) {
        const cs::Jacobian  jac = model.compute_totals(tsol::V{"loop.nl.z"}, tsol::V{"loop.nl.b"}, mode);
        const cs::JacBlock& b   = jac.block("loop.nl.z", "loop.nl.b");
        REQUIRE(b.data.size() == tsol::kN * tsol::kN);
        double worst = 0;
        for (std::size_t k = 0; k < b.data.size(); ++k) worst = std::max(worst, std::fabs(b.data[k] - ref.dz_db[k]));
        INFO((mode == cs::DerivMode::forward ? "forward" : "reverse") << ": largest difference " << worst << ", tolerance " << 1e-3 * ref_max + 1e-4);
        CHECK(worst <= 1e-3 * ref_max + 1e-4);
    }
}

TEST_CASE("solvers: placement", "[solvers]") {
    // The reference: a model-level solver on the subsystem `loop`.
    cs::Model base = cs::create_model<tsol::LoopTop<tsol::Loop>>(default_config());
    base.setup();
    base.set_solver("loop", std::make_unique<cs::NewtonKrylov>());
    base.run();
    base.wait();
    const std::vector<float> want = tsol::loop_z(base);

    SECTION("the System attaches its own solver in setup()") {
        cs::Model model = cs::create_model<tsol::LoopTop<tsol::LoopSelf>>(default_config());
        model.setup();
        model.run();
        model.wait();
        CHECK(model.solve_result("loop").converged);
        const std::vector<float> z = tsol::loop_z(model);
        REQUIRE(z.size() == want.size());
        for (std::size_t i = 0; i < z.size(); ++i) CHECK(std::fabs(z[i] - want[i]) <= 1e-5f);
    }
    SECTION("a solver at the root") {
        cs::Model model = cs::create_model<tsol::LoopTop<tsol::Loop>>(default_config());
        model.setup();
        model.set_solver(std::make_unique<cs::NewtonKrylov>());
        model.run();
        model.wait();
        CHECK(model.solve_result("").converged);
        const std::vector<float> z = tsol::loop_z(model);
        REQUIRE(z.size() == want.size());
        for (std::size_t i = 0; i < z.size(); ++i) CHECK(std::fabs(z[i] - want[i]) <= 1e-5f);
    }
}

TEST_CASE("solvers: failure handling", "[solvers]") {
    SECTION("not converged is reported") {
        cs::Model model = cs::create_model<tsol::SellarLoop>(default_config());
        model.setup();
        model.set_solver(std::make_unique<cs::NewtonKrylov>(cs::NewtonOptions{.max_iter = 0, .atol = 0.0, .rtol = 0.0}));
        model.run();
        model.wait();
        CHECK_FALSE(model.solve_result("").converged);
        CHECK_THROWS_AS(model.solve_result("d1"), std::logic_error);
    }
    SECTION("raise_on_failure throws SolverError from run()") {
        cs::Model model = cs::create_model<tsol::SellarLoop>(default_config());
        model.setup();
        auto nk = std::make_unique<cs::NewtonKrylov>(cs::NewtonOptions{.max_iter = 0, .atol = 0.0, .rtol = 0.0});
        nk->raise_on_failure = true;
        model.set_solver(std::move(nk));
        REQUIRE_THROWS_AS(model.run(), cs::SolverError);
    }
}
