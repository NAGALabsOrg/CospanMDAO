#pragma once

// =============================================================================
// naga/cospan/solver.hpp — nonlinear solvers for cycles (derivatives, phase 2). Standard library only.
// =============================================================================
//
// A cycle is a nonlinear system r(z) = z - G(z) = 0 over the cycle variables z (host doubles). The Model builds the
// NonlinearProblem (model.hpp, SystemProblem); a Solver drives it. GMRES is the linear solver of NewtonKrylov and of the
// coupled derivative solves.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace naga::cospan {

struct SolverError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// A nonlinear system r(z) = z - G(z) = 0 over z in R^n (host doubles).
class NonlinearProblem {
public:
    virtual ~NonlinearProblem() = default;
    virtual std::size_t size() const                                              = 0;
    virtual void        get_state(std::span<double> z)                            = 0;  // the current state
    virtual void        evaluate(std::span<const double> z, std::span<double> r)  = 0;  // state := z, r = z - G(z)
    virtual void        jvp(std::span<const double> v, std::span<double> out)     = 0;  // (I - G_z) v at the last evaluated point
    virtual void        vjp(std::span<const double> v, std::span<double> out)     = 0;  // (I - G_z)^T v at the last evaluated point
};

struct SolveResult {
    bool                converged  = false;
    int                 iterations = 0, n_evals = 0, n_jvp = 0;
    double              r0_norm = 0, r_norm = 0;
    std::vector<double> forcing_history;
    std::string         message;
};

class Solver {
public:
    virtual ~Solver()                                  = default;
    virtual SolveResult solve(NonlinearProblem& p)     = 0;
    virtual std::string name() const                   = 0;
    bool                raise_on_failure = false;
};

struct GmresOptions {
    int    restart = 30, max_iter = 300;
    double rtol = 1e-5, atol = 1e-10;
};
struct GmresResult {
    bool   converged = false;
    int    iterations = 0;
    double residual_norm = 0;
};

namespace detail_solver {
inline double dot(std::span<const double> a, std::span<const double> b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline double norm2(std::span<const double> a) { return std::sqrt(dot(a, a)); }
}  // namespace detail_solver

/// Restarted GMRES(m): modified Gram-Schmidt, Givens rotations, no preconditioner; x is the initial guess and the result.
inline GmresResult gmres(std::size_t n, const std::function<void(std::span<const double>, std::span<double>)>& apply,
                         std::span<const double> b, std::span<double> x, const GmresOptions& o) {
    using detail_solver::dot;
    using detail_solver::norm2;
    const double tol   = std::max(o.atol, o.rtol * norm2(b));
    int          total = 0;
    const int    m     = std::max(1, o.restart);

    std::vector<double>              r(n), w(n);
    std::vector<std::vector<double>> V(static_cast<std::size_t>(m) + 1, std::vector<double>(n));
    std::vector<std::vector<double>> H(static_cast<std::size_t>(m) + 1, std::vector<double>(static_cast<std::size_t>(m)));
    std::vector<double>              g(static_cast<std::size_t>(m) + 1), cs(static_cast<std::size_t>(m)), sn(static_cast<std::size_t>(m));

    while (true) {
        apply(std::span<const double>{x.data(), n}, std::span<double>{w});
        for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - w[i];
        const double beta = norm2(r);
        if (beta <= tol) return {true, total, beta};
        if (total >= o.max_iter) return {false, total, beta};

        for (std::size_t i = 0; i < n; ++i) V[0][i] = r[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);

        int k = 0;  // columns built
        for (int jj = 0; jj < m; ++jj) {
            if (total >= o.max_iter) break;
            const auto j = static_cast<std::size_t>(jj);
            apply(std::span<const double>{V[j]}, std::span<double>{w});
            ++total;
            for (std::size_t i = 0; i <= j; ++i) {
                H[i][j] = dot(w, V[i]);
                for (std::size_t q = 0; q < n; ++q) w[q] -= H[i][j] * V[i][q];
            }
            const double hn = norm2(w);
            H[j + 1][j]     = hn;
            for (std::size_t i = 0; i < j; ++i) {  // the earlier rotations on column j
                const double t = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                H[i + 1][j]    = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                H[i][j]        = t;
            }
            const double denom = std::hypot(H[j][j], hn);
            if (denom == 0.0) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = H[j][j] / denom;
                sn[j] = hn / denom;
            }
            H[j][j]     = denom;
            H[j + 1][j] = 0.0;
            g[j + 1]    = -sn[j] * g[j];
            g[j]        = cs[j] * g[j];
            k           = jj + 1;
            const double res = std::fabs(g[j + 1]);
            if (hn < 1e-14 * beta) break;  // happy breakdown
            for (std::size_t q = 0; q < n; ++q) V[j + 1][q] = w[q] / hn;
            if (res <= tol) break;
        }

        std::vector<double> y(static_cast<std::size_t>(k), 0.0);
        for (int ii = k - 1; ii >= 0; --ii) {
            const auto i = static_cast<std::size_t>(ii);
            double     s = g[i];
            for (std::size_t l = i + 1; l < static_cast<std::size_t>(k); ++l) s -= H[i][l] * y[l];
            y[i] = H[i][i] == 0.0 ? 0.0 : s / H[i][i];
        }
        for (std::size_t i = 0; i < y.size(); ++i)
            for (std::size_t q = 0; q < n; ++q) x[q] += y[i] * V[i][q];
    }
}

struct NewtonOptions {
    int          max_iter = 30;
    double       atol = 1e-5, rtol = 1e-5;
    GmresOptions gmres{};
    double       eta_0 = 0.5, eta_max = 0.9, ew_gamma = 0.9;
    int          max_ls = 10;
    double       armijo = 1e-4;
};

/// Inexact Newton-Krylov with Eisenstat-Walker forcing and a backtracking (Armijo) line search.
class NewtonKrylov final : public Solver {
public:
    NewtonOptions opt;
    NewtonKrylov() = default;
    explicit NewtonKrylov(NewtonOptions o) : opt{o} {}

    SolveResult solve(NonlinearProblem& p) override {
        using detail_solver::norm2;
        const std::size_t   n = p.size();
        SolveResult         res;
        std::vector<double> z(n), F(n), d(n), zt(n), Ft(n), rhs(n);
        p.get_state(z);
        p.evaluate(z, F);
        res.n_evals = 1;
        double fn   = norm2(F);
        res.r0_norm = fn;
        const double tol = std::max(opt.atol, opt.rtol * fn);
        double       prev_fn = fn, prev_eta = opt.eta_0;

        const auto finish = [&](bool ok, const char* msg) {
            res.converged = ok;
            res.r_norm    = fn;
            res.message   = msg;
            return res;
        };
        for (int k = 0; k < opt.max_iter; ++k) {
            if (fn <= tol) return finish(true, "converged");
            double eta;
            if (k == 0) {
                eta = opt.eta_0;
            } else {
                eta = opt.ew_gamma * (fn / prev_fn) * (fn / prev_fn);
                if (opt.ew_gamma * prev_eta * prev_eta > 0.1) eta = std::max(eta, opt.ew_gamma * prev_eta * prev_eta);
            }
            eta = std::max(eta, 0.5 * tol / fn);
            eta = std::min(eta, opt.eta_max);
            res.forcing_history.push_back(eta);
            prev_eta = eta;

            for (std::size_t i = 0; i < n; ++i) rhs[i] = -F[i];
            std::fill(d.begin(), d.end(), 0.0);
            GmresOptions go = opt.gmres;
            go.rtol         = eta;
            go.atol         = 0.0;
            gmres(n,
                  [&](std::span<const double> in, std::span<double> out) {
                      ++res.n_jvp;
                      p.jvp(in, out);
                  },
                  rhs, d, go);

            double lambda   = 1.0;
            bool   accepted = false;
            for (int t = 0; t <= opt.max_ls; ++t) {
                for (std::size_t i = 0; i < n; ++i) zt[i] = z[i] + lambda * d[i];
                p.evaluate(zt, Ft);
                ++res.n_evals;
                const double ftn = norm2(Ft);
                if (ftn <= (1.0 - opt.armijo * lambda) * fn) {
                    z        = zt;
                    F        = Ft;
                    prev_fn  = fn;
                    fn       = ftn;
                    accepted = true;
                    ++res.iterations;
                    break;
                }
                lambda /= 2.0;
            }
            if (!accepted) {
                p.evaluate(z, F);
                ++res.n_evals;
                fn = norm2(F);
                return finish(false, "line search failed");
            }
        }
        // Deliberate addition to the loop above (it checks convergence only at the top of an iteration): the step taken in the
        // last allowed iteration may have converged, so check once more before giving up on max_iter.
        if (fn <= tol) return finish(true, "converged");
        return finish(false, "max_iter reached");
    }
    std::string name() const override { return "NewtonKrylov"; }
};

struct BgsOptions {
    int    max_iter = 50;
    double atol = 1e-5, rtol = 1e-5;
};

/// Fixed-point (block Gauss-Seidel) iteration z <- G(z).
class BlockGaussSeidel final : public Solver {
public:
    BgsOptions opt;
    BlockGaussSeidel() = default;
    explicit BlockGaussSeidel(BgsOptions o) : opt{o} {}

    SolveResult solve(NonlinearProblem& p) override {
        using detail_solver::norm2;
        const std::size_t   n = p.size();
        SolveResult         res;
        std::vector<double> z(n), r(n);
        p.get_state(z);
        p.evaluate(z, r);
        res.n_evals = 1;
        double rn   = norm2(r);
        res.r0_norm = rn;
        const double tol = std::max(opt.atol, opt.rtol * rn);
        for (int k = 0; k < opt.max_iter; ++k) {
            if (rn <= tol) break;
            for (std::size_t i = 0; i < n; ++i) z[i] -= r[i];  // G(z)
            p.evaluate(z, r);
            ++res.n_evals;
            ++res.iterations;
            rn = norm2(r);
        }
        res.r_norm    = rn;
        res.converged = rn <= tol;
        res.message   = res.converged ? "converged" : "max_iter reached";
        return res;
    }
    std::string name() const override { return "BlockGaussSeidel"; }
};

}  // namespace naga::cospan
