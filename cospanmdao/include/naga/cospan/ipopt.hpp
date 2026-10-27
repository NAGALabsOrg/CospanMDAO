#pragma once

// =============================================================================
// naga/cospan/ipopt.hpp — a reduced-space interior-point optimizer in the style of IPOPT (derivatives, phase 3)
// =============================================================================
//
// Standard library only; all host double, dense linear algebra. A primal-dual interior-point method with a filter line search,
// an L-BFGS Hessian, a simplified feasibility restoration and no second-order correction; slack variables for inequality rows.
// docs/optimizers.md summarizes it; the numbered steps below follow the phase-3 specification.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "naga/cospan/optimizer.hpp"

namespace naga::cospan {

struct IpoptOptions {
    double tol = 1e-4, acceptable_tol = 1e-3;
    int    acceptable_iter = 15, max_iter = 200, lbfgs_history = 6, print_level = 0;
    double mu_init = 0.1, kappa_eps = 10, kappa_mu = 0.2, theta_mu = 1.5, tau_min = 0.99, kappa1 = 1e-2, kappa2 = 1e-2, bound_mult_init = 1.0;
};

namespace detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
inline double norm_inf(const std::vector<double>& a) {
    double m = 0;
    for (const double v : a) m = std::max(m, std::fabs(v));
    return m;
}
inline double norm_1(const std::vector<double>& a) {
    double m = 0;
    for (const double v : a) m += std::fabs(v);
    return m;
}

/// Dense row-major n x n L-BFGS matrix (BFGS with skipping, scalar init) from the pairs (s, y), oldest first.
inline std::vector<double> lbfgs_matrix(std::size_t n, const std::vector<std::pair<std::vector<double>, std::vector<double>>>& pairs) {
    std::vector<double> B(n * n, 0.0);
    if (pairs.empty()) {
        for (std::size_t i = 0; i < n; ++i) B[i * n + i] = 1.0;
        return B;
    }
    const auto& [sn, yn] = pairs.back();
    const double sts     = dot(sn, sn);
    double       sigma   = sts > 0 ? dot(sn, yn) / sts : 1.0;
    if (!std::isfinite(sigma)) sigma = 1.0;
    sigma = std::min(std::max(sigma, 1e-8), 1e8);
    for (std::size_t i = 0; i < n; ++i) B[i * n + i] = sigma;
    std::vector<double> Bs(n);
    for (const auto& [s, y] : pairs) {
        const double sy = dot(s, y);
        if (sy <= 1e-8 * std::sqrt(dot(s, s)) * std::sqrt(dot(y, y))) continue;  // skip a pair with no positive curvature
        for (std::size_t i = 0; i < n; ++i) {
            double v = 0;
            for (std::size_t j = 0; j < n; ++j) v += B[i * n + j] * s[j];
            Bs[i] = v;
        }
        const double sBs = dot(s, Bs);
        if (sBs <= 1e-300) continue;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) B[i * n + j] += -Bs[i] * Bs[j] / sBs + y[i] * y[j] / sy;
    }
    return B;
}

/// Dense LU with partial pivoting; solves A x = b in place (b becomes x). False if a pivot is below 1e-12 max(1, |A|inf) or the
/// result is not finite.
inline bool lu_solve(std::vector<double> A, std::size_t n, std::vector<double>& b) {
    double anorm = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double r = 0;
        for (std::size_t j = 0; j < n; ++j) r += std::fabs(A[i * n + j]);
        anorm = std::max(anorm, r);
    }
    const double tol = 1e-12 * std::max(1.0, anorm);
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t piv = k;
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::fabs(A[i * n + k]) > std::fabs(A[piv * n + k])) piv = i;
        if (!(std::fabs(A[piv * n + k]) >= tol)) return false;
        if (piv != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
            std::swap(b[k], b[piv]);
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double f = A[i * n + k] / A[k * n + k];
            if (f == 0.0) continue;
            for (std::size_t j = k + 1; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            b[i] -= f * b[k];
        }
    }
    for (std::size_t kk = n; kk-- > 0;) {
        double s = b[kk];
        for (std::size_t j = kk + 1; j < n; ++j) s -= A[kk * n + j] * b[j];
        b[kk] = s / A[kk * n + kk];
    }
    for (const double v : b)
        if (!std::isfinite(v)) return false;
    return true;
}

}  // namespace detail

class IpoptOptimizer final : public Optimizer {
public:
    IpoptOptions opt;
    IpoptOptimizer() = default;
    explicit IpoptOptimizer(IpoptOptions o) : opt{o} {}

    OptimizeResult optimize(NlpProblem& p) override {
        using detail::dot;
        using detail::norm_1;
        using detail::norm_inf;
        using Vec = std::vector<double>;
        constexpr double inf = std::numeric_limits<double>::infinity();

        const std::size_t n = p.n_vars(), m = p.n_cons();
        Vec               xl(n), xu(n), gl(m), gu(m), x(n);
        p.bounds(xl, xu, gl, gu);
        p.initial_point(x);

        // ---- 2.1 formulation: equalities, inequalities (with slacks), dropped rows
        std::vector<std::size_t> rowE, rowI;
        for (std::size_t i = 0; i < m; ++i) {
            if (std::isfinite(gl[i]) && std::isfinite(gu[i]) && gl[i] == gu[i]) rowE.push_back(i);
            else if (std::isfinite(gl[i]) || std::isfinite(gu[i])) rowI.push_back(i);
        }
        const std::size_t mc = rowE.size(), md = rowI.size();
        Vec               dl(md), du(md);
        for (std::size_t j = 0; j < md; ++j) {
            dl[j] = gl[rowI[j]];
            du[j] = gu[rowI[j]];
        }
        std::vector<char> hxl(n), hxu(n), hsl(md), hsu(md);
        for (std::size_t i = 0; i < n; ++i) {
            hxl[i] = std::isfinite(xl[i]);
            hxu[i] = std::isfinite(xu[i]);
        }
        for (std::size_t j = 0; j < md; ++j) {
            hsl[j] = std::isfinite(dl[j]);
            hsu[j] = std::isfinite(du[j]);
        }
        std::size_t n_z = 0;
        for (std::size_t i = 0; i < n; ++i) n_z += (hxl[i] ? 1 : 0) + (hxu[i] ? 1 : 0);
        for (std::size_t j = 0; j < md; ++j) n_z += (hsl[j] ? 1 : 0) + (hsu[j] ? 1 : 0);

        OptimizeResult res;
        // The cached evaluation at the current x.
        double f = 0;
        Vec    gf(n), g(m), J(m * n);

        const auto finish = [&](OptStatus st, int iterations) {
            res.status     = st;
            res.iterations = iterations;
            res.x          = x;
            res.f          = f;
            res.g          = g;
            double viol    = 0;
            for (std::size_t i = 0; i < m; ++i) {
                if (std::isfinite(gl[i])) viol = std::max(viol, gl[i] - g[i]);
                if (std::isfinite(gu[i])) viol = std::max(viol, g[i] - gu[i]);
            }
            res.constr_viol = viol;
            return res;
        };
        const auto finite_all = [](const Vec& v) {
            for (const double e : v)
                if (!std::isfinite(e)) return false;
            return true;
        };
        // f and g at a point. False on an evaluation failure or a non-finite value.
        const auto eval_fg = [&](const Vec& xx, double& fo, Vec& go) {
            ++res.n_f;
            if (!p.eval_f(xx, fo) || !std::isfinite(fo)) return false;
            if (m > 0 && (!p.eval_g(xx, go) || !finite_all(go))) return false;
            return true;
        };
        const auto eval_derivs = [&](const Vec& xx, Vec& gfo, Vec& Jo) {
            ++res.n_grad;
            if (!p.eval_grad_f(xx, gfo) || !finite_all(gfo)) return false;
            if (m > 0 && (!p.eval_jac_g(xx, Jo) || !finite_all(Jo))) return false;
            return true;
        };

        // ---- 2.2 initialisation
        const auto push_into = [&](double v, double lo, double hi) {
            const bool hl = std::isfinite(lo), hu = std::isfinite(hi);
            const double width = hl && hu ? hi - lo : inf;
            if (hl) {
                const double pl = std::min(opt.kappa1 * std::max(1.0, std::fabs(lo)), opt.kappa2 * width);
                v               = std::max(v, lo + pl);
            }
            if (hu) {
                const double pu = std::min(opt.kappa1 * std::max(1.0, std::fabs(hi)), opt.kappa2 * width);
                v               = std::min(v, hi - pu);
            }
            return v;
        };
        for (std::size_t i = 0; i < n; ++i) x[i] = push_into(x[i], xl[i], xu[i]);
        if (!eval_fg(x, f, g) || !eval_derivs(x, gf, J)) return finish(OptStatus::invalid_evaluation, 0);

        Vec s(md);
        for (std::size_t j = 0; j < md; ++j) s[j] = push_into(g[rowI[j]], dl[j], du[j]);
        Vec yc(mc, 0.0), yd(md, 0.0), zL(n, opt.bound_mult_init), zU(n, opt.bound_mult_init), vL(md, opt.bound_mult_init), vU(md, opt.bound_mult_init);
        double mu = opt.mu_init;

        const auto theta_of = [&](const Vec& gv, const Vec& sv) {
            double t = 0;
            for (std::size_t k = 0; k < mc; ++k) t += std::fabs(gv[rowE[k]] - gl[rowE[k]]);
            for (std::size_t j = 0; j < md; ++j) t += std::fabs(gv[rowI[j]] - sv[j]);
            return t;
        };
        // The barrier objective; +inf outside the interior.
        const auto phi_of = [&](double fv, const Vec& xv, const Vec& sv, double mu_) {
            double b = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (hxl[i]) {
                    if (!(xv[i] - xl[i] > 0)) return inf;
                    b += std::log(xv[i] - xl[i]);
                }
                if (hxu[i]) {
                    if (!(xu[i] - xv[i] > 0)) return inf;
                    b += std::log(xu[i] - xv[i]);
                }
            }
            for (std::size_t j = 0; j < md; ++j) {
                if (hsl[j]) {
                    if (!(sv[j] - dl[j] > 0)) return inf;
                    b += std::log(sv[j] - dl[j]);
                }
                if (hsu[j]) {
                    if (!(du[j] - sv[j] > 0)) return inf;
                    b += std::log(du[j] - sv[j]);
                }
            }
            return fv - mu_ * b;
        };

        const double theta0     = theta_of(g, s);
        const double theta_max  = 1e4 * std::max(1.0, theta0);
        const double theta_min  = 1e-4 * std::max(1.0, theta0);
        std::vector<std::pair<double, double>> filter{{theta_max, -inf}};

        // J^T [yc; yd] over the rows of the full Jacobian.
        const auto jt_y = [&](const Vec& Jm, const Vec& ycv, const Vec& ydv) {
            Vec out(n, 0.0);
            for (std::size_t k = 0; k < mc; ++k)
                for (std::size_t i = 0; i < n; ++i) out[i] += Jm[rowE[k] * n + i] * ycv[k];
            for (std::size_t j = 0; j < md; ++j)
                for (std::size_t i = 0; i < n; ++i) out[i] += Jm[rowI[j] * n + i] * ydv[j];
            return out;
        };

        // Optimality error E_mu (mu = 0 gives E_0).
        const auto opt_error = [&](double mu_) {
            double zv1 = 0;
            for (std::size_t i = 0; i < n; ++i) zv1 += (hxl[i] ? std::fabs(zL[i]) : 0) + (hxu[i] ? std::fabs(zU[i]) : 0);
            for (std::size_t j = 0; j < md; ++j) zv1 += (hsl[j] ? std::fabs(vL[j]) : 0) + (hsu[j] ? std::fabs(vU[j]) : 0);
            const double dd = static_cast<double>(mc + md + n_z);
            const double sd = dd > 0 ? std::max(100.0, (norm_1(yc) + norm_1(yd) + zv1) / dd) / 100.0 : 1.0;
            const double sc = n_z > 0 ? std::max(100.0, zv1 / static_cast<double>(n_z)) / 100.0 : 1.0;
            Vec          dual = jt_y(J, yc, yd);
            for (std::size_t i = 0; i < n; ++i) dual[i] += gf[i] - (hxl[i] ? zL[i] : 0) + (hxu[i] ? zU[i] : 0);
            double e_dual = norm_inf(dual) / sd;
            double e_s    = 0;
            for (std::size_t j = 0; j < md; ++j) e_s = std::max(e_s, std::fabs((hsu[j] ? vU[j] : 0) - (hsl[j] ? vL[j] : 0) - yd[j]));
            e_dual = std::max(e_dual, e_s / sd);
            double e_pri = 0;
            for (std::size_t k = 0; k < mc; ++k) e_pri = std::max(e_pri, std::fabs(g[rowE[k]] - gl[rowE[k]]));
            for (std::size_t j = 0; j < md; ++j) e_pri = std::max(e_pri, std::fabs(g[rowI[j]] - s[j]));
            double e_c = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (hxl[i]) e_c = std::max(e_c, std::fabs((x[i] - xl[i]) * zL[i] - mu_));
                if (hxu[i]) e_c = std::max(e_c, std::fabs((xu[i] - x[i]) * zU[i] - mu_));
            }
            for (std::size_t j = 0; j < md; ++j) {
                if (hsl[j]) e_c = std::max(e_c, std::fabs((s[j] - dl[j]) * vL[j] - mu_));
                if (hsu[j]) e_c = std::max(e_c, std::fabs((du[j] - s[j]) * vU[j] - mu_));
            }
            return std::max({e_dual, e_pri, e_c / sc});
        };

        const auto filter_ok = [&](double th, double ph) {
            for (const auto& [tf, pf] : filter)
                if (th >= (1.0 - 1e-5) * tf && ph >= pf - 1e-5 * tf) return false;
            return true;
        };

        // The largest alpha in (0,1] keeping value + alpha dir within the fraction-to-boundary of its bounds.
        const auto frac_bound = [&](const Vec& v, const Vec& dir, const Vec& lo, const Vec& hi, const std::vector<char>& hl,
                                    const std::vector<char>& hu, double tau, double a) {
            for (std::size_t i = 0; i < v.size(); ++i) {
                if (hl[i] && dir[i] < 0) a = std::min(a, -tau * (v[i] - lo[i]) / dir[i]);
                if (hu[i] && dir[i] > 0) a = std::min(a, tau * (hi[i] - v[i]) / dir[i]);
            }
            return a;
        };
        const auto frac_mult = [&](const Vec& mv, const Vec& dmv, const std::vector<char>& has, double tau, double a) {
            for (std::size_t i = 0; i < mv.size(); ++i)
                if (has[i] && dmv[i] < 0) a = std::min(a, -tau * mv[i] / dmv[i]);
            return a;
        };

        // Multiplier recalculation by least squares (step 9), when the point is nearly feasible.
        const auto recalc_multipliers = [&]() {
            const std::size_t mh = mc + md;
            if (mh == 0 || theta_of(g, s) > 1e-6) return;
            const std::size_t nn = n + md;
            Vec               Jh(mh * nn, 0.0);
            for (std::size_t k = 0; k < mc; ++k)
                for (std::size_t i = 0; i < n; ++i) Jh[k * nn + i] = J[rowE[k] * n + i];
            for (std::size_t j = 0; j < md; ++j) {
                for (std::size_t i = 0; i < n; ++i) Jh[(mc + j) * nn + i] = J[rowI[j] * n + i];
                Jh[(mc + j) * nn + n + j] = -1.0;
            }
            Vec gw(nn);
            for (std::size_t i = 0; i < n; ++i) gw[i] = gf[i] - (hxl[i] ? zL[i] : 0) + (hxu[i] ? zU[i] : 0);
            for (std::size_t j = 0; j < md; ++j) gw[n + j] = (hsu[j] ? vU[j] : 0) - (hsl[j] ? vL[j] : 0);
            Vec M(mh * mh, 0.0), rhs(mh, 0.0);
            for (std::size_t a = 0; a < mh; ++a) {
                for (std::size_t b = 0; b < mh; ++b) {
                    double v = 0;
                    for (std::size_t i = 0; i < nn; ++i) v += Jh[a * nn + i] * Jh[b * nn + i];
                    M[a * mh + b] = v + (a == b ? 1e-12 : 0.0);
                }
                double v = 0;
                for (std::size_t i = 0; i < nn; ++i) v += Jh[a * nn + i] * gw[i];
                rhs[a] = -v;
            }
            if (!detail::lu_solve(M, mh, rhs) || norm_inf(rhs) > 1000.0) return;
            for (std::size_t k = 0; k < mc; ++k) yc[k] = rhs[k];
            for (std::size_t j = 0; j < md; ++j) yd[j] = rhs[mc + j];
        };

        std::vector<std::pair<Vec, Vec>> pairs;
        int skips = 0, acceptable_count = 0;

        for (int k = 0; k < opt.max_iter; ++k) {
            // 1. evaluation at x is cached (f, g, gf, J)
            // 2. optimality error
            double Emu = opt_error(mu);
            const double E0 = opt_error(0.0);
            if (E0 <= opt.tol) return finish(OptStatus::solved, k);
            acceptable_count = E0 <= opt.acceptable_tol ? acceptable_count + 1 : 0;
            if (acceptable_count >= opt.acceptable_iter) return finish(OptStatus::solved_to_acceptable, k);

            // 3. barrier update
            while (Emu <= opt.kappa_eps * mu && mu > opt.tol / 10.0) {
                mu = std::max(opt.tol / 10.0, std::min(opt.kappa_mu * mu, std::pow(mu, opt.theta_mu)));
                filter.assign(1, {theta_max, -inf});
                Emu = opt_error(mu);
            }

            // 4. Hessian
            const Vec B = detail::lbfgs_matrix(n, pairs);

            // 5. step
            Vec Sx(n, 0.0), Ss(md, 0.0), gphi_x(n), gphi_s(md, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                gphi_x[i] = gf[i];
                if (hxl[i]) {
                    Sx[i] += zL[i] / (x[i] - xl[i]);
                    gphi_x[i] -= mu / (x[i] - xl[i]);
                }
                if (hxu[i]) {
                    Sx[i] += zU[i] / (xu[i] - x[i]);
                    gphi_x[i] += mu / (xu[i] - x[i]);
                }
            }
            Vec tail(md, 0.0);  // mu/(s-dl) - mu/(du-s)
            for (std::size_t j = 0; j < md; ++j) {
                if (hsl[j]) {
                    Ss[j] += vL[j] / (s[j] - dl[j]);
                    gphi_s[j] -= mu / (s[j] - dl[j]);
                    tail[j] += mu / (s[j] - dl[j]);
                }
                if (hsu[j]) {
                    Ss[j] += vU[j] / (du[j] - s[j]);
                    gphi_s[j] += mu / (du[j] - s[j]);
                    tail[j] -= mu / (du[j] - s[j]);
                }
            }
            const std::size_t N = n + mc + md;
            Vec               r(N, 0.0);
            for (std::size_t i = 0; i < n; ++i) r[i] = -gphi_x[i];
            for (std::size_t kk = 0; kk < mc; ++kk) r[n + kk] = -(g[rowE[kk]] - gl[rowE[kk]]);
            for (std::size_t j = 0; j < md; ++j) r[n + mc + j] = -(g[rowI[j]] - s[j]) + tail[j] / Ss[j];

            Vec    sol;
            bool   solved_step = false;
            double dw = 0, dc = 0;
            const auto attempt = [&](double dwv, double dcv) {
                Vec K(N * N, 0.0);
                for (std::size_t i = 0; i < n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) K[i * N + j] = B[i * n + j];
                    K[i * N + i] += Sx[i] + dwv;
                }
                for (std::size_t kk = 0; kk < mc; ++kk)
                    for (std::size_t i = 0; i < n; ++i) {
                        K[i * N + n + kk]  = J[rowE[kk] * n + i];
                        K[(n + kk) * N + i] = J[rowE[kk] * n + i];
                    }
                for (std::size_t kk = 0; kk < mc; ++kk) K[(n + kk) * N + n + kk] = -dcv;
                for (std::size_t j = 0; j < md; ++j) {
                    for (std::size_t i = 0; i < n; ++i) {
                        K[i * N + n + mc + j]          = J[rowI[j] * n + i];
                        K[(n + mc + j) * N + i]        = J[rowI[j] * n + i];
                    }
                    K[(n + mc + j) * N + n + mc + j] = -1.0 / Ss[j] - dcv;
                }
                Vec sv = r;
                if (!detail::lu_solve(K, N, sv)) return false;
                double res_inf = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    double v = -r[i];
                    for (std::size_t j = 0; j < N; ++j) v += K[i * N + j] * sv[j];
                    res_inf = std::max(res_inf, std::fabs(v));
                }
                if (!(res_inf <= 1e-8 * (1.0 + norm_inf(r)))) return false;
                sol = std::move(sv);
                return true;
            };
            if (attempt(0.0, 0.0)) {
                solved_step = true;
            } else {
                dc = 1e-8 * std::pow(mu, 0.25);
                if (attempt(0.0, dc)) {
                    solved_step = true;
                } else {
                    for (dw = 1e-4; dw <= 1e20; dw *= 8.0)
                        if (attempt(dw, dc)) {
                            solved_step = true;
                            break;
                        }
                }
            }
            if (!solved_step) return finish(OptStatus::error, k);

            Vec dx(sol.begin(), sol.begin() + static_cast<std::ptrdiff_t>(n));
            Vec yhc(mc), yhd(md), ds(md), dzL(n, 0.0), dzU(n, 0.0), dvL(md, 0.0), dvU(md, 0.0), dyc(mc), dyd(md);
            for (std::size_t kk = 0; kk < mc; ++kk) {
                yhc[kk] = sol[n + kk];
                dyc[kk] = yhc[kk] - yc[kk];
            }
            for (std::size_t j = 0; j < md; ++j) {
                yhd[j] = sol[n + mc + j];
                dyd[j] = yhd[j] - yd[j];
                ds[j]  = (yhd[j] + tail[j]) / Ss[j];
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (hxl[i]) dzL[i] = mu / (x[i] - xl[i]) - zL[i] - zL[i] / (x[i] - xl[i]) * dx[i];
                if (hxu[i]) dzU[i] = mu / (xu[i] - x[i]) - zU[i] + zU[i] / (xu[i] - x[i]) * dx[i];
            }
            for (std::size_t j = 0; j < md; ++j) {
                if (hsl[j]) dvL[j] = mu / (s[j] - dl[j]) - vL[j] - vL[j] / (s[j] - dl[j]) * ds[j];
                if (hsu[j]) dvU[j] = mu / (du[j] - s[j]) - vU[j] + vU[j] / (du[j] - s[j]) * ds[j];
            }

            const double theta = theta_of(g, s);
            const double phi   = phi_of(f, x, s, mu);

            // 10. stationary infeasible point
            double dmax = std::max(norm_inf(dx), norm_inf(ds));
            if (dmax < 1e-14 && theta > opt.tol) return finish(OptStatus::infeasible_stationary, k);

            // 6. line search
            const double tau = std::max(opt.tau_min, 1.0 - mu);
            double       amax = frac_bound(x, dx, xl, xu, hxl, hxu, tau, 1.0);
            amax              = frac_bound(s, ds, dl, du, hsl, hsu, tau, amax);
            double az         = frac_mult(zL, dzL, hxl, tau, 1.0);
            az                = frac_mult(zU, dzU, hxu, tau, az);
            az                = frac_mult(vL, dvL, hsl, tau, az);
            az                = frac_mult(vU, dvU, hsu, tau, az);
            double gd = dot(gphi_x, dx) + dot(gphi_s, ds);

            constexpr double gth = 1e-5, gph = 1e-5, eta_phi = 1e-8, delta = 1.0, s_phi = 2.3, s_th = 1.1, g_alpha = 0.05;
            double alpha_min;
            if (gd < 0 && theta <= theta_min)
                alpha_min = g_alpha * std::min({gth, gph * theta / (-gd), delta * std::pow(theta, s_th) / std::pow(-gd, s_phi)});
            else if (gd < 0)
                alpha_min = g_alpha * std::min(gth, gph * theta / (-gd));
            else
                alpha_min = g_alpha * gth;

            bool   accepted = false, h_step = false;
            double alpha = amax, ftr = 0;
            Vec    xt(n), st(md), gtr(m);
            while (alpha >= alpha_min) {
                for (std::size_t i = 0; i < n; ++i) xt[i] = x[i] + alpha * dx[i];
                for (std::size_t j = 0; j < md; ++j) st[j] = s[j] + alpha * ds[j];
                if (!eval_fg(xt, ftr, gtr)) {
                    alpha /= 2.0;
                    continue;
                }
                const double th_t = theta_of(gtr, st), ph_t = phi_of(ftr, xt, st, mu);
                if (!std::isfinite(ph_t) || !filter_ok(th_t, ph_t)) {
                    alpha /= 2.0;
                    continue;
                }
                const bool switching = gd < 0 && alpha * std::pow(-gd, s_phi) > delta * std::pow(theta, s_th);
                if (switching && theta <= theta_min) {
                    if (ph_t <= phi + eta_phi * alpha * gd) {
                        accepted = true;
                    }
                } else if (th_t <= (1.0 - gth) * theta || ph_t <= phi - gph * theta) {
                    accepted = true;
                    h_step   = true;
                }
                if (accepted) break;
                alpha /= 2.0;
            }

            if (accepted) {
                // 7. accept
                if (h_step) filter.emplace_back((1.0 - gth) * theta, phi - gph * theta);
                const Vec x_old = x, gf_old = gf, J_old = J;
                x = xt;
                s = st;
                for (std::size_t kk = 0; kk < mc; ++kk) yc[kk] += alpha * dyc[kk];
                for (std::size_t j = 0; j < md; ++j) yd[j] += alpha * dyd[j];
                for (std::size_t i = 0; i < n; ++i) {
                    zL[i] += az * dzL[i];
                    zU[i] += az * dzU[i];
                }
                for (std::size_t j = 0; j < md; ++j) {
                    vL[j] += az * dvL[j];
                    vU[j] += az * dvU[j];
                }
                constexpr double kS = 1e10;
                const auto clip = [&](double v, double gap) { return std::min(std::max(v, mu / (kS * gap)), kS * mu / gap); };
                for (std::size_t i = 0; i < n; ++i) {
                    if (hxl[i]) zL[i] = clip(zL[i], x[i] - xl[i]);
                    if (hxu[i]) zU[i] = clip(zU[i], xu[i] - x[i]);
                }
                for (std::size_t j = 0; j < md; ++j) {
                    if (hsl[j]) vL[j] = clip(vL[j], s[j] - dl[j]);
                    if (hsu[j]) vU[j] = clip(vU[j], du[j] - s[j]);
                }
                f = ftr;
                g = gtr;
                if (!eval_derivs(x, gf, J)) return finish(OptStatus::invalid_evaluation, k);

                // the L-BFGS pair
                Vec sk(n), yk = jt_y(J, yc, yd);
                const Vec old_l = jt_y(J_old, yc, yd);
                for (std::size_t i = 0; i < n; ++i) {
                    sk[i] = x[i] - x_old[i];
                    yk[i] = (gf[i] + yk[i]) - (gf_old[i] + old_l[i]);
                }
                if (dot(sk, yk) <= 1e-8 * std::sqrt(dot(sk, sk)) * std::sqrt(dot(yk, yk))) {
                    if (++skips > 2) pairs.clear();
                } else {
                    skips = 0;
                    pairs.emplace_back(std::move(sk), std::move(yk));
                    if (static_cast<int>(pairs.size()) > opt.lbfgs_history) pairs.erase(pairs.begin());
                }
                recalc_multipliers();
            } else {
                // 8. restoration
                filter.emplace_back(theta, phi);
                bool restored = false;
                for (int it = 0; it < 20 && !restored; ++it) {
                    const std::size_t mh = mc + md, nn = n + md;
                    const double      th_cur = theta_of(g, s);
                    Vec               h(mh), Jh(mh * nn, 0.0);
                    for (std::size_t kk = 0; kk < mc; ++kk) {
                        h[kk] = g[rowE[kk]] - gl[rowE[kk]];
                        for (std::size_t i = 0; i < n; ++i) Jh[kk * nn + i] = J[rowE[kk] * n + i];
                    }
                    for (std::size_t j = 0; j < md; ++j) {
                        h[mc + j] = g[rowI[j]] - s[j];
                        for (std::size_t i = 0; i < n; ++i) Jh[(mc + j) * nn + i] = J[rowI[j] * n + i];
                        Jh[(mc + j) * nn + n + j] = -1.0;
                    }
                    Vec D(nn, 0.0);
                    if (mh > 0) {
                        Vec M(mh * mh, 0.0), w = h;
                        for (std::size_t a = 0; a < mh; ++a)
                            for (std::size_t b = 0; b < mh; ++b) {
                                double v = 0;
                                for (std::size_t i = 0; i < nn; ++i) v += Jh[a * nn + i] * Jh[b * nn + i];
                                M[a * mh + b] = v + (a == b ? 1e-8 : 0.0);
                            }
                        if (!detail::lu_solve(M, mh, w)) break;
                        for (std::size_t i = 0; i < nn; ++i) {
                            double v = 0;
                            for (std::size_t a = 0; a < mh; ++a) v += Jh[a * nn + i] * w[a];
                            D[i] = -v;
                        }
                    }
                    Vec rdx(D.begin(), D.begin() + static_cast<std::ptrdiff_t>(n)), rds(D.begin() + static_cast<std::ptrdiff_t>(n), D.end());
                    double ra = frac_bound(x, rdx, xl, xu, hxl, hxu, tau, 1.0);
                    ra        = frac_bound(s, rds, dl, du, hsl, hsu, tau, ra);
                    bool moved = false;
                    for (int hv = 0; hv <= 10; ++hv, ra /= 2.0) {
                        for (std::size_t i = 0; i < n; ++i) xt[i] = x[i] + ra * rdx[i];
                        for (std::size_t j = 0; j < md; ++j) st[j] = s[j] + ra * rds[j];
                        double ft;
                        Vec    gt(m);
                        if (!eval_fg(xt, ft, gt)) continue;
                        if (theta_of(gt, st) <= 0.9 * th_cur) {
                            x = xt;
                            s = st;
                            f = ft;
                            g = gt;
                            moved = true;
                            break;
                        }
                    }
                    if (!moved) break;
                    if (!eval_derivs(x, gf, J)) return finish(OptStatus::invalid_evaluation, k);
                    if (filter_ok(theta_of(g, s), phi_of(f, x, s, mu))) restored = true;
                }
                if (!restored) return finish(OptStatus::restoration_failed, k);
                std::fill(zL.begin(), zL.end(), opt.bound_mult_init);
                std::fill(zU.begin(), zU.end(), opt.bound_mult_init);
                std::fill(vL.begin(), vL.end(), opt.bound_mult_init);
                std::fill(vU.begin(), vU.end(), opt.bound_mult_init);
                std::fill(yc.begin(), yc.end(), 0.0);
                std::fill(yd.begin(), yd.end(), 0.0);
                alpha = 0.0;
                recalc_multipliers();
            }

            if (opt.print_level >= 1)
                std::fprintf(stderr, "%4d f=%.8e theta=%.3e E=%.3e mu=%.3e |dx|=%.3e alpha=%.3e\n", k, f, theta_of(g, s), Emu, mu, dmax, alpha);
        }
        return finish(OptStatus::max_iter, opt.max_iter);
    }
};

}  // namespace naga::cospan
