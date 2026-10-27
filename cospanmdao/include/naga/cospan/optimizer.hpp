#pragma once

// =============================================================================
// naga/cospan/optimizer.hpp — the optimizer interface (derivatives, phase 3). Standard library only.
// =============================================================================
//
// An NlpProblem is  min f(x)  s.t.  gl <= g(x) <= gu,  xl <= x <= xu  over host doubles. Model::optimize (model.hpp) builds one
// from the model's design variables, objective and constraints; an Optimizer solves it.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace naga::cospan {

struct OptimizerError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class NlpProblem {
public:
    virtual ~NlpProblem()                = default;
    virtual std::size_t n_vars() const   = 0;
    virtual std::size_t n_cons() const   = 0;
    /// +-infinity allowed (std::numeric_limits<double>::infinity()); an equality has gl[i] == gu[i].
    virtual void bounds(std::span<double> xl, std::span<double> xu, std::span<double> gl, std::span<double> gu) const = 0;
    virtual void initial_point(std::span<double> x) const                                                      = 0;
    /// An eval_* that returns false is an evaluation failure.
    virtual bool eval_f(std::span<const double> x, double& f)                    = 0;
    virtual bool eval_g(std::span<const double> x, std::span<double> g)          = 0;
    virtual bool eval_grad_f(std::span<const double> x, std::span<double> gf)    = 0;
    virtual bool eval_jac_g(std::span<const double> x, std::span<double> J)      = 0;  // dense row-major m x n
};

enum class OptStatus { solved, solved_to_acceptable, max_iter, infeasible_stationary, restoration_failed, invalid_evaluation, error };

struct OptimizeResult {
    OptStatus           status = OptStatus::error;
    double              f      = 0;
    std::vector<double> x, g;
    double              constr_viol = 0;
    int                 iterations = 0, n_f = 0, n_grad = 0;
};

class Optimizer {
public:
    virtual ~Optimizer()                              = default;
    virtual OptimizeResult optimize(NlpProblem& p)    = 0;
};

}  // namespace naga::cospan
