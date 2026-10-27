# Optimizers

Header: `naga/cospan/optimizer.hpp` (the interface, std only), `naga/cospan/ipopt.hpp` (the IPOPT-style optimizer, std only),
and `Model::optimize` in `model.hpp`.

## Interface

`NlpProblem` describes `min f(x)  s.t.  gl <= g(x) <= gu,  xl <= x <= xu`:

- `n_vars()`, `n_cons()`;
- `bounds(xl, xu, gl, gu)`: `+-inf` for an absent bound; an equality has `gl == gu`;
- `initial_point(x)`;
- `eval_f(x, f)`, `eval_g(x, g)`, `eval_grad_f(x, grad)`, `eval_jac_g(x, J)`. `J` is a dense row-major `m x n` matrix.

An `eval_*` returns `false` when the point cannot be evaluated (see "Evaluation failure").

`Optimizer::optimize(NlpProblem&)` returns an `OptimizeResult`: `status` (`OptStatus`), `x`, `f`, `g`, `constr_viol`,
`iterations`, `n_f`, `n_grad`. `OptimizerError` reports misuse.

## Model::optimize

`OptimizeResult Model::optimize(Optimizer&)` optimizes the marked design variables for the one marked objective, subject to the
marked constraints. It needs exactly one objective and at least one design variable (else `OptimizerError`). The model is run
first if it has not run.

Mark mapping:

- `x` is the design variables flattened in mark order; `g` is the constraints flattened in mark order. The element count of a
  mark is `shape.elements()`.
- Scaling is OpenMDAO's affine form per element: `x_s = (x - ref0) / (ref - ref0)`, for design variables, the objective and
  constraints. `ref == ref0` throws `OptimizerError`.
- Bounds are broadcast from size 1 and scaled; lower and upper swap when `ref < ref0`. A constraint with `equals` has
  `gl = gu =` the scaled value.
- The initial point is the current value of each design variable, scaled.

On return the model holds the result's `x` (run and waited), and `x`, `f`, `g` and `constr_viol` are in the model's UNscaled
units.

## Reduced-space formulation

Only the design variables are optimizer variables. Every evaluation writes `x` into the model, runs it and waits, so the
model, solvers included (Newton, block Gauss-Seidel), is converged inside every evaluation. Gradients come from
`compute_totals(DerivMode::automatic)`, once per point, with the chain rule for the scaling applied per element. Evaluations
are cached: the same `x` (bitwise) is not re-run.

## IPOPT

`IpoptOptimizer` follows Wachter and Biegler's primal-dual interior-point method, in double precision on the host:

- slacks for inequality rows, bounds handled by a log barrier;
- filter line search;
- L-BFGS approximation of the Lagrangian Hessian (dense, from the stored pairs; no exact second derivatives);
- augmented KKT system solved by dense LU, with inertia-free regularisation;
- a simplified feasibility restoration (minimises the constraint violation);
- no second-order correction.

`IpoptOptions` has the tolerances (`tol`, `acceptable_tol`), `max_iter` and the barrier, L-BFGS memory and line-search
parameters. Defaults suit models evaluated in f32: the model's values carry about 1e-7 relative error, so the default
tolerances are looser than IPOPT's own, and `solved_to_acceptable` is a normal outcome.

## Evaluation failure

An evaluation fails (returns `false`) when any node's last solve did not converge, when `f` or `g` is not finite, or when
`compute_totals` throws `DerivativeError`. The line search treats a failed trial point as rejected and backtracks; a failure at
the initial point, or one that persists, ends the run with a failure status.
