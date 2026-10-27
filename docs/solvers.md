# Solvers (phase 2)

A cycle (a variable read by a component that runs before its producer) is, without a solver, only a one-pass approximation: the
reader sees the producer's previous value. A solver attached to a System iterates the cycles that System owns to convergence.

## The residual form

Let `z` be the cycle variables the System owns (concatenated, in double) and `G(z)` the values their producers write when the
System's children run once with `z` in place. The solver drives

    r(z) = z - G(z) = 0.

`NonlinearProblem` (`solver.hpp`, standard library only) is this system: `get_state`, `evaluate(z, r)`, and the products
`jvp(v) = (I - G_z) v` and `vjp(v) = (I - G_z)^T v` at the last evaluated point.

## Attaching a solver

- Inside a System's `setup()`: `set_solver(std::make_unique<cs::NewtonKrylov>());`
- From main, after the model's `setup()`: `model.set_solver("path.to.system", std::make_unique<...>())`, and
  `model.set_solver(...)` for the root. A handle has `set_solver` too. A leaf throws `std::invalid_argument`.
- `model.solve_result("path")` returns the last `SolveResult` at that node (`std::logic_error` if none was recorded).
- `solver->raise_on_failure = true` makes `run()` throw `SolverError` when a solve does not converge.

## Ownership

A cycle variable is owned by the **deepest solved System above its producer (in scope) whose leaf range contains the variable's
earliest reader**. A cycle variable must have exactly one producer, be dense f32 and of known shape (`SolverError` otherwise). A
run policy anywhere inside a solved System is an error (`SolverError`).

## What `run()` does at a solved System

1. The **initial pass**: the System's children run once, every time (a warm start from the previous solve's state).
2. If the System owns no cycle variable, the result is "no cycle variables".
3. Otherwise the solver iterates; every `evaluate` writes `z` into the producers' data, runs the children once and reads `G(z)`
   back.

## Solvers

**NewtonKrylov** (`NewtonOptions`): inexact Newton with GMRES(`restart`, `max_iter`) on `J d = -F`, `J = I - G_z` (matrix-free,
through `jvp`).
- Forcing term: `eta_0` at the first step; then `ew_gamma * (|F_k| / |F_{k-1}|)^2`, safeguarded by
  `ew_gamma * eta_{k-1}^2` when that is above 0.1, floored at `0.5 * tol / |F_k|` and capped at `eta_max`. GMRES runs with
  relative tolerance `eta` and no absolute tolerance.
- Line search: backtracking from lambda = 1, halving, accepting when `|F(z + lambda d)| <= (1 - armijo * lambda) |F|`, at most
  `max_ls` halvings. If none is accepted the state is restored and the solve reports "line search failed".
- Convergence: `|F| <= max(atol, rtol * |F_0|)`, checked at the top of every iteration and once more after the last allowed one.

**BlockGaussSeidel** (`BgsOptions`): the fixed-point iteration `z <- G(z)`, with the same convergence test. It needs no
derivatives.

`SolveResult` has `converged`, `iterations`, `n_evals`, `n_jvp`, `r0_norm`, `r_norm`, `forcing_history` and `message`.

## Totals through cycles

`compute_totals` no longer throws on a cycle. At a converged state `z = G(x, z)`:
- forward mode solves `(I - G_z) dz = G_x dx` by GMRES (one coupled solve per element of `wrt`);
- reverse mode solves `(I - G_z^T) delta = p1` (one per element of `of`), where `p1` is what the first reverse sweep leaves on the
  readers placed before the producer, and seeds the producer side with the original seed plus `delta`.

These totals **assume the cycle is converged**: run the model with a solver first. The coupled solves use GMRES with the
options set by `model.set_linear_options(GmresOptions)` (default `restart 30, max_iter 300, rtol 1e-5, atol 1e-10`); if one
does not converge, `DerivativeError` is thrown.

## Limitation: the Jacobian at G(z)

While iterating, a cycle variable's buffer holds `G(z)`, so the derivative kernels of the readers placed **before** the producer
see `G(z)` as their primal input instead of `z`. The difference is the residual, so the Jacobian (and the products `jvp` and
`vjp` the Newton solver uses) is exact at convergence.
