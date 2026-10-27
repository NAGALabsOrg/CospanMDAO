# Derivatives (phase 1)

A component may define derivative kernels next to `compute_forward`. `Model::compute_totals` chains them over the model into
Jacobian blocks, and `Model::check_partials` tests one component's kernels. Phase 1 covers partials, activity analysis and
forward/reverse totals. There are no solvers and no optimizers yet.

## The two kernels

Both are optional static members. They take **the same item** as `compute_forward` (`sycl::id<R>` or `cs::GroupItem<R>`) and, for
a group-range component, the same `LocalMem` as the fourth parameter.

```cpp
// forward mode (a Jacobian-vector product): dy += J dx
static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dy);
// reverse mode (a vector-Jacobian product): xbar += J^T ybar
static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar);
```

`x` is the forward's own input data. `dx`, `dy`, `ybar` and `xbar` are the same port types pointed at derivative buffers. For
`y = a x^2`:

```cpp
static void compute_jvp(sycl::id<1> i, const Inputs& x, const Inputs& dx, const Outputs& dy) {
    dy.y.add(i, 2.0f * x.a[0] * x.x[i] * dx.x[i] + x.x[i] * x.x[i] * dx.a[0]);
}
static void compute_vjp(sycl::id<1> i, const Inputs& x, const Outputs& ybar, const Inputs& xbar) {
    const float g = ybar.y.get(i);
    xbar.x.add(i, 2.0f * x.a[0] * x.x[i] * g);
    xbar.a.add(std::size_t{0}, x.x[i] * x.x[i] * g);   // a is shared by every i: accumulate
}
```

## Accumulate only

A derivative buffer is zeroed before a sweep. A kernel **reads** with `operator[]` (tangent inputs, `x`) or `Output::get`
(`ybar`) and **writes only with `add`**, an atomic relaxed device-scope `fetch_add`. `Input::add` and `Output::add` exist for
this; never assign into a derivative buffer. Pass flat indices as `std::size_t`.

## Inactive variables

A variable is active when it is reachable from the free inputs asked about (`wrt`) and reaches the outputs asked about (`of`).
For an inactive variable the kernel reads zeros (a shared per-device zero buffer) and its writes land in a per-device sink
buffer. A component launches only if it has at least one active input and at least one active output; otherwise it is skipped.
So a component with no derivative kernels is an error only when it is on an active path: `DerivativeError: 'nd' needs
compute_jvp`.

## compute_totals

```cpp
cs::Jacobian J = model.compute_totals(of, wrt, cs::DerivMode::automatic);   // vectors of root-relative paths
const cs::JacBlock& b = J.block("obj.f", "z");                               // rows = numel(of), cols = numel(wrt), row-major
```

- `wrt` must be free inputs (or a tie alias), `of` outputs, all dense f32. Paths resolve as `add_design_var` does. `block()`
  throws `std::out_of_range` for a pair that was not asked.
- A prior `run()` is required (else `std::logic_error`); the model is waited on first.
- Mode: `forward` does one sweep per element of `wrt` (JVP, components in run order); `reverse` one per element of `of` (VJP,
  reverse order). `automatic` is reverse iff the total elements of `of` is less than that of `wrt`.
- `compute_totals(mode)` uses the declared objectives and constraints as `of` and the design variables as `wrt`.
- `last_deriv_stats()` reports the active variables and the components the last sweep launched.

## check_partials

`model.check_partials("c")` tests one component whose inputs are all free. With random directions (xorshift64\*, `CheckOptions`)
it compares the JVP with a central difference of the forward (step `h_rel * max(1, max|x|)`): a direction passes if
`||jvp - fd|| <= rtol * max(||fd||, ||jvp||) + atol`. If the component has `compute_vjp` it also runs the dot test
`<JVP(v), w> = <v, VJP(w)>`, which passes if the two sides differ by at most `1e-3 (|s1| + |s2|) + 1e-6`. Defaults: `h_rel 1e-2`,
`rtol 2e-2`, `atol 1e-3`, 3 directions. The report has `fd_pass`, `dot_pass` and the errors. The model is left with its original
inputs.

## Limits of phase 1

- No cycles: a cycle on the derivative path throws `DerivativeError` (a solver is needed).
- No ragged ports. f32 only. One device per variable.
- No run policies in scope (`DerivativeError`).
- A derivative launch uses the forward's launch configuration (range, work-group shape, local memory).
- Custom-op and host components have no derivatives.
