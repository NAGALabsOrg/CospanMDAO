# Triage: OpenMP `device scan` failure (2026-10-26)

Investigation only. No change to `device_scan.hpp`, the fork (`/home/jack/Projects/acpp-fork`, `naga/develop`, read-only) or any
test. Tree: `feat/derivatives` at 86cbe55 (b937876 plus the derivatives work, which does not touch the scan).

## Symptom

`device scan: exclusive scan of u32, n + 1 outputs, the last is the total` (`tests/test_device_scan.cpp:65`, tag `[scan]`)
fails on OpenMP (`ACPP_VISIBILITY_MASK=omp`, default env) at n = 1000 and n = 4099 (the CHECK at line 70). Every other n in that
test passes (0, 1, 2, 37, 255, 256, 257), as do the other two `[scan]` cases (including 256*256+17). CUDA and OpenCL pass.
Jack ran this suite on this machine before without this failure.

Machine: Intel Core Ultra 7 265, 20 logical CPUs (`sys cpu` and `/proc/cpuinfo`; `Cpus_allowed_list: 0-19`).
`nproc` is not installed in this shell (no coreutils), so that figure is from those sources.

## Measurements (verbatim)

### 1. `[scan]` 20 times on OpenMP, nothing else running

```
run 1 .. run 20: test cases:  3 |  2 passed | 1 failed      (20 of 20 runs failed)
```

Mismatch counts and indices, parsed from the Catch2 expansion of every failing CHECK. All 20 runs gave **one** signature
(`distinct_signatures: 1` for each n):

```
n = 1000: 12 mismatches
  index: 256,257,512,513,514,515,768,769,770,771,772,773
  dev:   0,4,1283,1284,1284,1294,2557,2566,2574,2581,2587,2592
  ref:   1283,1287,2557,2558,2558,2568,3833,3842,3850,3857,3863,3868
n = 4099: 42 mismatches
  index: 256,257,258,259,260,512,513,768,769,770,771,772,773,774,1280,1281,1282,1283,1284,1285,1536,1537,1538,2304,2305,2306,2307,2308,2309,2310,2560,2561,2562,2563,2816,3584,3585,3586,3587,3588,3840,3841
  dev:   0,4,7,9,10,1283,1284,2557,2566,2574,2581,2587,2592,2596,5122,5125,5127,5128,5128,5138,6402,6402,6412,10243,10245,10246,10246,10256,10265,10273,11520,11530,11539,11547,12788,16637,16646,16654,16661,16667,17913,17919
  ref:   1283,1287,1290,1292,1293,2557,2558,3833,3842,3850,3857,3863,3868,3872,6402,6405,6407,6408,6408,6418,7673,7673,7683,11520,11522,11523,11523,11533,11542,11550,12788,12798,12807,12815,14080,17913,17922,17930,17937,17943,19202,19208
```

Only the first few elements of a block after the first are wrong; the rest of the block (and block 0) match the reference.

### 2. `OMP_NUM_THREADS`, `[scan]` 5 times each

```
OMP_NUM_THREADS=1:     0/5 runs failed
OMP_NUM_THREADS=2:     0/5 runs failed
OMP_NUM_THREADS=4:     5/5 runs failed
OMP_NUM_THREADS=8:     5/5 runs failed
OMP_NUM_THREADS=unset: 5/5 runs failed     (unset = 20 threads here)
```

CI runner CPU count: `/tmp/ci-cospan-39738752550/linux-64.log` does not exist any more, so none could be read (none printed).

### 3. `scan_group_size` on the OpenMP device (scratch program `/tmp/cospan-deriv/triage/wgsize.cpp`)

```
device: AdaptiveCpp OpenMP host device
max_work_group_size: 1024
max_compute_units: 20
scan_group_size: 256
```

So n = 1000 is 4 work-groups and n = 4099 is 17; the block sums are scanned by the one-work-group kernel in both.

### 4. One failing n = 1000 run (scratch program `scan_repro.cpp`: kernels (a), (b), (c) copied from `device_scan.hpp`, `s[]` and `b[]` in host USM)

Built `--acpp-targets=generic`, run with `ACPP_VISIBILITY_MASK=omp`:

```
wg=256 groups=4
s[]: 1283 1274 1276 1164
b[]: 0 1283 2557 3833 4997
expected b[]: 0 1283 2557 3833 4997
total mismatches: 12
256 0 1283  <-- mismatch
257 4 1287  <-- mismatch
512 1283 2557  <-- mismatch
513 1284 2558  <-- mismatch
514 1284 2558  <-- mismatch
515 1294 2568  <-- mismatch
```

(columns: index, device, reference; the full 250-520 listing is in `/tmp/cospan-deriv/triage/repro-1.log`; 768..773 are the other six mismatches.)
At the end of the run `s[]` and `b[]` are both correct. Same 12 indices as the test. The wrong values equal the chunk-local
result plus the previous block's offset (index 512: 0 + b[1] = 1283; 768: 0 + b[2] = 2557; 256: 0 + b[0] = 0).

### 5. The fork's OpenMP host implementation (describe only)

The tests are built for the generic (SSCP) target; the OpenMP device runs the SSCP host builtins.

- `src/libkernel/sscp/host/scan_exclusive.cpp`, lines 91-165 (`__acpp_sscp_work_group_exclusive_scan_*`), and
  `src/libkernel/sscp/host/reduction.cpp`, lines ~80-115 (`__acpp_sscp_work_group_reduce_*`): each takes a scratch pointer from
  `__acpp_sscp_host_get_internal_local_memory()` and calls `wg_host_scan<true>` (`include/hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_host.hpp`,
  lines 24-72), which writes `shrd_mem[wg_lid + 1] = x`, then runs log2(wg) rounds of read, work-group barrier, write, work-group barrier.
- `src/libkernel/sscp/host/localmem.cpp:31-34`: `__acpp_sscp_host_get_internal_local_memory()` returns `__acpp_cbs_sscp_internal_local_memory` (aligned to 512).
- `src/runtime/omp/omp_queue.cpp:260-324` (`launch_kernel_from_so`): the scratch is `static thread_local std::vector<char> internal_local_memory` of
  1024*8 bytes plus 64K*8 bytes of static local memory, in two places: lines 283-289 (one work-group and no dynamic local memory:
  runs on the submitting thread) and lines 305-310 inside `#pragma omp parallel` (several work-groups: per OpenMP thread; the
  work-groups are distributed by `#pragma omp for collapse(3)`, lines 312-323). So per-thread (thread_local), not static and not
  shared across work-groups.
- The library-path (non-SSCP) host versions are in `include/hipSYCL/sycl/libkernel/host/group_functions.hpp` (reduce at 341-351, exclusive scan at 375-403); not read in detail since the suite does not use that path.

### 6. History of `device_scan.hpp`

```
587f138 Metal: lowest common denominator: 32-bit ragged columns and scan, 32-bit device tests, Metal gates CI; roadmap: device capabilities
34530d2 layout: the library package moves to cospanmdao/; the root is the workspace only; SlabKit c1291a5 (slabkit/ subdirectory)
dc7bcac docs: internal design notes move to internal/; Furo theme; getting started lists every platform
ca0e6fe Resize events: uninitialized allocations, the per-variable resize-event slot, repin SlabKit 4f57086
bf9e339 P3: containers own events; holders own containers; launch events registered at submit
4f114a2 Orchestration and data-model rework (D144, spec r02)
```

Last change: 587f138, 2026-10-22 03:08:27 +0000 (the 32-bit change; it also changed `tests/test_device_scan.cpp`).

### 7. H5 / H6 / H7 measurements

- `pixi.lock`: `git status --short pixi.lock` empty; `git diff --stat b937876 -- pixi.lock` empty (unchanged). Last lock commits: 6e08324 (2026-10-22 06:51), ed61a20 (2026-10-22 01:55, "toolchain 2026.10.21 build 2"), dc18866.
- `pixi list -e default` (acpp, slabkit, cospan):

```
naga-acpp-llvm      23.0git     hfc7a621_0                  prefix.dev/jackm97/naga-labs-staging
naga-acpp-runtime   2026.10.21  llvm23_0_0_h2b02c85_2       prefix.dev/jackm97/naga-labs-staging
slabkit             0.3.0       h3a9d3be_0                  git ... rev 08d6bee7a4a769743be9523db589d8906ca14dfe
cospanmdao, cospanmdao-examples, cospanmdao-tests  (path packages, rebuilt 2026-10-26 19:57:25)
```
- `conda-meta` mtimes: `naga-acpp-llvm`, `naga-acpp-runtime`, `slabkit` all 2026-10-26 19:26:42 -0700 (the env was (re)installed at the start of today's session, so the earlier environment's state cannot be compared).
- AdaptiveCpp cache: `~/.acpp` (`apps/` inside; rule from `src/common/filesystem.cpp:216-231`: `$XDG_DATA_HOME` else `$HOME/.acpp`), 19.07 MB apparent / 23.27 MB on disk, `apps/` mtime 2026-10-26 20:05:02 (written by the runs above).
- `[scan]` 10 times with `~/.acpp` moved aside (restored afterwards): **10/10 failed**.
- Concurrency: the baseline OpenMP run (log written 19:27:06) and the CUDA run (19:28:03) were separate sequential tool calls; the 20 runs of measurement 1 and the 10 cache-less runs ran with no build or test process alive (`ps` filter on cospan-tests/cmake/ninja/clang/acpp/pixi was empty; load average 1.63 right before, left over from the earlier builds).

## Hypotheses

| | Hypothesis | Status |
|---|---|---|
| H1 | race in OpenMP group-algorithm scratch with concurrent work-groups | **weakened: passed earlier on this machine.** Also the fork's scratch is `thread_local`, per OpenMP thread, not shared between work-groups (item 5); the failure does track the thread count (item 2). Not supported as stated. |
| H2 | unordered-queue dependency not honoured on the OpenMP host, so kernel (c) reads `b` before (b) writes it | **supported in part, not proven.** Only the first few elements of each later block are wrong, the rest and the final `s[]`/`b[]` are right (items 1, 4): that is the shape of an early reader. It needs 4 or more threads (item 2). Unexplained: the wrong value is the previous block's offset (b[g-1]), not a zero or stale `b[g]`. |
| H3 | in-group scan error for groups after the first, independent of thread count | **contradicted**: depends on thread count (item 2); the interior of every block is right and block 0 is right. |
| H4 | the 32-bit change (587f138, 2026-10-22) shows only at this machine's work-group size | **untested.** Work-group size here is 256 (item 3). It is not known whether Jack's earlier passing runs were before or after 587f138. |
| H5 | environment drift (runtime or toolchain resolved differently) | **not supported by the lock, otherwise untestable**: `pixi.lock` unchanged at b937876; installed `naga-acpp-runtime` is build `_2`, as in the lock; but the environment files date from today's reinstall, so the earlier environment cannot be compared. |
| H6 | stale or corrupted AdaptiveCpp JIT/appdb cache | **contradicted**: 10/10 failures with `~/.acpp` moved aside. |
| H7 | concurrency or load from another build or test run | **contradicted**: 20/20 failures with nothing else running; baselines were sequential. |

## Next experiments

1. In `scan_repro.cpp`, add an `ev_b.wait()` before launching (c) (and, separately, `ev_a.wait()` before (b)), then run at `OMP_NUM_THREADS=4` and 20. If the 12 mismatches vanish, the dependency between the kernels (H2) is the cause and the next step is the unordered queue's handling of events on the OpenMP device; if not, H2 is out and the (a) kernel's group algorithms at 4+ concurrent work-groups (H1/H3 variants) are next.
2. Check out 587f138's parent for `device_scan.hpp` and `test_device_scan.cpp` into a scratch copy and run `[scan]` at `OMP_NUM_THREADS=4` (H4). Together with the earlier passing runs' thread settings (if Jack remembers any `OMP_NUM_THREADS`), this separates "the code changed" from "the thread count changed".
