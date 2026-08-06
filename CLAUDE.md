# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Group-lasso variable selection over foot marker data from an incline-running motion-capture dataset. The question: which foot markers can be dropped while still predicting the remaining markers (or the ground reaction forces)? A group-lasso penalty on kernel lengthscales (GP) or linear coefficients drives whole markers to zero as λ grows; a run sweeps λ and records the regularization path.

## Trust no logged number

**Every quantitative claim in this repo is a historical claim, not evidence.** Read this before acting on any "measured" figure in this file, `docs/HANDOFF.md`, `docs/ROADMAP.md`, or `docs/RESULTS.md`:

- **Every result pickle ever written was deleted on 2026-08-04** (seven generations, 416 MB + 74 MB mid-flight). Nothing quoted below can be recomputed without rerunning.
- **Provenance is broken.** Every `meta.json` across all seven generations records the same `git_revision` (`b7c4dc6`) because nothing was committed between them — seven generations that differ *only* by code changes all claim one revision. The only record of what distinguished v4 from v7 is prose. Commit before every result generation, or the ledger is fiction.
- **`converged=True` does not certify a fit.** Recomputed from the pickles before deletion (`docs/RESULTS.md`): late-path primal residuals *graze* the tolerance (ratios 0.9947–0.9996) rather than fall below it, so the tolerance — not the optimum — decides where a fit stops and which group dies next. The dual test reduces to `‖Δz‖ < tol·‖u‖` (rho cancels), so it *loosens* as `‖u‖` grows along the path. And rho tracks λ through the adaptation rule, so the ratio `λ/rho` that actually kills groups is non-monotone even when λ is monotone — part of the λ axis is illusory.
- **The benchmark tables are not mutually comparable.** Cold-state repeats, warmstarted iterations, and compile-included fits were quoted against each other. The single defensible solver comparison (one knot, one warmstart, end to end) is L-BFGS-B `maxiter=5 ls=5`: 67.4 s / 16 ADMM iterations vs optimistix 488.7 s / 35 — and even that is one knot, not a path.
- Several tuned defaults (`history_length=40`, chunk sizing, inner tolerances) rest on measurements taken against *older versions of the problem* (different box, different solver, different parametrization). Treat them as plausible starting points to re-measure, not settings that were validated.

The **mathematical** facts below (evenness, absorbing death, box dynamics) are derivations checkable from the code; they survive the deleted pickles. The **numbers** do not.

## Where this is going (2026-08-06)

A complete restructure is planned. What changed to enable it:

- **`jaxvlse` 0.1.1 shipped 2026-08-06 and is a dependency.** Bounded L-BFGS-B at `from vlse.optim import minimise`, signature `minimise(fun, x0, bounds, args=..., tol=..., max_iterations=..., history_length=..., max_linesearch=...)`, scipy-style `fun(x, *args)`, returns `LBFGSBState` with `.x`, jit/vmap/`lax.map` compatible. The x-update no longer needs the unconstrained-solve-plus-projection workaround; the box can be the solver's own feasible set.
- **Multi-start optimization is now affordable.** `vlse.minimise` vmaps over starts (batched solves were the only configuration where GPU beat CPU in the solver battery). This attacks the actual disease: the GP objective is nonconvex with `theta = 0` a local minimum at every λ > 0, so a single-start fit's answer is decided entirely by its warmstart. The entire upward-only continuation walk, the probe-calibration dance, and the "never hand a fit a sparser warmstart" rule are *workarounds for single-start fitting*. Multi-start over initializations replaces trust-the-warmstart with compare-the-basins.
- **The current sweep architecture is not to be preserved.** Its convergence stamps are untrustworthy (see above), its λ axis is entangled with rho adaptation, and its path is a continuation artifact. When restructuring, the math invariants below still bind; the sweep strategy, stopping rule, and tuned budgets do not.

## Layout

```
src/feetgp/        the library: admm.py, glassogp.py, linear.py, inclinerunning.py
                   the entry points: run.py, plots.py, summarize_runs.py
bench/             one-off diagnostics: uv run python -m bench.bench_knot
slurm/             batch scripts, submitted from the repo root: sbatch slurm/run_gp_v7.slurm
docs/              RESULTS.md (what the deleted result generations showed), HANDOFF.md, ROADMAP.md
tests/             pytest, synthetic data only
logs/              slurm stdout/stderr and watcher logs
```

src layout: `import feetgp` resolves to the installed package, never the cwd. Entry points live *in* the package, invoked as `python -m feetgp.run` — modules, not scripts, no `sys.path` manipulation anywhere. `run.py`'s `git_revision()` runs `git rev-parse HEAD` with `cwd=os.path.dirname(__file__)` (inside the work tree, so still correct).

## Commands

```bash
# single run (GP model, marker targets, both feet)
uv run python -m feetgp.run --subsample 20 --target markers --feet both \
    --chunk_size 39 --maxiter 300 --inner_maxiter 50 --inner_tol 1e-4

# linear ablation instead of GP
uv run python -m feetgp.run --linear_model --subsample 20 --tol 1e-6

# fast bounded solver
uv run python -m feetgp.run --solver lbfgsb ...

# render plotly HTML for every finished run under results/
uv run python -m feetgp.plots --results_dir results

# one line per run dir: lambda grid, active-group trajectory, R2, convergence counts
uv run python -m feetgp.summarize_runs results

# cluster
sbatch slurm/run_gp.slurm            # a100/v100/h100 — float64 throughput ranks the cards, never l40s
./watcher.sh                         # submit + poll sacct + resubmit non-COMPLETED array tasks

uv run pytest tests                  # synthetic data, no dataset needed
```

No linter config in the repo.

## Runs are resumable, and that shapes everything

`run.py` writes one pickle per λ to `results/<run_name>/lambda=<%.9e>.pkl`; on startup any λ with a pickle is loaded instead of refit (`fit_or_load`), which is what lets preempted SLURM tasks just be resubmitted. `--overwrite` forces a refit. The cache key is the λ value parsed from the filename (`find_cached`, `np.isclose(rtol=1e-6)`), not the filename format. Do not return to a format that rounds: `%013.6f` collapsed every λ below ~5e-7 onto the λ=0 pickle.

Each pickle carries the full `admm_state`, so a resume warmstarts the dual variable and rho, not just parameters. **A warmstart carries pathology across generations** — seeding v7 from v6's frozen pickles reproduced the freeze exactly — so a code fix needs a fresh grid, not a resumed one.

Pickles are versioned by `STATE_FORMAT` in `run.py` (currently 5: shared ADMM machinery, `(groups, members)` layout, nugget in `aux`). Format 4 is losslessly converted (`admm_state_from_legacy`); older formats load as results but are refused as warmstarts — reading one format's `w` as another's is a wrong-nugget bug that leaves no trace. Bump `STATE_FORMAT` whenever the parametrization moves; convert instead of refusing only when provably lossless. The field-only `GLASSOADMMState` stubs in `glassogp.py`/`linear.py` exist so a stray format-4 pickle (none survive in this repo) stays unpicklable-by-name; keep them.

`run.py` writes `meta.json` per run dir (argv, `group_size`, columns, `group_labels`, git revision). New labelling work belongs there, not in the path parser (`labels_from_run_dir` is fallback only).

## Math invariants — these survive any restructure

The GP objective is **exactly even in `theta`**: the kernel sees only `theta**2`, so `K(theta) = K(-theta)` bitwise and `∇f(0) = 0`. Everything below follows.

- **`theta = 0` satisfies group-lasso stationarity at every λ > 0** (`‖∇_g f‖ = 0 ≤ λ`). The all-dead point is always a local minimum, the problem is nonconvex, and which solution a single-start ADMM finds is decided by the warmstart. Any single-start path is a continuation path, not a global-optimum path.
- **Death is absorbing** (under the box). Once a group has `x = z = 0`, `u` stops moving, the x-update target `z - u = -u` is nonpositive and clips back to zero, and no term can resurrect it. Hence the current sweep's hard rule: never hand a fit a warmstart sparser than its own solution — walking λ downward chained onto sparser neighbours produced whole paths at 0 active. Multi-start is the structural fix; until then the rule stands.
- **`theta >= 0` is load-bearing dynamically, free statistically.** Unconstrained, a dead group's fixed point requires `u = 0` exactly (measure zero) and ADMM limit-cycles between the two mirror wells instead. With the box, the x-update clips to exactly zero and stationarity relaxes to the inequality `u >= 0`, satisfied by a whole set. The box must hold on **both** `x` and `z` (the optimistix path projects after solving; the lbfgsb path has it as the feasible set).
- **The mirror well is killed by `relu` in the likelihood, not by steering rho.** At a fixed point the x-update target is `z_g (1 - l1/(rho‖z_g‖))`, which goes negative as a group approaches death; on an even objective that makes the reflected well the deeper one. `admm_x_update_loss` feeds `jax.nn.relu(theta)` to the kernel — the negative orthant becomes a likelihood plateau (C1 across the boundary, since evenness forces zero gradient there; `test_x_update_objective_is_flat_below_zero`) — while the augmented term keeps **raw** `theta` so a stray negative coordinate is still pulled back. If the relu ever saw the augmented term too, the orthant would be entirely flat and nothing could escape it. **Do not reintroduce a rho floor**: `rho_floor` held every group a fixed factor from its own death threshold and froze whole paths at full support. It conflated a global pathology (all targets negative at once) with the legitimate death of one marginal group.
- **Sparsity lives in `z`, not `x`.** `z` is the prox output and is exactly zero by construction; `fit` reports `state.z`. Do not read `state.x` for `n_active`.
- **The nugget saturates, it is not clipped.** `g = g_min + (g_max - g_min)·sigmoid(w)`, `w` unbounded, riding in `ADMMState.aux` where `admm.py` never touches it — structurally outside the consensus constraint and both residuals. It was once a clipped column of `x`, and any output noisier than the ceiling then held the primal residual above tol at *every* λ. Do not bound `w`; do not move it out of `aux`. The `g_range` floor (1e-4) is load-bearing: without it the marginal likelihood runs away to interpolation as `g → 0`. `nugget_from_w`/`w_from_nugget` are the only conversion sites.
- **GP runs plain ADMM, `alpha = 1.0`.** Over-relaxation's `x_hat = alpha·x + (1-alpha)·z` leaves the box whenever `x` and `z` straddle (feeding `u` the overshoot, reopening the mirror well), and at λ=0 — where the prox is the identity — it turns the primal residual into a noise meter for the x-update's jitter (`r = (alpha-1)‖x - z_prev‖`) with a floor no budget crosses. `linear.py` keeps `alpha=1.6`: no box, objective not even.
- **rho is reset at the λ=0 handoff, not gated in `check_residuals`.** At λ=0 the prox is the identity, `u ≡ 0`, the primal residual is identically zero, and `dual > 10·primal` fires every iteration — rho decaying toward `RHO_MIN` is *locally correct* (it makes the x-update the exact MLE λ=0 wants, and it is the only way λ=0's dual test can pass at all). The damage is only at the handoff — a warmstart at rho≈1e-6 has a prox threshold that kills every group on contact — so `run.py` does `states[0.0]._replace(rho=jnp.array(1.0))`. Gating the adaptation on residuals clearing `eps_abs` instead was tried and burns the whole budget at λ=0.
- **`eps_abs = sqrt(p)·EPS`, not `EPS`** (Boyd §3.3.1): both residuals are norms of p-element vectors, and at λ=0 the dual criterion has no relative term left, so an unscaled floor asks the inexact x-update for a residual below its own noise.
- **rho adaptation is clamped to `[RHO_MIN, RHO_MAX]` and frozen after `adapt_rho_iters`** (`None` → `max_iterations // 2` — keep that coupling; otherwise retuning the budget silently moves the freeze point). Unclamped, an irreducible primal floor feeds the ×2 rule to rho ~1e22, where the augmented term swamps float64. `u` is rescaled by the applied ratio, not by `scale`. But note the skepticism section: even clamped, rho tracking λ is what made part of the λ axis illusory — the adaptation rule itself is restructure material.

## ADMM structure (both models)

`admm.py` owns the algorithm; `glassogp.py`/`linear.py` own only their x-update, passed to `admm.solve` as a closure `(state, iteration) -> (state, exact)`.

- **Layout convention `(... g)`**: leading axes index groups, last axis holds one group's members. That removes `group_size` from `admm.py` entirely — the prox is a norm over the last axis. Models convert with `admm.to_groups`/`to_outputs` (`rearrange(v, "o (d g) -> d (o g)", g=group_size)`), pooling one marker across every output and coordinate.
- **`ADMMState.aux`** carries parameters the x-update owns but the consensus must not see (the GP nugget `w`). `admm.py` threads it through untouched.
- **The x-update reports whether it was exact**; `solve` suppresses the convergence break otherwise, so a cheap early iteration under the ramping inner budget (`inner_maxiter_init · 2**iter` up to the cap) cannot fake convergence.
- **`bounds` are a `solve` argument, not state** — always rederived from data, nothing to strip before pickling.
- x-update: linear = closed-form ridge (always exact). GP = per-output L-BFGS over `jax.lax.map(..., batch_size=chunk_size)` — a chunked vmap; a `lax.while_loop` under vmap cannot exit per element, so a chunk costs the max over its members. Prefer `chunk_size` a divisor of the output count (78 = 2×39).
- `GroupLassoGaussianProcess.fit` returns `(model, loglik, admm_state, info)`; linear returns `(model, loss, admm_state, info)`. `info` (`converged`, `iterations`, residuals) goes in the pickle and drives `plots.py`'s crossed-out unconverged λ — bookkeeping lives there, not in `ADMMState`. Both models expose `theta`, `predict`, and a `warmstart=` kwarg so `run.py` treats them interchangeably.
- Prediction and `unpack_parameters` use `lax.scan` over outputs, not vmap — one n×n kernel live at a time, deliberate OOM avoidance. `predict` skips `Kxx` unless `covariance=True`.

## The two inner solvers (`--solver`)

`optimistix` (default) is unconstrained L-BFGS plus projection onto the box; `lbfgsb` is vlse's bounded solver, box as feasible set. **No budget or tolerance transfers between them:**

- `inner_maxiter`: optimistix counts single steps, lbfgsb whole line searches — unset resolves to 50 vs 5.
- `inner_tol`: optimistix stops on a *step* criterion (loose), lbfgsb on the projected-gradient infinity norm (scipy's `pgtol`, tight) — unset resolves to 1e-4 vs 1e-2.
- `--inner_max_linesearch` (lbfgsb only, default 5) is the straggler's leash under `lax.map`.
- The ramp's first rung `inner_maxiter_init` is 1 for lbfgsb vs 20 for optimistix — a shared 20 would sit above lbfgsb's cap from iteration 0, leaving the ramp inert and stamping every x-update `exact=True`.
- lbfgsb's `solver_bounds` are the `[0, theta_max]` box in output layout with a `±inf` column appended for `w` (which must stay unbounded — it saturates).

The inner budget is the dominant cost lever (`iterations × outputs × inner_steps`); the device is compute-bound (one n×n float64 Cholesky saturates an A100), so `chunk_size` barely matters and batching over *starts* — not outputs — is where the GPU wins.

## Column layout is load-bearing

`InclineRunning` builds marker columns as `[LCAL1, RCAL1, LCUB, RCUB, ...]` — left/right of the *same* marker adjacent, each with X/Y/Z. That interleaving is what makes `group_size=6` one marker across both feet; `--ungroup_feet` or a single-foot run drops to 3. Reordering the columns silently changes what a "group" means.

Other data facts before touching `inclinerunning.py`:

- Train/test split is deterministic even/odd rows — consecutive motion-capture frames, so **test is not independent of train**. (Known methodology debt, never addressed.)
- `--target markers` sets `autoregressive=True`: each output's own marker group is zeroed from its inputs so it cannot predict itself.
- `--target forces` cube-roots the forces and block-averages the force file (5 rows) to match marker rate.
- `--relative` (bare) subtracts the LMAL/MMAL midpoint per foot; `--relative MARKER` subtracts a named marker and drops its now-zero column.
- Normalization keeps constant columns at 0 rather than dividing by zero.

## Conventions

- **float64 everywhere** (`jax_enable_x64` in every entry point). The kernel computes squared distance as `|z1|² + |z2|² - 2 z1·z2` — one matmul, but it cancels, which is why float64 is non-negotiable. Re-check the cancellation if lengthscale bounds ever widen a lot.
- Models are `NamedTuple`s, jitted via `eqx.filter_jit` where a static field is present.
- `run.py` strips `x_train`/`y_train` before pickling — loaded models can compute `group_norms` but cannot `predict` without re-attaching data.
- `data/`, `results*/`, `*.pkl`/`*.tsv`/`*.xlsx`/`*.html` are gitignored. `*.slurm` is gitignored but `!slurm/*.slurm` overrides: batch scripts are the only record of what each generation ran with — commit them with the code change, **and commit the code before submitting** (see the provenance failure above).
