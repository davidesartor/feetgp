# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Group-lasso variable selection over foot marker data from an incline-running motion-capture dataset. The question being answered: which foot markers can be dropped while still predicting the remaining markers (or the ground reaction forces)? A group-lasso penalty on kernel lengthscales (or on linear coefficients) drives whole markers to zero as λ grows; a run sweeps λ and records the regularization path.

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

src layout, so `import feetgp` resolves to the **installed** package, never to whatever directory
python happened to start in. `uv run` keeps the editable install current; nothing depends on the
repo root being on `sys.path`, which is why `bench/` scripts work either as `-m bench.bench_knot`
or by path.

The entry points live *in* the package and are invoked as `python -m feetgp.run`, not by path.
They are modules, not scripts: they import `feetgp.*` absolutely like everything else, and there
is no `sys.path` manipulation anywhere. `run.py`'s `git_revision()` runs `git rev-parse HEAD` with
`cwd=os.path.dirname(__file__)`, which is now `src/feetgp` — still inside the work tree, so the
recorded revision is unchanged.

**Every pickle ever written by this repo was deleted on 2026-08-04, and the compatibility shims
went with them.** `src/{admm,glassogp,linear}.py` used to re-export the package modules so that
results written before the src move — which name their classes `glassogp.GroupLassoGaussianProcess`,
`admm.ADMMState`, i.e. by top-level module path — stayed loadable. With no such pickle left in
existence they are dead weight. If an old result ever resurfaces from a backup, a three-line
`src/glassogp.py` doing `from feetgp.glassogp import *` brings it back; nothing else is needed.
The field-only `GLASSOADMMState` stubs inside `glassogp.py` / `linear.py` and the format-4
`admm_state_from_legacy` converter are **kept** — they cost nothing and they are the only path
back for a stray pickle someone else is holding.

## Commands

```bash
# single run (GP model, marker targets, both feet)
uv run python -m feetgp.run --subsample 20 --target markers --feet both \
    --chunk_size 39 --maxiter 300 --inner_maxiter 50 --inner_tol 1e-4

# linear ablation instead of GP
uv run python -m feetgp.run --linear_model --subsample 20 --tol 1e-6

# render plotly HTML for every finished run under results/
uv run python -m feetgp.plots --results_dir results

# one line per run dir: lambda grid, active-group trajectory, R2, convergence counts
uv run python -m feetgp.summarize_runs results

# cluster: array job over the 10 ablation configs
sbatch slurm/run_gp.slurm            # a100/v100/h100, sub=20
sbatch slurm/run_gp_1080ti.slurm     # 1080ti, sub=100
sbatch slurm/run_linear.slurm
./watcher.sh                         # submit + poll sacct + resubmit non-COMPLETED array tasks
```

```bash
uv run pytest tests            # model-level tests, synthetic data, no dataset needed
```

No linter config in the repo.

## Runs are resumable, and that shapes everything

`run.py` writes one pickle per λ to `results/<run_name>/lambda=<%.9e>.pkl`. On startup, any λ that already has a pickle is loaded instead of refit (`fit_or_load`). This is why preempted SLURM tasks can just be resubmitted — `watcher.sh` relies on it. `--overwrite` forces a refit.

The cache key is the **λ value parsed out of the filename** (`find_cached`, matched with `np.isclose(rtol=1e-6)`), not the filename format, so the older `%013.6f` names are still found. Do not go back to a format that rounds: `%013.6f` collapsed every λ below ~5e-7 onto the λ=0 pickle.

Each pickle also carries the full `admm_state`, so a resumed run warmstarts the dual variable and rho, not just the parameters.

Pickles are versioned by `STATE_FORMAT` in `run.py`. Through format 4 it said what the last column of `admm_state.x` meant: 1 a linear `g`, 2 `w = log(g - g_min)`, 3 `w = logit((g - g_min) / (g_max - g_min))`, 4 the same `w` but with `theta` back in the nonnegative orthant, so a format-3 state's negative thetas would silently clip to zero. 5 (current) is the shared ADMM machinery: same parametrization as 4, but the iterates are laid out `(groups, group members)` and the nugget moved out of `x` into `aux`. That is a pure relabelling, so format 4 is **converted** (`admm_state_from_legacy`) rather than refused. Any other version is still loaded as a *result* but refused as a *warmstart* (`fit_or_load` warmstarts cold) — reading one format's `w` as another's is a wrong-nugget bug that leaves no trace. Bump `STATE_FORMAT` whenever the parametrization moves, and convert instead of refusing only when the change is provably lossless. **Every result was deleted on 2026-08-04** — 416 MB of stale generations plus the pre-2026-08-03 archive (which predated `admm_state` and `meta.json` entirely), then the 74 MB of `results_v7` as well, mid-flight. What they showed is in `docs/RESULTS.md`, per-run numbers in `docs/results_summary.jsonl`; there is no pickle left to warmstart from, so the next run starts cold and rebuilds its own λ grid.

`run.py` writes a `meta.json` per run dir (argv, `group_size`, `x_columns`/`y_columns`, `group_labels`, git revision). `plots.py` reads it; `labels_from_run_dir` is only the fallback for runs saved before it existed. New labelling work belongs in `meta.json`, not in the path parser.

## λ sweep strategy

Not a fixed grid. `run.py` fits λ=0 first, then walks λ **upward** (`*= lambda_step`) until zero groups survive, each fit warmstarted from the λ below it, and finally bisects any interval where more than one group died at once.

**The walk is upward-only, and that is a correctness requirement, not a convergence preference.** Death is absorbing in the GP model. The objective is exactly even in `theta`, so `∇f(0) = 0`; once a group has `x = z = 0`, its `u` stops moving (`u += x - z` adds nothing), the x-update target `z - u = -u` is nonpositive and clips straight back to zero, and no term in the problem can push it off zero again. A fit handed a warmstart sparser than its own solution therefore *keeps* the extra dead groups. The sweep used to walk down from `lambda_pivot` chaining each fit onto its sparser neighbour, and since the pivot usually landed already dead, entire paths came back 0 active at every λ — measured on `markers/right_only`, λ = 98 … 280 all 0/13, and on `right_only --relative` a non-monotone 2 / 1 / 3 active as λ *increased* through 23 / 30 / 39. So: never hand a fit a warmstart sparser than what it should return. The refinement pass obeys the same rule — it warmstarts each bisection from `states[lo]`, the denser end of the interval.

The same evenness means `theta = 0` satisfies group-lasso stationarity at *every* λ > 0: the all-dead point is always a local minimum, the problem is nonconvex, and which local solution ADMM finds is decided entirely by the warmstart. The path is a continuation path, not a global-optimum path. That is why the grid can only be built in one direction.

The grid's bottom is scaled off the data, not off a likelihood pivot. `LAMBDA_START_FRACTION * max(group_norms at λ=0)` (0.02) is the first probe; the useful λ range sits on the scale of those norms, since a group dies once the prox threshold `l1 / rho` reaches its own norm, and they span well under a decade. The old `lambda_pivot` — where the penalty balances the log-likelihood gain over the null model — landed 20–30× above the whole band (323 against norms of 9.6–15.6) and put the entire path in the downward walk, i.e. in the broken direction. `LAMBDA_START_FRACTION` is only a guess at the scale, so the start is calibrated first by dividing by `lambda_step` until some λ holds the full support; those probes warmstart from **λ=0**, never from the probe above them, which is what makes descending safe there and unsafe in the walk proper.

The whole grid hangs off the λ=0 fit, so anything that perturbs it (iteration budget, inner-solver schedule) shifts every subsequent λ. A resumed run reads λ=0 from cache and therefore keeps its original grid.

## ADMM structure (both models)

`admm.py` owns the algorithm; `glassogp.py` and `linear.py` own only their x-update. The loop is still the same three phases — `x_update` → `z_and_u_update` → `check_residuals` — but only the first is per-model, passed to `admm.solve` as a closure `(state, iteration) -> (state, exact)`. The two models used to carry a `GLASSOADMMState` each with all three phases as methods; those classes still exist in both modules as **field-only stubs** so format-4 pickles unpickle at all (pickle resolves a class by module + name, so deleting them makes such a result unreadable, not merely un-warmstartable). No such pickle survives in this repo, but the stubs cost nothing and are the only way back for one held elsewhere. `admm_state_from_legacy` converts one; `admm_state_from_pickle` is what the benches and `run.py` should call.

- **the layout convention is `(... g)`: leading axes index groups, the last axis holds one group's members.** That is what removes `group_size` from `admm.py` entirely — the prox is then just a norm over the last axis, and `check_residuals`/`solve` never need to know what a group is. Both models store parameters per output, so they convert on the way in and out with `admm.to_groups` / `admm.to_outputs` (`rearrange(v, "o (d g) -> d (o g)", g=group_size)` and back), which pools one marker across every output and every coordinate — the same element set the old `"o (d g) -> (o g) d"` + `axis=0` norm took.
- **`ADMMState.aux` is for parameters the x-update owns but the consensus must not see.** Nothing in `admm.py` reads or writes it; it is threaded through the loop and returned. The GP's nugget `w` rides there. That is structural, replacing the old trick of giving `w` a column in `x` with `±inf` bounds so `z` happened to copy it exactly and contribute nothing to either residual.
- **the x-update reports whether it was exact**, and `solve` suppresses the convergence break when it was not. This generalizes the GP's `maxiter == inner_maxiter` gate; the linear model's closed-form solve always returns `True`.
- **`bounds` are a `solve` argument, not state.** They are always rederived from the data, so nothing has to strip them before pickling.

- **x-update.** Linear: closed-form ridge solve. GP: `optimistix.LBFGS` on the negative marginal log-likelihood plus the augmented-Lagrangian term (or vlse's bounded L-BFGS-B under `--solver lbfgsb`, see below), run over outputs with `jax.lax.map(..., batch_size=chunk_size)` — a chunked `vmap`, which is what `--chunk_size` sets. This is the expensive part, and the whole cost of a fit is `iterations * outputs * inner_steps`. It is inexact by design: the budget ramps `inner_maxiter_init * 2**iter` up to `inner_maxiter` (50), and the convergence break is suppressed until the budget reaches the cap, so a cheap early iteration cannot fake convergence.
- **the inner budget is the single biggest cost lever, and it is a cap, not a tolerance.** optimistix's LBFGS terminates on a **step** criterion, `norm(y_new - y_old) < atol + rtol * norm(y)` under `max_norm` — not on the gradient. At the old `rtol = atol = EPS = 1.49e-8` that criterion essentially never fires on a 79-dimensional vector: measured on the real problem (n=1350, d=o=78, λ=0), 47/78 outputs hit `max_steps=1000` and the mean was 872 steps, at 175.7s for one x-update. Relaxing to `1e-4` reaches the same augmented-Lagrangian value to 4e-6 relative in 429 mean steps with 0/78 at the cap; `1e-3` costs 2.2e-4 relative at 168 steps. So `--inner_tol` defaults to 1e-4, but the operating point is `--inner_maxiter 50`: a cheap inexact x-update run many times beats an exact one run a few times (measured 8.1s per ADMM iteration at `chunk_size=39` against ~97s under the old settings). Inexact ADMM tolerates this — subproblem errors only need to be summable.
- **`chunk_size` barely matters, because the device is compute-bound.** One 1350×1350 float64 Cholesky already saturates an A100, so batching outputs buys little: measured 8.90s / 9.37s / 8.10s per ADMM iteration at chunk 26 / 32 / 39. Prefer a divisor of the output count (78 = 2×39 = 3×26) so no ragged trailing chunk pays for a full one. The `lax.map` straggler tax — a `lax.while_loop` under `vmap` cannot exit per element, so a chunk costs the max over its members — measured only 1.15x at chunk 32, and is ~1.00x once every output hits the same cap.
- **`history_length` defaults to 40** rather than the usual 10: measured 221 steps at 40 against 735 at 10 and 627 for dense BFGS. Taken under the box, which is what the solver runs under again, but with a collapsed active set of ~2 free dimensions — the setting is probably still right, the number is not current evidence.
- **z/u-update.** Group-soft-thresholding prox (`admm.group_soft_threshold`), a norm over the last axis under the `(... g)` layout.
- **`theta >= 0` is load-bearing, and not for statistical reasons.** The kernel sees `theta` only through `theta**2`, so the objective is *exactly* even in every coordinate (`max|K(theta) - K(-theta)| = 0`, likelihoods bit-identical) and `dloglik/dtheta_d = 0` at `theta_d = 0`. The box was once dropped on that argument. **It is free statistically and not free dynamically**, and putting it back is the single change that made ADMM converge at all:
  - At an ADMM fixed point for a *dead* group, the x-update sees target `z - u = -u` and must return `0`. Unconstrained, stationarity reads `rho*u = -∇f(0) = 0` — the even objective's gradient vanishes at zero, so the condition is `u = 0` exactly, a measure-zero event. ADMM never reaches it and limit-cycles instead: the minimiser lands near `-u` (large), the prox kills it again, `u += x - z` collapses `u` back, period 2. Measured at λ=116.6, sub=200: group 11's norm went 2.90 → **15.21** → 2.96 over 20 iterations, the active count flickered 11/12/13, and `r` oscillated 9.8 → 15.2 → 9.7 → 21.5 → 12.0 for 150 iterations with rho pinned. At λ=256.2 the active count sat at 0 for 80 iterations while x-norms bounced 3–15.
  - With the box, the minimiser clips to *exactly* zero and stationarity relaxes to the **inequality** `u >= 0`, satisfied by a whole set. Same trace, boxed: `r` falls monotonically 23.2 → 11.3 → 6.8 → 5.4 → 3.3 → 2.7, x-norms drop to O(1), active count settles at 13.
  - So the box must hold on **both** `x` and `z`. The GP's `x_update` closure runs unconstrained L-BFGS (`x_update_solve`) and then projects onto `bounds`; `admm.z_and_u_update` clips when it is passed them. An unprojected `x` is also what made λ=0 unconvergeable: the prox is the identity there (`max(0, 1 - 0/·) = 1`), so `z = x + u` and the primal residual *must* be identically zero, but `x` sat outside the box (`max|θ| = 2.878` against `theta_max = 2.6645`) and `z` was clipped to meet it, holding `r = 0.285` — of which `r_box = 0.285`, i.e. all of it — forever. Projected, λ=0 gives `primal = 0.00e+00` exactly, `n_over = 0`.
- **sparsity still lives in `z`, not `x`.** `x` reaches exact zeros again now that it is projected, but `z` is the prox output and is exactly zero by construction, so `fit` keeps reporting `state.z`. `unpack_parameters` takes `|theta|`, now a no-op. Do not go back to reading `state.x` for `n_active`.
- **the nugget saturates, it is not clipped.** `g = g_min + (g_max - g_min) * sigmoid(w)`, so `g` is inside `g_range` by construction and `w` needs no bound. The nugget is out of the consensus constraint structurally — `w` lives in `ADMMState.aux`, which `admm.py` never touches, and the augmented-Lagrangian term covers only `theta` (`target_theta`). It used to be `g_min + exp(w)` with `w` clipped to `log(g_range)` in `z_and_u_update`, and that clip was fatal: any output whose noise exceeded `g_range[1]` sat permanently at `x != z`, holding the primal residual above `tol` **at every λ**, so no fit ever converged. On real data 32% of nuggets were pinned at the old ceiling of 1.0. Do not reintroduce a bound on `w`, and do not move it back out of `aux`.
- **`g_range` is now `(1e-4, 100.0)`.** The floor is load-bearing: without it the marginal likelihood runs away to interpolation as `g -> 0`, and 66% of real nuggets sit on it. The ceiling used to be `1.0`, which cannot express "this output is noisier than its signal". Even with saturation a binding ceiling costs iterations (103 against 52 on the toy fit) because the output sits on a flat ridge; with headroom `g` settles at a genuine interior optimum, measured identical at ceilings 10/100/1000. Model `g` is linear, state `w` is not — `nugget_from_w`/`w_from_nugget` are the only places that conversion should happen.
- **the absolute tolerance is `sqrt(p) * EPS`, not `EPS`** (Boyd §3.3.1). Both residuals are norms of `p`-element vectors, and at λ=0 the prox is the identity so `u` is exactly zero and the dual criterion loses its relative term entirely, leaving only the absolute floor. Unscaled, that floor asks the *inexact* x-update for a dual residual below its own noise: measured flat at 5.6e-7 across iterations 50→100 at `inner_tol=1e-4`, against an `EPS` of 1.49e-8. λ=0 would have spun to the iteration cap with `primal = 0` and a converged solution in hand. `sqrt(6084) * EPS = 1.16e-6` clears it.
- **the GP runs plain ADMM, `alpha = 1.0`, and Boyd's 1.5–1.8 over-relaxation band is wrong here.** Two independent reasons, both traced back to `x_hat = alpha*x + (1-alpha)*z`. (1) That combination leaves the box whenever `x` and `z` straddle, so `x_hat` goes *negative* and feeds `u` the overshoot — reopening the second well the `theta >= 0` box exists to close. (2) At λ=0 the prox is the identity, so `z = x_hat + u` and `u = 0`, which makes `r = ||x - x_hat|| = (alpha-1)||x - z_prev||`: the inexact x-update's own step-to-step jitter, amplified, not a consensus error, with a noise floor it can never cross. Measured at `alpha=1.6`, λ=0, sub=20: `r` stuck at 1.02 / 1.06 / 1.72 / 1.55 across iterations 51/101/151/201 with rho pinned, heading for the 400-iteration cap. At `alpha=1.0` the same fit reports `r = 0.000e+00` at iteration 1 and **converges in 25 iterations**. Every measurement that validated the box was taken at `alpha=1.0`; `run.py` never passed `alpha`, so it was silently running the one setting the box cannot tolerate. `admm.solve` therefore defaults to `alpha=1.0` and `glassogp.fit` pins it there. `linear.py` keeps `alpha=1.6` — it has no box, so only the cosmetic λ=0 effect applies there, and its paths are measured good.
- **rho is deliberately *not* inherited from the λ=0 fit** (`run.py` does `states[0.0]._replace(rho=jnp.array(1.0))`). λ=0 walks rho all the way down to `RHO_MIN`, because the prox is the identity there, so `u ≡ 0`, the primal residual is identically zero, and `dual > 10 * primal` is a tautology that fires every iteration. That decay is *correct locally* — the augmented term is vacuous at λ=0 and `rho → 0` is what makes the x-update the exact MLE that λ wants — and it is also what lets λ=0 pass its dual test at all: with `u = 0` the dual criterion has no relative term, so `dual = rho * ||z - z_prev||` clears the absolute floor only because rho is tiny (measured `s = 7.04e-07` at `rho = 2e-6`, i.e. `||Δz|| = 0.35`). Do not "fix" that by gating the adaptation on the residuals clearing `eps_abs`: that pins rho at 1.0 and then the dual test demands `||Δz|| < 1.16e-6` from an `inner_maxiter=20` inexact solve, which never fires and burns the whole 400-iteration budget. The damage was only ever at the *handoff* — a next-λ warmstart at rho=2e-6 has a prox threshold `l1 / (rho * norm)` that kills every group on contact — so the reset belongs there, not in `check_residuals`.
- **the negative x-update target is handled by a relu inside the likelihood, not by a floor on rho.** At an ADMM fixed point `u_g` points along `z_g` with norm `l1 / rho`, so the x-update target reads `z_g (1 - l1 / (rho ||z_g||))`, which goes negative on a group as soon as the prox threshold `l1 / (rho ||z_g||)` passes 1. On an even objective a negative target makes the *mirror* well the deeper one: individual outputs fall into it and project back onto the box at 0, and the primal residual then sits on a handful of output rows that no iteration budget clears. Measured on `markers/both` at λ=34.97 (rho=1, smallest live group 6.85, so the target was 5× negative): `r` stalled at 2.7–4.6 for the full 400 iterations, 13/13 still active, residual carried by 5 of 78 rows (1.56 / 1.30 / 0.76 / 0.64 / 0.64 against a bulk of 0.05). **The inner budget is not the lever**: `inner_maxiter=200` gave `4.61` against 50's `4.65` for 2.4× the wall time.
  - The fix is `kernel(jax.nn.relu(theta), ...)` in `admm_x_update_loss`: the negative orthant becomes a likelihood *plateau*, so there is no mirror well to fall into, while the augmented term keeps the **raw** `theta` so a coordinate on the wrong side of zero is still pulled back to its target. It costs no smoothness — evenness forces `dloglik/dtheta_d = 0` at `theta_d = 0`, so `loglik(relu(theta))` is C1 across the boundary (`test_x_update_objective_is_flat_below_zero`). If the relu ever saw the augmented term too, the whole orthant would be flat and a stray coordinate could never escape.
  - **Do not reintroduce `rho_floor`.** The earlier fix was `RHO_TARGET_MARGIN * l1 / min(live group norms)` with margin 2.0, clamped in `check_residuals`. It did clear the stalled knots (λ=34.97 in 45 iterations, λ=22.08 in 49, against 400-iteration stalls) and it is also what froze the whole GP path at full support: forcing `l1 / rho <= min_norm / 2` holds *every* group at a factor of 2 from its own death threshold, permanently. Death needs the prox threshold at 1.0; measured over the live path it was pinned at 0.12–0.21 while rho tracked λ upward (λ=26.9 rho=1 → λ=814.8 rho=1.2e3), so 30 λ ran out to 815 at 13/13 with R² flat at 0.9998. The floor conflated a genuine pathology — rho too small globally, every live group's target negative at once — with the legitimate death event of one marginal group's target going negative.
- **residuals.** Standard primal/dual check with rho adaptation (×2 / ÷0.5) clamped to `[RHO_MIN, RHO_MAX]`, `RHO_MAX = 1e6`, disabled after `adapt_rho_iters` so the problem stops moving and can actually converge. The clamp is not decorative: with an irreducible primal residual the ×2 rule is a positive feedback loop, and rho was observed reaching `1.89e+22`, at which point the augmented term swamps the likelihood in float64 and the residual can never fall again. `u` is rescaled by the *applied* ratio `rho_old / rho_new`, not by `scale`, so clamping stays consistent. `adapt_rho_iters=None` means `max_iterations // 2`, which is what the freeze used to be hardwired to — keep that default. It matters because otherwise retuning the iteration budget silently moves the freeze point, which is a behaviour change disguised as a budget change.

`GroupLassoGaussianProcess.fit` returns `(model, loglik, admm_state, info)`; `GroupLassoLinear.fit` returns `(model, loss, admm_state, info)`. `info` is a plain dict (`converged`, `iterations`, `primal_residual`, `dual_residual`) that `run.py` stores in each pickle and `plots.py` uses to cross out unconverged λ — convergence bookkeeping goes there, not into `ADMMState`, which stays lean. Both expose `theta`, `predict`, and accept a `warmstart=`/`**kwargs` signature so `run.py` can call them interchangeably. The GP's `warmstart` takes either an `ADMMState` (preferred: carries `x`, `z`, `u`, `rho`, `aux`) or a fitted model (`theta`/`g` only).

Prediction and parameter unpacking use `jax.lax.scan` over outputs rather than `vmap` — deliberate, to avoid OOM on the `n×n` kernels. `predict` factorizes `Koo` once per output and skips the `m×m` `Kxx` entirely unless `covariance=True`.

## `converged=True` does not certify a λ, and three measurements say why

Recomputed from every saved `admm_state` in v4–v7 on 2026-08-04, before those directories were
deleted (`docs/RESULTS.md` carries the tables). Read this before trusting a path or "fixing" the
stopping test.

- **Late-path fits graze the tolerance rather than fall below it.** Over the second half of
  `markers/both` the primal residual sits at 0.264194 against an `eps_pri` of 0.264400 — ratio
  0.9992, and 0.9947 / 0.9970 / 0.9996 at the neighbouring λ. Every one is stamped
  `converged=True` on a margin of 0.1%. So the tolerance, not the optimum, decides where a fit
  stops, and therefore which group dies next.
- **The dual test reduces to `‖Δz‖ < tol · ‖u‖`.** `dual = rho * ‖Δz‖` and
  `eps_dual = sqrt(p)*EPS + tol * rho * ‖u‖`, so rho cancels wherever the absolute term is not
  binding. On `markers/both_ungrouped` at λ=0.2166, `eps_dual = 1.11e-3` against a floor of
  `sqrt(p)*EPS = 1.16e-6`: three orders above it, with `‖u‖ = 17.7`. What turns λ=3.7006 from a
  400-iteration stall into convergence in 8 is `‖u‖` growing 4×, loosening the test — `‖Δz‖` is
  0.061 there against 0.085 in the stalling fit. **A dual floor tied to inner-solver accuracy
  therefore addresses a mechanism that is not the one operating**; the earlier handoff note
  proposing one is wrong on this point.
- **rho tracks λ, so part of the λ axis is illusory.** With the primal residual pinned near
  tolerance, the `primal > 10*dual` rule doubles rho as λ climbs: λ=26.9 → rho=1, λ=459 → 4,
  λ=1034 → 16, λ=3490 → 128, λ=5236 → 512. The prox threshold is `l1 / rho`, so λ/rho — the ratio
  that actually kills groups — goes 27, 115, 65, 27, 10 while λ increases monotonically. Deaths
  happen when rho lags λ. Same failure the `rho_floor` produced, arriving through the adaptation
  rule instead.

The speed comparisons in `docs/HANDOFF.md` do not all measure the same thing: in job 62585927 the
"49.4 s per x-update" repeats all start from the *same cold state*, while "30.2 s per ADMM
iteration" is a 3-iteration `fit` including compile with iterations 2–3 warmstarted, so it is not
3× the first row. The one comparison measured end to end from a single warmstart at a single knot
is **L-BFGS-B at `maxiter=5 ls=5`, 67.4 s and 16 ADMM iterations, against optimistix's 488.7 s and
35**, at equal-or-better log-likelihood.

## the bounded L-BFGS lives in `vlse` now, not here

Nothing else in the JAX ecosystem ships a bounded L-BFGS — `optimistix` and `optax` have LBFGS only, `jaxopt` has `LBFGSB` but is deprecated — so this repo carried its own Byrd–Lu–Nocedal–Zhu port (`lbfgsb.py`) plus a scipy-fidelity battery (`tests/test_lbfgsb.py`). Both were deleted on 2026-08-04: the solver ships in `vlse` (`jaxvlse` on PyPI, imports as `vlse`), and its correctness tests go with it.

Everything that used it was poured into the main work tree on 2026-08-04 from the `lbfgsb` worktree (which held zero commits and 846 uncommitted lines, and is now gone): `bench/bench_solver.py`, `bench/bench_budget.py`, `tests/test_x_update_bounded.py`, and the x-update itself. `bench/bench_lbfgsb.py` was already there. They all import `from vlse.lbfgsb import minimise` — a placeholder path, since the upstream release is still in flight. `vlse` is about ready to ship **0.1.0**; when it lands, `uv add jaxvlse` and the imports resolve as written. Until then anything importing `vlse` raises `ModuleNotFoundError: No module named 'vlse.lbfgsb'`; the test skips on it (`pytest.importorskip`), the benches do not.

**The two inner solvers are a `--solver` switch, and `optimistix` is still the default** precisely because `vlse` is not installable yet. `glassogp.py` imports `vlse.lbfgsb` under `try`/`except ImportError` and `x_update_solve_bounded` raises a pointed `ImportError` if it is called without it, so nothing is broken by the missing dependency until someone asks for the solver by name. What changes when they do:

- **`inner_maxiter` is not the same currency.** optimistix counts single steps, `lbfgsb` counts whole line searches, so its budget runs an order of magnitude smaller — `run.py` resolves an unset `--inner_maxiter` to 5 for `lbfgsb` and 50 for `optimistix`.
- **`inner_tol` is not the same criterion either.** optimistix stops on a *step*, `lbfgsb` on the infinity norm of the *projected gradient* (scipy's `pgtol`). An unset `--inner_tol` resolves to 1e-2 for `lbfgsb` and 1e-4 for `optimistix`. Do not carry a number across.
- **`--inner_max_linesearch` (default 5) is the straggler's leash.** A `lax.while_loop` under `lax.map` cannot exit per element, so one output stuck in a long line search charges its whole chunk. Measured at `inner_maxiter=12`, cutting it from 30 to 5 took one x-update from 28.24 s to 9.45 s for the same objective to eight digits and the same projected gradient.
- **the inner ramp's first rung is solver-dependent**, `inner_maxiter_init = 1` for `lbfgsb` against 20 for `optimistix`. A shared 20 would sit above `lbfgsb`'s own cap from iteration 0, making `min(inner_maxiter, inner_maxiter_init * 2**iter)` inert — and, worse, reporting `exact=True` on every x-update, so the convergence break the ramp exists to suppress could fire on a five-line-search solve. This is the one deviation from what the worktree ran; its 67.4 s figure below was measured with the ramp inert.
- **the box is real, not a projection.** `solver_bounds` is the same `[0, theta_max]` box `z` lives in, converted to output layout with a `±inf` column appended for the nugget `w` — `w` saturates through a sigmoid and must stay unbounded (see the nugget bullet above). The x-update still clips its result to `bounds` afterwards, which is a no-op on this path and the actual projection on the optimistix one.

Measured end to end at one knot from a single warmstart: **`lbfgsb` at `inner_maxiter=5 max_linesearch=5` took 67.4 s and 16 ADMM iterations against optimistix's 488.7 s and 35**, at equal-or-better log-likelihood. That is the only comparison in `docs/HANDOFF.md` where both sides start from the same state (see the speed table in `docs/RESULTS.md` for why the other rows are not comparable), and it is a single knot — it is a reason to try the switch, not evidence the whole path is 7× faster.

**The benchmark measures call overhead, not algorithms, and that is the useful finding.** `bench/bench_lbfgsb.py` (`sbatch --partition=gpu --gres=gpu:v100:1 --job-name=bench_v100 slurm/bench_lbfgsb.slurm`) times both solvers on 48 functions:

- one problem at a time on CPU we are 5.8–8.4x scipy, but a control against *numpy* objectives (`scipy` never touching JAX) still gives 2.1–2.7x, and dispatch accounts for 44–53% of scipy's time when the objective is JAX. At `d = 2…8` a solve is 4–65 evaluations of nothing, so what is being timed is the per-evaluation Python round trip, which scipy pays once per evaluation and we pay once per solve. Mean iteration counts agree (39.4 against 40.8) — the algorithms are the same algorithm.
- **on a GPU, one solve at a time is a loss** (0.33–0.67x scipy): a single `value_and_grad` dispatch costs 255 µs (l40s) to 715 µs (a100) against 122 µs on CPU. Batched over 1024 starts with `vmap` it is 46–131x, which is the only reason to put this on a GPU at all.
- **float64 is what ranks the cards.** Batched per-solve summed over the battery: v100 4.43 ms, a100 4.62 ms, l40s 6.27 ms, 1080ti 12.78 ms, one CPU core 64.8 ms. The l40s loses to a five-year-old v100 because it is 1:64 FP64 against the v100's 1:2, and this kernel cannot leave float64. Ask for v100/a100/h100, not l40s.
- **batching is a GPU-only win.** On CPU, `vmap` over 1024 starts costs 65–102 ms per solve against 39–51 ms sequential, because a `lax.while_loop` under `vmap` cannot exit per element and the batch pays for its slowest member (DeJong5: 19.0 ms batched against 0.67 ms sequential). Threads do not help either — one pinned core beat all 16 (39.1 ms against 51.1 ms).

## Column layout is load-bearing

`InclineRunning` builds marker columns as `[LCAL1, RCAL1, LCUB, RCUB, ...]` — left and right of the *same* marker adjacent, each with X/Y/Z. That interleaving is what makes `group_size=6` a single marker across both feet; `--ungroup_feet` (or a single-foot run) drops to `group_size=3`. Reordering those columns silently changes what a "group" means.

Other data facts worth knowing before touching `inclinerunning.py`:

- Train/test split is deterministic even/odd rows (`x[::2]` / `x[1::2]`), not random — consecutive motion-capture frames, so test is not independent of train.
- `--target markers` sets `autoregressive=True`, which makes the GP zero out each output's own marker group from its inputs (in both the design matrix and the parameter vector) so an output cannot predict itself.
- `--target forces` cube-roots the force values and averages the force file in blocks of 5 rows to match the marker sample rate.
- `--relative` (bare) subtracts the LMAL/MMAL midpoint per foot per coordinate; `--relative MARKER` subtracts a named marker and drops that now-zero column.
- Normalization keeps constant columns at 0 rather than dividing by zero.

## Conventions

- JAX runs in float64 (`jax_enable_x64`) in every entry point; the GP kernel is not numerically forgiving without it.
- `kernel` computes the squared distance as `|z1|² + |z2|² - 2 z1·z2` (one matmul) rather than broadcasting an `n×m×d` difference tensor. That expansion is why float64 is non-negotiable here: it cancels. Measured against the broadcast form on fitted models, `max|ΔK| ≈ 2e-14` and relative Δloglik ≈ `1e-12`, far under the `1e-4` nugget floor — but re-check that if the lengthscale bounds ever widen a lot.
- Models are `NamedTuple`s, jitted via `eqx.filter_jit` where a static field (`group_size`) is present.
- Before pickling, `run.py` strips `x_train`/`y_train` off the model (`_replace(x_train=None, y_train=None)`) — loaded models can compute `group_norms` but **cannot** `predict` without re-attaching the data.
- `data/`, `results/`, and all `*.pkl`/`*.tsv`/`*.xlsx`/`*.html` outputs are gitignored. `*.slurm` is gitignored too, but `!slurm/*.slurm` overrides it: the batch scripts are the only record of what each generation of results was submitted with, so commit them alongside the code change they ran.
