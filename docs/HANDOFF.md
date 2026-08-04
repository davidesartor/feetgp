# feetgp — handoff, 2026-08-04

> **Audited 2026-08-04 against the saved `admm_state`s before those results were deleted; four
> claims below did not survive.** See `RESULTS.md`: the near-zero-λ dual band is not the absolute
> floor `sqrt(p)·EPS` collapsing (`eps_dual` is 1.11e-3 there, `‖u‖` growth is what loosens the
> test); the scipy-baseline speed rows are not comparable to each other (30.2 s/ADMM-iteration
> against 49.4 s/x-update in the same job); job 62586894 is TIMEOUT, not RUNNING; and v8 (array
> 62588680) never ran — it was PENDING with no `results_v8` on disk. Also, paths in this file
> predate the 2026-08-04 restructure: library **and** entry points are in `src/feetgp/` and run as
> `python -m feetgp.run`, benches in `bench/` (run with `-m`), batch scripts in `slurm/`. Every
> result directory this file refers to, `results_v7` included, has been deleted; `RESULTS.md` and
> `results_summary.jsonl` are all that is left of them.

Paste this whole file into a new chat to resume. Repo `/home/dsartor_umass_edu/feetgp`, branch
`main`, worktree `/home/dsartor_umass_edu/feetgp-lbfgsb` on branch `lbfgsb`. Read `CLAUDE.md` and
`ROADMAP.md` first — they carry the invariants; this file only carries state and the open threads.

## The standing goal (user's words)

> working means:
> - run produces good trajectories for the groups (as lambda increases, they are progressively
>   driven to zero. not all at the same time, one by one. accuracy or model compatibility degrades
>   smoothly as lambda increases)
> - run is significantly faster than the baseline using scipy.minimize and syncing to host
>   (approx 24h now)

## Standing constraints — do not relitigate

- **Never hand a fit a warmstart sparser than its own solution.** Death is absorbing in the GP;
  the λ walk is upward-only and that is a correctness requirement, not a preference.
- `adapt_rho and iter < max_iterations // 2` — the rho-freeze cutoff is deliberate, to avoid
  oscillatory behaviour. Do not change its effective behaviour silently.
- Do not add residual-logging fields to `ADMMState`. Convergence bookkeeping goes in `info`.
- Do not hand a warmstart sparser than the fit's own solution, and do not read `state.x` for
  `n_active` — sparsity lives in `z`.
- Do not reintroduce `rho_floor` (it froze every path at full support — ROADMAP defect 9).
- Do not reintroduce a bound on the nugget's `w` column (the old clip made every λ unconvergeable).
- GP runs `alpha = 1.0`. Boyd's 1.5–1.8 over-relaxation band is wrong here; `linear.py` keeps 1.6.
- `uv` does all Python: `uv run`, `uv add`/`uv remove`. Never bare `python`/`python3`, never raw
  `pip`, never hand-edit `pyproject.toml`/`uv.lock`.
- SLURM: always request **both** partitions, `--partition=gpu,gpu-preempt`. Timing jobs use
  `--qos=short` (limit: one submitted job per user, so it is often unavailable).
- The P1 methodology track (train/test split change, λ selection, debiasing, NLPD, `llk` key split)
  was never approved. Do not start it.

## Where the speed stands

| | s per ADMM iteration | s per x-update |
|---|---|---|
| scipy baseline (job 62585927, A100-PCIE-40GB, worktree `feetgp-baseline` @ 16e0f91) | 30.2 | 49.4 |
| current main, optimistix LBFGS, chunk 39 | 7.77 | — |
| chunk 78 (full vmap) | 7.53 | — |
| L-BFGS-B at the shipped budget (`maxiter=5 ls=5`), measured at the λ=34.97 knot | 4.21 | — |

~4× per iteration for optimistix against the baseline, ~7× for L-BFGS-B, before counting the
iterations the old code wasted — and it wastes fewer of them too (16 against 35 at the knot).

## Where the trajectories stand — results_v7 (array 62584583, still running)

v7 is the **optimistix baseline**, deliberately left to finish under the old solver so v8 has
something to be compared against. As of 2026-08-04: tasks 0, 2, 3, 6, 8, 9 COMPLETED, tasks 4, 5, 7
RUNNING (~5h elapsed), task 1 PENDING. Do not resubmit it against the new solver.

Rendered: `uv run python plots.py --results_dir results_v7`.

- **markers/both** (38 λ, done): 13/13 flat to λ=459, then 12→11→…→1→0 one at a time, R²
  0.9998 → 0.859 at 4 groups. 3 λ unconverged, all inside the flat region. **This is the target
  shape.**
- **markers/right_only** (36 λ): 13→11→10→9→8→7→5→4→2→0, R² smooth to 0.986 at 8 groups, collapses
  below 4. 4 λ unconverged.
- **markers/left_only** (48 λ): same shape with one non-monotone bump (8 at λ=566, back to 9 at
  λ=849).
- **forces/both** (43 λ): clean 13→12→11→10→9→8 to λ=16.95, then broken — see open thread 2.
- **markers/both_ungrouped** (18 λ so far): 26/26 through λ=142, plus the dual-only band, thread 1.

## Thread 0 — the ADMM extraction (**shipped on main, 2026-08-04, uncommitted**)

The algorithm now lives in `admm.py`; `glassogp.py` and `linear.py` own only their x-update. This
landed *after* the lbfgsb worktree branched, so it is the thing thread 3's merge has to be replayed
onto — see "The planned integration" below.

**API.** `admm.solve(x_update, state, l1, *, max_iterations, tol, bounds, alpha, adapt_rho,
adapt_rho_iters, log_every) -> (ADMMState, info)`. The caller passes one closure,
`XUpdate = (state, iteration) -> (state, exact)`; the library owns `x_hat`, the prox, the u-update,
the residuals, rho adaptation, the loop, `info`, and the tqdm/stdout logging.

- **Layout convention `(... g)`** — leading axes index groups, the last axis holds one group's
  members. That is what deletes `group_size` from `admm.py`: the prox is a norm over the last axis
  and nothing else needs to know what a group is. Models convert at the boundary with
  `admm.to_groups` / `admm.to_outputs` (`rearrange(v, "o (d g) -> d (o g)", g=group_size)`), which
  pools one marker across every output and coordinate — the same element set the old
  `"o (d g) -> (o g) d"` + `axis=0` norm took.
- **`ADMMState(x, z, u, rho, aux)`.** `aux` is threaded through untouched, for parameters the
  x-update owns but the consensus must not see. The GP nugget `w` rides there, which makes
  structural what used to be the trick of giving `w` a column in `x` with `±inf` bounds so that `z`
  happened to copy it exactly.
- **`exact`** generalizes the GP's `maxiter == inner_maxiter` gate: a subproblem solved under budget
  suppresses the convergence break, so a cheap early iteration cannot certify convergence. The
  linear model's closed-form solve always returns `True`.
- **`bounds` are a `solve` argument, not state** — always rederived from the data, so nothing has to
  strip them before pickling.
- `alpha` is a float on `solve`, default 1.0. GP pins 1.0, linear keeps 1.6. Unchanged behaviour,
  but it is now explicit rather than a default `run.py` silently never passed.

**Pickle compatibility. `STATE_FORMAT` 4 → 5.** Format 5 is the same parametrization as 4 with a new
layout, so it is **converted, not refused** — `admm_state_from_legacy` in each module, plus
`admm_state_from_pickle(results)` which is what benches should call. Field-only `GLASSOADMMState`
stubs stay in both modules because pickle resolves a class by module + name: deleting them makes
every old result *unreadable*, not merely un-warmstartable. Verified on a real v7 pickle — theta
round-trips bit-identical, group norms match the old `rearrange` reference, `rho` preserved — and
end to end by resuming a copied v7 dir, which loads cached format-4 λ and warmstarts the next λ off
them.

**Also touched.** All seven benches ported (`bench_signs` / `bench_inner` / `bench_sync` onto
`glassogp.x_update_solve` + `admm.*` — the last two were already stale against the format-3/4 loss
signature and are runnable again; `bench_chunk` / `bench_knot` / `bench_death` / `bench_rho` onto
`admm_state_from_pickle`). `EPS` deleted from `glassogp.py`, dead — `admm.EPS` is the one.
`tests/test_admm.py` is new (13 tests); `tests/test_models.py` updated for the 6-element
`admm_x_update_loss` args and the new layout. **22 pass on main.**

## Open thread 1 — the near-zero-λ dual-only band (cost, not correctness)

`markers/both_ungrouped`, λ = 0.2166 … 2.467: seven consecutive fits burn the full 400 iterations
at 26/26 active, then λ=3.70 converges in **8**.

| λ | primal | dual | rho | iters |
|---|---|---|---|---|
| 0 | 0.000e+00 | 2.30e-07 | 1e-6 | 21 |
| 0.2166 | 3.56e-02 | 5.32e-03 | 6.25e-2 | 400 |
| 2.4671 | 1.09e-01 | 3.12e-02 | 2.50e-1 | 400 |
| 3.7006 | 1.26e-01 | 1.52e-02 | 2.50e-1 | **8** |

λ=3.70 passes with a *larger* primal residual than λ=0.2166 which fails, so the binding test is the
dual one. `eps_dual`'s relative term is `tol * rho * ||u||`; just above λ=0 the prox barely shrinks
anything, `u` stays near zero, and the criterion collapses onto the absolute floor
`sqrt(p) * EPS = 1.16e-6` — four orders under the inexact x-update's own noise. It is the λ=0
degeneracy leaking into a neighbourhood (at λ=0 exactly it is harmless: `u ≡ 0`, rho decays to
`RHO_MIN`, and `rho * ||Δz||` clears the floor because rho is tiny).

R² is 0.99979 across the whole band, identical to λ=0, so the path shape is unaffected. The bill is
7 × 400 = 2800 wasted ADMM iterations. **Do not fix by loosening `tol` globally** — the dual
criterion needs a floor tied to the inner solver's accuracy, not to machine epsilon.

Written up in `ROADMAP.md` under "A second, distinct knot".

## Open thread 2 — bisection lands in a different basin (this one is correctness)

`forces/both` tail, verbatim (λ, active, R²_test, converged, iterations):

```
      16.94915    8  r2= 0.89169  ok     3
      16.96258    4  r2= 0.89952  ok   158      <- 8 -> 4 in one bisection step, R2 goes UP
      17.84445    6  r2= 0.89731  NOT  400      <- back to 6
      26.76668    5  r2= 0.84468  ok   294
     135.50632    4  r2= 0.71184  ok   343
     137.23421    3  r2=-0.00009  ok     5      <- 3 groups but R2 collapses to 0
     138.98414    0  r2=-0.00012  ok     6
     142.55122    1  r2=-0.00008  ok     6      <- 0 -> 1 as lambda increases
```

The problem is nonconvex — `theta = 0` satisfies stationarity at every λ > 0, so which local
solution ADMM finds is decided entirely by the warmstart. Two mechanisms, both now measured. rho is
**not** one of them: it sits at 1–4 across the whole region and the smallest live group's prox
threshold `l1 / (rho ||z_g||)` is 7–230, i.e. far past 1 everywhere, so nothing is being held off
its death threshold.

**(a) Refinement fits rubber-stamp their warmstart.** Look at the iteration counts: λ = 16.85,
16.91, 16.94, 16.95 all converge in **3** ADMM iterations at 8 active with `min|g| = 0.114`, then
λ=16.96258 runs **158** and lands at 4 active with `min|g| = 1.186` and a *higher* R². Three is the
earliest convergence the code permits — the inner budget ramps `inner_maxiter_init * 2**iter` and
the convergence break is suppressed until it reaches the cap — so those fits pass the residual test
before the solution has had a chance to move at all. They converge to the *previous* λ's solution.
Only the one fit that ran long escaped to the better basin.

**(b) Adjacent λ in the final path come from different continuation chains.** Refinement warmstarts
each bisection from `states[lo]`, the denser end of *its own* interval, which is the right rule in
isolation; but the result is then printed next to a λ that was reached down a different chain. That
is what the `left_only` tail shows — λ=3014.1 at 1 active, λ=3170.8 at 2 active — and it is
presentational as much as it is real.

**Fix direction.** Process refinement points in increasing λ and warmstart each from the nearest
*already-fitted lower* λ, so the final sorted grid is one chain; and make the convergence test
require that `z` actually moved, so a fit cannot certify its own warmstart in 3 iterations. Neither
is written yet.

## Thread 3 — the L-BFGS-B solver swap (**decided: it wins, 8.7× at the shipped budget**)

**Why.** `theta >= 0` is load-bearing for ADMM convergence (dead-group stationarity relaxes from
the measure-zero `u = 0` to the inequality `u >= 0`). Main currently gets the box from
`jax.nn.relu(theta)` inside the likelihood plus a projection after an *unconstrained* solve. A
Byrd–Lu–Nocedal–Zhu bounded L-BFGS makes the box the solver's actual feasible set, so the relu and
the projection both become unnecessary.

**The solver is not in this repo any more — it lives in `vlse`.** The Byrd–Lu–Nocedal–Zhu port was
written here (`lbfgsb.py` + the scipy-fidelity battery `tests/test_lbfgsb.py`) and has been moved
out into the `vlse` package (`jaxvlse` on PyPI, imports as `vlse`), which is **about ready to ship
0.1.0**. Nothing else in the JAX ecosystem has a bounded L-BFGS — optimistix and optax have LBFGS
only, jaxopt has `LBFGSB` but is deprecated — which is why it exists at all. Both files were
deleted from `main` on 2026-08-04; the correctness tests go with the package.

Consequence for the merge: `feetgp` should **depend** on it rather than carry it.
`uv add jaxvlse`, then `from vlse.lbfgsb import minimise`. The worktree still has the local
`lbfgsb.py` and does a bare `import lbfgsb` (`glassogp.py:9`, call site `glassogp.py:220`) — that
is what v8 is running under, and it is the same code, so the numbers below carry over. Switch the
import as part of the merge, once 0.1.0 is on the index. `bench_lbfgsb.py` on main already
anticipates this and imports `from vlse.lbfgsb import minimise`, so it is broken until then.

**Status.** Fully wired and tested in the `feetgp-lbfgsb` worktree — **208 tests passed** at the
time of the swap (196 `test_lbfgsb.py` + 12 `test_models.py`); on main only the 12 model tests
remain, the other 196 having gone to `vlse`. Port validated against scipy on the 26-function
battery (jobs 62586544 L40S / 62586572 1080Ti / 62586573 V100): mean iterations 27.2 ours vs 28.6
scipy — same path — at 2.24× / 4.08× / 3.57× sequential and 388× / 280× / 811× batched.

**The blocker, since resolved.** End-to-end at the knot λ=34.97 warmstarted from λ=26.90, all on
A100, `inner_tol=1e-2`, `--maxiter 60 --chunk_size 39`:

```
optimistix reference:              35 iters,  488.7s (13.96 s/iter), primal=1.63e-1, 13/13
L-BFGS-B inner_maxiter=50 ls=30:   27 iters, 1316.2s (48.75 s/iter), primal=1.28e-1, 13/13, loglik=489762
L-BFGS-B inner_maxiter= 5 ls=30:   19 iters,  236.8s (12.46 s/iter), primal=2.06e-1, 13/13, loglik=489801
L-BFGS-B inner_maxiter= 3 ls=30:   16 iters,  135.9s ( 8.49 s/iter), primal=2.18e-1, 13/13, loglik=489796
```

At `inner_maxiter=50` it was **2.7× slower** despite converging in fewer ADMM iterations (27 vs 35).
At `inner_maxiter=3` it is **3.6× faster than optimistix end-to-end** (135.9s against 488.7s) and
takes **16 ADMM iterations against 35**, at a *better* log-likelihood (489796 against 489762 for the
50-budget run). The cheaper the inner solve, the fewer ADMM iterations — inexact ADMM in its
intended regime, and the same lesson the repo already learned once with optimistix.

**Cause of the original blocker: the budget unit is not the same currency.**
`lbfgsb.minimise(max_iterations=50)` is 50 *outer* iterations, each a line search of up to
`max_linesearch=30` function+gradient evaluations. `optx.minimise(max_steps=50)` is ~50 evaluations
total. `--inner_maxiter 50` carried over from the optimistix defaults and meant 3–4× the work.

**What a unit of budget costs** (job 62587527, `bench_budget.py`, one real x-update, A100):

```
maxiter=  3 max_linesearch=30:   7.60s  objective=-485132.40  evals/output mean=12.3  at_cap=74/78  pgrad median=2.711e-01
maxiter=  5 max_linesearch=30:  12.49s  objective=-485138.69  evals/output mean=16.6  at_cap=56/78  pgrad median=2.130e-01
maxiter=  8 max_linesearch=30:  19.21s  objective=-485141.61  evals/output mean=18.5  at_cap=45/78  pgrad median=1.343e-01
maxiter= 12 max_linesearch=30:  28.24s  objective=-485145.38  evals/output mean=20.9  at_cap=42/78  pgrad median=6.222e-02
maxiter= 20 max_linesearch=30:  46.15s  objective=-485157.74  evals/output mean=24.5  at_cap=30/78  pgrad median=1.275e-02
```

Two things fall out.

**Wall time is linear in the outer-iteration budget, not in mean evaluations** — 2.53 / 2.50 / 2.40
/ 2.35 / 2.31 s per outer iteration across the sweep, while mean evals/output only goes 12.3 → 24.5.
That is the `lax.map` straggler tax: a `lax.while_loop` under `vmap` cannot exit per element, so the
chunk pays for whichever output is still running a full 30-evaluation line search, and 30–74 of 78
outputs are at the cap at every budget. **Cutting `max_linesearch` attacks the term that actually
sets wall time; cutting `maxiter` only scales it.** Stage 2 of the budget job measures that.

**The objective barely moves.** Across the whole 3 → 20 range it spans 25.3 out of 485 000, i.e.
5.2e-5 relative — well inside what inexact ADMM tolerates (subproblem errors only need to be
summable). The projected gradient does move, 2.7e-1 → 1.3e-2, so the budget is buying accuracy that
the augmented-Lagrangian value does not reflect.

Against optimistix's 13.96 s per whole ADMM iteration, the cost-matched budget is
**`inner_maxiter ≈ 5`**, not the 12–15 first guessed.

**Both levers at once is the operating point** (job 62587874):

```
optimistix reference:            35 iters, 488.7s (13.96 s/iter), loglik 489762
L-BFGS-B maxiter=50 ls=30:       27 iters, 1316.2s (48.75 s/iter), loglik 489762
L-BFGS-B maxiter=12 ls= 5:       25 iters,  272.6s (10.90 s/iter), loglik 489785
L-BFGS-B maxiter= 5 ls= 5:       16 iters,   67.4s ( 4.21 s/iter), loglik 489799
L-BFGS-B maxiter= 3 ls= 5:       17 iters,   55.9s ( 3.29 s/iter), loglik 489799
```

**8.7× faster than optimistix end-to-end at `maxiter=5 ls=5`**, in fewer than half the ADMM
iterations, at the best log-likelihood in the whole sweep. Cold λ=0 verified separately under a
capped line search: `primal = 0.000e+00` exactly, 22 iterations, loglik 496094.

One anomaly worth not over-reading: `maxiter=20 ls=5` **failed** to converge in 60 iterations
(primal 1.17). The budget is not monotone in quality, so do not assume a larger one is the safe
choice.

**`max_linesearch` is the straggler's leash, and it is free.** Same x-update at `maxiter=12`:

```
max_linesearch= 5:   9.45s  objective=-485145.36  evals/output mean=13.5 max=22  pgrad median=6.222e-02
max_linesearch=10:  17.19s  objective=-485145.38  evals/output mean=16.5 max=65  pgrad median=6.222e-02
max_linesearch=20:  27.89s  objective=-485145.38  evals/output mean=20.6 max=76  pgrad median=6.222e-02
max_linesearch=30:  28.24s  objective=-485145.38  evals/output mean=20.9 max=78  pgrad median=6.222e-02
```

Identical objective to 8 digits and an identical projected gradient, for **3× the wall time**. The
cap is not an accuracy knob under `lax.map` — it only bounds the worst output, and the worst output
is what the whole chunk waits for (max evals/output 78 → 22).

**Shipped.** `run.py` defaults are now `--inner_maxiter 5 --inner_max_linesearch 5 --inner_tol 1e-2`
(was 50 / — / 1e-4; 50 was an optimistix number and cost 8.7× here). Plumbing smoke-tested through
`run.py` end to end on CPU at `--subsample 600` before launch. **v8 is running as array 62588680**,
launched *from the worktree* so the still-running v7 array cannot pick the new solver up mid-flight,
writing to `/home/dsartor_umass_edu/feetgp/results_v8`.

**The merge is gated on v8, and it is the next thing to do.** The main tree still runs optimistix;
the worktree exists only so the still-running v7 array cannot pick the new solver up mid-flight, and
it should be deleted (`git worktree remove ../feetgp-lbfgsb`) once merged. The merge is a *replay*,
not a copy — see "The planned integration" below, because main has since had the ADMM extraction and
the worktree has not. The gate is v8's *paths*, not its speed: speed is already measured. What to
look for is one-group-at-a-time deaths, R² degrading smoothly, and no repeat of the `forces/both`
tail in thread 2.

## Jobs in flight (as of 2026-08-04)

| id | what | state | log |
|---|---|---|---|
| 62584583_* | v7 GP array, the optimistix baseline | 4/5/7 RUNNING, 1 PENDING, rest COMPLETED | `logs/gp/v7_*` |
| 62588680_0..9 | **v8 GP array**, L-BFGS-B at the shipped budget, → `results_v8` | PENDING | `feetgp-lbfgsb/logs/gp/v8_62588680_*` |
| 62586894 | `bench_solver`: knot at `inner_tol` 1e-2 (done), 1e-3, 1e-4, then cold λ=0 | RUNNING | `feetgp-lbfgsb/logs/gp/solver_62586894.out` |
| 62587527 | `bench_budget`: evals/output vs `maxiter`, then `max_linesearch` at `maxiter=12` | done, table above | `feetgp-lbfgsb/logs/gp/budget_62587527.out` |
| 62587731 | `bench_solver_budget`: knot at `inner_maxiter` ∈ {3,5,8} | done | `feetgp-lbfgsb/logs/gp/solvbud_62587731.out` |
| 62587755 | `bench_solver_linesearch`: `maxiter` ∈ {12,20} at `ls=5`, then cold λ=0 | done | `feetgp-lbfgsb/logs/gp/solvls_62587755.out` |
| 62587874 | `bench_solver_corner`: `maxiter` ∈ {3,5} at `ls=5` — produced the shipped operating point | done | `feetgp-lbfgsb/logs/gp/solvcnr_62587874.out` |

(62587561 was the same job with the grid {8,12,20}; cancelled once the budget bench showed the
cost-matched point is near 5, not 12. 62586894 can be killed — its remaining stages sweep
`inner_tol` at `inner_maxiter=50`, a budget that is no longer the operating point.)

## What is already staged in the worktree

`glassogp.py` (x_update on `lbfgsb.minimise`, no relu, no post-projection; `fit` takes `inner_tol`
and `inner_max_linesearch`), `run.py` (`--inner_tol` is now a projected-gradient tolerance,
`--inner_max_linesearch` added, and the three budget defaults changed), `tests/test_models.py`
(relu-plateau test replaced by a dead-group test and a feasibility test), `lbfgsb.py` and
`tests/test_lbfgsb.py` (**do not copy these to main — they went to `vlse`**), `bench_solver.py`,
`bench_budget.py`, `bench_solver_budget.slurm`, `bench_solver_linesearch.slurm`,
`bench_solver_corner.slurm`, `run_gp_v8.slurm`.

## The planned integration — lbfgsb onto `admm.py`

**Copying the worktree files over main is no longer the merge.** The worktree branched before the
ADMM extraction, so its `glassogp.py` and `run.py` are format-4, `GLASSOADMMState`-shaped files;
dropping them in reverts thread 0. Replay the solver swap onto main's structure instead. It is a
small replay, because the extraction put the whole swap inside one function.

1. **`glassogp.x_update_solve` is the only place the solver appears.** Swap `optx.LBFGS` +
   `optx.minimise` for `minimise(...)` from the bounded port, drop `jax.nn.relu` from
   `admm_x_update_loss`, and pass `bounds` in (converted to the `(... g)` layout with
   `admm.to_groups`, or kept per output — pick one and say which in the signature). The `fit`
   closure then loses its `jnp.clip(theta, *bounds)`, since feasibility becomes the solver's job.
   `admm.z_and_u_update` should still be handed `bounds`: the box has to hold on **both** `x` and
   `z`.
2. **Which port? — decided: `vlse`.** The solver was written here but now lives in the `vlse`
   package (`jaxvlse` on PyPI, imports as `vlse`), **about ready to ship 0.1.0**; main deleted
   `lbfgsb.py` and `tests/test_lbfgsb.py` on 2026-08-04 and `bench_lbfgsb.py` already imports
   `from vlse.lbfgsb import minimise`. So: `uv add jaxvlse` and import from `vlse` — do **not** bring
   the worktree's `lbfgsb.py` back. The worktree's bare `import lbfgsb` is the same code, so every
   benchmark number above carries over unchanged; only the import line moves. Until 0.1.0 is on the
   index, `bench_lbfgsb.py` on main is broken, and that is the one thing blocking this step.
3. **`run.py`**: `--inner_tol` becomes a projected-gradient tolerance and `--inner_max_linesearch`
   is added, defaults `--inner_maxiter 5 --inner_max_linesearch 5 --inner_tol 1e-2`. No
   `STATE_FORMAT` bump — the solver swap does not move the parametrization, so format-5 pickles stay
   valid warmstarts. Bump only if `w` or `theta` changes meaning.
4. **Tests**: replace `test_x_update_objective_is_flat_below_zero` (the relu plateau is gone) with
   the worktree's dead-group and feasibility tests. `tests/test_admm.py` is solver-agnostic and
   should pass untouched — if it does not, the swap leaked into the library.
5. **`CLAUDE.md`** sections to rewrite: the x-update bullet (`optimistix.LBFGS`), "the inner budget
   is a cap, not a tolerance" (step criterion → projected gradient, and its numbers are optimistix
   numbers), and the relu bullet with its `test_x_update_objective_is_flat_below_zero` reference.
   "the bounded L-BFGS lives in `vlse` now" is already correct and only needs its last line updated,
   from "fix the import when `vlse` publishes" to the shipped dependency.
6. Then `sbatch run_gp_v8.slurm`.

Both changes are uncommitted on main; commit the extraction before starting the replay so a bad
merge is one `git checkout` away.

## The path ahead, in order

1. **Watch v8 (62588680), then replay the merge.** Gate above, six-step recipe in "The planned
   integration". This is the only in-flight task. Step 2 of it waits on `vlse` 0.1.0.
2. **Update `CLAUDE.md` and `ROADMAP.md` for the L-BFGS-B x-update** — step 5 of that recipe lists
   the exact sections.
3. **Fix refinement chaining** (thread 2, correctness). Process refinement points in increasing λ,
   warmstart each from the nearest already-fitted *lower* λ so the final sorted grid is one
   continuation chain, and make the convergence test require `z` to have actually moved so a fit
   cannot certify its own warmstart in 3 iterations. Not written yet.
4. **The near-zero-λ dual band** (thread 1, cost only — 2800 wasted iterations, no path damage).
   Needs a dual floor tied to the inner solver's accuracy rather than to machine epsilon. Lower
   priority than 3; it costs time, not correctness.

### Follow-ups, not started, not approved

- **Multi-GPU output sharding.** The x-update is `lax.map` over outputs on one device. `shard_map`
  across GPUs is the obvious next parallelism lever, estimated ~2× at 2 GPUs — outputs are
  independent given `theta`, so it is close to embarrassingly parallel, minus the straggler tax
  which sharding does not remove. Nobody has measured it.
- **The whole P1 methodology track** (train/test split change, λ selection, debiasing, NLPD, `llk`
  key split). Never approved. Listed here only so it is not rediscovered as if it were new.

## Handy commands

```bash
uv run pytest tests                                   # 22 on main, 208 in the worktree; CPU only
uv run python plots.py --results_dir results_v7
squeue -u $USER -o "%.10i %.10j %.8T %.9M"
sbatch run_gp_v8.slurm
```

Path dump used above (needs `PYTHONPATH` so the pickles can find `glassogp`):

```bash
PYTHONPATH=/home/dsartor_umass_edu/feetgp uv run python - <<'EOF'
import glob, pickle, re, sys
import numpy as np
d = "results_v7/model=gp/target=forces/feet=both/inclines=all_sub=20"
paths = sorted(glob.glob(d + "/lambda=*.pkl"),
               key=lambda p: float(re.search(r"lambda=([0-9.e+-]+)\.pkl", p).group(1)))
for p in paths:
    l1 = float(re.search(r"lambda=([0-9.e+-]+)\.pkl", p).group(1))
    r = pickle.load(open(p, "rb"))
    n_active = int((np.asarray(r["group_norms"]) > 0).sum())
    info = r["info"]
    print(f"{l1:14.5f} {n_active:4d}  r2={np.median(r['r2_test']):8.5f}  "
          f"{'ok ' if info['converged'] else 'NOT'} {info['iterations']:5d}  "
          f"rho={float(r['admm_state'].rho):.3e}")
EOF
```
