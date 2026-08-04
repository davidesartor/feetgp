# What the deleted result directories showed

Seven generations of pickles (`results_v4` … `results_v7_seeded_from_v6`, plus `results_refine`,
`results_smoke`, `results_bench`, and the `results_pre_2026-08-03` archive) were deleted on
2026-08-04, 416 MB in total. This file is what survives them. Machine-readable per-run summaries —
λ grid size, active-group trajectory, R², convergence counts — are in
[results_summary.jsonl](results_summary.jsonl), regenerable for future runs with
`uv run python -m feetgp.summarize_runs results_<dir>`.

`results_v7` was kept a few hours longer — SLURM array 62584583 was still writing it — and then
deleted too, 74 MB, mid-flight. **No pickle from any generation still exists.** Everything below
was measured off those states before they went; none of it can be recomputed without rerunning.

## The provenance problem, first

Every `meta.json` in all seven generations records the same `git_revision`, `b7c4dc6` — the
2026-07-03 commit. Nothing has been committed since, so seven generations of results that differ
*only* by code changes all claim to come from one revision. The recorded configs are nearly
identical too:

| dir | subsample | maxiter | inner_maxiter | inner_tol | lambda_step | lambda_refine |
|---|---|---|---|---|---|---|
| v4 | 20 | 400 | 50 | 1e-4 | 1.3 | 100 |
| v5 | 20 | 400 | 50 | 1e-4 | 1.3 | 25 |
| v6 | 20 | 400 | 50 | 1e-4 | 1.3 | 25 |
| v7 | 20 | 400 | 50 | 1e-4 | 1.5 | 30 |

So the *only* record of what distinguished v4 from v7 is the prose in `ROADMAP.md`. That is the
single strongest argument for the restructure: commit before each generation, or the ledger is
fiction. Nothing below can be re-derived from the pickles alone.

## Timeline

| dir | written | what it was | verdict |
|---|---|---|---|
| `results_pre_2026-08-03` | 2026-07-04 → 07-07 | 725 GP pickles with no `admm_state`, `%013.6f` λ names, λ values like `98758233950707.4` | unusable even as results (predates defect 3); archived, now deleted |
| `results_smoke` | 08-03 12:41 | sub=200, `inner_maxiter=20` plumbing smoke test | 12/12 λ unconverged; served its purpose |
| `results_refine` | 08-03 13:12 | linear, sub=400, `tol=1e-6`, `chunk_size=8` | 28/35 λ unconverged at that tolerance |
| `results_v4` | 08-03 14:00 → 18:11 | first post-defect-3 (θ ≥ 0 box) runs | GP grid built by the **downward** walk of defect 7 → GP paths unusable; **linear paths were the acceptance evidence** |
| `results_v5` | 08-03 18:30 → 20:43 | first upward-walk GP runs | every GP path frozen at full support (defect 8/9 era) |
| `results_v6` | 08-03 20:59 → 08-04 00:05 | GP continued, seeded from v5's converged pickles | still frozen: 13/13 for 38 λ; one death at λ=2153 |
| `results_v7_seeded_from_v6` | 08-04 00:09 | v6 copied to seed a fresh run | the seed reproduced the freeze — see below |
| `results_v7` | 08-04 00:09 → still running | relu-in-likelihood, no rho floor, α=1.0 | **first generation with real trajectories** |

## The four GP failure modes, each visible in the data

**v4 — downward walk (defect 7).** GP runs finished in 1–9 λ and jumped straight to zero support:
`markers/right_only` 13 active at λ=0 → 0 at λ=98; `markers/both` 13 → 12 at λ=323 → 0 at λ=420.
`right_only --relative` was non-monotone as λ *increased*: 2 active at λ=23, 4 at λ=30, 3 at λ=39.
Death is absorbing, so chaining each fit onto a sparser neighbour keeps the extra dead groups.

**v5/v6 — the rho floor (defect 9).** The mirror image: nothing died at all. v5 GP is 13/13 at every
λ in all six marker configs; v6 is 13/13 for 38 λ in `markers/both`, and `right_only` holds 13/13
until λ=2153. Holding `l1/rho ≤ min_norm/2` keeps every group a factor of two from its own death
threshold, permanently.

**v7_seeded_from_v6 — the freeze is inherited.** Seeding a new run from v6's pickles reproduced v6
exactly: 37 λ at 13/13 in `markers/both`, 28 λ at 13/13 in `left_only`, every fit "converged". A
warmstart carries the pathology across generations, which is why a code fix needs a fresh grid, not
a resumed one.

**v7 — real paths, with the tail still broken.** `markers/both`, the target shape:

```
λ      0 …  459   13 active   R² 0.9998      (13/13 across three decades of λ)
λ    689 → 3490   12 → 5      R² 0.9996 → 0.925   one group at a time
λ   5236          4           R² 0.859
λ   5302          3           R² 0.349      <- collapse
λ   5370 → 7853   2 → 0       R² ~0.0003
```

`right_only` and `left_only` have the same shape; `left_only` has one non-monotone bump (8 active at
λ=566, back to 9 at λ=849), and `right_only --relative` skips (9 → 7 → 4 → 1). `forces/both` is
clean 13 → 8 to λ=16.7 and then breaks:

```
16.94915   8 active   R²=0.892   ok    3 iterations
16.96258   4 active   R²=0.900   ok  158 iterations   <- 4 groups die in one bisection step, R² rises
17.84445   6 active   R²=0.897   NOT 400 iterations   <- back to 6
137.23421  3 active   R²=-0.0001 ok    5 iterations   <- support survives, model does not
138.98414  0 active
142.55122  1 active   <- support grows as λ grows
```

## Three findings the pickles support that the handoff notes do not

**1. Late-path "converged" flags mean almost nothing.** The stopping test is
`primal < sqrt(p)·EPS + tol·max(‖x‖,‖z‖)` with `tol=1e-3`. Over the entire second half of
`markers/both`, the primal residual does not fall — it *grazes* the threshold:

| λ | primal | eps_pri | ratio |
|---|---|---|---|
| 1034 | 0.264194 | 0.264400 | 0.9992 |
| 1551 | 0.263000 | 0.264400 | 0.9947 |
| 2448 | 0.263600 | 0.264400 | 0.9970 |
| 2850 | 0.264300 | 0.264400 | 0.9996 |

Every one of those is stamped `converged=True` on a margin of 0.1 %. The tolerance, not the
optimum, is deciding where each fit stops — and therefore which group dies next.

**2. The dual criterion loosens as the path advances, which is what the "near-zero-λ dual band"
actually is.** `eps_dual = sqrt(p)·EPS + tol·rho·‖u‖`, and `dual = rho·‖Δz‖`, so rho cancels and the
test reduces to `‖Δz‖ < tol·‖u‖`. In `markers/both_ungrouped`:

| λ | ‖u‖ | ‖Δz‖ | eps_dual | outcome |
|---|---|---|---|---|
| 0.2166 | 17.7 | 0.085 | 1.11e-3 | 400 iterations, NOT converged |
| 2.4671 | 50.3 | 0.125 | 1.26e-2 | 400 iterations, NOT converged |
| 3.7006 | 75.5 | 0.061 | 1.89e-2 | **converged in 8** |

The handoff explains this band as the criterion "collapsing onto the absolute floor
`sqrt(p)·EPS = 1.16e-6`". It does not: `eps_dual` is 1.11e-3 at λ=0.2166, three orders *above* that
floor, and `‖u‖` is 17.7, not near zero. What changes between the failing and passing fits is that
`‖u‖` has grown 4×, loosening the test, while `‖Δz‖` is the same size on both sides. So the proposed
fix — a dual floor tied to inner-solver accuracy — addresses a mechanism that is not the one
operating, and z really is still moving 0.06–0.09 per iteration in the fits that pass.

**3. rho tracks λ, so part of the λ axis is illusory.** rho doubles whenever
`primal > 10·dual`, and with the primal residual pinned near tolerance it climbs with λ:
λ=26.9 → rho=1, λ=459 → 4, λ=1034 → 16, λ=3490 → 128, λ=5236 → 512. The prox threshold is `l1/rho`,
so the ratio λ/rho — the thing that actually kills groups — is non-monotone (27, 115, 65, 27, 10)
even though λ is monotone. Deaths happen when rho lags λ, not when λ crosses a group's norm. This is
the same failure the `rho_floor` produced, arriving through the adaptation rule instead.

## The linear ablation is the part that met its acceptance bar

`results_v4/model=linear` and `results_v5/model=linear`, 10 configs, `--lambda_refine 100`: a
gap-free ladder from full support to zero, one group per step, 33–60 λ per config, 13s–9m06s each on
CPU. Example, `markers/both` (42 λ): 13 @ 0 → 12 @ 0.15 → 11 @ 0.20 → 10 @ 0.57 → 9 @ 1.63 → 8 @ 2.12
→ 7 @ 6.07 → 6 @ 49.5 → 5 @ 83.6 → 4 @ 141 → 3 @ 404 → 2 @ 887 → 1 @ 1499 → 0 @ 4280, R² 0.9999 down
to 0.641 at one group. The linear objective is not even, so death is not absorbing there and the
walk direction does not matter.

One caveat that carried through every generation: on `--target forces` the linear model scores
**R² = −1.94 at λ=0**, i.e. worse than predicting the mean before any penalty is applied. Its force
paths measure regularization of a model that never fit.

## Speed numbers, and what they are not comparable to

| measurement | value | measured how |
|---|---|---|
| scipy baseline, cold x-update | 49.4 s | `bench_baseline.py`, 3 repeats from the *same* cold state (A100, job 62585927) |
| scipy baseline, "per ADMM iteration" | 30.2 s | a 3-iteration `fit` in the same job — includes compile, and iterations 2–3 warmstart, so it is not 3 × the row above |
| optimistix, per ADMM iteration | 7.77 s | `bench_chunk`, 10 iterations at chunk 39 from a v6 warmstart (job 62584713) |
| optimistix, per ADMM iteration | 13.96 s | 35 iterations to convergence at the λ=34.97 knot |
| L-BFGS-B `maxiter=5 ls=5` | 4.21 s | 16 iterations at the same knot (job 62587874) |

The two optimistix rows differ by 1.8× because one runs 10 iterations under the ramping inner budget
(`inner_maxiter_init · 2^iter`, cheap early) and the other runs to convergence at the cap. Headline
ratios quoted against whichever row is convenient — "4× against baseline", "8.7× against
optimistix" — are not the same comparison. The defensible statement is the one measured end to end
at a single knot from a single warmstart: **L-BFGS-B at `maxiter=5 ls=5` took 67.4 s and 16 ADMM
iterations where optimistix took 488.7 s and 35**, at an equal-or-better log-likelihood.

Also worth keeping from `results_bench` (the vlse solver battery, 48 functions): batched over 1024
starts, per-solve time summed over the battery was v100 4.43 ms, a100 4.62 ms, l40s 6.27 ms,
1080ti 12.78 ms, one CPU core 64.8 ms. float64 throughput is what ranks the cards — ask for
v100/a100/h100, never l40s.

## If these paths are ever rerun

The three findings above say the current stopping rule cannot certify a path. Before spending
another generation of GPU time: make the convergence test require that `z` actually moved, decide
what a converged fit means when the primal residual is pinned at 0.999× tolerance, and record a real
git revision in `meta.json`.
