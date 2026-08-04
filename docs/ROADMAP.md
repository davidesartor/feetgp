# Roadmap: make the GP fit fast and reliable on the real problem

Written 2026-08-03, against `b7c4dc6` + working tree. Everything below is anchored to
measurements in `logs/bench/`, not estimates.

Real problem: `n=1350 d=78 o=78 group_size=6` (13 marker groups), 10 ablation configs,
~30 λ per config.

## Where it started

Measured, A100, `chunk_size=32`, λ=0, `tol=1e-3` (`logs/bench/fit_a100_62567726.out`):

| ADMM iters | wall | marginal/iter | loglik | converged |
|---|---|---|---|---|
| 5 | 129.3s | 25.9s | 496142.124 | no |
| 10 | 665.4s | 107.2s | 496281.113 | no |
| 20 | 1579.5s | 91.4s | 496386.850 | no |

~95-100s per ADMM iteration in steady state, not converging, objective still climbing
at iteration 20. At the old `max_iterations=1000` that is ~27h for a *single* λ.

Two independent defects were behind it. Both are fixed.

## Defect 1 — the inner tolerance never fired (speed)

`optx.LBFGS` terminates on a **step** criterion, `norm(y_new - y_old) < atol + rtol *
norm(y)` under `max_norm` — not a gradient one. It was called with
`rtol = atol = EPS = 1.49e-8`, i.e. a 1.49e-8 step on a 79-dimensional vector, which
essentially never fires. The documented "inexact early, exact later" ramp had degenerated
into "burn the full cap, every iteration, for all 78 outputs".

Measured (`logs/bench/inner_a100_62568370.out`, cold x-update, chunk 32):

| inner rtol | wall | steps mean/med/max | hit cap | loss sum |
|---|---|---|---|---|
| 1.49e-08 | 175.7s | 872 / 1000 / 1000 | 47/78 | -498014.3982 |
| 1.00e-06 | 169.2s | 863 / 1000 / 1000 | 45/78 | -498014.3982 |
| 1.00e-04 | 126.5s | 429 / 404 / 805 | 0/78 | -498012.2522 |
| 1.00e-03 | 57.1s | 168 / 163 / 377 | 0/78 | -497905.6158 |

1e-4 reaches the same augmented-Lagrangian value as EPS to 4e-6 relative, in half the
steps, with no output hitting the cap.

**Landed:** `inner_rtol`/`inner_atol` (default 1e-4) exposed on
`GroupLassoGaussianProcess.fit` and as `--inner_tol` on `run.py`; `inner_maxiter`
dropped to 50 as the operating point with the outer budget raised to compensate. Total
cost is `ADMM_iters × outputs × inner_steps` and the device is compute-bound, so only
total step count matters — a cheap inexact x-update run many times beats an exact one
run rarely (Eckstein & Bertsekas: convergence holds while subproblem errors are
summable).

Result at 10 ADMM iterations: **~8 s/iter against 66.5 s/iter**, and ahead on wall-clock
(89s → loglik 496187 vs the old 129.3s → 496142).

**Not done:** replacing the `inner_maxiter_init * 2**iter` cap ramp with a *tolerance*
ramp tied to the current residual. Still the better construction — a cap silently
returns a non-solution, a tolerance returns a solution of known quality — and it would
also remove the 7 recompiles per fit that the static `maxiter` rungs force. Deferred
because the cap at 50 is no longer the binding cost.

## Defect 1b — `lax.map` pays the max over each chunk

A `lax.while_loop` under `vmap` cannot exit per element, so a chunk costs the max over
its members. Measured tax at `chunk_size=32`: **1.15×** (78000 output-steps paid vs
67990 needed) — small enough that chunk sorting is not worth it.

Chunk sweep at the new inner budget, 10 ADMM iterations:

| chunk | wall | s/iter | loglik |
|---|---|---|---|
| 26 | 89.0s | 8.90 | 496187.265 |
| 32 | 93.7s | 9.37 | 496191.265 |
| 39 | 81.0s | 8.10 | 496195.883 |
| 78 | 76.3s | 7.63 | 496197.115 |

The device is compute-bound — one 1350×1350 float64 Cholesky saturates an A100 — so
`chunk_size` is a minor lever. **`39` is the setting**: an even divisor of 78, near-best,
and not a full vmap. `run.py` clamps it to the output count so `--target forces` (3
outputs) does not pad.

## Defect 2 — the nugget clip pinned the primal residual (reliability)

Speed alone did not make the fit converge: it plateaued at primal ≈ 3.5 and stayed there
(`logs/bench/fit2_a100_62568612.out`, 200 iterations).

The nugget was `g_min + exp(w)` with `w` **clipped to `log(g_range)` in
`z_and_u_update`**. The augmented-Lagrangian term covers only `theta`
(`target_theta = (z - u)[:-1]`), so `w` is not really part of the consensus — but the
clip made `x ≠ z` on that column anyway. Any output whose noise exceeded `g_range[1]`
therefore held the primal residual permanently above `tol`, **at every λ**. No fit could
ever converge, and `plots.py` was drawing paths made entirely of unconverged points.

On real data **32% of fitted nuggets sat exactly on the old ceiling of 1.0** and 66% on
the floor — only ~1% were interior. Not a toy artifact.

**Landed:** `g = g_min + (g_max - g_min) * sigmoid(w)` (`nugget_from_w` /
`w_from_nugget`), `w` bounds set to `±inf`. `g` is inside `g_range` by construction, the
last column satisfies `x = z` identically, and it contributes zero to both residuals.

Toy fit, 200-iteration budget, before → after:

| ceiling | before | after |
|---|---|---|
| 1 | no, primal 7.0e-1 | **yes, 103 iters** |
| 10 | no, primal 1.1e+1 | **yes, 52 iters** |
| 100 | no, primal 2.98e+1 | **yes, 52 iters** |
| 1000 | no, primal 2.15e+1 | **yes, 52 iters** |

Raising the ceiling used to make things *worse* — that was purely the clip artifact.
With saturation, `g` settles at a genuine interior optimum (1.894, identical at ceilings
10/100/1000). A binding ceiling still costs iterations because the output sits on a flat
ridge, so **`g_range` default is now `(1e-4, 100.0)`**. The floor stays: without it the
marginal likelihood runs away to interpolation as `g -> 0`.

## Defect 2b — rho adaptation was unbounded

The ×2 / ÷0.5 rule has no fixed point when the primal residual has an irreducible floor:
rho was observed reaching **1.89e+22**, at which point the augmented term swamps the
likelihood in float64 and the residual can never fall again — positive feedback.

**Landed:** clamp to `[RHO_MIN, RHO_MAX] = [1e-6, 1e6]`, with `u` rescaled by the
*applied* ratio `rho_old / rho_new` rather than by `scale`, so the clamp stays
consistent. Both `glassogp.py` and `linear.py`.

## The rho freeze

`adapt_rho and iter < max_iterations // 2` tied "when rho stops moving" to "how long we
are allowed to run", so dropping the iteration budget silently moved the freeze point —
a behaviour change riding along on what looked like a budget change.

**Landed:** wired to `adapt_rho_iters`, the argument that already existed for it and was
never used. `None` defaults to `max_iterations // 2`, preserving the existing behaviour
exactly. The half-way freeze itself is deliberate (it avoids oscillation) and was kept.

## Defect 3 — the even objective made every fixed point measure-zero (reliability)

This was the one that mattered. After defects 1, 2 and 2b were fixed, *no* λ converged
— including λ=0, where the group-lasso prox is exactly the identity
(`max(0, 1 - 0/(rho*norm)) = 1`), so `z = x + u`, `u_new = 0`, and the primal residual
is algebraically forced to zero. It was 6.16 and **growing**.

Two causes, one change:

- **λ=0 was the box clip.** `z_and_u_update` clipped `z` to `self.bounds`, `x_update`
  did not project, so an `x` that wanted to sit outside the box (`max|θ| = 2.878`
  against `theta_max = 2.6645`) held `x != z` forever. Measured share of the residual:
  `r = 2.857e-01` of which `r_box = 2.850e-01` — all of it.
- **λ>0 was a limit cycle.** The kernel sees `theta` only through `theta**2`, so
  `-loglik` is exactly even and `∇f(0) = 0`. For a dead group the x-update target is
  `z - u = -u`, and stationarity at `x = 0` reads `rho*u = -∇f(0) = 0` — u exactly zero,
  measure zero, never reached. ADMM oscillates instead: the minimiser lands near `-u`,
  the prox kills it, `u += x - z` collapses `u` back. Period 2. Measured at λ=116.6,
  sub=200: group 11 at 2.90 → **15.21** → 2.96 over 20 iterations, active count
  flickering 11/12/13, `r` oscillating 9.8 / 15.2 / 9.7 / 21.5 / 12.0 for 150 iterations
  with rho pinned at 4.0. At λ=256.2 the active count sat at 0 for 80 iterations while
  x-norms bounced 3–15.

The box had been dropped earlier on the argument that the objective is even so the sign
carries no information. That argument is correct and the conclusion was wrong: **the box
is free statistically and load-bearing dynamically.** Under `theta >= 0` the x-update
clips a dead coordinate to *exactly* zero and stationarity relaxes to the inequality
`u >= 0`, which a whole set satisfies.

**Landed:** `theta_bounds` back to `[0, theta_max]` in `fit`, and `x_update` projects
its L-BFGS output onto `self.bounds` before returning, so `x` and `z` live in the same
set. The `w` column's bounds are `±inf`, so the saturating nugget is untouched and stays
out of the consensus residual. `STATE_FORMAT` bumped to 4 — a format-3 state's negative
thetas would silently clip to zero if warmstarted.

**Measured after, same settings as before:**

| | before | after |
|---|---|---|
| λ=0, sub=20, 50 iters | `r` 3.17 → 4.46 → 6.16, rising | `r = 0.00e+00`, `n_over = 0`, 13/13 active |
| λ=116.6, sub=200, 150 iters | `r` oscillating 9.8–21.5 | `r` monotone 23.2 → 11.3 → 6.8 → 5.4 → 3.3 → 2.7 |
| x group norms | 3–15, bouncing | O(1), decaying |
| sign flips per group | many | 0 |

## Defect 4 — the absolute tolerance was unscaled

With λ=0 finally at `primal = 0` exactly, it *still* did not converge: `u` is exactly
zero there, so the dual criterion loses its relative term and reduces to the bare
absolute floor. That floor was `EPS = 1.49e-8` on the norm of a 6084-element vector,
below the inexact x-update's own noise — the dual residual measured flat at 5.6e-7
across iterations 50→100 at `inner_tol=1e-4`. It would have spun to the cap with a
converged solution in hand.

**Landed:** `eps_abs = sqrt(x.size) * EPS` (Boyd §3.3.1) in `check_residuals`, in both
`glassogp.py` and `linear.py`. `sqrt(6084) * EPS = 1.16e-6` clears the noise floor.

## Defect 5 — over-relaxation is incompatible with the theta box (reliability)

With defects 3 and 4 landed, the *bench* reported `primal = 0.00e+00` at λ=0 but `run.py`
still stalled there: `r` sat at 1.02 / 1.06 / 1.72 / 1.55 across iterations 51/101/151/201
with rho pinned, heading for the 400-iteration cap. The whole λ grid hangs off the λ=0 fit,
so nothing downstream was trustworthy.

The difference was `alpha`. `GroupLassoGaussianProcess.fit` defaulted to Boyd's `alpha=1.6`
and `run.py` never passed one, while every measurement that validated the box was taken
through `bench_signs.py` / `bench_fit.py`, both of which default to `alpha=1.0`. So the box
was being validated under one setting and shipped under another.

Two independent failures, both from `x_hat = alpha*x + (1-alpha)*z`:

1. **It leaves the box.** When `x` and `z` straddle, `1.6*x - 0.6*z` goes negative, and `u`
   accumulates the overshoot — reopening the second well that `theta >= 0` exists to close.
2. **At λ=0 it turns the primal residual into a noise meter.** The prox is the identity, so
   `z = x_hat + u` and `u = 0`, giving `r = ||x - x_hat|| = (alpha-1)||x - z_prev||`. That is
   the inexact x-update's own step-to-step jitter amplified by 0.6, not a consensus error,
   and it has a floor no iteration count can cross.

**Landed:** `alpha: float = 1.0` in `glassogp.py`, documented at `z_and_u_update` so it does
not get "optimized" back to Boyd's band. λ=0 now reports `r = 0.000e+00` at iteration 1 and
**converges in 25 iterations** (was: 400+, stalled). `linear.py` keeps `alpha=1.6` — no box,
so only the cosmetic λ=0 effect applies, and its paths are measured good.

## Defect 6 — λ=0 walked rho to RHO_MIN and poisoned the next warmstart

Immediately visible once λ=0 converged: it finished at `rho = 2e-6`, i.e. `RHO_MIN`, and
handed that to the next λ as a warmstart. At that rho the prox threshold `l1 / (rho * norm)`
kills every group on contact, so the first penalized fit starts from a wiped support.

Cause: the ×2/÷0.5 rule compares the two residuals, which is only evidence when both are
informative. At λ=0 the primal residual is identically zero *by construction*, so
`dual > 10 * primal` is a tautology and fired on every one of the 25 iterations.

**First attempt, reverted.** Gating both directions on the *other* residual clearing
`eps_abs` looked principled and was wrong. It pins rho at 1.0 through λ=0, and the dual test
there has no relative term (`u = 0`), so it then demands `||z - z_prev|| < 1.16e-6` from an
`inner_maxiter=20` inexact solve. That never fires. Measured: λ=0 still at iteration 1 after
6 minutes with no convergence in sight, against 25 iterations before the gate — it would
have spent the entire 400-iteration budget. The tell was in the number that had looked like
success: `s = 7.041e-07` at `rho = 2e-6` means `||Δz|| = 0.35`, so the dual residual was
passing *because* rho was small, not because the parameters had stopped moving.

The decay is correct locally. At λ=0 the augmented term is vacuous — `u ≡ 0`, `z ≡ x` — and
`rho → 0` is precisely what turns the x-update into the exact MLE that λ=0 wants.

**Landed:** the reset moved to the handoff. `run.py` does
`unpenalized_warmstart = states[0.0]._replace(rho=jnp.array(1.0))`, so λ=0 keeps its decay
and the next λ starts from a usable rho. `check_residuals` is back to the plain Boyd rule in
both `glassogp.py` and `linear.py`, which also means linear's measured paths are unchanged.
The refinement pass never warmstarts from λ=0 (its gap filter requires `lo > 0`).

## Defect 7 — the λ sweep walked downward, and death is absorbing

The GP paths came back flat dead. Whole configs reported 0 active groups at *every*
penalized λ (`markers/right_only`, λ = 98 / 127 / 166 / 215 / 280, all 0/13), and where they
were not flat they were non-monotone: `right_only --relative` gave 2 / 1 / 3 active as λ
*increased* through 23.0 / 29.9 / 38.9. Neither is a path.

Cause, and it is the same evenness that produced defect 3. The kernel sees `theta` only
through `theta**2`, so `∇f(0) = 0` exactly. Two consequences:

1. **`theta = 0` is a local minimum at every λ > 0.** The group-lasso stationarity condition
   for a dead group is `||∇_g f|| ≤ λ`, and here the left side is 0. So the all-dead point
   always satisfies it, the problem is nonconvex, and which local solution ADMM lands in is
   decided entirely by the warmstart. The path is a continuation path, not a global one.
2. **Death is absorbing.** Once a group has `x = z = 0`, `u += x - z` stops moving it, the
   x-update target `z - u = -u` is nonpositive, the box clips it back to 0, and `∇f(0) = 0`
   supplies no restoring force. Nothing in the problem can resurrect it.

`run.py` fitted `lambda_pivot` first, then walked *down* from it chaining each fit onto its
sparser neighbour. The pivot — where the penalty balances the log-likelihood gain over the
null model — landed 20–30× above the useful range (323 against λ=0 group norms of 9.6–15.6),
so it was usually already all-dead, and every smaller λ inherited that and kept it. The whole
useful path sat in the downward walk, i.e. in the one direction the model cannot support.

**Landed:** the sweep is a single upward chain. λ=0, then the grid bottom calibrated by
dividing `LAMBDA_START_FRACTION * max(group_norms)` (0.02) by `lambda_step` until some λ
matches the λ=0 support — those probes warmstart from λ=0, never from the probe above, since
a dense warmstart has no dead group to inherit — then `*= lambda_step` upward, each fit
warmstarted from the λ below it, until nothing survives. The break tests
`n_active >= path[0.0]`, not `== n_groups`: `both_ungrouped` comes out of λ=0 with groups
already at exactly zero norm. `lambda_pivot` and the null-model likelihood are gone. The
refinement pass already warmstarted from `states[lo]`, the denser end, and is unchanged.

Verified on the linear model at sub=100 (`--lambda_step 1.5`, 29 λ): `maxdrop = 1` across the
whole grid, gap-free 13 → 0, R² 0.9995 → 0.9964 → 0.9956 → … → 0.3607 → −0.0044.

## Defect 8 — rho drifted under the level at which the x-update target leaves the box

With defect 7 fixed, the v5 GP array walked λ upward cleanly and cheaply for a while — task 0
(`markers/both`) converged in 18 iterations at λ=0, then 11 / 43 / 13 / 9 / 6 / 6 / 10 / 7 / 6 /
16 / 5 / 11 / 11 / 16 / 12 / 15 / 53 up to λ=26.9, ~1 minute per λ — and then hit a wall. λ=34.97
ran the full 400-iteration cap (~51 min) with `r` flat at 3.89 / 3.45 / 4.34 / 3.26 / 3.22, rho
pinned at 1.0, and 13/13 still active. `left_only` did the same at λ=22.08 and 28.71.

**Cause.** At an ADMM fixed point `u_g` points along `z_g` with norm `l1 / rho`. The x-update
target is therefore `z_g (1 - l1 / (rho ||z_g||))`, which flips sign on every live group as soon
as rho drops under `l1 / ||z_g||`. The GP objective is exactly even in theta, so a negative target
makes the mirror well the deeper one; individual outputs' L-BFGS falls into it and the projection
onto the box returns ~0 for that row. Measured directly in the cached states: at λ=34.97 the
primal residual was `r_theta = 2.693` with `r_w = 0.0000` exactly, carried by 5 of 78 output rows
(1.56 / 1.30 / 0.76 / 0.64 / 0.64) against a bulk of 0.05. `l1 / rho` was 34.97 against a smallest
live group norm of 6.85 — 5× into the sign-flipped regime.

**Bench** (`bench_knot.py`, warmstarted from the cached λ=26.9 exactly as the sweep does, 40
iterations each):

| setting | primal | dual | s/iter |
|---|---|---|---|
| `inner_maxiter=50`, rho=1 (baseline) | 4.6542 | 1.4353 | 6.66 |
| `inner_maxiter=200`, rho=1 | 4.6083 | 1.5468 | 15.70 |
| `inner_maxiter=50`, **rho=8** | **0.4086** | 0.4885 | 6.55 |

So the inner budget is *not* the lever — 4× of it buys nothing for 2.4× the wall time, which also
rules out the inexact-ADMM error floor. rho is the lever, 11× on the primal residual at equal
cost. And the floor has to hold *during* the fit: at rho=8 the residual balancer halved rho to 2
within 11 iterations and `r` climbed back 1.13 → 1.87.

**First fix, since reverted.** `GLASSOADMMState.rho_floor` returned
`RHO_TARGET_MARGIN * l1 / min(live group norms)` (margin 2.0), and `check_residuals` clamped to it
in place of `RHO_MIN`. It cleared both stalled knots — `markers/both` λ=34.97 in 45 iterations,
`left_only` λ=22.08 in 49, against 400-iteration stalls — and it was wrong, see defect 9.

A second, distinct knot is still open: `both_ungrouped` at λ=0.2166 fails on the **dual only**
(primal 0.0518 against a tolerance of 0.2257, dual 0.0083 against 0.0011), with `l1 / rho = 3.47`
comfortably below the smallest live group norm of 5.26. That one is x-update jitter keeping
`||Δz||` off the floor, not a sign flip.

**Redone in v7**, without the floor: it is still there, and it is a *band*, not a point — the
seven λ from 0.2166 to 2.467 all burn the full 400 iterations at 26/26 active, then λ=3.70
converges in **8**:

| λ | primal | dual | rho | iterations |
|---|---|---|---|---|
| 0 | 0.000e+00 | 2.30e-07 | 1e-6 | 21 |
| 0.2166 | 3.56e-02 | 5.32e-03 | 6.25e-2 | 400 |
| 0.7310 | 9.27e-02 | 1.81e-02 | 1.25e-1 | 400 |
| 2.4671 | 1.09e-01 | 3.12e-02 | 2.50e-1 | 400 |
| 3.7006 | 1.26e-01 | 1.52e-02 | 2.50e-1 | **8** |
| 5.5510 | 7.12e-02 | 2.65e-02 | 5.00e-1 | 12 |

The λ=3.70 fit passes with a **larger** primal residual than the λ=0.2166 fit that fails, so the
binding test is the dual one and the mechanism is the tolerance, not the residual: `eps_dual`'s
relative term is `tol * rho * ||u||`, and just above λ=0 the prox barely shrinks anything, so `u`
stays near zero and the criterion collapses onto the absolute floor `sqrt(p) * EPS = 1.16e-6` —
four orders under the inexact x-update's own noise. It is the λ=0 degeneracy leaking into a
neighbourhood: at λ=0 exactly it is harmless (the prox is the identity, `u ≡ 0`, rho decays to
`RHO_MIN`, and `rho * ||Δz||` clears the floor because rho is tiny), but at λ=0.2166 rho settles at
6.25e-2 and `||Δz||` cannot.

Cost, not correctness: R² is 0.99979 across the whole band, identical to λ=0, and the path resumes
converging above it. The bill is 7 × 400 = 2800 wasted ADMM iterations on the critical path of one
config. Do **not** pay for it by loosening `tol` globally — the fix, if one is wanted, is that the
dual criterion needs a floor tied to the inner solver's accuracy rather than to machine epsilon.

## Defect 9 — the rho floor made death unreachable, so the whole path stayed at full support

The v6 GP array converged everywhere and selected nothing. `markers/both` ran 30 λ up to 815 at
**13/13 active** with R² flat at 0.9998; the smallest group norm merely shrank 9.64 → 1.56. Every
other config did the same.

**Cause.** A group dies when its prox threshold `l1 / (rho ||z_g||)` reaches 1. The floor pins
`l1 / rho <= min_norm / 2`, i.e. it holds *every* group at a factor of ≥2 from its own death
threshold, at every λ, by construction — raising λ just drags rho up with it. Measured over the
live path:

| λ | rho | l1/rho | max‖g‖ | prox threshold | active |
|---|---|---|---|---|---|
| 26.903 | 1 | 26.9 | 14.901 | 0.643 | 13 |
| 34.974 | 8.51 | 4.11 | 15.185 | 0.207 | 13 |
| 99.890 | 36 | 2.77 | 13.643 | 0.165 | 13 |
| 482.151 | 494 | 0.975 | 6.848 | 0.125 | 13 |
| 814.835 | 1.2e3 | 0.678 | 4.379 | 0.134 | 13 |

Defect 8 conflated two things: a genuine pathology (rho so small that *every* live group's target
goes negative at once, an ADMM tuning failure) and the legitimate death event (one marginal
group's target going negative, which is exactly what killing it looks like). Suppressing the
second to avoid the first suppresses selection itself.

**Fix.** Kill the mirror well at its source instead of steering rho around it. `admm_x_update_loss`
feeds `jax.nn.relu(theta)` to the kernel, so the negative orthant is a likelihood *plateau* and a
negative target has nothing to fall into; the augmented term keeps the **raw** theta, so a
coordinate on the wrong side of zero is still pulled back. Evenness forces `dloglik/dtheta_d = 0`
at the boundary, so `loglik(relu(theta))` is C1 there and L-BFGS is untroubled — no smoothness is
spent. `rho_floor` and `RHO_TARGET_MARGIN` are deleted, `check_residuals` clamps to
`[RHO_MIN, RHO_MAX]` again, and `fit` no longer raises the warmstart's rho.
`test_x_update_objective_is_even_in_theta` is replaced by
`test_x_update_objective_is_flat_below_zero`, which asserts the plateau and the zero boundary
gradient. `STATE_FORMAT` stays 4 — the parametrization did not move.

## Reliability gates

1. **Poisoned pickles.** All 725 GP pickles under `results/` carried no `admm_state` at
   all, old `%013.6f` filenames, and λ values like `98758233950707.437500`. Archived to
   `results_pre_2026-08-03/` (moved, not deleted; gitignored; reversible).
   `STATE_FORMAT` in `run.py` now versions what the last column of `admm_state.x` means
   — 1 linear `g`, 2 `log(g - g_min)`, 3 the logit, 4 the logit with `theta >= 0` — and
   a mismatch is loaded as a result but **refused as a warmstart**. Bump it whenever the
   parametrization moves. Everything under `results/` predates defect 3 and must not be
   reused even as a result. `results_v4/` holds the post-defect-3 runs but its GP grid was
   built by the downward walk of defect 7, so its GP paths are unusable as results too;
   the current runs go to `results_v5/`. `results_v4/`'s *linear* paths are unaffected
   (that objective is not even, so death is not absorbing there) and remain the linear
   acceptance evidence below. `results_v5/`'s GP runs all stalled at the defect-8 knot, so
   the GP runs continue in `results_v6/`, seeded from v5's converged pickles with the four
   unconverged ones dropped.
2. **Convergence recorded per λ.** `fit` returns a fourth value, an `info` dict
   (`converged`, `iterations`, `primal_residual`, `dual_residual`), persisted in each
   pickle. `plots.py` reads it and crosses out non-converged λ with `x-thin` markers, so
   a kinked path can no longer be mistaken for a real one. Deliberately *not* fields on
   `GLASSOADMMState`, which stays lean.
3. **Regression tests.** `tests/` is at 10 passing.
   `test_inner_solver_stops_before_its_cap` asserts the inner tolerance actually fires —
   it did not at EPS, which is what made `inner_maxiter` the operating point.
   `test_fit_converges_within_its_iteration_budget` is what caught both defect 2 and 2b.
4. **CLAUDE.md.** Updated for the x-update, the inner budget, chunk sizing, the
   saturating nugget, the rho clamp, `adapt_rho_iters`, the 4-tuple return, and
   `STATE_FORMAT`. Still stale: the `history_length=40` justification ("221 steps at 40
   against 735 at 10") was measured against the old *boxed* problem, whose active set
   had collapsed to ~2 free dimensions. The setting is probably still right; the number
   backing it is not evidence for the current unconstrained solver. Re-measure with the
   same harness that sweeps the tolerance.

## Acceptance criterion — linear model, met

All **10** configs, at `--lambda_refine 100`: every one has `maxdrop = 1`, i.e. a gap-free
ladder from full support down to zero with exactly one group dying per step. `--lambda_refine
30` was not enough — `both_ungrouped` still had an interval where 7 groups died at once, and
left/right had 2. Whole 10-config array: 13s to 9m06s per config on CPU.

| config | λ | conv | act 26→0 or 13→0, R² |
|---|---|---|---|
| markers/both | 42 | 41 | 0.9997 … 0.6651, 0 |
| markers/both_ungrouped | 56 | 47 | 0.9997 … 0.5562, 0 |
| markers/both rel | 39 | 38 | 0.9999 … 0.5141, 0 |
| markers/both_ungrouped rel | 52 | 50 | 0.9999 … 0.4747, 0 |
| markers/left | 45 | 40 | 0.9989 … 0.6879, 0 |
| markers/right | 42 | 41 | 0.9999 … 0.3384, 0 |
| forces/both | 39 | 32 | -1.779 at λ=0, then 0.516 … 0.089, 0 |
| forces/both_ungrouped | 48 | 37 | -1.779 at λ=0, then 0.527 … 0.013, 0 |

The `forces` λ=0 row is not a solver defect: the unpenalized linear fit overfits the test
half badly (R² −1.78), and the first nonzero λ recovers it to +0.52. The penalty helps there.

Detail, `results_v4/model=linear/target=markers/feet=both/inclines=all_sub=20`, 42 λ,
41 converged, **3m45s** for the whole path on CPU:

| λ | active | r2_test | | λ | active | r2_test |
|---|---|---|---|---|---|---|
| 0 | 13 | 0.9997 | | 49.5 | 6 | 0.9838 |
| 0.119 | 13 | 0.9988 | | 83.6 | 5 | 0.9734 |
| 0.154 | 12 | 0.9987 | | 141 | 4 | 0.9570 |
| 0.200 | 11 | 0.9986 | | 404 | 3 | 0.9014 |
| 0.572 | 10 | 0.9982 | | 887 | 2 | 0.7997 |
| 1.63 | 9 | 0.9976 | | 1499 | 1 | 0.6651 |
| 2.12 | 8 | 0.9975 | | 3293 | 1 | 0.2187 |
| 6.07 | 7 | 0.9963 | | 4280 | 0 | -0.0000 |

Monotone, one group at a time, R² degrading smoothly. λ=0 is the one unconverged fit:
the prox is the identity there, so `u = 0` and the dual criterion has only the absolute
floor to work with — it runs to the cap with a solution in hand (r2 0.9997).

## Open

- **Per-λ cost and the outer budget.** λ=0 at sub=20, chunk 39, `inner_maxiter=20`
  measured **6.2 s/iter** and converged by iteration 50 (`primal = 0`, dual 5.6e-7),
  against 66.5 s/iter under the pre-defect-1 settings. `--maxiter` and the SLURM
  `--time` are set from the v4 sweep, not from hope.
- **Path shape, GP.** Job `62571786` (`run_gp_smoke.slurm`, sub=20, `--lambda_step 1.3`,
  `--lambda_refine 100`) is the first sweep with defects 5 and 6 fixed. Still running.
  At sub=200 the transition was a cliff — 13 active at λ=116.6, 0 at λ=197.1 — but
  n=135 there against d=78, so that is a degenerate problem, not evidence about sub=20.
  If the deaths really do clump at sub=20, the standard remedy is an adaptive group
  lasso (weight each group by `1/||theta_d||` from the λ=0 fit), which is a change to
  the estimator and so needs asking about first.
- **The real sweep.** `run_gp_v4.slurm` (array 0-9, `--lambda_refine 100`) is staged, and
  `watcher.sh` now points at it with `INIT_ARRAY[run_gp_v4.slurm]="0-9"`. Held until the
  smoke run confirms the path shape.
- **Old results are poisoned.** Everything under `results/` predates defect 3 and is a
  limit cycle, not a fit. `fit_or_load` would reuse those pickles as results, so the v4
  runs write to `results_v4/` instead.
- **Not started, not approved.** The whole P1 methodology track: the deterministic
  even/odd train/test split (consecutive motion-capture frames, so test is not
  independent of train), λ selection, debiasing, NLPD.
