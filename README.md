# feetgp

Group-lasso variable selection over foot marker data from an incline-running motion-capture
dataset: which foot markers can be dropped while still predicting the remaining markers, or the
ground reaction forces? A group-lasso penalty on kernel lengthscales (GP) or on coefficients
(linear ablation) drives whole markers to zero as λ grows; a run sweeps λ and records the
regularization path.

```bash
uv run python -m feetgp.run --subsample 20 --target markers --feet both \
    --chunk_size 39 --maxiter 300
uv run python -m feetgp.run --linear_model --subsample 20 --tol 1e-6   # linear ablation
uv run python -m feetgp.plots --results_dir results                    # plotly HTML per run
uv run python -m feetgp.summarize_runs results                         # one line per run dir
uv run pytest tests                                                    # synthetic data, no dataset needed
```

```
src/feetgp/   glasso_admm.py gp.py linear.py inclinerunning.py   the library
              run.py plots.py summarize_runs.py                 entry points, python -m
tests/        pytest, synthetic data only
```

`CLEANUP-2026-08-08.md` carries the history: what was tried and failed, which measurements are
obsolete, and the defects diagnosed but never fixed. The invariants that used to sit in `CLAUDE.md`
