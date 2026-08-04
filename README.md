# feetgp

Group-lasso variable selection over foot marker data from an incline-running motion-capture
dataset: which foot markers can be dropped while still predicting the remaining markers, or the
ground reaction forces? A group-lasso penalty on kernel lengthscales (GP) or on coefficients
(linear ablation) drives whole markers to zero as λ grows; a run sweeps λ and records the
regularization path.

```bash
uv run python -m feetgp.run --subsample 20 --target markers --feet both \
    --chunk_size 39 --maxiter 300 --inner_maxiter 50 --inner_tol 1e-4
uv run python -m feetgp.run --linear_model --subsample 20 --tol 1e-6   # linear ablation
uv run python -m feetgp.plots --results_dir results                    # plotly HTML per run
uv run python -m feetgp.summarize_runs results                         # one line per run dir
uv run pytest tests                                                    # synthetic data, no dataset needed
sbatch slurm/run_gp.slurm                                              # cluster: 10-config array job
```

```
src/feetgp/   admm.py glassogp.py linear.py inclinerunning.py   the library
              run.py plots.py summarize_runs.py                 entry points, python -m
bench/  slurm/  tests/  docs/  logs/
```

`CLAUDE.md` carries the invariants — read it before changing the ADMM loop, the λ sweep direction,
or the column layout. `docs/RESULTS.md` is what the deleted result generations showed, with
per-run numbers in `docs/results_summary.jsonl`.
