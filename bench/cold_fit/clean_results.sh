#!/bin/bash
# Drop torn rows left by array tasks racing on results.csv, holding their lock.
set -euo pipefail

RESULTS=${RESULTS:-bench/cold_fit/results.csv}

flock "$RESULTS" uv run python -c "
import pandas as pd, sys
path = sys.argv[1]
rows = pd.read_csv(path)
clean = rows[rows.chip.notna() & rows.n.notna()]
clean.to_csv(path, index=False)
print(f'{path}: kept {len(clean)}, dropped {len(rows) - len(clean)}')
" "$RESULTS"
