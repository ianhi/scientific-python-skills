# Project Planning

## Target Libraries

### Phase 1 (Complete)
- zarr (zarr-developers/zarr-python) - v3.1.5
- xarray (pydata/xarray) - v2026.1.0
- icechunk (earth-mover/icechunk) - v1.1.18

### Phase 2 (Complete)
- numpy (numpy/numpy) - v2.4.2
- pandas (pandas-dev/pandas) - v3.0.0
- matplotlib (matplotlib/matplotlib) - v3.10.x

### Future
- scipy, polars
- Integration skill (zarr + xarray + icechunk together)

## Process per Library

### Research
1. Clone repo: `git clone --depth 1 <url> repos/<name>`
2. Read docs + source via Explore agents (parallel per lib)
3. Fetch issues/PRs via `scripts/fetch_issues.sh <owner/repo> research/<lib>`
4. Cluster by topic via `uv run python scripts/summarize_topics.py research/<lib>`

### Write
5. Synthesize into single skill file using `skills/TEMPLATE.md`

### Review
6. Launch fresh-eyes reviewer agents that read ONLY the skill file + source code
7. Apply critical fixes from review

### Test
8. Launch test agents that read ONLY the skill file, write a script, and execute it
9. Apply fixes found by testing

## Agent Coordination Notes
- Explore agents can't write files - extract their output yourself
- Background agents can't get interactive Bash approval - pre-build scripts in `scripts/`
- 3 parallel agents per phase (one per library) is the sweet spot

## Library-Specific Notes
- **zarr v3**: `codecs=` only on ShardingCodec, NOT create_array(). Use serializer=/compressors=/filters=. `from_array()` is store-first, data keyword-only.
- **xarray**: `ds.dims` deprecated for dict access → use `ds.sizes`. `use_cftime` deprecated → use `CFDatetimeCoder`.
- **icechunk**: `save_config()` takes no args. `flush()` requires message. Use `open_or_create()`.
- **numpy**: NEP 50 type promotion silently causes overflow/precision loss. `copy=False` raises ValueError if copy needed.
- **pandas 3.0**: CoW always on - chained assignment silently fails. `axis=None` reduces all axes. Offset aliases changed (`ME` not `M`).
- **matplotlib**: Always OO + `layout='constrained'`. `legend(loc='best')` is slow. Colorbar needs explicit `ax=`.
