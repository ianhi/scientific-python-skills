# Combined Skill File Test Report

**Date**: 2026-02-11
**Libraries tested**: zarr 3.1.5, xarray 2026.1.0, icechunk 1.1.18
**Skill files tested**:
- `/skills/zarr.md`
- `/skills/xarray.md`
- `/skills/icechunk.md`

## Test Script

`/tests/combined_test/test_combined_skill.py`

All 10 steps passed on the first run with zero code modifications needed.

---

## Patterns That Worked Correctly in Combination

### 1. Icechunk repository creation + xarray write via `to_icechunk`

The icechunk skill documents `icechunk.in_memory_storage()` + `Repository.create()` + `to_icechunk(ds, session, mode="w")` and this worked exactly as described. The xarray dataset (created per xarray skill patterns) wrote cleanly through the icechunk session.

### 2. Branching, tagging, and time-travel reads

The icechunk skill documents:
- `repo.create_tag("v1.0", snapshot_id=snapshot_v1)` -- worked
- `repo.lookup_branch("main")` + `repo.create_branch("dev", snapshot_id=...)` -- worked
- `repo.readonly_session(tag="v1.0")` for time-travel reads -- worked
- `repo.readonly_session(branch="dev")` for branch reads -- worked

Reading with `xr.open_zarr(session.store, consolidated=False)` correctly returns data at the requested point in history.

### 3. Append workflow with `to_icechunk(ds_new, session, append_dim="time")`

The icechunk skill documents appending with `append_dim="time"`. This worked correctly on the dev branch: the original 100 timesteps + 50 appended = 150 total, verified by reading back.

### 4. Session lifecycle: new session after commit

The icechunk skill warns that sessions become read-only after `commit()` and you must create a new writable session. This pattern was followed throughout and worked correctly.

### 5. `xr.date_range` with `unit="s"`

The xarray skill documents `xr.date_range("2000-01-01", periods=365, freq="D", unit="s")` for controlling time resolution. This worked and produced `datetime64[s]` coordinates, which are necessary to avoid the nanosecond overflow issues the skill warns about.

### 6. `apply_ufunc` with `dask='parallelized'`

The xarray skill documents:
- Core dims are moved to the **last** axis (-1) -- correct, the function operated on `axis=-1`
- MUST specify `output_dtypes` with `dask='parallelized'` -- correct, omitting it would fail
- `input_core_dims` and `output_core_dims` pattern -- worked as documented

### 7. Writing multiple groups with `to_zarr` and `mode="a"`

The icechunk skill documents: "Subsequent groups need `mode='a'`" when using `to_zarr`. Writing to `group="processed/anomalies"` with `mode="a"`, `zarr_format=3`, `consolidated=False` worked correctly.

### 8. `consolidated=False` mandatory for icechunk reads

The icechunk skill warns that `consolidated=False` is mandatory. All `xr.open_zarr()` calls used `consolidated=False` and worked. This is a critical cross-library pattern.

### 9. Zarr v3 format verification

The zarr skill documents `zarr.open_group()` with `mode='r'` and accessing members via `group['path']`. Checking `arr.metadata.zarr_format` confirmed all arrays were stored in v3 format. The zarr skill's group member iteration via `list(grp.members())` worked correctly.

### 10. Icechunk ancestry browsing

The icechunk skill documents `repo.ancestry(branch="main")` with `ancestor.id` and `ancestor.message`. This worked and correctly showed the full commit chain.

---

## Patterns That Were Wrong, Misleading, or Required Adjustment

### No errors encountered

All patterns from the three skill files worked correctly on the first attempt. The documentation was accurate and the code examples were directly usable without modification.

---

## What Was Missing: Integration Patterns Not Covered

### 1. `to_icechunk` with groups

The icechunk skill documents `to_icechunk(ds, session, mode="w")` but does NOT show how to write to a specific group (e.g., `group="processed/anomalies"`). For writing to a sub-group, I had to fall back to `ds.to_zarr(session.store, group="...", zarr_format=3, consolidated=False, mode="a")` using the icechunk gotcha about multiple groups. It would be helpful if the icechunk skill documented whether `to_icechunk` supports a `group=` parameter, or if `to_zarr` is the correct approach for sub-group writes.

### 2. Reading from a specific group with icechunk

The icechunk skill shows `xr.open_zarr(session.store, consolidated=False)` for reading, but does not document how to pass `group="processed/anomalies"` when reading a specific group within an icechunk repo. This works by passing `group=` to `xr.open_zarr()`, but it is not documented in any of the three skill files for the icechunk context.

### 3. Chunking data read from icechunk for dask processing

The xarray skill documents `ds.chunk({"time": -1, "lat": 15})` for making data dask-backed, and the icechunk skill documents `to_icechunk` handling dask data. However, none of the skill files explain the workflow of: read from icechunk (which returns dask-backed data by default via `open_zarr`) -> process with dask -> write results back. The interaction between icechunk's dask-backed reads and `to_icechunk`'s dask write handling is undocumented.

### 4. `to_zarr` vs `to_icechunk` decision guide

The icechunk skill says `to_zarr` "works fine for non-distributed (in-memory) data" and `to_icechunk` is needed for dask/distributed writes. But there is no clear guidance on when to prefer one over the other in a mixed workflow (some data is numpy-backed, some is dask-backed). A decision matrix would be helpful:
- Non-dask data, root group: either works, `to_icechunk` is simpler
- Non-dask data, sub-group: `to_zarr` with `group=` (no documented `to_icechunk` equivalent)
- Dask data, root group: must use `to_icechunk`
- Dask data, sub-group: unclear

### 5. Encoding patterns across the three libraries

The xarray skill documents zarr v3 encoding (`compressors` plural key, `zarr.codecs`). The zarr skill documents codec pipelines. The icechunk skill does not mention encoding at all. It is unclear whether encoding options work the same way when writing through icechunk (either via `to_icechunk` or `to_zarr` with `session.store`). Does icechunk's transactional layer interact with encoding in any way?

### 6. `drop_encoding()` when reading from one icechunk snapshot and writing to another

The xarray skill warns about stale encoding causing errors when rechunking and re-writing zarr data. It is undocumented whether this applies to icechunk workflows (e.g., read from tag "v1.0", process, write to a new branch). In our test, reading and writing with default encoding worked, but the interaction is not documented.

### 7. DataTree + icechunk

The xarray skill documents `DataTree` and `xr.open_datatree("store.zarr", engine="zarr")`. The icechunk skill does not mention DataTree at all. Can you use `xr.open_datatree()` with an icechunk store? Can you write a DataTree to icechunk? This is a significant gap for hierarchical data management.

---

## Were There Cases Where One Skill File Contradicted Another?

### No direct contradictions found

The three skill files were internally consistent. The main area of potential confusion (not contradiction) is:

1. **xarray skill says `ds.to_zarr("store.zarr", consolidated=True)` is "default"**, while the **icechunk skill says `consolidated=False` is mandatory**. These are not contradictory (icechunk overrides the general zarr advice), but a user following the xarray skill's zarr patterns without reading the icechunk skill would hit errors. The icechunk skill does document this clearly as a gotcha.

2. **xarray skill documents `ds.to_zarr()` as the primary write method for zarr**, while the **icechunk skill says to use `to_icechunk()` instead**. Again, not contradictory (icechunk is a specialized store), but the relationship could be clearer.

---

## Suggestions for a Future "Integration" Skill File

A cross-library integration skill file covering zarr + xarray + icechunk patterns should include:

1. **Complete write workflows**: Decision tree for `to_icechunk` vs `to_zarr` with `session.store`, covering root groups, sub-groups, dask vs non-dask data, and encoding options.

2. **Complete read workflows**: How to read from branches, tags, and snapshots, including sub-groups and DataTree hierarchies.

3. **Round-trip encoding patterns**: Document `drop_encoding()` guidance specific to icechunk (when is it needed? when is it safe to skip?).

4. **Multi-group repository patterns**: Show the full lifecycle of a repository with multiple groups -- initial creation, reading individual groups, appending to specific groups, and verifying structure.

5. **DataTree + icechunk**: Document whether DataTree I/O works with icechunk stores, and if so, what the patterns look like.

6. **Processing pipeline patterns**: Read from icechunk -> chunk with dask -> process with `apply_ufunc` or `map_blocks` -> write results back to icechunk (same or different group/branch). This is a very common scientific workflow.

7. **Zarr v3 format enforcement**: Clarify that icechunk requires zarr v3 format and that `zarr_format=3` should always be passed to `to_zarr` (or is it automatic?). In our test, `to_icechunk` automatically used v3, but `to_zarr` requires the explicit parameter.

8. **Conflict resolution in data pipelines**: The icechunk skill documents `ConflictError` and `rebase`, but there is no guidance on how this interacts with xarray/dask write patterns (e.g., what happens if two dask jobs try to append to the same branch concurrently?).

9. **Performance tuning across the stack**: Chunk size recommendations that account for zarr v3 codec defaults, xarray's chunk alignment requirements, and icechunk's transaction overhead.

10. **Version info compatibility matrix**: Which versions of zarr, xarray, and icechunk are tested together and known to work.
