# xarray Skill File Test Report

**Date**: 2026-02-11
**Skill file**: `/Users/ian/Documents/dev/scientific-python-skills/skills/xarray.md`
**Test script**: `/Users/ian/Documents/dev/scientific-python-skills/tests/xarray_test/test_xarray_skill.py`
**xarray version**: 2026.1.0+
**zarr version**: 3.1.5+
**Python**: 3.12

---

## Results Summary

| # | Section | Result |
|---|---------|--------|
| 1 | Create Dataset with `xr.date_range(unit="s")` | PASS |
| 2 | `apply_ufunc` with dask (`dask='parallelized'`, `output_dtypes`, `input_core_dims`, `output_core_dims`) | PASS |
| 3 | GroupBy / Resample (`"ME"` frequency, `groupby("time.season").map(...)`, `groupby("time.month").mean()`) | PASS |
| 4 | Write to zarr v3 (`zarr_format=3`, `BloscCodec`, `compressors=(codec,)`) | PASS |
| 5 | `open_mfdataset` with `compat='override'`, `coords='minimal'` | PASS |
| 6 | Encoding gotchas (`drop_encoding()` before re-saving) | PASS |
| 7 | DataTree (`from_dict`, navigation, `subtree_with_keys`, `map_over_datasets`) | PASS |
| 8 | `CFDatetimeCoder(time_unit="s")` passed to `decode_times=` | PASS |

**All 8 sections passed.**

---

## What Worked Correctly

### 1. Dataset creation with `xr.date_range(unit="s")`
The skill file's pattern `xr.date_range("2000-01-01", periods=365, freq="D", unit="s")` worked exactly as documented. The `unit="s"` parameter produced `datetime64[s]` coordinates.

### 2. `apply_ufunc` with dask
The skill file's guidance was accurate and critical:
- **`dask='parallelized'` requires `output_dtypes`** -- confirmed; omitting it would fail.
- **Core dims are moved to last axis (-1)** -- confirmed; the standardize function using `axis=-1` worked correctly.
- **`input_core_dims` and `output_core_dims` syntax** -- the list-of-lists format `[["time"]]` is correct and worked.

### 3. GroupBy / Resample with modern frequency codes
- `resample(time="ME")` worked correctly (month-end). The skill file's note that `"M"` is deprecated in favor of `"ME"` is accurate.
- `groupby("time.season").map(lambda x: x - x.mean("time"))` for seasonal anomalies worked.
- `groupby("time.month").mean()` for monthly climatology worked.

### 4. Zarr v3 encoding
The skill file's pattern is correct:
```python
from zarr.codecs import BloscCodec
compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
ds.to_zarr(path, zarr_format=3, encoding={"var": {"compressors": (compressor,)}})
```
Key points confirmed:
- The key must be `"compressors"` (plural), not `"compressor"`.
- The value must be a tuple `(codec,)`, not a bare codec instance.
- `zarr_format=3` is the correct kwarg.

### 5. `open_mfdataset` with performance options
The skill file's recommended pattern worked:
```python
xr.open_mfdataset("data/*.nc", compat="override", coords="minimal", data_vars="minimal")
```

### 6. Encoding gotchas / `drop_encoding()`
`ds.drop_encoding()` correctly clears the encoding dict. The skill file's warning about encoding persisting through operations (and causing errors when re-writing to a different store) is accurate and important.

### 7. DataTree
All DataTree patterns from the skill file worked:
- `xr.DataTree.from_dict({"/": ds, "/group": ds1})` -- correct constructor.
- `dt["/surface"]` navigation -- works.
- `dt.children` -- returns dict of child nodes.
- `dt.subtree_with_keys` -- iterates over `(path, node)` pairs. Paths are relative strings (`.`, `surface`, `upper_air`), not absolute paths starting with `/`.
- `dt.map_over_datasets(lambda ds: ds.mean())` -- works.

### 8. CFDatetimeCoder
The skill file's pattern is correct:
```python
coder = xr.coders.CFDatetimeCoder(time_unit="s")
ds = xr.open_dataset("file.nc", decode_times=coder)
```
The result has `datetime64[s]` dtype as expected.

---

## What Was Wrong or Misleading

### No outright errors in the skill file code patterns

All code patterns tested from the skill file executed successfully. However, there are observations about things that could mislead:

### 1. `Dataset.dims` returns a warning-emitting object, not a plain dict
The skill file shows `ds.dims` as if it returns a standard mapping. In xarray 2026.1.0, accessing `ds.dims` emits:
```
FutureWarning: The return type of `Dataset.dims` will be changed to return a set
of dimension names in future, in order to be more consistent with `DataArray.dims`.
To access a mapping from dimension names to lengths, please use `Dataset.sizes`.
```
The skill file does not mention `Dataset.sizes` anywhere. Users following the skill file and writing `ds.dims["time"]` will get noisy warnings. The skill file should recommend `ds.sizes` instead.

### 2. Zarr v3 consolidated metadata warning
When writing zarr with `zarr_format=3`, a warning is emitted:
```
ZarrUserWarning: Consolidated metadata is currently not part in the Zarr format 3
specification. It may not be supported by other zarr implementations and may change
in the future.
```
The skill file mentions `consolidated=True` as a performance tip but does not note this v3 incompatibility. Users may want to pass `consolidated=False` when using `zarr_format=3`.

### 3. `subtree_with_keys` path format
The skill file shows:
```python
for path, node in dt.subtree_with_keys:
    print(path, node.dataset)
```
This implies `path` is a string like `"/"` or `"/group1"`. In practice, paths are relative names like `"."`, `"surface"`, `"upper_air"` (not absolute paths starting with `/`). This could confuse users expecting the same path format used in `DataTree.from_dict`.

---

## What Was Missing from the Skill File

### 1. NetCDF backend dependency requirements
The skill file shows `to_netcdf()` and `open_dataset()` patterns but never mentions that these require a backend library (`h5netcdf`, `netCDF4`, or `scipy`) to be installed. When none is present, xarray raises:
```
ValueError: cannot write NetCDF files because none of the suitable backend
libraries (netCDF4, h5netcdf, scipy) are installed
```
And `h5netcdf` itself requires `h5py`. The skill file should note these dependencies, especially since the `pyproject.toml` for a zarr-focused project may not include them.

### 2. `Dataset.sizes` as the preferred way to get dimension lengths
As noted above, `ds.dims` is being deprecated for dict-like access. The skill file should mention `ds.sizes` as the preferred replacement.

### 3. `consolidated=False` recommendation for zarr v3
The skill file should note that `consolidated=True` (the default) emits a warning with `zarr_format=3` since consolidated metadata is not yet part of the zarr v3 spec.

### 4. `exclude_dims` for `apply_ufunc` when reducing dimensions
The skill file mentions `exclude_dims=set()` in the rules section but doesn't show a concrete example of when you need it (e.g., when the output has fewer core dims than the input and the dim changes size).

---

## Suggestions for Improving the Skill File

1. **Add a "Dependencies" section** at the top listing what backends are needed for NetCDF I/O (`h5netcdf` + `h5py`, or `netCDF4`, or `scipy`). This is a common stumbling block.

2. **Replace `ds.dims` with `ds.sizes`** throughout. The `dims` attribute on `Dataset` is transitioning to return a set (like `DataArray.dims`), so `ds.sizes` is the forward-compatible way to get a `{dim_name: length}` mapping.

3. **Add `consolidated=False` to the zarr v3 example.** Since consolidated metadata is not part of the zarr v3 spec, the example should either pass `consolidated=False` or note the warning.

4. **Clarify `subtree_with_keys` path format.** Note that paths are relative node names (like `"."`, `"surface"`), not absolute paths (like `"/surface"`).

5. **Add a concrete `exclude_dims` example** for `apply_ufunc` to complement the brief mention in the rules.

6. **Note that `open_mfdataset` with `engine="h5netcdf"` requires `h5py`** -- the skill file recommends this engine for parallel reads but doesn't flag the dependency.

7. **Minor: add a note about `FutureWarning` noise.** Users running xarray 2026.x will see several `FutureWarning` messages related to the `dims` transition. Acknowledging this helps users distinguish expected warnings from real problems.
