# xarray Skill File Review

Reviewed against xarray source at `/Users/ian/Documents/dev/scientific-python-skills/repos/xarray/` (appears to be post-2025.01 based on deprecation markers).

---

## ACCURACY

### Function Signatures

1. **DataTree constructor (line 213-221)**: CORRECT. The `dataset` parameter name is verified (`dataset: Dataset | Coordinates | None = None`). The note about renaming from `ds` is helpful.

2. **`xr.date_range_like(ds.time)` (line 619)**: **WRONG**. The actual signature is `date_range_like(source, calendar, use_cftime=None)` -- the `calendar` parameter is **required**, not optional. The skill file shows `new_times = xr.date_range_like(ds.time)` which would raise a `TypeError`. Should be `new_times = xr.date_range_like(ds.time, calendar="standard")`.

3. **`xr.date_range(..., unit="s")` (line 611)**: CORRECT. The source confirms `unit: PDDatetimeUnitOptions = "ns"` parameter exists.

4. **`xr.coders.CFDatetimeCoder(time_unit="s")` (lines 107, 636)**: CORRECT. `xr.coders` re-exports from `xarray.coding.times`. The parameter is `time_unit`, confirmed in source.

5. **`apply_ufunc` signature (lines 251-311)**: CORRECT. The `dask`, `output_dtypes`, `dask_gufunc_kwargs`, `vectorize`, `input_core_dims`, `output_core_dims` parameters all match source. One note: the source also has `output_sizes` as a top-level parameter (not just inside `dask_gufunc_kwargs`), but the skill file only shows the `dask_gufunc_kwargs` route. Both work, but the simpler top-level `output_sizes` param could be mentioned.

6. **`open_mfdataset` defaults**: The skill file (line 77) shows `combine='by_coords'` as a recommended parameter. The source confirms this is already the **default** (`combine: Literal["by_coords", "nested"] = "by_coords"`). So specifying it explicitly is redundant but not wrong.

7. **`ds.to_zarr("store.zarr", zarr_format=3)` (line 444)**: CORRECT. Source confirms `zarr_format: int | None = None` parameter.

8. **`open_zarr` defaults**: Correct that `open_zarr` is in `xarray.backends.zarr`, not `backends.api`. The signature matches.

9. **`DataTree` parameter `dataset=None` with comment "renamed from 'ds' param" (line 214)**: CORRECT and useful context.

### Anti-Patterns: Would the "WRONG" Code Actually Fail?

1. **Zarr v3 compression (lines 9-23)**: ACCURATE. With zarr v3, `numcodecs.blosc.Blosc` and the singular `compressor` key do not work. The plural `compressors` tuple syntax is correct for zarr v3.

2. **apply_ufunc missing output_dtypes (lines 36-44)**: ACCURATE. With `dask='parallelized'`, omitting `output_dtypes` will raise a ValueError. This is a very common error.

3. **apply_ufunc core dimensions axis=0 vs axis=-1 (lines 48-57)**: ACCURATE. Core dims are moved to the last axis. Using `axis=0` gives wrong results silently (not an error, which is worse). Excellent anti-pattern to include.

4. **GroupBy without flox (lines 62-69)**: PARTIALLY ACCURATE. The "WRONG" code will not fail -- it will just be very slow. The label "WRONG" is misleading. This is a performance issue, not a correctness issue. The comment says "extremely slow" which is fair, but the heading style suggests it's broken code.

5. **`open_mfdataset` slow (lines 72-84)**: The "WRONG" example `xr.open_mfdataset('data/*.nc')` works fine -- it just may be slow for many files with identical coordinates. The "RIGHT" version shows `combine='by_coords'` which is already the default, so that line is redundant. The real value is `compat='override'` and `coords='minimal'`.

6. **String coordinates and zarr (lines 88-96)**: ACCURATE for older xarray/zarr combos. May be less relevant with newer zarr v3 string handling, but still a valid gotcha.

7. **Datetime nanosecond overflow (lines 99-109)**: ACCURATE. The `pd.date_range("0001-01-01", periods=365)` will indeed cause an `OutOfBoundsDatetime` error with nanosecond resolution.

8. **DataTree import (lines 112-120)**: ACCURATE. `datatree` was merged into xarray in v2024.10.

### Issue Numbers

- **#10032** (Zarr v3 compression): Could not verify from source alone, but the content is consistent with zarr v3 migration issues.
- **#1385** (open_mfdataset slow): Plausible.
- **#3476** (String coords and zarr): Plausible.
- **#2304** (int16 scale_factor precision): Plausible.
- **#5475** (Zarr fill_value vs _FillValue): Plausible.
- **#1672** (No native append to NetCDF): Plausible.
- **#2799** (Performance of sel/isel in loops): Plausible.
- **#5065** (Chunk alignment with zarr): Plausible.
- **#9914** (Chunk alignment): Plausible.
- **#2237** (GroupBy with dask destroys chunks): Plausible.
- **#2064** (concat_dim added to all variables): Plausible.
- **#1077** (MultiIndex serialization): Plausible.
- **#3980** (Subclassing): Plausible.

### Code Snippets That Would Not Work

1. **Line 619**: `new_times = xr.date_range_like(ds.time)` -- **BROKEN**. Missing required `calendar` argument.

2. **Line 709**: `ds = xr.open_mfdataset("*.nc", concat_dim="time")` -- **WOULD ERROR**, but not for the stated reason. Since `combine` defaults to `"by_coords"`, passing `concat_dim` with `combine="by_coords"` raises a `ValueError`: "When combine='by_coords', passing a value for `concat_dim` has no effect." The gotcha label says "concat_dim added to all variables" which describes old behavior (pre-combine refactor). This anti-pattern is stale.

3. **Line 703**: `anomaly = ds.groupby("time.month") - ds.groupby("time.month").mean()` -- This is a functional pattern, but the typical pattern for computing anomalies is: `climatology = ds.groupby("time.month").mean(); anomaly = ds.groupby("time.month") - climatology`. The shown syntax creates two separate groupby objects and may not do what users expect (though it should work).

---

## USEFULNESS (1-5 per section)

| Section | Rating | Notes |
|---|---|---|
| CRITICAL: Outdated Patterns | **5** | Best section. Zarr v3 migration, apply_ufunc pitfalls -- these are exactly the things Claude gets wrong. |
| Quick Reference - Installation | **2** | Claude already knows this. Space wasted. |
| Quick Reference - Common Operations | **3** | Decent cheat sheet, but Claude knows basic xarray. The resample frequency codes ("ME" not "M") are the only high-value bit here. |
| Core API - DataArray/Dataset/DataTree | **3** | DataTree constructor details are valuable. DataArray/Dataset constructors are well-known. |
| Patterns - Indexing | **3** | The inclusive-slice gotcha is great (rating 5 if standalone). The rest is standard. |
| Patterns - apply_ufunc | **5** | THE most error-prone xarray API. Five patterns covering real use cases. Invaluable. |
| Patterns - GroupBy/Resample/Rolling | **4** | Custom groupers (SeasonGrouper, BinGrouper) are new and not well-known. Frequency string deprecation (ME/YE/QE) is critical. |
| Data Combination | **3** | Standard API, but `combine_nested` with nested lists is non-obvious. |
| I/O - NetCDF | **4** | Encoding examples are genuinely useful. The open_mfdataset optimization flags are high-value. |
| I/O - Zarr | **4** | Region writes, zarr_format=3, DataTree zarr -- all modern and useful. |
| Encoding Gotchas | **5** | Encoding persistence, _FillValue=0 masking zeros, drop_encoding() -- these cause real bugs. |
| Dask Integration | **4** | Chunk alignment with zarr is critical. Lazy evaluation patterns are helpful. |
| Integration (pandas/numpy/matplotlib) | **2** | Too basic. Claude knows these. |
| Integration (rioxarray/dask.distributed) | **3** | Niche but useful when relevant. |
| Time Series | **4** | cftime_range, CF decoding, time accessors are important for climate/weather users. |
| Gotchas & Common Mistakes | **5** | Every item here prevents real bugs. |
| Known Limitations & Bugs | **3** | Good to know, but rarely actionable. |
| Performance Tips | **4** | Good summary, mostly duplicates earlier content though. |

### Sections to Cut (to save space)

1. **Quick Reference - Installation & Import** (lines 125-138): Remove entirely. Claude knows `import xarray as xr`. The optional dependencies list could be a single-line comment instead.

2. **Integration - With pandas** (lines 542-554): Too basic. `ds.to_dataframe()` and `df.to_xarray()` are self-documenting.

3. **Integration - With NumPy** (lines 557-562): Five lines that say nothing non-obvious. Remove.

4. **Integration - With matplotlib** (lines 565-584): Basic plotting. Claude knows this. Maybe keep the faceted plots example (`col="time"`) and the `attrs` auto-label trick; cut the rest.

5. **Integration - With dask.distributed** (lines 597-603): Too basic. `Client()` then `.compute()` is standard dask.

### Sections to Expand

1. **apply_ufunc** (lines 251-311): Already the best section. Consider adding a pattern for functions with keyword arguments (using the `kwargs` parameter of `apply_ufunc`).

2. **Encoding Gotchas** (lines 453-481): Add guidance on reading/modifying/writing encoding programmatically. Also add the `use_cftime` deprecation (since 2025.01.1): users should now pass `CFDatetimeCoder(use_cftime=True)` to `decode_times` instead of using the `use_cftime` kwarg directly.

3. **DataTree** (lines 211-221): This is extremely thin for a major new feature. Add `DataTree.from_dict()`, `DataTree.to_dict()`, `group_subtrees()`, and show how to iterate/transform tree structures. The `map_over_datasets` function is mentioned but not illustrated.

---

## MISSING

### Common Patterns Not Covered

1. **`xr.where(cond, x, y)`** -- The top-level `xr.where` function (distinct from `da.where(cond)`) is exported and commonly used but never mentioned. Important distinction: `da.where(cond)` keeps the shape with NaN fill, while `xr.where(cond, x, y)` is a three-argument ternary.

2. **`xr.full_like()`, `xr.zeros_like()`, `xr.ones_like()`** -- Common array creation functions. All exported from top-level.

3. **`xr.dot()`** -- Einstein summation for labeled arrays. Very useful for users coming from numpy's `einsum`.

4. **`xr.corr()` and `xr.cov()`** -- Correlation and covariance functions with proper dimension handling.

5. **`xr.map_blocks()`** -- Alternative to `apply_ufunc` for dask workflows. The skill file covers `apply_ufunc` extensively but never mentions `map_blocks`, which is often simpler for block-wise operations.

6. **`xr.open_dataarray()`** -- Mentioned in `__init__.py` exports but absent from the skill file. Common for single-variable files.

7. **`xr.load_dataset()` / `xr.load_dataarray()`** -- Open-load-close pattern. Important distinction from `open_dataset`.

8. **`xr.open_datatree()` and `xr.open_groups()`** -- The DataTree I/O side. `open_groups()` is important when groups are not alignable.

9. **Accessor registration pattern** -- The `@xr.register_dataarray_accessor` / `@xr.register_dataset_accessor` pattern is mentioned in passing (line 739) but never shown with code. This is the recommended extension mechanism.

10. **`ds.assign()` / `ds.assign_coords()`** -- Extremely common for adding/modifying variables/coords. Not shown anywhere.

11. **`ds.pipe()`** -- Method chaining pattern. Common in modern xarray code.

12. **`ds.swap_dims()`** -- Changing which coordinate is the dimension coordinate.

13. **`ds.set_index()` / `ds.reset_index()`** -- MultiIndex creation and destruction. The skill file mentions `reset_index()` in passing but never shows the pattern.

14. **`ds.stack()` / `ds.unstack()`** -- Reshaping dimensions. The skill file shows `stack` once in the pandas section but doesn't explain the pattern.

### Gotchas That Should Be Added

1. **`use_cftime` is deprecated (2025.01.1)**: The skill file uses `use_cftime` without mentioning its deprecation. Should recommend `decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)` instead.

2. **`ds["new_var"] = da` silently drops `da.name`**: When assigning to a Dataset, the variable name comes from the key, not from `da.name`. This causes confusion.

3. **Coordinate alignment is automatic and silent**: `da1 + da2` automatically aligns coordinates, which can silently introduce NaN values when coordinates don't fully overlap. This is a massive source of bugs for beginners.

4. **`.values` vs `.data` vs `.to_numpy()`**: The skill file touches this but doesn't emphasize the critical distinction -- `.values` triggers dask computation while `.data` preserves laziness.

5. **Dimension order matters for broadcasting but not for named operations**: xarray broadcasts by name not position, but when converting to/from numpy, dimension order matters. This bidirectional gotcha is not covered.

6. **`ds.copy(deep=True)` does not copy dask arrays**: With dask backing, `copy(deep=True)` only copies the task graph reference, not the underlying data. This surprises users.

7. **`ds.sel(time="2020")` partial string indexing**: Works for selecting all of year 2020 with datetime coordinates. This is a pandas-inherited feature that many users don't know about.

8. **`consolidated=None` default in `to_zarr`**: The default changed. Users should be aware that consolidation behavior depends on version.

### Important API Functions Not Covered

From `__init__.py` exports that are absent from the skill file:

- `xr.Variable` -- low-level but important for custom backends
- `xr.Coordinates` -- new class for coordinate handling
- `xr.decode_cf()` -- for manual CF decoding after `decode_times=False`
- `xr.infer_freq()` -- frequency inference for time coordinates
- `xr.set_options()` / `xr.get_options()` -- only `keep_attrs` is shown; other options like `display_expand_attrs`, `display_width`, `arithmetic_join` are missing
- `xr.testing` -- assert_equal, assert_identical, assert_allclose (critical for testing)
- `xr.tutorial` -- `xr.tutorial.open_dataset("air_temperature")` for examples
- `xr.map_over_datasets()` -- top-level function for DataTree operations
- `xr.group_subtrees()` -- for iterating over matched DataTree nodes

---

## SUGGESTIONS

### Specific Edits

1. **Fix `date_range_like` example (line 619)**:
   ```python
   # CURRENT (broken):
   new_times = xr.date_range_like(ds.time)
   # FIXED:
   new_times = xr.date_range_like(ds.time, calendar="standard")
   ```

2. **Fix `open_mfdataset` concat_dim anti-pattern (lines 707-713)**: The current "WRONG" example would actually raise a `ValueError` because `concat_dim` cannot be used with `combine="by_coords"` (the default). Either update the example to pass `combine="nested"` in both WRONG and RIGHT, or remove this anti-pattern entirely since it describes pre-refactor behavior. Suggested replacement:
   ```python
   # GOTCHA: concat_dim requires combine='nested'
   # WRONG - raises ValueError because combine='by_coords' is default
   ds = xr.open_mfdataset("*.nc", concat_dim="time")

   # RIGHT - use combine='nested' with concat_dim
   ds = xr.open_mfdataset("*.nc", combine="nested", concat_dim="time", data_vars="minimal")
   ```

3. **Add `use_cftime` deprecation warning**: After the datetime section (around line 647), add:
   ```python
   # DEPRECATED (2025.01+): use_cftime kwarg is deprecated
   # WRONG:
   ds = xr.open_dataset("file.nc", use_cftime=True)
   # RIGHT:
   coder = xr.coders.CFDatetimeCoder(use_cftime=True)
   ds = xr.open_dataset("file.nc", decode_times=coder)
   ```

4. **Clarify GroupBy flox anti-pattern label (lines 62-69)**: Change from "WRONG" to "SLOW" or "PERFORMANCE". The code is not wrong -- it works, just slowly.

5. **Trim the Quick Reference section**: The Common Operations block (lines 141-181) duplicates content covered in detail in later sections. Either cut it or reduce it to a 5-line "most common operations" cheat sheet.

6. **Add `xr.where()` to Common Operations**:
   ```python
   # Conditional selection (three-argument form)
   result = xr.where(ds.temp > 300, ds.temp, np.nan)
   ```

7. **Add coordinate alignment gotcha to the Gotchas section**:
   ```python
   # GOTCHA: automatic alignment introduces NaN
   a = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
   b = xr.DataArray([10, 20, 30], dims=["x"], coords={"x": [1, 2, 3]})
   result = a + b  # x=[0,1,2,3], with NaN at x=0 and x=3!
   ```

8. **Add `xr.testing` mention**: At minimum:
   ```python
   # Testing (useful for verifying results)
   xr.testing.assert_equal(da1, da2)      # values + coords
   xr.testing.assert_identical(da1, da2)  # values + coords + attrs
   xr.testing.assert_allclose(da1, da2)   # approximate equality
   ```

9. **Expand DataTree section significantly**: Add `from_dict`, tree iteration, and `map_over_datasets` examples:
   ```python
   # Create from dict
   dt = xr.DataTree.from_dict({
       "/": xr.Dataset({"global_attr": ...}),
       "/group1": ds1,
       "/group1/subgroup": ds2,
   })

   # Iterate over all nodes
   for path, node in dt.subtree_with_keys:
       print(path, node.dataset)

   # Map function over all datasets in tree
   result = dt.map_over_datasets(lambda ds: ds.mean())
   ```

### Ordering Changes

The current order places anti-patterns first, which is good for preventing mistakes. However, the document flow has some issues:

1. The "Encoding Gotchas" section (line 453) is buried inside "I/O Patterns" but is so important it should be its own top-level section, perhaps merged with the "Gotchas & Common Mistakes" section.

2. The "Performance Tips" section at the end is a bullet-point summary that mostly restates earlier content. Either integrate these tips into their respective sections (e.g., flox tip goes in GroupBy section) or keep it as a quick-reference summary but remove the duplicated prose elsewhere.

3. Consider grouping "GroupBy, Resample, Rolling" closer to "Dask Integration" since dask interaction is the main complexity there.

### Things That Would Confuse Claude

1. **Redundant `combine='by_coords'` in examples**: Since this is the default, showing it explicitly might cause Claude to think it's required or that the default is something else. Either annotate it with `# (default)` or remove it.

2. **The "WRONG" label for performance-only issues** (flox, open_mfdataset speed): Claude will treat these as errors and refuse to generate the "WRONG" code, when actually the code works correctly. Use "SLOW" or "SUBOPTIMAL" labels to distinguish from actual bugs.

3. **Mixing zarr v2 and v3 advice**: The skill file covers both but doesn't clearly delineate when each applies. A user working with zarr v2 might be confused by the v3 advice and vice versa. Consider adding a clear note: "If using zarr v3 (zarr >= 3.0), use plural keys (compressors, filters). If using zarr v2, use singular keys."

4. **`save_mfdataset` is shown without explanation (line 412)**: `xr.save_mfdataset(datasets, paths)` appears as a one-liner with no context. Either explain what it does (write multiple datasets to multiple files) or remove it.

5. **The `open_mfdataset` concat_dim anti-pattern (line 709)** would actively mislead Claude into thinking the code works but has a subtle bug, when in reality it raises an immediate error. This could cause Claude to write incorrect explanations of what goes wrong.

---

## OVERALL ASSESSMENT

**Grade: B+**

**Strengths:**
- The anti-patterns section at the top is excellent and addresses the highest-impact mistakes
- apply_ufunc coverage is thorough and well-structured with five distinct patterns
- Encoding gotchas are genuinely useful and hard to find elsewhere
- Zarr v3 migration guidance is timely and important
- Good coverage of dask chunk alignment issues

**Weaknesses:**
- Two code examples are broken (`date_range_like`, `open_mfdataset` concat_dim)
- Too much basic content that wastes context window (imports, basic plotting, pandas conversion)
- DataTree coverage is too thin for a major new feature
- Missing important top-level functions (`xr.where`, `xr.dot`, `xr.testing`, etc.)
- Performance issues labeled as "WRONG" may cause Claude to be overly cautious
- No mention of `use_cftime` deprecation despite showing the old pattern
- Missing the automatic coordinate alignment gotcha, which is arguably the single most confusing xarray behavior for new users

**Priority fixes (in order):**
1. Fix the broken `date_range_like` example
2. Fix or rewrite the `open_mfdataset` concat_dim anti-pattern
3. Add `use_cftime` deprecation notice
4. Relabel performance-only anti-patterns as "SLOW" not "WRONG"
5. Add coordinate alignment gotcha
6. Expand DataTree section
7. Cut basic integration sections to save space
8. Add `xr.where()`, `xr.testing`, and `xr.map_blocks` coverage
