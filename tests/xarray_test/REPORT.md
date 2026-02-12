# xarray Skill Assessment

Tested against xarray v2026.1.0, zarr v3.1.5, dask v2026.1.2, h5netcdf v1.8.1.

54 tests executed, 0 failures. Several skill file accuracy issues found below.

## Verified Correct

- **DataTree constructor**: `xr.DataTree(dataset=ds, children=None, name="root")` works. The `ds` param is fully removed (TypeError if used). Skill file's `dataset=` param is correct.
- **DataTree.from_dict**: Works with hierarchical paths (`"/group1/subgroup"` etc). Children with conflicting dim sizes vs parent raise ValueError (alignment constraint).
- **DataTree.map_over_datasets**: `dt.map_over_datasets(lambda ds: ds.mean())` works correctly.
- **DataTree I/O**: `dt.to_zarr()` + `xr.open_datatree(path, engine="zarr")` round-trips correctly.
- **apply_ufunc core dims**: Core dims are moved to last axis (-1), confirmed with assertion inside the function.
- **apply_ufunc output_dtypes**: Required with `dask='parallelized'`, works as documented.
- **apply_ufunc output_sizes**: `dask_gufunc_kwargs={"output_sizes": {...}}` works for dimension-changing functions.
- **apply_ufunc multiple returns**: `output_core_dims=[[], []]` with `output_dtypes=[float, float]` correctly returns two arrays.
- **apply_ufunc vectorize**: `vectorize=True` with `dask='parallelized'` works for scalar functions.
- **map_blocks**: `xr.map_blocks(lambda block: block - block.mean(), ds)` works on chunked datasets.
- **concat existing dim**: `xr.concat([ds1, ds2], dim="time")` joins along existing dimension.
- **concat new dim**: `xr.concat([ds1, ds2], dim="ensemble")` creates a new dimension.
- **merge different vars**: Combines datasets with different variable names.
- **merge conflict detection**: Raises `MergeError` on conflicting values.
- **combine_by_coords**: Auto-infers concat dimensions from coordinates.
- **Custom groupers**: `BinGrouper`, `TimeResampler`, `SeasonGrouper` all import from `xarray.groupers` and work as documented.
- **Resample freq aliases**: `"ME"` and `"YE"` work correctly. Old `"M"` raises ValueError (completely removed, not just deprecated).
- **open_mfdataset performance**: `compat="override"`, `coords="minimal"`, `data_vars="minimal"`, `engine="h5netcdf"` all work together with `parallel=True`.
- **Zarr v3 encoding**: `BloscCodec` from `zarr.codecs` with `compressors=(codec,)` (plural key) works correctly.
- **drop_encoding()**: Clears encoding dict, enables clean re-write to different format/chunking.
- **CFDatetimeCoder**: `xr.coders.CFDatetimeCoder(time_unit="s")` passed to `decode_times=` works correctly.
- **CFDatetimeCoder(use_cftime=True)**: Constructor accepts the parameter (requires cftime package at runtime).
- **sel() inclusive slicing**: `da.sel(x=slice(10, 30))` returns values at 10, 20, and 30 (both ends inclusive).
- **.values vs .data**: `.data` returns dask array (lazy), `.values` triggers compute and returns numpy.
- **concat_dim requires combine='nested'**: `open_mfdataset(concat_dim="time")` without `combine="nested"` raises ValueError.
- **Encoding persistence**: Encoding is present after `open_zarr` and persists through operations.
- **_FillValue=0 gotcha**: Zeros become NaN when `_FillValue=0`; `mask_and_scale=False` prevents this.
- **String coordinates**: Object-dtype strings need `astype(str)` before writing to zarr.
- **ds.sizes**: Works as documented replacement for `ds.dims` dict access.
- **DataTree import**: `from xarray import DataTree` works (not from `datatree` package).
- **use_cftime deprecation**: `open_dataset(use_cftime=True)` emits DeprecationWarning as documented.
- **register_dataarray_accessor**: Exists for the subclassing alternative pattern.

## Issues Found

### 1. Automatic alignment NaN gotcha is OUTDATED (High Impact)
- **Skill file claim** (line 202-210): "a + b with partially overlapping coords fills gaps with NaN" and shows result with `x=[0,1,2,3]` including NaN.
- **Actual behavior in v2026.1.0**: The default `arithmetic_join` is now `'inner'`, NOT `'outer'`. The result is `x=[1,2]` with values `[12, 23]` and NO NaN.
- **Impact**: This is the first gotcha in the skill file and it's now wrong for the current version. The old behavior only occurs with `xr.set_options(arithmetic_join="outer")`.
- **Fix**: Update the gotcha to explain that the default changed. With `arithmetic_join='inner'` (the new default), partial overlaps silently DROP non-overlapping coords instead of introducing NaN. The `xr.align(join="inner")` fix is now the default behavior. The NEW gotcha should be that non-overlapping data is silently dropped (rather than producing NaN).

### 2. "Attributes are dropped by default" claim is PARTIALLY WRONG (Medium Impact)
- **Skill file claim** (line 227-231): `result = da1 + da2  # result.attrs == {}` with comment "Attributes are dropped by default in operations".
- **Actual behavior**: With the default `keep_attrs='default'`, IDENTICAL attrs ARE preserved. `DataArray([1,2,3], attrs={"units":"K"}) + DataArray([4,5,6], attrs={"units":"K"})` yields `result.attrs == {"units": "K"}`. Only CONFLICTING attrs are dropped (e.g., `"units":"K"` vs `"units":"C"` yields `{}`).
- **Fix**: Update the example to use arrays with different attrs to demonstrate the dropping behavior, or clarify the nuanced behavior.

### 3. DataTree.from_dict alignment constraint not mentioned (Low Impact)
- **Skill file** (line 14-19): Shows `from_dict` with `/` having `("x", [1,2,3])` and `/group1` using `ds1`. If `ds1` uses the same dimension name `x` with a different size, this raises `ValueError` due to parent-child alignment.
- **Fix**: Add a brief note that child nodes sharing dimension names with parents must have compatible sizes, or use different dimension names.

## Missing Content

- **`arithmetic_join` default change**: The skill file should mention that the default `arithmetic_join` changed from `'outer'` to `'inner'`. This fundamentally changes how misaligned coordinate arithmetic behaves and is the biggest behavioral change users will encounter.
- **`arithmetic_compat` default**: The default is now `'minimal'`, working together with the inner join default.
- **DataTree alignment constraints**: The `from_dict` documentation should mention that nodes sharing dimension names must have compatible sizes with their parents.
- **`use_new_combine_kwarg_defaults` option**: xarray has this option (`False` by default) that will change combine/concat defaults in the future. Worth mentioning for forward compatibility.

## Overall Assessment

The skill file is **quite good overall** -- the vast majority of claims verified correct. The apply_ufunc documentation is excellent and all patterns work exactly as described. GroupBy, encoding, I/O, and zarr patterns are all accurate.

The two significant issues both relate to **behavioral defaults that changed in recent xarray versions**:
1. `arithmetic_join` defaulting to `'inner'` makes the NaN alignment gotcha outdated -- this is the highest-impact issue
2. `keep_attrs='default'` now preserving identical attrs makes the "attrs dropped" claim partially wrong

These should be updated since they affect how users understand xarray's fundamental behavior. The DataTree alignment note is minor but would prevent confusion.
