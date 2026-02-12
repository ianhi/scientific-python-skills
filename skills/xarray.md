# xarray - Claude Code Skill

> Skill for writing idiomatic xarray code. Version: 2026.01.0+

## DataTree (New - in xarray since v2024.10)

```python
xr.DataTree(
    dataset=None,            # Dataset for this node (renamed from 'ds' param)
    children=None,           # dict of {name: DataTree}
    name=None,
)

# Create from dict (most common constructor)
dt = xr.DataTree.from_dict({
    "/": xr.Dataset({"global_temp": ("x", [1, 2, 3])}),
    "/group1": ds1,
    "/group1/subgroup": ds2,
})

# Map a function over every dataset in the tree
result = dt.map_over_datasets(lambda ds: ds.mean())

# I/O: read/write hierarchical formats (zarr, HDF5)
dt = xr.open_datatree("store.zarr", engine="zarr")
dt.to_zarr("out.zarr")
```

## apply_ufunc

The #1 pain point. Key rules:
- Core dimensions are moved to the **last** axis (-1), not axis=0
- With `dask='parallelized'`, MUST specify `output_dtypes`

```python
# Reduction along core dimensions
def detrend(arr):
    """arr has core dims as LAST axes (-1), not axis=0."""
    return scipy.signal.detrend(arr, axis=-1)

result = xr.apply_ufunc(
    detrend, da,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],  # same dims if not reducing
    dask="parallelized",
    output_dtypes=[float],         # REQUIRED with dask='parallelized'
)

# Function that changes dimension size - must specify output_sizes
result = xr.apply_ufunc(
    coarsen_func, da, 2,
    input_core_dims=[["time"], []],
    output_core_dims=[["time_coarse"]],
    dask="parallelized",
    output_dtypes=[float],
    dask_gufunc_kwargs={"output_sizes": {"time_coarse": len(da.time) // 2}},
)

# Multiple return values
mean, std = xr.apply_ufunc(
    decompose, da,
    input_core_dims=[["time"]],
    output_core_dims=[[], []],
    dask="parallelized",
    output_dtypes=[float, float],
)

# vectorize=True for scalar functions over broadcast dims
result = xr.apply_ufunc(
    scalar_func, da1, da2,
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
)
```

**`xr.map_blocks()`** is simpler when you want labeled xarray objects per chunk:
```python
result = xr.map_blocks(lambda block: block - block.mean(), ds)
```

## Data Combination - Subtle Differences

```python
# concat: stack datasets along a dimension (must share other dims/vars)
combined = xr.concat([ds1, ds2], dim="time")       # existing dim
combined = xr.concat([ds1, ds2], dim="ensemble")    # NEW dim (adds it)

# merge: combine datasets with DIFFERENT variables into one
merged = xr.merge([ds_temp, ds_precip])  # fails if same var has conflicting values

# GOTCHA: concat vs merge confusion
# concat requires same variables, joins along a dim
# merge requires different variables (or compatible overlaps), no concat dim
# If you want to add new variables, use merge. If extending a dim, use concat.

# combine_by_coords: auto-infers concat dims from coordinates
# Use for tiled/multi-file data where coords tell you the layout
combined = xr.combine_by_coords([ds1, ds2, ds3, ds4])
```

## GroupBy Performance

**Install `flox` for 10-100x faster groupby/resample with dask arrays.** It's automatically used when installed.

```python
# Custom groupers (v2024.01+)
from xarray.groupers import BinGrouper, TimeResampler, SeasonGrouper
ds.groupby(x=BinGrouper(bins=[0, 25, 50])).mean()
ds.groupby(time=TimeResampler(freq="ME")).mean()
ds.groupby(time=SeasonGrouper(["DJF", "MAM", "JJA", "SON"])).mean()  # v2024.09+

# Resample freq aliases: use "ME" not "M", "YE" not "Y", "QE" not "Q"
ds.resample(time="ME").mean()   # month-end
ds.resample(time="YE").mean()   # year-end
```

## I/O Gotchas

### open_mfdataset performance (#1385)
```python
# SLOW - default settings reindex every coordinate across all files
ds = xr.open_mfdataset("data/*.nc")

# FAST - skip redundant coordinate checking when files share coords
ds = xr.open_mfdataset(
    "data/*.nc",
    parallel=True,
    compat="override",       # trust coords from first file
    coords="minimal",        # only read coords from first file
    data_vars="minimal",     # only read data vars that vary
    engine="h5netcdf",       # h5netcdf is thread-safe for parallel reads
)
```

### Zarr v3 encoding (#10032)
```python
# Use zarr.codecs (not numcodecs) with zarr v3
from zarr.codecs import BloscCodec
compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
ds.to_zarr("out.zarr", encoding={"var": {"compressors": (compressor,)}})

# Note: encoding keys are PLURAL in v3: compressors (not compressor), filters

# When migrating zarr v2 -> v3, drop inherited encoding:
ds = xr.open_zarr("old_v2.zarr")
ds = ds.drop_encoding()
ds.to_zarr("new_v3.zarr")
```

### String coordinates (#3476)
```python
# Cast object-dtype strings to str before writing to zarr
for coord in ds.coords:
    if ds[coord].dtype == object:
        ds[coord] = ds[coord].astype(str)
```

### Encoding gotchas
```python
# GOTCHA: encoding persists through operations and can cause errors
ds = xr.open_zarr("input.zarr")
ds_subset = ds.isel(time=slice(0, 10))
ds_subset.to_zarr("output.zarr")  # ERROR - stale chunk encoding
# Fix: always drop_encoding() when changing chunk layout or format
ds_subset = ds_subset.drop_encoding()
ds_subset.to_zarr("output.zarr")

# GOTCHA: int16 with scale_factor decodes to float32 (#2304)
# netCDF4 library promotes to float64, but xarray uses float32 for int16
# Fix: mask_and_scale=False, or control dtype in encoding

# GOTCHA: _FillValue=0 in NetCDF will mask legitimate zeros
encoding = {"var": {"_FillValue": -9999}}  # use a sentinel instead
```

### Chunk alignment with zarr (#5065, #9914)
```python
# CRITICAL: dask chunks must be multiples of zarr chunks
ds = ds.chunk({"x": 150})  # if zarr chunks are 100, this FAILS
ds.to_zarr("out.zarr")

# RIGHT - ensure chunk alignment
ds = ds.chunk({"x": 200})  # multiple of zarr chunk size
# For sharded zarr: align to SHARD boundaries, not just internal chunks
```

### CF Time Decoding
```python
# Control time decoding (v2024.01+)
coder = xr.coders.CFDatetimeCoder(time_unit="s")
ds = xr.open_dataset("file.nc", decode_times=coder)

# Use cftime for non-standard calendars or dates outside 1678-2262
coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_dataset("file.nc", decode_times=coder)
```

## Gotchas & Common Mistakes

### Automatic coordinate alignment silently introduces NaN
```python
# GOTCHA: a + b with partially overlapping coords fills gaps with NaN
a = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
b = xr.DataArray([10, 20, 30], dims=["x"], coords={"x": [1, 2, 3]})
result = a + b  # x=[0,1,2,3], with NaN at x=0 and x=3!

# Fix: explicitly align first
a2, b2 = xr.align(a, b, join="inner")  # only shared coords
result = a2 + b2  # x=[1,2], no NaN
```

### sel() slices are inclusive on both ends
```python
da.sel(x=slice(10, 30))  # Returns x=[10, 20, 30] -- includes 30!
# This differs from Python/numpy slicing where end is exclusive
```

### Zarr fill_value != _FillValue (#5475)
```python
# Zarr fill_value (chunk background) != CF _FillValue (missing data indicator)
# int array with fill_value=0 -> legitimate 0 values become NaN
# Workaround: xr.open_dataset("store.zarr", engine="zarr", mask_and_scale=False)
```

### Attributes are dropped by default in operations
```python
result = da1 + da2  # result.attrs == {}
with xr.set_options(keep_attrs=True):
    result = da1 + da2  # attrs preserved
```

### Performance: avoid iterating with .sel/.isel (#2799)
```python
# SLOW - 1000x slower than numpy for element-wise access
for i in range(n):
    val = da.isel(x=i).values  # massive overhead per call

# FAST - extract numpy array first
arr = da.values  # or .data to keep dask lazy
```

### .values vs .data
```python
ds["temp"].values   # triggers compute, returns numpy (downloads entire array!)
ds["temp"].data     # keeps dask lazy, returns dask array
```

### concat_dim requires combine='nested' (#2064)
```python
# WRONG - ValueError
ds = xr.open_mfdataset("*.nc", concat_dim="time")
# RIGHT
ds = xr.open_mfdataset("*.nc", combine="nested", concat_dim="time")
# Or omit concat_dim entirely (default combine='by_coords' auto-detects)
```

## Deprecation Reference

| Deprecated | Replacement | Since |
|------------|-------------|-------|
| `open_dataset(use_cftime=True)` | `open_dataset(decode_times=CFDatetimeCoder(use_cftime=True))` | v2025.01 |
| `resample(time="M")` | `resample(time="ME")` (month-end) | pandas v2.2+ |
| `resample(time="Y")` | `resample(time="YE")` (year-end) | pandas v2.2+ |
| `from datatree import DataTree` | `from xarray import DataTree` | v2024.10 |
| `ds.dims` (as dict) | `ds.sizes` | Ongoing |

## Known Limitations

- **No native append to NetCDF** (#1672): Use zarr's `append_dim` instead.
- **MultiIndex serialization** (#1077): Use `ds.reset_index()` before writing.
- **Zarr v3 fill_value masking** (#5475): Being actively fixed.
- **Subclassing** (#3980): Use accessors (`@xr.register_dataarray_accessor`) instead.

## Performance Tips

- **Install flox** for 10-100x faster groupby/resample with dask
- **Install bottleneck** for faster NaN-aware reductions
- **Use `engine="h5netcdf"`** for parallel NetCDF reads (thread-safe)
- **Use `compat="override", coords="minimal"`** with open_mfdataset (#1385)
- **Avoid `.sel()`/`.isel()` in loops** - extract `.values` first (#2799)
- **Call `.unify_chunks()`** before writing to zarr
- **Use `ds.drop_encoding()`** when re-writing data to a different format
- **Avoid `.values` on large dask arrays** - use `.data` to keep lazy
- **Prefer zarr over NetCDF** for cloud storage and parallel writes
