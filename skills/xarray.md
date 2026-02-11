# xarray - Claude Code Skill

> Skill for writing idiomatic xarray code. Version: 2026.01.0+

## CRITICAL: Do NOT Use These Outdated Patterns

### Zarr v3 compression encoding (#10032)
```python
# WRONG - numcodecs.blosc no longer works with zarr v3
from numcodecs.blosc import Blosc
compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
ds.to_zarr("out.zarr", encoding={"var": {"compressor": compressor}})

# RIGHT - use zarr.codecs for zarr v3 (note plural "compressors" key)
from zarr.codecs import BloscCodec
compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
ds.to_zarr("out.zarr", encoding={"var": {"compressors": (compressor,)}})

# ALSO RIGHT - drop inherited encoding when re-writing zarr v2 data as v3
ds = xr.open_zarr("old_v2.zarr")
ds = ds.drop_encoding()
ds.to_zarr("new_v3.zarr")
```

### Encoding key names changed for zarr v3
```python
# WRONG - zarr v2 style singular keys
encoding = {"var": {"compressor": codec, "filters": [filter1]}}

# RIGHT - zarr v3 style plural keys
encoding = {"var": {"compressors": (codec,), "filters": (filter1,)}}
```

### apply_ufunc with dask - missing output_dtypes
```python
# WRONG - fails with dask arrays when output_dtypes not specified
result = xr.apply_ufunc(my_func, dask_da, dask='parallelized')

# RIGHT - always specify output_dtypes with dask='parallelized'
result = xr.apply_ufunc(
    my_func, dask_da,
    dask='parallelized',
    output_dtypes=[float],
)
```

### apply_ufunc core dimensions - wrong axis assumption
```python
# WRONG - assumes function operates on axis=0
def my_func(arr):
    return np.mean(arr, axis=0)
result = xr.apply_ufunc(my_func, da, input_core_dims=[['time']])

# RIGHT - core dims are moved to LAST axis (-1)
def my_func(arr):
    return np.mean(arr, axis=-1)
result = xr.apply_ufunc(my_func, da, input_core_dims=[['time']])
```

### GroupBy with dask - not using flox
```python
# SLOW - extremely slow groupby with dask (non-contiguous groups destroy chunks)
result = ds.groupby('time.month').mean()  # without flox installed

# FAST - install flox for 10-100x faster groupby with dask
# pip install flox
# flox is automatically used when installed, no code change needed
result = ds.groupby('time.month').mean()
```

### open_mfdataset slow with identical coordinates (#1385)
```python
# SLOW - reindexes every file (extremely slow for many files)
ds = xr.open_mfdataset('data/*.nc')

# FAST - skip redundant coordinate checking
ds = xr.open_mfdataset(
    'data/*.nc',
    # combine='by_coords' is already the default
    compat='override',     # skip coordinate checking
    coords='minimal',      # only read coords from first file
    data_vars='minimal',   # only read data vars that vary
)
```

### String coordinates and zarr (#3476)
```python
# WRONG - object dtype strings fail when writing to zarr
ds.to_zarr("out.zarr")  # TypeError with object-dtype string coords

# RIGHT - explicitly cast string coords to str before writing
for coord in ds.coords:
    if ds[coord].dtype == object:
        ds[coord] = ds[coord].astype(str)
ds.to_zarr("out.zarr")
```

### Datetime nanosecond overflow
```python
# WRONG - dates outside 1678-2262 fail with default ns resolution
times = pd.date_range("0001-01-01", periods=365)  # overflows

# RIGHT - use cftime for non-standard calendars or wide date ranges
times = xr.cftime_range("0001-01-01", periods=365, calendar="standard")

# RIGHT - control time resolution on decode (v2024.01+)
coder = xr.coders.CFDatetimeCoder(time_unit="s")
ds = xr.open_dataset("file.nc", decode_times=coder)
```

### DataTree import location
```python
# WRONG - old external package import
from datatree import DataTree

# RIGHT - DataTree is now in xarray (since v2024.10)
from xarray import DataTree
# or
dt = xr.DataTree(...)
```

## Quick Reference

### Common Operations (Copy-Paste Ready)
```python
# Create DataArray
da = xr.DataArray(
    np.random.randn(3, 4),
    dims=["x", "y"],
    coords={"x": [10, 20, 30], "y": pd.date_range("2020", periods=4)},
    attrs={"units": "K"},
    name="temperature",
)

# Create Dataset
ds = xr.Dataset(
    {"temp": (("x", "y"), np.random.randn(3, 4)),
     "precip": (("x",), [1.0, 2.0, 3.0])},
    coords={"x": [10, 20, 30], "y": pd.date_range("2020", periods=4)},
)

# Select data
ds.sel(x=20)                          # label-based
ds.isel(x=0)                          # integer-based
ds.sel(y=slice("2020-01", "2020-03")) # time slice (inclusive)
ds.sel(x=15, method="nearest")        # nearest neighbor

# Aggregate
ds.mean(dim="x")
ds.groupby("y.month").mean()
ds.resample(y="ME").mean()   # "ME" = month-end frequency
ds.rolling(y=3).mean()
ds.coarsen(y=2).mean()

# Conditional selection
xr.where(ds.temp > 300, ds.temp, np.nan)  # three-argument form: cond, x, y
ds.temp.where(ds.temp > 0)                # keeps shape, fills with NaN
ds.temp.where(ds.temp > 0, drop=True)     # drops coords where all NaN

# Compute (dask)
ds.compute()   # trigger lazy computation
ds.persist()   # persist in distributed memory
ds.load()      # load into memory

# I/O
ds = xr.open_dataset("file.nc")
ds.to_netcdf("out.nc")
ds = xr.open_zarr("store.zarr")
ds.to_zarr("out.zarr")
```

## Core API

### DataArray
```python
xr.DataArray(
    data,                    # array-like, scalar, or dask array
    coords=None,             # dict mapping dim names -> labels
    dims=None,               # sequence of dimension names
    name=None,               # str
    attrs=None,              # dict of metadata
)
# Key properties: .values, .data, .dims, .coords, .attrs, .name, .encoding
# .values returns numpy array; .data returns underlying array (dask, numpy, etc.)
```

### Dataset
```python
xr.Dataset(
    data_vars=None,          # dict: {name: (dims, data)} or {name: DataArray}
    coords=None,             # dict: {name: data} or {name: (dims, data)}
    attrs=None,              # dict of metadata
)
# Key properties: .data_vars, .coords, .dims, .attrs, .encoding
# Access variables: ds["var"] or ds.var (if no name conflict)
# Drop variables: ds.drop_vars(["var1"])
# Rename: ds.rename({"old": "new"})
```

### DataTree
```python
xr.DataTree(
    dataset=None,            # Dataset for this node (renamed from 'ds' param)
    children=None,           # dict of {name: DataTree}
    name=None,               # str
)
# Navigate: dt["/group/subgroup"], dt.children, dt.parent
# Iterate: dt.subtree, dt.leaves
# Map over all nodes: dt.map_over_datasets(func)

# Create from dict (most common constructor)
dt = xr.DataTree.from_dict({
    "/": xr.Dataset({"global_temp": ("x", [1, 2, 3])}),
    "/group1": ds1,
    "/group1/subgroup": ds2,
})

# Iterate over all nodes (breadth-first)
for path, node in dt.subtree_with_keys:
    print(path, node.dataset)

# Map a function over every dataset in the tree
result = dt.map_over_datasets(lambda ds: ds.mean())

# Group matching subtrees across multiple DataTrees
for path, *nodes in xr.group_subtrees(dt1, dt2):
    print(path, nodes)

# I/O: read/write hierarchical formats (zarr, HDF5)
dt = xr.open_datatree("store.zarr", engine="zarr")
dt.to_zarr("out.zarr")
```

## Patterns & Idioms

### Indexing
```python
# Label-based (.sel) - uses coordinate values
da.sel(x=10)                   # exact match
da.sel(x=[10, 20])             # multiple values
da.sel(x=slice(10, 30))        # range (inclusive both ends!)
da.sel(x=15, method="nearest") # nearest, pad, backfill
da.sel(x=15, method="nearest", tolerance=2)

# Integer-based (.isel) - uses positions
da.isel(x=0)
da.isel(x=slice(0, 2))
da.isel(x=[0, -1])

# .loc - label-based, positional dims (DataArray only)
da.loc[10]            # first dim
da.loc[10:30, "2020"] # multiple dims

# Boolean indexing
da.where(da > 0)              # keeps shape, fills with NaN
da.where(da > 0, drop=True)   # drops coordinates where all NaN

# Important: .sel slices are INCLUSIVE on both ends (unlike Python)
da.sel(x=slice(10, 30))  # includes x=30!
```

### apply_ufunc (major pain point for users)
```python
# Pattern 1: simple element-wise function
result = xr.apply_ufunc(np.square, da)

# Pattern 2: reduction along core dimensions
def detrend(arr):
    """arr has core dims as last axes."""
    return scipy.signal.detrend(arr, axis=-1)

result = xr.apply_ufunc(
    detrend, da,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],  # same dims if not reducing
    dask="parallelized",
    output_dtypes=[float],
)

# Pattern 3: function that changes dimension size
def coarsen_func(arr, factor):
    return arr[..., ::factor]

result = xr.apply_ufunc(
    coarsen_func, da, 2,
    input_core_dims=[["time"], []],
    output_core_dims=[["time_coarse"]],
    dask="parallelized",
    output_dtypes=[float],
    dask_gufunc_kwargs={"output_sizes": {"time_coarse": len(da.time) // 2}},
)

# Pattern 4: multiple return values
def decompose(arr):
    return arr.mean(axis=-1), arr.std(axis=-1)

mean, std = xr.apply_ufunc(
    decompose, da,
    input_core_dims=[["time"]],
    output_core_dims=[[], []],
    dask="parallelized",
    output_dtypes=[float, float],
)

# Pattern 5: vectorize scalar function over broadcast dims
def scalar_func(x, y):
    return x ** 2 + y

result = xr.apply_ufunc(
    scalar_func, da1, da2,
    vectorize=True,  # numpy.vectorize under the hood
    dask="parallelized",
    output_dtypes=[float],
)
```

**apply_ufunc rules:**
- Core dimensions are moved to the **last** axis (-1) of the input arrays
- `vectorize=True` loops over non-core (broadcast) dimensions automatically
- With `dask='parallelized'`, MUST specify `output_dtypes`
- For new output dimensions, use `dask_gufunc_kwargs={"output_sizes": {...}}`
- `exclude_dims=set()` to allow dimensions to change size

**Alternative: `xr.map_blocks()`** -- for dask block-wise operations where the
function expects a DataArray/Dataset (not raw numpy). Simpler than `apply_ufunc`
when you want to operate on labeled xarray objects per chunk:
```python
def process_block(ds_block):
    # ds_block is a Dataset/DataArray with coordinates
    return ds_block - ds_block.mean()

result = xr.map_blocks(process_block, ds)  # template inferred if possible
```

### GroupBy, Resample, Rolling
```python
# GroupBy
ds.groupby("time.season").mean()
ds.groupby("time.month").map(lambda x: x - x.mean("time"))  # anomalies
ds.groupby_bins("lat", bins=np.arange(-90, 91, 30)).mean()

# Custom groupers (v2024.01+)
from xarray.groupers import UniqueGrouper, BinGrouper, TimeResampler
ds.groupby(x=BinGrouper(bins=[0, 25, 50])).mean()
ds.groupby(time=TimeResampler(freq="ME")).mean()

# Season grouper (v2024.09+)
from xarray.groupers import SeasonGrouper
ds.groupby(time=SeasonGrouper(["DJF", "MAM", "JJA", "SON"])).mean()

# Resample (time dimension only)
ds.resample(time="D").mean()    # daily
ds.resample(time="ME").mean()   # month-end
ds.resample(time="YE").mean()   # year-end
ds.resample(time="QE").mean()   # quarter-end
# NOTE: "M", "Y", "Q" are deprecated; use "ME", "YE", "QE"

# Rolling (overlapping windows)
ds.rolling(time=7, center=True).mean()
ds.rolling(time=7, min_periods=3).mean()  # allow partial windows

# Coarsen (non-overlapping blocks)
ds.coarsen(time=7, boundary="trim").mean()
ds.coarsen(x=2, y=2).mean()  # spatial downsampling

# Weighted operations
weights = np.cos(np.deg2rad(ds.lat))
ds.weighted(weights).mean(dim="lat")
```

### Data Combination
```python
# Concatenate along existing/new dimension
combined = xr.concat([ds1, ds2], dim="time")
combined = xr.concat([ds1, ds2], dim="ensemble")  # new dimension

# Merge different variables into one Dataset
merged = xr.merge([ds_temp, ds_precip])
merged = xr.merge([ds1, ds2], compat="override")  # skip checking

# Auto-combine from coordinates
combined = xr.combine_by_coords([ds1, ds2, ds3, ds4])

# Nested combine (explicit file layout)
combined = xr.combine_nested(
    [[ds00, ds01], [ds10, ds11]],
    concat_dim=["x", "y"],
)

# Align datasets to common coordinates
ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="outer")
# join: "inner", "outer", "left", "right", "exact"

# Broadcast
da1_b, da2_b = xr.broadcast(da1, da2)
```

## I/O Patterns

### NetCDF
```python
# Read
ds = xr.open_dataset("file.nc")
ds = xr.open_dataset("file.nc", engine="h5netcdf")  # alternative backend
ds = xr.open_dataset("file.nc", chunks={"time": 100})  # lazy with dask

# Write with encoding
encoding = {
    "temperature": {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "_FillValue": -9999.0,
        "chunksizes": (1, 100, 100),  # NetCDF4 chunk sizes
    },
    "time": {
        "units": "days since 1900-01-01",
        "calendar": "standard",
        "dtype": "float64",
    },
}
ds.to_netcdf("out.nc", encoding=encoding, format="NETCDF4")

# Multiple files
ds = xr.open_mfdataset(
    "data/*.nc",
    combine="by_coords",   # (default)
    parallel=True,          # use dask.delayed to open files in parallel
    compat="override",      # trust coords from first file
    coords="minimal",
    data_vars="minimal",
    engine="h5netcdf",      # h5netcdf can be faster for parallel reads
)
xr.save_mfdataset(datasets, paths)

# Append to unlimited dimension (NetCDF4)
ds.to_netcdf("out.nc", unlimited_dims=["time"])
# NOTE: appending to existing NetCDF is NOT natively supported (#1672)
# Use mode="a" with netCDF4 library directly for append workflows
```

### Zarr
```python
# Read (automatically uses dask if installed)
ds = xr.open_zarr("store.zarr")
ds = xr.open_zarr("s3://bucket/store.zarr")  # cloud storage

# Write
ds.to_zarr("store.zarr")
ds.to_zarr("store.zarr", mode="w")             # overwrite
ds.to_zarr("store.zarr", mode="a")             # modify existing arrays

# Append along dimension
ds.to_zarr("store.zarr", append_dim="time")

# Region write (update a slice of existing store)
ds.to_zarr("store.zarr", region={"time": slice(10, 20)})
ds.to_zarr("store.zarr", region="auto")  # auto-determine region

# Delayed write with dask
delayed = ds.to_zarr("store.zarr", compute=False)
delayed.compute()

# Zarr v3 format
ds.to_zarr("store.zarr", zarr_format=3)

# Consolidated metadata (faster open for many variables)
ds.to_zarr("store.zarr", consolidated=True)  # default

# DataTree to/from zarr
dt = xr.open_datatree("store.zarr", engine="zarr")
dt.to_zarr("out.zarr")
```

### Encoding Gotchas
```python
# GOTCHA: int16 with scale_factor decodes to float32, losing precision (#2304)
# netCDF4 library promotes to float64, but xarray uses float32 for int16
ds = xr.open_dataset("file.nc")
# Fix: explicitly control dtype in encoding or mask_and_scale=False
ds = xr.open_dataset("file.nc", mask_and_scale=False)

# GOTCHA: encoding persists through operations and can cause errors
ds = xr.open_zarr("input.zarr")
ds_subset = ds.isel(time=slice(0, 10))
# May fail because chunk encoding from original store is incompatible
ds_subset.to_zarr("output.zarr")  # ERROR
# Fix: drop encoding
ds_subset = ds_subset.drop_encoding()
ds_subset.to_zarr("output.zarr")

# GOTCHA: time encoding units must match data range
encoding = {"time": {"units": "days since 2000-01-01", "dtype": "float64"}}
# If time values are before 2000, negative values are fine
# But be careful with "hours since..." for long time series (precision loss)

# GOTCHA: _FillValue=0 in NetCDF will mask legitimate zeros
# To preserve actual zero values, use a different fill value
encoding = {"var": {"_FillValue": -9999}}

# View current encoding
print(ds["var"].encoding)
```

## Dask Integration

### Chunking Patterns
```python
# Read with chunks
ds = xr.open_dataset("file.nc", chunks={"time": 100, "x": 500, "y": 500})
ds = xr.open_dataset("file.nc", chunks="auto")    # let dask decide
ds = xr.open_zarr("store.zarr")                   # uses zarr chunks

# Rechunk
ds = ds.chunk({"time": -1, "x": 100})  # -1 = single chunk along dim
ds = ds.chunk(time=100)                 # keyword syntax also works

# Unify chunks across variables (important before writing)
ds = ds.unify_chunks()

# Check chunk sizes
print(ds.chunks)
print(ds["var"].chunks)
```

### Lazy Evaluation Patterns
```python
# All operations are lazy until .compute(), .load(), or .values
result = ds["temp"].mean(dim="time")  # builds task graph only
result_array = result.compute()       # triggers computation
result.load()                         # computes in-place

# Write lazily
delayed = ds.to_netcdf("out.nc", compute=False)
delayed.compute()

# Persist keeps data in distributed memory
ds = ds.persist()  # useful with dask.distributed

# Avoid: calling .values triggers immediate compute
# bad_idea = ds["temp"].values  # downloads entire array!
```

### Chunk Alignment with Zarr (#5065, #9914)
```python
# CRITICAL: dask chunks must be multiples of zarr chunks
# WRONG - misaligned chunks cause errors or corruption
ds = ds.chunk({"x": 150})  # if zarr chunks are 100, this fails
ds.to_zarr("out.zarr")

# RIGHT - ensure chunk alignment
ds = ds.chunk({"x": 200})  # multiple of zarr chunk size
ds.to_zarr("out.zarr")

# Or use safe_chunks=False to bypass (use with caution)
ds.to_zarr("out.zarr", safe_chunks=False)

# For sharded zarr: align to SHARD boundaries, not just internal chunks
```

## Integration

### With rioxarray (geospatial)
```python
import rioxarray
ds = xr.open_dataset("file.nc")
ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
ds = ds.rio.write_crs("EPSG:4326")
ds.rio.to_raster("output.tif")
```

## Time Series Handling

### Creating Time Coordinates
```python
# Standard datetime
times = xr.date_range("2000-01-01", periods=365, freq="D")
times = xr.date_range("2000-01-01", periods=365, freq="D", unit="s")

# Non-standard calendars (cftime)
times = xr.cftime_range("2000-01-01", periods=365, calendar="360_day")
times = xr.cftime_range("2000-01-01", periods=365, calendar="noleap")

# Match existing time coordinate (calendar is required)
new_times = xr.date_range_like(ds.time, calendar="standard")
```

### Time Accessors
```python
ds.time.dt.year
ds.time.dt.month
ds.time.dt.day
ds.time.dt.dayofyear
ds.time.dt.hour
ds.time.dt.season        # "DJF", "MAM", "JJA", "SON"
ds.time.dt.is_leap_year
ds.time.dt.strftime("%Y-%m-%d")
```

### CF Time Decoding
```python
# Control time decoding resolution (v2024.01+)
coder = xr.coders.CFDatetimeCoder(time_unit="s")
ds = xr.open_dataset("file.nc", decode_times=coder)

# DEPRECATED (2025.01+): use_cftime as a direct kwarg is deprecated
# WRONG:
ds = xr.open_dataset("file.nc", use_cftime=True)
# RIGHT - pass CFDatetimeCoder to decode_times instead:
ds = xr.open_dataset("file.nc", decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))

# Disable time decoding entirely
ds = xr.open_dataset("file.nc", decode_times=False)

# Decode manually later
ds = xr.decode_cf(ds)

# For non-standard calendars, xarray uses cftime automatically
# datetime64[ns] range: 1678-2262
# Outside this range -> cftime objects are used
```

## Gotchas & Common Mistakes

### Zarr fill_value != _FillValue (#5475)
Zarr's `fill_value` (background value for uninitialized chunks) is NOT the same
as CF's `_FillValue` (missing data indicator). xarray historically treated them
as identical. For zarr v3, `fill_value` is always required in metadata.
```python
# This causes valid 0 values to become NaN when reading zarr:
# int array with fill_value=0 (zarr default) -> 0 becomes NaN

# Workaround: disable masking
ds = xr.open_dataset("store.zarr", engine="zarr", mask_and_scale=False)

# Or for v3: xarray now uses _FillValue attribute instead of fill_value
```

### sel() slices are inclusive on both ends
```python
da = xr.DataArray([1, 2, 3, 4], dims=["x"], coords={"x": [10, 20, 30, 40]})
da.sel(x=slice(10, 30))  # Returns x=[10, 20, 30] -- includes 30!
# This differs from Python/numpy slicing where end is exclusive
```

### Automatic coordinate alignment silently introduces NaN
```python
# GOTCHA: a + b with partially overlapping coords fills gaps with NaN
a = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
b = xr.DataArray([10, 20, 30], dims=["x"], coords={"x": [1, 2, 3]})
result = a + b  # x=[0,1,2,3], with NaN at x=0 and x=3!

# Fix: explicitly align first to control behavior
a2, b2 = xr.align(a, b, join="inner")  # only keep shared coords
result = a2 + b2  # x=[1,2], no NaN

# Or use join="exact" to raise on mismatch
a3, b3 = xr.align(a, b, join="exact")  # raises ValueError
```

### Attributes are dropped by default in operations
```python
# Attributes are lost after arithmetic
result = da1 + da2  # result.attrs == {}

# Preserve with context manager
with xr.set_options(keep_attrs=True):
    result = da1 + da2  # result.attrs preserved
```

### Performance: avoid iterating with .sel/.isel (#2799)
```python
# WRONG - 1000x slower than numpy for element-wise access
for i in range(n):
    val = da.isel(x=i).values  # massive overhead per call

# RIGHT - extract numpy array first, then iterate
arr = da.values
for i in range(n):
    val = arr[i]

# BEST - use vectorized operations instead of loops
result = da.mean(dim="x")
```

### GroupBy with dask destroys chunk structure (#2237)
Non-contiguous groupby labels (like month-of-year) rechunk the entire array
into a single chunk. Install `flox` to fix this.
```python
# With flox installed, this preserves chunks:
anomaly = ds.groupby("time.month") - ds.groupby("time.month").mean()
```

### concat_dim requires combine='nested' (#2064)
```python
# WRONG - raises ValueError because concat_dim cannot be used with
# combine='by_coords' (the default)
ds = xr.open_mfdataset("*.nc", concat_dim="time")

# RIGHT - use combine='nested' when specifying concat_dim
ds = xr.open_mfdataset(
    "*.nc", combine="nested", concat_dim="time", data_vars="minimal",
)

# RIGHT - or omit concat_dim and let combine='by_coords' (default) auto-detect
ds = xr.open_mfdataset("*.nc", data_vars="minimal")
```

### Encoding not cleared on rechunk (#5065)
```python
# Reading zarr, rechunking, and writing to new zarr can fail
ds = xr.open_zarr("input.zarr")
ds = ds.chunk({"time": 50})
# Stale encoding from input causes chunk mismatch errors
ds.to_zarr("output.zarr")  # FAILS

# Fix: always drop_encoding() when changing chunk layout
ds = ds.drop_encoding()
ds.to_zarr("output.zarr")
```

### Testing utilities
```python
xr.testing.assert_equal(da1, da2)      # values + coords must match
xr.testing.assert_identical(da1, da2)  # values + coords + attrs must match
xr.testing.assert_allclose(da1, da2)   # approximate equality (for floats)
```

## Known Limitations & Bugs

- **No native append to NetCDF** (#1672): `to_netcdf` cannot append to existing files.
  Use `mode="a"` with the `netCDF4` library directly, or use zarr's `append_dim`.
- **MultiIndex serialization** (#1077): MultiIndex round-tripping to netCDF/zarr
  has edge cases. Use `ds.reset_index()` before writing if issues arise.
- **Zarr v3 fill_value masking** (#5475): Being actively fixed. For zarr v3,
  xarray now reads `_FillValue` from attributes, not zarr's `fill_value`.
- **Boolean zarr arrays**: With zarr v3, `fill_value` must be `true` or `false`,
  so `_FillValue`-based masking would make one boolean value unmaskable.
- **Subclassing** (#3980): xarray objects are not designed to be subclassed.
  Use accessors instead: `@xr.register_dataarray_accessor`.

## Performance Tips

- **Install flox** for 10-100x faster groupby/resample with dask arrays
- **Install bottleneck** for faster NaN-aware reductions (nanmean, etc.)
- **Use `engine="h5netcdf"`** for parallel NetCDF reads (thread-safe)
- **Use `chunks="auto"`** to let dask choose chunk sizes matching file layout
- **Use `open_mfdataset(..., parallel=True)`** to open files concurrently
- **Avoid repeated `.sel()`/`.isel()`** in loops; extract `.values` first (#2799)
- **Use `combine="by_coords", compat="override", coords="minimal"`** with
  `open_mfdataset` when files share identical non-concat coordinates (#1385)
- **Call `.unify_chunks()`** before writing to avoid chunk mismatch errors
- **Use `ds.drop_encoding()`** when re-writing data to a different store format
- **Prefer zarr over NetCDF** for cloud storage and parallel write workflows
- **Set `consolidated=True`** (default) when writing zarr for faster metadata reads
- **Avoid `.values` on large dask arrays** - use `.data` to keep lazy
