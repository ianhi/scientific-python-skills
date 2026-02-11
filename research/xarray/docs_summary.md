# XARRAY LIBRARY RESEARCH SUMMARY

## Version and Release Information

**Current Version:** 2026.01.0+ (development version, with unreleased v2026.02.0 in progress)
- Latest released: v2026.01.0 (January 28, 2026)
- Repository is actively maintained with regular commits
- Apache License 2.0
- Originally released May 2014, renamed from "xray" in January 2016

### Recent Changes (v2026.01.0)
- Improved DataTree HTML representation with collapsible groups
- DataArray.set_xindex and Dataset.set_xindex now automatically replace existing indexes
- Automatic temporary PandasIndex creation for 1D coordinate selection
- Minimum h5netcdf version now 1.4.0
- New `arithmetic_compat` option for coordinate comparison

## Core Data Model and Concepts

### Three Primary Data Structures

1. **DataArray** - xarray's labeled, multi-dimensional array
   - Properties: `values` (underlying array), `dims` (dimension names), `coords` (coordinate labels), `attrs` (metadata), `name` (optional)
   - Builds on NumPy with dimension names instead of just axis numbers
   - Coordinates enable fast label-based indexing similar to pandas
   
2. **Dataset** - dict-like container of aligned DataArray objects
   - Multi-dimensional generalization of pandas.DataFrame
   - All variables share a common coordinate system
   - Variables can have different dtypes and dimensions
   - Automatic alignment based on dimension names
   
3. **DataTree** - tree-like hierarchical container (newer, still evolving)
   - Organizes DataArrays into groups (like nested Datasets)
   - Similar to a recursive dict of Datasets
   - Coordinate inheritance down to children
   - Supports filesystem-like syntax for navigation

### Core Concepts

- **Dimensions**: Named axes (e.g., "time", "x", "y")
- **Coordinates**: Arrays labeling each point along dimensions
  - Dimension coordinates: 1D coordinates with same name as dimension
  - Non-dimension (auxiliary) coordinates: additional labels
- **Attributes**: Arbitrary metadata dictionaries (strictly user-level)
- **Encoding**: Information for serialization/deserialization (time units, compression, etc.)

## Public API Surface

### Top-Level Functions

**Data Creation:**
- `DataArray()`, `Dataset()`, `DataTree()` - constructors
- `full_like()`, `zeros_like()`, `ones_like()` - array factories

**I/O Operations:**
- `open_dataset()`, `load_dataset()` - read single netCDF
- `open_dataarray()`, `load_dataarray()` - read single DataArray
- `open_mfdataset()`, `save_mfdataset()` - multiple file operations
- `open_zarr()`, `open_datatree()`, `open_groups()` - other formats
- `load_datatree()` - load DataTree from file

**Data Combination:**
- `concat()` - concatenate along dimension
- `merge()` - combine multiple datasets by variables
- `combine_by_coords()`, `combine_nested()` - smart combining
- `align()` - align based on coordinates
- `broadcast()` - broadcast dimensions

**Computation:**
- `apply_ufunc()` - apply functions with core dimensions
- `map_blocks()` - parallel map over chunks (dask)
- `where()` - conditional selection
- `dot()`, `cross()` - linear algebra
- `corr()`, `cov()` - correlation/covariance
- `polyval()` - polynomial evaluation

**Time Functions:**
- `date_range()`, `cftime_range()`, `date_range_like()` - create time coordinates
- `decode_cf()` - decode CF conventions

**Other:**
- `show_versions()` - diagnostic info
- `set_options()`, `get_options()` - configuration
- `register_dataarray_accessor()`, `register_dataset_accessor()` - extension mechanism

### Key Submodules

- `xarray.backends` - I/O backends
- `xarray.coders` - encoding/decoding (CF, time, etc.)
- `xarray.groupers` - groupby grouper objects
- `xarray.indexes` - indexing implementations
- `xarray.testing` - testing utilities
- `xarray.tutorial` - example datasets
- `xarray.ufuncs` - universal functions (numpy dispatch)

## Indexing and Selection Patterns

### Four Indexing Modes

| Mode | Dimension | Index | DataArray Syntax | Dataset Syntax |
|------|-----------|-------|------------------|----------------|
| Positional by integer | Positional | By integer | `da[:, 0]` | N/A |
| Positional by label | Positional | By label | `da.loc[:, 'IA']` | N/A |
| By name | By name | By integer | `da.isel(space=0)` | `ds.isel(space=0)` |
| By name | By name | By label | `da.sel(space='IA')` | `ds.sel(space='IA')` |

### Key Methods

- `.loc[]` - label-based indexing (inclusive on both ends, like pandas)
- `.isel()` - integer selection by dimension name
- `.sel()` - label selection by dimension name
- `.head()`, `.tail()`, `.thin()` - convenience slicing
- `.reindex()`, `.reindex_like()` - reindex to new coordinates
- `.interp()`, `.interp_like()` - interpolation

### Advanced Indexing

- Boolean indexing: `da[da > 0]`
- Array indexing: `da.isel(x=[0, 2, 4])`
- Slice objects: `da.sel(time=slice('2000-01-01', '2000-12-31'))`
- Label ranges: `da.loc['2000-01':'2000-12']` (inclusive)
- Multi-dimensional: `da.sel(x=2, y=[10, 20, 30])`

Note: v2026.01.0 now automatically creates temporary PandasIndex for 1D coordinate selection without explicit indexes

## GroupBy, Resample, and Rolling Operations

### GroupBy (Split-Apply-Combine)

```python
# Basic groupby
ds.groupby('letter').mean()
ds.groupby('letter')['variable']  # access specific group

# Binning
ds.groupby_bins('x', bins=[0, 25, 50])

# Groupby with multiple variables (requires flox package for best performance)
# Available through xarray.groupers
```

**Key Methods:**
- `.groups` - see group labels
- `.map()` - apply function to each group
- `.mean()`, `.sum()`, `.std()` - aggregations
- `.reduce()` - custom reduction function

**Performance Note:** Install `flox` package for substantially improved groupby performance, especially with dask

### Resample (Time Series)

```python
ds.resample(time='1D').mean()  # daily mean
ds.resample(time='1M').sum()   # monthly sum
```

**Frequency Strings:** 'D' (day), 'W' (week), 'M' (month), 'Q' (quarter), 'Y' (year), 'h' (hour), 'm' (minute), 's' (second)

### Rolling and Coarsen Windows

```python
ds.rolling(time=7).mean()      # 7-element rolling window
ds.coarsen(time=7).mean()      # 7-element non-overlapping average
```

## I/O Patterns and Backends

### Primary Formats

**NetCDF (Recommended for xarray):**
```python
# Read
ds = xr.open_dataset('file.nc')
ds = xr.open_dataset('file.nc', engine='netcdf4')  # explicit engine

# Write
ds.to_netcdf('output.nc')
ds.to_netcdf('output.nc', encoding={'var': {'zlib': True, 'complevel': 4}})

# Multiple files
ds = xr.open_mfdataset('data/*.nc')
xr.save_mfdataset(datasets, paths=['file1.nc', 'file2.nc'])
```

**Zarr (Cloud Storage Optimized):**
```python
# Read - loads as dask array by default if dask installed
ds = xr.open_zarr('path/to/data.zarr')

# Write
ds.to_zarr('path/to/data.zarr')
ds.to_zarr('s3://bucket/path.zarr')  # cloud storage

# Read with dask
ds = xr.open_zarr('path/to/data.zarr')  # already chunked
```

**HDF5:**
```python
ds = xr.open_dataset('file.h5', engine='h5netcdf')
```

### Backend Selection

Xarray automatically tries engines in order: **netcdf4 → h5netcdf → scipy → pydap → zarr**

Configure with: `xr.set_options(netcdf_engine_order=['h5netcdf', 'netcdf4', 'scipy'])`

### Encoding/Decoding

```python
# CF convention decoding (automatic on read)
ds = xr.decode_cf(ds)

# Time decoding with specific resolution
coder = xr.coders.CFDatetimeCoder(time_unit='s')
ds = xr.decode_cf(ds, decode_times=coder)

# Manual encoding specification
encoding = {
    'temperature': {
        'zlib': True,
        'complevel': 4,
        'units': 'K',
        'long_name': 'Surface Temperature'
    }
}
ds.to_netcdf('file.nc', encoding=encoding)
```

## Computation Patterns

### Basic Operations

```python
# Arithmetic (broadcasts by dimension name)
result = da1 + da2  # dimensions align automatically

# Aggregations (by dimension name)
mean = da.mean(dim='time')
sum_all = da.sum()

# Element-wise functions
result = np.sin(da)
result = xr.where(da > 0, 'positive', 'negative')

# Linear algebra
dot_product = da1 @ da2
correlation = xr.corr(da1, da2, dim='time')
covariance = xr.cov(da1, da2, dim='time')
```

### apply_ufunc - Core Computation Pattern

```python
def custom_func(arr):
    # arr has only core dimensions
    return np.mean(arr, axis=-1)

result = xr.apply_ufunc(
    custom_func,
    da,
    input_core_dims=[['x', 'y']],
    output_core_dims=[['x']],
    vectorize=True,  # handle broadcast dims
    dask='parallelized',  # for dask arrays
    output_dtypes=[float]
)
```

**Key Parameters:**
- `input_core_dims` - dimensions to pass to function
- `output_core_dims` - expected output dimensions
- `vectorize` - automatically broadcast non-core dims
- `dask` - 'parallelized', 'forbidden', or None
- `output_dtypes` - specify output types for dask
- `keep_attrs` - preserve metadata
- `kwargs` - pass additional arguments to function

### Computation with Dask

```python
# Read with chunking
ds = xr.open_dataset('big_file.nc', chunks={'time': 10})

# Operations are lazy
result = ds['temp'].mean(dim='x')  # not computed yet

# Force computation when needed
computed = result.compute()

# Persist to avoid recomputation
persisted = ds.persist()

# Write with dask
ds.to_netcdf('output.nc')  # dask chunks preserved
ds.to_zarr('output.zarr')  # optimal for zarr
```

## Dask Integration

### Key Concepts

- **Chunking**: Dask divides arrays into manageable pieces
- **Lazy Evaluation**: Operations queue up as tasks
- **Parallelism**: Chunks processed independently
- **Transparent**: Most xarray methods work automatically with dask

### Common Patterns

```python
# Specify chunks on read
ds = xr.open_dataset('file.nc', chunks={'time': 100})
ds = xr.open_zarr('data.zarr')  # loads as dask by default

# Manual chunking
ds_chunked = ds.chunk({'time': 50, 'space': 100})

# Dask-aware operations
ds.groupby('letter').mean()  # works with dask
ds.rolling(time=7).mean()    # works with dask

# Control computation
delayed = ds.to_netcdf('output.nc', compute=False)
result = delayed.compute()

# Rechunk before write
ds = ds.unify_chunks()
ds.to_netcdf('output.nc')
```

## Time Series Data Handling

### Creating Time Coordinates

```python
# Using pandas
times = pd.date_range('2000-01-01', periods=365)
ds = xr.Dataset({'var': ('time', data)}, coords={'time': times})

# Using xarray
times = xr.date_range('2000-01-01', periods=365)
times = xr.date_range('2000-01-01', periods=365, unit='s')  # specify resolution

# Using cftime for non-standard calendars
times = xr.cftime_range('2000-01-01', periods=365, calendar='360_day')
```

### Datetime Indexing

```python
# String indexing
ds.sel(time='2000-01-01')
ds.sel(time='2000-01')  # entire month
ds.sel(time=slice('2000-01-01', '2000-12-31'))  # inclusive

# Time accessors
ds.time.dt.year
ds.time.dt.month
ds.time.dt.dayofyear

# Inferring frequency
freq = xr.infer_freq(ds.time)
```

### Time Encoding

```python
# CF convention: "days since 2000-01-01"
# Stored in attrs: units, calendar

# Automatic encoding/decoding on read/write
ds = xr.open_dataset('file.nc')  # automatically decoded

# Control decoding resolution
coder = xr.coders.CFDatetimeCoder(time_unit='s')
ds = xr.decode_cf(raw_ds, decode_times=coder)

# Non-standard calendars use cftime
# Resolution limitations: datetime64[ns] covers 1678-2262
# Older/newer dates or non-Gregorian calendars use CFTimeIndex
```

## Reshaping and Reorganizing Data

### Dimension Operations

```python
# Transpose
ds.transpose('y', 'x', 'time')
ds.transpose(..., 'time')  # move time to end

# Expand dimensions
ds.expand_dims('new_dim')
da.expand_dims(z=3)  # expand with size 3

# Squeeze out size-1 dimensions
ds.squeeze('dim')
da.squeeze()  # squeeze all size-1 dims

# Swap dimensions
ds.swap_dims({'x': 'space'})

# Rename
ds.rename({'old_name': 'new_name'})
ds.rename_vars({'var1': 'temperature'})
ds.rename_dims({'x': 'lon', 'y': 'lat'})
```

### Convert Between Structures

```python
# Dataset to DataArray
arr = ds.to_dataarray()  # combines variables along new dim

# DataArray to Dataset
ds = da.to_dataset(name='temperature')
ds = da.to_dataset(dim='variable')  # use as new dimension

# To/from pandas
series = da.to_series()
df = ds.to_dataframe()
da = df.to_xarray()

# To/from NumPy
arr = da.to_numpy()
values = da.values
```

### Stack/Unstack

```python
# Stack multiple dimensions
stacked = ds.stack(location=('x', 'y'))

# Unstack back
unstacked = stacked.unstack('location')

# With MultiIndex support
ds = ds.set_index(location=('x', 'y'))
```

## Zarr-Specific Information

### Zarr Encoding Specification

**Format Versions:**
- **Zarr V2**: Uses `_ARRAY_DIMENSIONS` attribute for dimension names
- **Zarr V3**: Uses native `dimension_names` field (official spec)
- **NCZarr**: Supported for reading; uses `dimrefs` field

**Consolidated Metadata:**
```python
# Default behavior
ds.to_zarr('data.zarr')  # writes .zmetadata file

# Disable if needed
ds.to_zarr('data.zarr', consolidated=False)
```

**Cloud Storage:**
```python
# S3 example
ds.to_zarr('s3://bucket/path.zarr', consolidated=True)

# Requires fsspec and s3fs
```

**Sharded Zarr Warning:**
- Dask chunk boundaries must align with shard boundaries
- Not just internal Zarr chunk boundaries
- Can cause silent data corruption if misaligned

## Plotting API

### Basic Plotting

```python
# Line plot
da.plot()
da.plot.line()

# 2D plots
da.plot.pcolormesh()
da.plot.contourf()
da.plot.imshow()

# Multiple variables
ds.plot()
ds['temp'].plot()
```

### Customization

```python
# Using matplotlib attributes
da.plot(figsize=(10, 5), cmap='viridis', vmin=0, vmax=100)

# Attributes for labels
da.attrs['long_name'] = 'Temperature'
da.attrs['units'] = 'K'
da.plot()  # automatically labels from attrs

# Dataset plotting
ds.plot(col='variable')  # subplot per variable
ds.plot(row='lat')       # subplot per latitude
```

### Faceting

```python
# Multiple subplots
ds['temp'].plot(col='month', row='region', figsize=(12, 8))
```

## Extending and Accessors

### Register Custom Accessors

```python
@xr.register_dataarray_accessor("custom")
class CustomAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def do_something(self):
        return self._obj.mean()

# Usage
result = da.custom.do_something()
```

## Missing Value Handling

```python
# Detect missing values
mask = da.isnull()
mask = da.notnull()

# Count non-missing
count = da.count(dim='time')

# Remove missing
clean = da.dropna(dim='x')

# Fill missing
filled = da.fillna(0)
forward_fill = da.ffill(dim='time')
backward_fill = da.bfill(dim='time')

# Conditional replacement
result = xr.where(da > threshold, da, fill_value)
```

## Options and Configuration

### Available Options

```python
# Control display
xr.set_options(
    display_expand_data=False,  # truncate large arrays
    display_expand_coords=True,
    display_expand_attrs=True,
    display_expand_data_vars=True,
    display_max_rows=12,
    display_style='html'  # or 'text'
)

# Control behavior
xr.set_options(
    arithmetic_join='exact',      # how to join coordinates on ops
    keep_attrs=False,             # preserve attrs in operations
    use_bottleneck=True,          # use bottleneck library
    arithmetic_compat=False       # coordinate comparison on ops
)

# I/O configuration
xr.set_options(
    file_cache_maxsize=128,
    netcdf_engine_order=['netcdf4', 'h5netcdf', 'scipy'],
    warn_on_unclosed_files=True
)

# Plotting
xr.set_options(
    cmap_divergent='RdBu_r',
    cmap_sequential='viridis'
)

# Use as context manager
with xr.set_options(keep_attrs=True):
    result = ds1 + ds2
```

## Deprecations and Breaking Changes

### Recent Changes (v2026.01.0+)

**Breaking Change in v2026.02.0:**
- `FutureWarning` now used consistently for all deprecations (previously mixed with `DeprecationWarning`)

**New in v2026.01.0:**
- `set_xindex()` now replaces existing indexes automatically
- Automatic temporary PandasIndex creation for 1D coordinate selection
- h5netcdf minimum version: 1.4.0
- DataTree HTML representation improved

### General Deprecation Strategy

- Deprecations announced via FutureWarning
- Usually 2-3 release cycles of warning before removal
- Check release notes for migration paths

## Key Data Structures Summary

### DataArray Constructor
```python
xr.DataArray(
    data=array_like,
    coords={
        'dim1': coord_array,
        'dim2': coord_array,
        'non_dim_coord': (('dim1', 'dim2'), array)
    },
    dims=['dim1', 'dim2'],
    attrs={'units': 'K', 'long_name': 'Temperature'},
    name='temperature'
)
```

### Dataset Constructor
```python
xr.Dataset(
    data_vars={
        'var1': (('x', 'y'), array1),
        'var2': (('x',), array2)
    },
    coords={
        'x': x_coord,
        'y': y_coord
    },
    attrs={'source': 'model_output'}
)
```

### DataTree Constructor
```python
xr.DataTree(
    name='root',
    ds=xr.Dataset(...),  # data at this node
    children={
        'child1': DataTree(...),
        'child2': DataTree(...)
    }
)
```

## Integration Ecosystems

### Primary Integrations

- **Dask**: Lazy/parallel computation on large datasets
- **Zarr**: Cloud-optimized chunked storage
- **netCDF**: Scientific data standard
- **Pandas**: Interoperability for DataFrames
- **NumPy**: Underlying array operations
- **Matplotlib**: Plotting
- **CF Conventions**: Metadata standards

### Specialized Extensions

- **rioxarray**: Geospatial raster data (GeoTIFF, etc.)
- **flox**: Optimized groupby operations
- **nc-time-axis**: cftime plotting support
- **Cartopy**: Cartographic visualization
- **HoloViews/GeoViews**: Interactive dashboards

---

## Key References to Core Files

**Core Data Structures:**
- `/xarray/core/dataarray.py` - DataArray implementation
- `/xarray/core/dataset.py` - Dataset implementation
- `/xarray/core/variable.py` - Variable and IndexVariable classes
- `/xarray/core/coordinates.py` - Coordinate management

**I/O and Backends:**
- `/xarray/backends/api.py` - open_dataset, open_zarr, etc.
- `/xarray/backends/` - netcdf4, h5netcdf, zarr, pydap implementations

**Computation:**
- `/xarray/computation/apply_ufunc.py` - apply_ufunc implementation
- `/xarray/core/groupby.py` - groupby operations
- `/xarray/computation/rolling.py` - rolling/coarsen operations

**Time Handling:**
- `/xarray/coding/cftimeindex.py` - cftime support
- `/xarray/coding/cftime_offsets.py` - time ranges

**Indexing:**
- `/xarray/core/indexing.py` - indexing machinery
- `/xarray/core/indexes.py` - Index implementations

---

This comprehensive research covers xarray's complete public API, core concepts, common patterns, and integration points. The library is production-ready for scientific computing workflows involving multi-dimensional labeled arrays, with particular strength in geoscience and climate data processing.