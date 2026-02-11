# Zarr - Claude Code Skill

> Skill for writing idiomatic zarr v3 code. Current version: 3.1.5+ (Python 3.11+, NumPy 2.0+)

## CRITICAL: Do NOT Use These Outdated Patterns

Zarr v3 is a complete rewrite. Claude frequently generates v2 code that will fail silently or raise errors in v3.

### 1. Do NOT import codecs from zarr namespace
```python
# WRONG (v2) - these do not exist
from zarr import Blosc
from zarr import Zlib, Delta

# RIGHT (v3 arrays)
from zarr.codecs import BloscCodec, ZstdCodec, GzipCodec, BytesCodec

# RIGHT (v2 arrays only)
from numcodecs import Blosc, Zlib, Delta
```

### 2. Do NOT construct Array/Group directly
```python
# WRONG - constructor signatures completely changed in v3
arr = zarr.Array(store, shape=(100,), dtype='f4')
grp = zarr.Group(store)

# RIGHT
arr = zarr.create_array(store=store, shape=(100,), dtype='f4')
arr = zarr.open_array('data.zarr', mode='r')
grp = zarr.create_group(store='data.zarr')
grp = zarr.open_group('data.zarr', mode='r')
```

### 3. Do NOT use dot notation for group members
```python
# WRONG - not supported in v3
arr = group.array_name
subgrp = group.subgroup_name

# RIGHT
arr = group['array_name']
arr = group['path/to/nested/array']
```

### 4. Do NOT use v2 store classes
```python
# WRONG - removed in v3
from zarr import DirectoryStore, FSStore, MemoryStore as V2MemoryStore
from zarr import DBMStore, LMDBStore, SQLiteStore, N5Store

# RIGHT
from zarr.storage import LocalStore, MemoryStore, ZipStore, FsspecStore, ObjectStore
```

### 5. Do NOT use compressor=/filters= with v3 format
```python
# WRONG - v2 keyword args on v3 array
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      compressor=Blosc(), filters=[Delta()])

# RIGHT - v3 uses codecs= or compressors=/serializer=/filters=
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])
# Or simpler with compressors shorthand (v3 keyword, not v2 compressor):
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      compressors=zarr.codecs.ZstdCodec(level=3))
```

### 6. Do NOT use FSMap with v3 (#2706)
```python
# WRONG - FSMap no longer accepted as store
import fsspec
mapper = fsspec.get_mapper('s3://bucket/data.zarr')
z = zarr.open(store=mapper)  # TypeError: Unsupported type for store_like: 'FSMap'

# RIGHT - use FsspecStore or pass URL directly
store = zarr.storage.FsspecStore.from_url('s3://bucket/data.zarr',
                                          storage_options={'anon': True})
z = zarr.open_array(store=store, mode='r')

# Or pass URL string directly (simplest)
z = zarr.open_array('s3://bucket/data.zarr', mode='r',
                    storage_options={'anon': True})
```

### 7. Do NOT use resize with separate args
```python
# WRONG (v2)
z.resize(1000, 1000)

# RIGHT (v3) - pass tuple
z.resize((1000, 1000))
```

### 8. Do NOT mix v2/v3 codec types
```python
# WRONG - passing numcodecs objects to v3 array creation
import numcodecs
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      compressors=[numcodecs.Blosc()])  # TypeError (#2964)

# RIGHT - use zarr.codecs for v3
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      compressors=zarr.codecs.BloscCodec(cname='zstd'))
# numcodecs objects are only for zarr_format=2 via legacy zarr.create()
z = zarr.create(shape=(100,), dtype='f4', store=store,
                compressor=numcodecs.Blosc(), zarr_format=2)
```

### 9. Do NOT use v2 object dtype or assume v2 dtype encoding for v3
```python
# BROKEN - object dtype does not exist in v3
z = zarr.create_array(store, shape=(100,), dtype='|O')    # fails in v3

# PREFER named dtypes for clarity (endian-prefixed like '<i4' may work
# via numpy parsing, but named dtypes are the v3 convention)
z = zarr.create_array(store, shape=(100,), dtype='int32')    # clear v3 style
z = zarr.create_array(store, shape=(100,), dtype='float64')  # clear v3 style
z = zarr.create_array(store, shape=(100,), dtype='string')   # vlen strings (v3)
```

### 10. Do NOT use removed modules
```python
# WRONG - all removed in v3
from zarr import attrs, context, hierarchy, indexing, meta, sync, util
from zarr.sync import Synchronizer
from zarr import n5  # use n5py instead

# RIGHT - v3 replacements
from zarr import Array, Group, config
from zarr.storage import LocalStore, MemoryStore
```

## Quick Reference

```python
import zarr
import zarr.codecs
from zarr.storage import LocalStore, MemoryStore, FsspecStore
import numpy as np
# Install extras: pip install zarr[remote] (S3/GCS/Azure), zarr[gpu] (cupy)
```

### Common Operations
```python
# In-memory array
z = zarr.create_array(store={}, shape=(1000, 1000), chunks=(100, 100), dtype='float32')
z[:] = np.random.random((1000, 1000))

# Disk array
z = zarr.create_array(store='data.zarr', shape=(1000,), dtype='f4')
z2 = zarr.open_array('data.zarr', mode='r+')

# Save/load numpy convenience
zarr.save_array('data.zarr', np_array)
loaded = zarr.load('data.zarr')

# From existing array (store is first arg, data is keyword-only)
z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))

# Group hierarchy
root = zarr.open_group('data.zarr', mode='w')
grp = root.create_group('measurements')
arr = grp.create_array('temperature', shape=(365, 24), dtype='float32')
arr2 = root['measurements/temperature']  # access by path
```

## Core API

### zarr.create_array()
```python
zarr.create_array(
    store: StoreLike,          # path str, dict, None, Store instance, URL
    *,
    name: str | None = None,   # path within store (None = root)
    shape: ShapeLike | None = None,  # None if data is provided
    dtype: ZDTypeLike | None = None, # None if data is provided
    data: np.ndarray | None = None,  # initialize with data (sets shape/dtype)
    chunks: tuple[int, ...] | Literal['auto'] = 'auto',
    shards: ShardsLike | None = None,  # v3 only: tuple or 'auto'
    compressors: CompressorsLike = 'auto',   # shorthand (v3)
    serializer: SerializerLike = 'auto',     # shorthand (v3)
    filters: FiltersLike = 'auto',           # array-to-array transforms (v3)
    fill_value: Any = DEFAULT_FILL_VALUE,
    attributes: dict = None,
    dimension_names: DimensionNames = None,
    zarr_format: ZarrFormat | None = 3,      # 2 or 3 (default: 3)
    overwrite: bool = False,
    config: ArrayConfigLike = None,
) -> Array
# NOTE: zarr.create() is a legacy wrapper with v2-compatible params (compressor=, filters=).
# Prefer zarr.create_array() for new code.
```

### zarr.open_array()
```python
zarr.open_array(
    store: StoreLike = None,
    *,
    zarr_format: ZarrFormat = None,
    path: str = '',
    storage_options: dict = None,
    **kwargs,  # mode is passed here: mode='r'|'r+'|'a'|'w'|'w-'
) -> Array
# NOTE: mode is a hidden kwarg (not an explicit parameter).
# IDEs will not autocomplete it, but it works: zarr.open_array('data.zarr', mode='r')
```

### zarr.open_group() / zarr.create_group()
```python
zarr.open_group(
    store: StoreLike = None,
    *,
    mode: AccessModeLiteral = 'a',
    path: str = None,
    zarr_format: ZarrFormat = None,
    use_consolidated: bool | str = None,
    attributes: dict = None,
    storage_options: dict = None,
) -> Group

zarr.create_group(
    store: StoreLike,
    *,
    path: str = None,
    zarr_format: ZarrFormat = None,
    overwrite: bool = False,
    attributes: dict = None,
) -> Group
```

### Array Methods
```python
z[:]                        # read all
z[0:100, 50:150]            # slice
z[:] = data                 # write all
z[0, :] = row_data          # write slice
z.resize((new_dim0, ...))   # resize (tuple!)
z.append(data, axis=0)      # append along axis
z.attrs['key'] = 'value'    # JSON-serializable attrs
z.info                      # quick diagnostics
z.info_complete()           # detailed diagnostics
z.nchunks_initialized       # count stored chunks
```

### Group Methods
```python
grp['child']                         # access child by path
grp.create_array('name', ...)        # create child array
grp.create_group('name')             # create child group
grp.require_array('name', ...)       # get-or-create array
list(grp.members())                  # iterate (key, value) pairs
grp.tree()                           # display hierarchy (needs rich)
```

## Patterns & Idioms

### Store Configuration
```python
# Local filesystem
store = zarr.storage.LocalStore('path/to/data', mode='w')

# In-memory (testing/demos)
store = zarr.storage.MemoryStore()
# Or pass dict/None:
z = zarr.create_array(store={}, shape=(100,), dtype='f4')

# Zip archive (must close explicitly!)
store = zarr.storage.ZipStore('data.zip', mode='w')
# ... write data ...
store.close()  # REQUIRED to flush

# Cloud via fsspec (S3, GCS, Azure)
store = zarr.storage.FsspecStore.from_url(
    's3://bucket/data.zarr',
    storage_options={'anon': True}
)

# Cloud via obstore (experimental, Rust-based, faster)
from obstore.store import S3Store as ObS3
store = zarr.storage.ObjectStore(ObS3(bucket='mybucket', region='us-east-1'))

# String/URL shortcuts - zarr auto-resolves:
z = zarr.open_array('path/to/local.zarr')       # -> LocalStore
z = zarr.open_array('s3://bucket/data.zarr',     # -> FsspecStore
                    storage_options={'anon': True})
```

### Codec/Compression Patterns (v3)
```python
# Default: BytesCodec + ZstdCodec (good default, just works)
z = zarr.create_array(store, shape=(1000,), dtype='f4')

# Explicit codec pipeline
z = zarr.create_array(
    store, shape=(1000,), dtype='f4',
    codecs=[
        zarr.codecs.BytesCodec(endian='little'),
        zarr.codecs.BloscCodec(cname='zstd', clevel=5,
                               shuffle=zarr.codecs.BloscShuffle.shuffle),
    ]
)

# Shorthand (compressors= is a v3 convenience, NOT v2 compressor=)
z = zarr.create_array(
    store, shape=(1000,), dtype='f4',
    compressors=zarr.codecs.ZstdCodec(level=3)
)

# No compression
z = zarr.create_array(store, shape=(1000,), dtype='f4',
                      compressors=None)

# Checksum only
z = zarr.create_array(
    store, shape=(1000,), dtype='f4',
    codecs=[zarr.codecs.BytesCodec(), zarr.codecs.Crc32cCodec()]
)

# Variable-length strings
z = zarr.create_array(store, shape=(100,), dtype='string')
# Uses VLenUTF8Codec + ZstdCodec automatically
```

### Sharding
```python
# Sharding groups many small chunks into larger storage objects
# Critical for cloud/object stores with many small chunks
z = zarr.create_array(
    store='data.zarr',
    shape=(100_000, 100),
    chunks=(100, 100),       # logical chunks (access granularity)
    shards=(10_000, 100),    # physical shards (100 chunks per shard)
    zarr_format=3,           # sharding is v3 only
)

# Auto-sharding (config.array.target_shard_size_bytes defaults to None;
# when None, uses heuristic: shard if array_dim // chunk_dim > 8)
z = zarr.create_array(
    store='data.zarr',
    shape=(100_000, 100),
    chunks=(100, 100),
    shards='auto',
)

# Explicit ShardingCodec in pipeline
z = zarr.create_array(
    store='data.zarr',
    shape=(1000, 1000),
    dtype='f4',
    codecs=[
        zarr.codecs.ShardingCodec(
            chunk_shape=(100, 100),
            codecs=[zarr.codecs.BytesCodec(), zarr.codecs.ZstdCodec()],
            index_location='end',
        )
    ],
    chunks=(1000, 1000),  # shard shape = chunks when using ShardingCodec directly
)
```

### Consolidated Metadata
```python
# Consolidate: store all metadata in root (avoids per-array reads)
root = zarr.open_group('data.zarr', mode='w')
for i in range(100):
    root.create_array(f'arr_{i}', shape=(1000,), dtype='f4')

# Write consolidated metadata
zarr.consolidate_metadata('data.zarr')

# Read with consolidated metadata (fast open, no per-array I/O)
root = zarr.open_group('data.zarr')  # auto-detects consolidated
# Or explicitly:
root = zarr.open_consolidated('data.zarr')
# Or force:
root = zarr.open_group('data.zarr', use_consolidated=True)

# Access arrays - metadata already loaded, no store read needed
arr = root['arr_50']
```

### Configuration
```python
# Global config
zarr.config.set({'array.order': 'C'})
zarr.config.set({'async.concurrency': 10})
zarr.config.set({'threading.max_workers': 4})

# Temporary context
with zarr.config.set({'array.write_empty_chunks': True}):
    z = zarr.create_array(...)

# Per-array config
z = zarr.create_array(
    store, shape=(1000,), dtype='f4',
    config={'write_empty_chunks': False, 'order': 'C'}
)

# Environment variables: ZARR_<SECTION>__<KEY>=value
# ZARR_ARRAY__ORDER=F
# ZARR_ASYNC__CONCURRENCY=8
```

## Integration

### With xarray
```python
import xarray as xr

# Read zarr store
ds = xr.open_zarr('data.zarr')
ds = xr.open_zarr('s3://bucket/data.zarr', storage_options={'anon': True})

# Write zarr store
ds.to_zarr('output.zarr', mode='w')

# Consolidated metadata (recommended for cloud)
ds.to_zarr('output.zarr', mode='w')
zarr.consolidate_metadata('output.zarr')
ds = xr.open_zarr('output.zarr')  # uses consolidated metadata

# With explicit zarr group
store = zarr.storage.FsspecStore.from_url('s3://bucket/data.zarr',
                                          storage_options={'anon': True})
ds = xr.open_zarr(store, consolidated=True)
```

### With Dask
```python
import dask.array as da

# Read zarr into dask (lazy)
z = zarr.open_array('data.zarr', mode='r')
darr = da.from_zarr('data.zarr')

# Write dask to zarr
da.to_zarr(darr, 'output.zarr')

# Note: passing dask arrays directly to zarr may raise TypeError (#962)
# Use da.to_zarr() or compute first
```

### Cloud Storage with Caching (#2988)
```python
# fsspec caching requires fsspec >= 2025.7.0 for v3 compatibility
root = zarr.open_consolidated(
    "simplecache::gs://bucket/data.zarr",
    storage_options={
        "gs": {"asynchronous": True},
        "simplecache": {"cache_storage": "/local/cache/dir"},
    }
)
# Note: caching was broken with zarr v3 before fsspec 2025.7.0
```

## Gotchas & Common Mistakes

### from_array() has store-first, data keyword-only signature
```python
# WRONG - data is NOT the first argument
z = zarr.from_array(np_array, store='data.zarr')  # TypeError

# RIGHT - store is the first positional arg, data is keyword-only
z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))
```

### Blosc compression ratio degradation v2 vs v3 (#2171, #2766)
Blosc typesize defaults to 1 byte in v3 (instead of dtype size), causing 10-20x worse compression with shuffle.
```python
# BAD - shuffle is ineffective because typesize=1 internally
z = zarr.create_array(store, shape=(1000,), dtype='float64',
                      codecs=[zarr.codecs.BytesCodec(),
                              zarr.codecs.BloscCodec(cname='lz4', clevel=1,
                                                     shuffle=zarr.codecs.BloscShuffle.shuffle)])
# Compressed chunks may be 10-20x larger than equivalent v2

# WORKAROUND - use ZstdCodec (default) which is not affected
z = zarr.create_array(store, shape=(1000,), dtype='float64',
                      compressors=zarr.codecs.ZstdCodec(level=3))
# Or explicitly set typesize in BloscCodec if the fix has landed
```

### write_empty_chunks default changed (#853)
```python
# v2 default: write_empty_chunks=True (stores all chunks)
# v3 default: write_empty_chunks=False (skips fill-value-only chunks)
# This saves storage but means missing chunks return fill_value silently

# If you need all chunks written (e.g., for external tools):
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      config={'write_empty_chunks': True})
```

### Missing chunks return fill_value, not errors (#486)
```python
# Reading a chunk that was never written returns fill_value (default 0)
# This is BY DESIGN - zarr is sparse
# If you need to distinguish "never written" from "written with fill_value":
print(z.nchunks_initialized)  # check how many chunks actually exist
```

### Mode semantics changed (#3062)
```python
# DANGER: mode='a' (default for open_array) was buggy before 3.0.8
# It could DELETE existing data. Always use 3.0.8+

# Safe mode choices:
zarr.open_array('data.zarr', mode='r')    # read only
zarr.open_array('data.zarr', mode='r+')   # read/write existing (fails if missing)
zarr.open_array('data.zarr', mode='w-')   # create only (fails if exists)
zarr.open_array('data.zarr', mode='w')    # overwrite (DESTRUCTIVE)
zarr.open_array('data.zarr', mode='a')    # append/create (safe in 3.0.8+)
```

### create_dataset deprecation (#2689)
```python
# Group.create_dataset() is deprecated but still available for h5py compat
# h5py API compatibility is NOT a design goal for v3
grp.create_dataset('name', data=data)  # works but deprecated

# Preferred:
grp.create_array('name', shape=data.shape, dtype=data.dtype, data=data)
# Or:
arr = zarr.from_array(store, data=data)
```

### Attributes must be JSON-serializable
```python
# WRONG
z.attrs['timestamp'] = datetime.now()     # not JSON-serializable
z.attrs['data'] = np.array([1, 2, 3])    # not JSON-serializable

# RIGHT
z.attrs['timestamp'] = datetime.now().isoformat()
z.attrs['data'] = [1, 2, 3]
z.attrs['config'] = {'rate': 0.01, 'steps': 1000}
```

### zarr.codecs.Blosc is NOT numcodecs.Blosc
```python
# CAUTION: zarr.codecs exports numcodecs WRAPPERS that look like the originals
from zarr.codecs.numcodecs import Blosc   # v3 wrapper class, NOT numcodecs.Blosc
from numcodecs import Blosc               # raw numcodecs class

# These are DIFFERENT types! Do not interchange them.
# For v3 arrays, use the native v3 codecs:
from zarr.codecs import BloscCodec, ZstdCodec  # native v3 codecs (preferred)
# The zarr.codecs.numcodecs wrappers exist for edge cases only.
```

### FsspecStore requires async filesystem (#2706)
```python
# WRONG - local filesystem is not async
import fsspec
fs = fsspec.filesystem("file")
store = zarr.storage.FsspecStore(fs, path="/local/path")  # may fail

# RIGHT - use LocalStore for local files
store = zarr.storage.LocalStore("/local/path")

# RIGHT - for cloud, use asynchronous=True
import s3fs
fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
store = zarr.storage.FsspecStore(fs, path="bucket/data.zarr")
```

## Known Limitations & Bugs

- **Structured dtypes not supported in v3** (#2134): `dtype=[('x', 'f4'), ('y', 'f4')]` works in v2 but not v3. Use separate arrays.
- **copy/copy_all incomplete** in v3: Use `zarr.from_array()` as workaround for array copying.
- **Performance regression in v3** (#2710): Large array writes can be slower due to async overhead and `all_equal` check. Being actively addressed.
- **Blosc shuffle typesize bug** (#2171, #2766): Blosc gets typesize=1 instead of actual dtype size, degrading shuffle effectiveness. Use ZstdCodec as default compressor to avoid.
- **fsspec caching broken before fsspec 2025.7.0** (#2988): `simplecache::` and `filecache::` protocols did not work with zarr v3's async store. Fixed in fsspec 2025.7.0.
- **Thread-safety of group/array init** (#1435): Initializing a group or array is not fully thread-safe.
- **F memory order**: v3 always stores in C order. Specifying `order='F'` emits a warning.
- **0-dimensional arrays** return scalars, not 0-d arrays.
- **moto mock_aws incompatible** with v3 async stores: Use moto server mode for testing.

## Performance Tips

- **Chunk size**: Target 1 MB+ uncompressed per chunk. Too small = filesystem overhead; too large = poor partial-read performance.
- **Sharding**: Use for datasets with many small chunks on cloud/object stores. Target 10-100+ chunks per shard.
- **Consolidated metadata**: Always use for read-heavy cloud data. Eliminates per-array metadata reads.
- **write_empty_chunks=False** (default): Saves storage for sparse data. Set True if external tools need all chunks present.
- **ZstdCodec over BloscCodec**: ZstdCodec is not affected by the typesize bug and provides good default compression.
- **Async concurrency**: Tune `zarr.config.set({'async.concurrency': N})` for cloud workloads.
- **ObjectStore (obstore)**: For production cloud I/O, Rust-based obstore can be faster than fsspec. Experimental.
- **Avoid iteration**: Row-by-row iteration over zarr arrays is significantly slower in v3 (#2529). Use bulk slicing.
