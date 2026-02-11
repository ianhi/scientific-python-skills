# Zarr-Python Research Summary

## Version Information

**Current Version:** 3.1.5 (released 2025-11-21)
**Python Support:** 3.11+
**NumPy Support:** 2.0+
**Status:** Major release (Version 3 was released Jan 9, 2025)
**Latest Release Location:** https://pypi.org/project/zarr/

### Key Version Details
- Zarr v3 represents a complete rewrite from v2 with breaking changes
- v2 support maintained on `support/v2` branch with 6+ months of maintenance post-v3
- Each major component has breaking changes from v2 -> v3

---

## Installation & Dependencies

### Required Dependencies
- numpy >= 2.0
- numcodecs >= 0.14
- packaging >= 22.0
- google-crc32c >= 1.5
- typing_extensions >= 4.12
- donfig >= 0.8

### Optional Dependencies (extras)
```bash
pip install zarr[remote]  # fsspec, obstore for S3/GCS/etc
pip install zarr[gpu]     # cupy-cuda12x
pip install zarr[cli]     # typer for CLI
```

### Development Dependencies
```bash
pip install zarr[test]     # pytest, hypothesis, benchmark testing
pip install zarr[docs]     # mkdocs-material for building docs
pip install zarr[optional] # rich (for tree display), universal-pathlib
```

---

## Public API Map

### Top-Level Functions (zarr.* namespace)

#### Array Creation Functions
- `zarr.create_array()` - Create a new array with full configuration
- `zarr.create()` - Legacy name, wraps create_array
- `zarr.empty()` - Create uninitialized array
- `zarr.zeros()` - Create array filled with zeros
- `zarr.ones()` - Create array filled with ones
- `zarr.full()` - Create array filled with scalar value
- `zarr.empty_like()` - Create empty array with same shape/dtype as input
- `zarr.zeros_like()` - Create zero-filled array from template
- `zarr.ones_like()` - Create one-filled array from template
- `zarr.full_like()` - Create filled array from template
- `zarr.from_array()` - Convert numpy/dask/xarray to zarr (concurrent streaming)
- `zarr.array()` - Create array from data immediately
- `zarr.open_like()` - Open or create array matching another array

#### Array I/O Functions
- `zarr.open_array()` - Open existing array or create with mode='a'
- `zarr.save_array()` - Save numpy array to zarr
- `zarr.load()` - Load array or group into memory as numpy
- `zarr.save()` - Save one or more numpy arrays to group

#### Group Operations
- `zarr.group()` - Create/open group (deprecated name, use open_group)
- `zarr.create_group()` - Create new group
- `zarr.open_group()` - Open or create group with file-mode semantics
- `zarr.save_group()` - Save multiple arrays to group
- `zarr.create_hierarchy()` - Batch create groups/arrays concurrently from metadata dict
- `zarr.open_consolidated()` - Alias for open_group with consolidated metadata

#### Utility Functions
- `zarr.open()` - Auto-detect and open array or group
- `zarr.consolidate_metadata()` - Consolidate hierarchy metadata to root
- `zarr.copy()`, `zarr.copy_all()`, `zarr.copy_store()` - Copy operations (not fully implemented v3)
- `zarr.tree()` - Display hierarchy (deprecated, use group.tree())
- `zarr.print_debug_info()` - Print debug info for bug reports
- `zarr.set_log_level()` - Configure logging
- `zarr.set_format()` - Configure log message format

### Core Classes

#### Array
- `zarr.Array` - Synchronous array interface (wraps AsyncArray)
- `zarr.AsyncArray` - Asynchronous array implementation (internal but public in v3.1.4+)
- **Key Methods:**
  - `__getitem__()`, `__setitem__()` - NumPy-style indexing
  - `.append()` - Append data to array
  - `.resize()` - Resize array dimensions (signature changed in v3: use resize((new_shape,)))
  - `.info` - Quick diagnostics
  - `.info_complete()` - Detailed diagnostics
  - `.attrs` - Custom attributes dict
  - `.to_numpy()` - Convert to memory
  - `.with_config()` - View with different runtime config
  - `.metadata` - ArrayMetadata object
  - `.nchunks_initialized` - Number of stored chunks

#### Group
- `zarr.Group` - Synchronous group interface (wraps AsyncGroup)
- `zarr.AsyncGroup` - Asynchronous group implementation
- **Key Methods:**
  - `__getitem__()` - Access members by path (e.g., `group['foo/bar']`)
  - `.create_array()` - Create array in group
  - `.create_group()` - Create subgroup
  - `.require_array()` - Get or create array (like h5py)
  - `.members()` - Iterate over immediate children
  - `.rmdir()` - Delete child group
  - `.tree()` - Display hierarchy (requires rich)
  - `.info` - Quick diagnostics
  - `.attrs` - Custom attributes
  - `.metadata` - GroupMetadata object

### Configuration Module
- `zarr.config` - donfig-based configuration object
  - `zarr.config.set()` - Set config values globally
  - `zarr.config.get()` - Get config values
  - `zarr.config.with_context()` - Temporary config context
  - `zarr.config.enable_gpu()` - Enable GPU support
  - **Config Options:**
    - `default_zarr_version` - Default format (2 or 3)
    - `array.order` - C or F memory order
    - `array.write_empty_chunks` - Write chunks filled with fill_value
    - `array.target_shard_size_bytes` - Auto shard size target
    - `async.concurrency` - Async operation concurrency
    - `threading.max_workers` - Thread pool size

---

## Storage Implementations

### Store Classes
Located in `zarr.storage.*`:

#### LocalStore
```python
zarr.storage.LocalStore(root='path/to/data')
# Directory-based storage on local filesystem
# Supports: mode='r'|'r+'|'a'|'w'|'w-', read_only flag
# Methods: open(), close(), list(), get(), set(), delete_dir()
```

#### MemoryStore
```python
zarr.storage.MemoryStore(store_dict={})
# In-memory dict-based storage
# Perfect for testing/demos
# store_dict can be passed pre-existing dict
```

#### ZipStore
```python
zarr.storage.ZipStore('file.zip', mode='w'|'r')
# Single-file zip archive storage
# Must call .close() explicitly
# Supports: mode='r'|'r+'|'w'
```

#### FsspecStore
```python
zarr.storage.FsspecStore(filesystem, root='')
zarr.storage.FsspecStore.from_url('s3://bucket/path', 
                                   storage_options={'anon': True})
# Cloud storage via fsspec (S3, GCS, Azure, HTTP, etc)
# Supports async operations for performance
# Can specify custom filesystem or infer from URL scheme
```

#### ObjectStore (Experimental)
```python
from obstore.store import S3Store
zarr.storage.ObjectStore(S3Store(...))
# Production-grade Rust-based object storage
# Supports S3, GCS, Azure via obstore
```

#### Wrapper/Logging Stores
- `zarr.storage.WrapperStore` - Base for wrapping other stores
- `zarr.storage.LoggingStore` - Logs all store operations
- `zarr.storage.GpuMemoryStore` - GPU memory-backed store

### StoreLike Interface
Can pass to any zarr function accepting `store` parameter:
- String path → LocalStore
- Dict {} or None → MemoryStore
- Path object → LocalStore
- 's3://url' → FsspecStore
- FSMap object → FsspecStore
- Store instance → Used directly

---

## Codec/Compression Configuration

### Zarr v3 Codecs (zarr.codecs.*)

#### Compression Codecs (bytes-to-bytes)
```python
zarr.codecs.BloscCodec(cname='zstd', clevel=5, 
                       shuffle=zarr.codecs.BloscShuffle.shuffle)
zarr.codecs.ZstdCodec(level=5)
zarr.codecs.GzipCodec(level=5)
zarr.codecs.Crc32cCodec()  # Checksum only
```

#### Serialization Codecs (array-to-bytes)
```python
zarr.codecs.BytesCodec(endian='little')  # Required
```

#### Transform Codecs (array-to-array)
```python
zarr.codecs.TransposeCodec(order=[2, 1, 0])
zarr.codecs.ShardingCodec(chunk_shape=(100, 100), 
                          index_codec_config=...,
                          index_location='end'|'start')
```

#### Variable-Length Codecs
```python
zarr.codecs.VLenUTF8Codec()   # For str dtype
zarr.codecs.VLenBytesCodec()  # For bytes dtype
```

### Zarr v2 Codecs (via numcodecs)
```python
from numcodecs import Blosc, Delta, Shuffle, BZ2, LZMA, LZ4
# Import from numcodecs, NOT zarr
compressor = Blosc(cname='zstd', clevel=5, shuffle='shuffle')
z = zarr.create_array(..., compressor=compressor, zarr_format=2)
```

### Default Codec Pipeline
- **v3 numeric:** BytesCodec + ZstdCodec (Zstandard compression)
- **v3 variable-length strings:** VLenUTF8Codec + ZstdCodec
- **v2 numeric:** Blosc with Zstandard + shuffle (if installed)
- **v2 objects:** Vlen-UTF8 or Vlen-Bytes codec required

### Codec Creation Pattern
```python
# Create with explicit codecs (v3)
codecs = [
    zarr.codecs.BytesCodec(),
    zarr.codecs.BloscCodec(cname='zstd', clevel=5),
]
z = zarr.create_array(..., codecs=codecs, zarr_format=3)

# Auto selection (v3)
z = zarr.create_array(...)  # Uses ZstdCodec by default

# Use compressor=None for no compression (v2)
z = zarr.create_array(..., compressor=None, zarr_format=2)
```

---

## Common Usage Patterns

### Basic Array Creation & Operations
```python
import zarr
import numpy as np

# Create in-memory array
store = zarr.storage.MemoryStore()
z = zarr.create_array(
    store=store, 
    shape=(1000, 1000), 
    chunks=(100, 100),  # or "auto"
    dtype='float32'
)

# Write data
z[:] = np.random.random((1000, 1000))
z[0, :] = 42

# Read data
data = z[:]  # Load all to numpy
first_row = z[0, :]
subset = z[100:200, 100:200]

# Resize
z.resize((2000, 2000))
z.append(more_data, axis=0)

# Metadata & diagnostics
print(z.info)
print(z.attrs)
z.attrs['description'] = 'My data'
```

### Group Hierarchy
```python
# Create group hierarchy
root = zarr.group(store='data.zarr')
measurements = root.create_group('measurements')
temperature = measurements.create_array(
    'temperature', 
    shape=(365, 24),
    dtype='float32'
)

# Access members
arr = root['measurements/temperature']
arr = root['measurements']['temperature']

# Inspect
print(root.tree())
print(root.info)

# Iterate members
for key, item in root.members():
    print(f"{key}: {item}")
```

### Persistent Storage (Disk)
```python
# Create on disk
z = zarr.create_array(
    store='data/myarray.zarr',
    shape=(1000, 1000),
    chunks=(100, 100),
    dtype='f4'
)
z[:] = data

# Re-open later
z2 = zarr.open_array('data/myarray.zarr', mode='r+')
z2[0:100, 0:100] = new_data

# Save/load convenience
zarr.save_array('data/simple.zarr', np_array)
loaded = zarr.load('data/simple.zarr')
```

### Cloud Storage
```python
# S3 with s3fs
z = zarr.create_array(
    store='s3://mybucket/data.zarr',
    shape=(10000, 10000),
    chunks=(1000, 1000),
    storage_options={'anon': True}  # Anonymous read
)

# Or explicit FsspecStore
import s3fs
store = zarr.storage.FsspecStore.from_url(
    's3://mybucket/data.zarr',
    storage_options={'anon': True}
)
z = zarr.open_array(store=store, mode='r')
```

### Sharding for Small Chunks
```python
# Problem: 1M tiny chunks = slow on filesystems
# Solution: Shard into larger objects

z = zarr.create_array(
    store='data.zarr',
    shape=(100_000_000, 100),
    chunks=(100, 100),      # Small chunks for access granularity
    shards=(10_000, 100),   # Shards store 100 chunks each
    zarr_format=3
)

# Or auto-shard
z = zarr.create_array(
    ...,
    shards='auto',  # Uses config.array.target_shard_size_bytes (128 MiB default)
    chunks=(100, 100)
)
```

### Configuration & Context
```python
# Global config
zarr.config.set({'array.order': 'F'})

# Temporary context
with zarr.config.set({'array.write_empty_chunks': True}):
    z = zarr.create_array(...)

# Per-array config
z = zarr.create_array(
    ...,
    config={'write_empty_chunks': False, 'order': 'C'}
)

# View with different config
z_f = z.with_config({'order': 'F'})  # Fortran order reads
```

### Consolidated Metadata (Performance)
```python
# Create hierarchy
root = zarr.create_group('data.zarr')
for i in range(100):
    root.create_array(f'array_{i}', shape=(1000,), dtype='f4')

# Consolidate (single pass to gather all metadata)
root = zarr.consolidate_metadata('data.zarr')

# Now opening group reads metadata once
root = zarr.open_group('data.zarr')  # Uses consolidated metadata if present
arr = root['array_50']  # No store read needed!
```

### Batch Creation
```python
from zarr import create_hierarchy
from zarr.core.group import GroupMetadata
from zarr.core.array import ArrayMetadata

nodes = {
    'foo/bar': GroupMetadata(),
    'foo/bar/baz': ArrayMetadata(shape=(100,), dtype='f4'),
    'foo/qux': ArrayMetadata(shape=(100, 100), dtype='i4'),
}

created = dict(create_hierarchy(
    store=zarr.storage.LocalStore('data.zarr'),
    nodes=nodes
))
```

### Working with Attributes
```python
# Arrays and groups have JSON-serializable attributes
z = zarr.create_array(...)
z.attrs['experiment'] = 'test_001'
z.attrs['created'] = '2025-02-11'
z.attrs['parameters'] = {'rate': 0.01, 'steps': 1000}

# Access
print(z.attrs['experiment'])
print(sorted(z.attrs.keys()))

# Delete
del z.attrs['created']

# Note: Only JSON-serializable types (str, int, float, bool, list, dict)
# NOT: datetime objects, numpy arrays, custom objects
```

---

## Key API Signatures

### zarr.create_array()
```python
zarr.create_array(
    store: StoreLike,
    *,
    shape: ShapeLike,
    chunks: ChunkLike = 'auto',
    dtype: ZDTypeLike = 'float64',
    
    # Compression/codecs
    codecs: Iterable[Codec] = None,           # v3 only
    compressor: CompressorLike = 'auto',      # v2 only
    filters: Iterable[Codec] = None,          # v2 only
    
    # Storage options
    fill_value: Any = DEFAULT_FILL_VALUE,
    order: MemoryOrder = None,  # 'C' or 'F'
    
    # Sharding (v3)
    shards: ShardsLike = None,  # (1000, 1000) or 'auto'
    
    # Metadata
    attributes: dict = None,
    dimension_names: DimensionNames = None,
    
    # Format & mode
    zarr_format: ZarrFormat = None,  # 2 or 3, defaults to config
    mode: AccessModeLiteral = 'w',
    
    # Config
    config: ArrayConfigLike = None,  # {'write_empty_chunks': False}
    
    # Data
    data: NDArrayLike = None,  # Initialize with data
    write_data: bool = True,
) -> Array
```

### zarr.open_array()
```python
zarr.open_array(
    store: StoreLike = None,
    *,
    mode: AccessModeLiteral = 'a',  # 'r'|'r+'|'a'|'w'|'w-'
    path: str = None,
    zarr_format: ZarrFormat = None,
    storage_options: dict = None,
) -> Array
```

### zarr.create_group()
```python
zarr.create_group(
    store: StoreLike,
    *,
    path: str = None,
    zarr_format: ZarrFormat = None,
    overwrite: bool = False,
    attributes: dict = None,
    storage_options: dict = None,
) -> Group
```

### zarr.open_group()
```python
zarr.open_group(
    store: StoreLike = None,
    *,
    mode: AccessModeLiteral = 'a',
    path: str = None,
    zarr_format: ZarrFormat = None,
    use_consolidated: bool | str = None,  # False|True|custom_key
    attributes: dict = None,
    storage_options: dict = None,
) -> Group
```

---

## V2 to V3 Migration - Critical Differences

### ANTI-PATTERNS (v3 will NOT work with v2 patterns):

1. **Don't import codecs from zarr namespace**
   ```python
   # ❌ WRONG (v2 pattern)
   from zarr import Blosc  # NO! Not available in v3
   
   # ✓ CORRECT (v3)
   from numcodecs import Blosc  # For v2 arrays
   from zarr.codecs import BloscCodec  # For v3 arrays
   ```

2. **Don't construct Array/Group directly**
   ```python
   # ❌ WRONG
   arr = zarr.Array(...)  # Constructor signature completely changed
   grp = zarr.Group(...)  # Constructor signature completely changed
   
   # ✓ CORRECT
   arr = zarr.create_array(...)
   arr = zarr.open_array(...)
   grp = zarr.create_group(...)
   grp = zarr.open_group(...)
   ```

3. **Don't use dot notation for group members**
   ```python
   # ❌ WRONG
   arr = group.array_name  # NOT supported in v3
   
   # ✓ CORRECT
   arr = group['array_name']
   arr = group['foo/bar/baz']  # Multi-level access
   ```

4. **Don't rely on zarr.sync or synchronizers**
   ```python
   # ❌ WRONG
   from zarr import sync  # Internal API, not stable
   from zarr.sync import Synchronizer  # Removed
   
   # ✓ CORRECT
   z = zarr.create_array(...)  # Already synchronous
   ```

5. **Don't use old store implementations**
   ```python
   # ❌ WRONG (v2)
   from zarr import DirectoryStore, FSStore, MemoryStore
   from zarr import DBMStore, LMDBStore, SQLiteStore, MongoDBStore, RedisStore, N5Store
   
   # ✓ CORRECT (v3)
   from zarr.storage import LocalStore, FsspecStore, MemoryStore, ZipStore, ObjectStore
   ```

6. **Don't use old codec pipeline**
   ```python
   # ❌ WRONG (v2)
   z = zarr.create_array(..., compressor=..., filters=[...])
   
   # ✓ CORRECT (v3)
   z = zarr.create_array(..., codecs=[...], zarr_format=3)
   ```

7. **Don't resize with separate args**
   ```python
   # ❌ WRONG (v2)
   z.resize(1000, 1000)
   
   # ✓ CORRECT (v3)
   z.resize((1000, 1000))
   ```

8. **Don't use copy/copy_all with v3 (not fully implemented)**
   ```python
   # ⚠️ INCOMPLETE
   zarr.copy(src, dst)  # Not fully implemented in v3
   zarr.copy_all(...)   # Not fully implemented in v3
   
   # WORKAROUND: Use from_array with explicit source/dest
   zarr.from_array(src_array, store=dest_store)
   ```

### Key Behavioral Changes:

- **Default format v3:** New arrays default to Zarr format 3 (not v2)
  - Set `zarr_format=2` to maintain v2 or `default_zarr_version=2` in config
  
- **Fill value semantics:** `fill_value` now defaults to `DEFAULT_FILL_VALUE` (type-appropriate zero)
  - Changed from literal `0` which caused issues with non-numeric types
  
- **Memory order (F) deprecation:** Zarr v3 always stores in C order, warning if F specified
  
- **Empty chunks:** `write_empty_chunks` defaults to `False` (don't store fill-value-only chunks)
  
- **Chunk encoding:** No positional arguments in creation functions anymore
  - All non-store args must be keyword-only

### Data Type Changes:

**v2 to v3 dtype string changes:**
```python
# v2 dtype encodings (NumPy-like strings)
'<i4'        # v2 little-endian int32
'|b1'        # v2 boolean
'|O'         # v2 object (ambiguous - needs codec to disambiguate)

# v3 dtype encodings (Zarr specification)
'int32'      # v3 int32 (endianness implicit)
'bool'       # v3 boolean
'string'     # v3 variable-length string (not |O!)
```

### Structured & Special Types (v3 Status):

Some complex data types still have limited support in v3:
- ✓ Numeric types (int, float, complex)
- ✓ Boolean
- ✓ Variable-length strings (str dtype)
- ✓ Variable-length bytes
- ⚠️ Fixed-length strings (partial support)
- ⚠️ Datetime/timedelta (supported but limited testing)
- ❌ Structured dtypes (dict-of-arrays)
- ❌ Object arrays (only vlen strings/bytes)
- ❌ Ragged arrays

---

## Integration Points

### xarray Integration
```python
import xarray as xr
import zarr

# xarray can read/write zarr stores directly
ds = xr.open_zarr('data.zarr')
ds.to_zarr('output.zarr', mode='w')

# Or explicit zarr Group
group = zarr.open_group('data.zarr')
ds = xr.open_zarr(group)
```

### Dask Integration
```python
import dask.array as da

# Dask integrates with zarr through standard array interface
z = zarr.open_array('data.zarr')
darr = da.from_delayed(z, shape=z.shape, dtype=z.dtype)

# Write dask array to zarr
da.to_zarr(darr, store='output.zarr')
```

### Concurrent Access
```python
# Multiple threads/processes can safely read/write same array
import concurrent.futures

z = zarr.open_array('data.zarr', mode='r+')

def write_chunk(i):
    z[i*1000:(i+1)*1000] = data[i]

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(write_chunk, range(10))
```

---

## Known Issues & Gotchas

### Critical Bug (Fixed in 3.0.8)
- **Issue #3062:** Opening with `mode='a'` (default) deleted existing data before v3.0.8
  - **Fix:** Update to 3.0.8+
  - **Workaround:** Explicitly specify `mode='r+'` for existing data

### Performance Considerations
1. **Empty chunk writes:** Disable with `config={'write_empty_chunks': False}` (default)
   - Avoids storing chunks that are all fill-value
   - Trades memory for CPU (does value comparison per chunk)
   
2. **Chunk size:** Recommend 1MB+ uncompressed for good I/O
   - Too small chunks → filesystem/object store overhead
   - Too large chunks → poor partial-read performance
   
3. **Sharding overhead:** Small shards (< 1 object) negate benefits
   - Use shards >= 10-100 chunks for typical I/O patterns
   
4. **Consolidated metadata:** Optimization for read-heavy, slowly-changing hierarchies
   - Not ideal for rapidly evolving data
   - Adds overhead to updates
   - Not supported by all stores

### API Stability

- **Private APIs** (zarr.core.*): Not covered by versioning policy, can change
- **Experimental Features** (zarr.experimental.*): Subject to change
  - CacheStore (v3.1.4+)
  - ObjectStore (v3.0.7+)
  - GPU buffers (v3.0.3+)

### Codec Limitations

- **numcodecs codecs in v3:** Available via `zarr.codecs.numcodecs.*` or `numcodecs.zarr3`
  - Some not in Zarr v3 spec but available for compatibility
  
- **Custom codecs:** Extension mechanism still in development
  - Registry-based registration via entry-points
  - API may change post-v3.0.0

### Type System Notes

- `zarr.dtype` module provides `ZDType` abstraction for cross-version types
- Zarr v2 uses NumPy dtypes, v3 uses `ZDType` instances
- `ZDType.to_native_dtype()` converts to NumPy for Python use

### Data Loss Prevention

- **Atomic writes:** LocalStore uses atomic operations (v3.1.3+)
- **ZipStore:** Must explicitly `.close()` to flush
- **Mode semantics:**
  - `'w'` = overwrite (destructive)
  - `'w-'` = create only, fail if exists (safe)
  - `'a'` = append/create (idempotent after v3.0.8)
  - `'r+'` = read/write existing (fails if doesn't exist)
  - `'r'` = read only

---

## Deprecations & Future Work

### Removed in v3
- `zarr.attrs` module
- `zarr.context` module  
- `zarr.hierarchy` (use Group instead)
- `zarr.indexing` module
- `zarr.meta` and `zarr.meta_v1`
- `zarr.sync` (internal only now)
- `zarr.types` (see zarr.types module for replacements)
- `zarr.util` module
- `zarr.n5` (use n5py instead)

### Planned But Not Yet Implemented (v3.0.0+)
- Full `zarr.copy()` functionality
- `zarr.Group.move()` method
- `chunk_store` parameter (separate chunk/metadata storage)
- `meta_array` parameter (hints for array type)
- `synchronizer` parameter (explicit locking)
- `dimension_separator` parameter
- `cache_metadata` parameter
- `cache_attrs` parameter

### Future Extensions (Post-v3.0.0)
- Custom data types
- Custom chunk grids (beyond rectangular)
- More built-in stores
- Further codec ecosystem development

---

## Configuration Reference

### Environment Variables
```bash
# Set config via env (ZARR_<SECTION>__<KEY>=value)
ZARR_ARRAY__ORDER=F
ZARR_ARRAY__WRITE_EMPTY_CHUNKS=true
ZARR_ASYNC__CONCURRENCY=8
ZARR_THREADING__MAX_WORKERS=4
```

### Default Configuration Values
```python
{
    'default_zarr_version': 3,
    'array': {
        'order': 'C',
        'write_empty_chunks': False,
        'target_shard_size_bytes': 134_217_728,  # 128 MiB
    },
    'async': {
        'concurrency': inf,  # No limit
    },
    'threading': {
        'max_workers': number of cores,
    },
    'codecs': {...},  # Codec implementations
    'buffers': {...}, # Buffer implementations
}
```

---

## Testing & Validation

### Creating Arrays
```python
# Validate basic creation
assert z.shape == (1000, 1000)
assert z.chunks == (100, 100)
assert z.dtype == np.dtype('f4')

# Check metadata
print(z.metadata)
print(z.compressors)

# Verify storage efficiency
print(z.info_complete())  # Shows bytes stored, compression ratio
```

### Edge Cases to Watch

1. **0-dimensional arrays:** Return scalars, not 0-d arrays
2. **Non-finite fill values:** Support inf, -inf, nan
3. **Endianness:** v2 only, v3 assumes native byte order
4. **Empty arrays:** Shape (0,) is valid, different from uninitialized
5. **String encodings:** v3 always UTF-8, v2 supports vlen-utf8 codec

---

## Gotcha Summary

| Issue | V2 | V3 | Solution |
|-------|----|----|----------|
| Array constructor | Works | Broken | Use create_array/open_array |
| Group.foo access | Works | Broken | Use group['foo'] |
| Blosc codec import | from zarr | Broken | from numcodecs |
| Codec configuration | compressor= | codes= | Use zarr.codecs.* |
| Mode='a' data loss | Bug rare | Fixed 3.0.8 | Upgrade to 3.0.8+ |
| Memory order F | Stored | Warning | C order only in v3 |
| Resize args | (1000,1000) | Broken | Use ((1000,1000),) |
| Structured dtypes | Works | Partial | v3 may not support |
| Empty chunks | Stored | Skip | Configurable |

---

This comprehensive research document should be saved to `/Users/ian/Documents/dev/scientific-python-skills/research/zarr/docs_summary.md` to serve as the foundation for the zarr skill file.