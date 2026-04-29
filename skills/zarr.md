# Zarr v3

Prescriptive guide for using the zarr-python v3 public API correctly. For v2 -> v3
migration patterns and anti-patterns, see `zarr-v2-v3-migration.md`.

## Mental Model

Zarr stores chunked, compressed, N-dimensional arrays. You interact with three
objects, in order of importance:

- **Array** — a chunked N-D array. This is what you read and write. Slice
  assignment (`arr[...] = data`) writes; slicing (`arr[0:100]`) reads.
- **Group** — a directory-like container of arrays and subgroups. Use bracket
  access (`group["name"]`) and `members()` to inspect.
- **Store** — where bytes live (filesystem, memory, S3, zip). You usually pass
  a path string and let zarr build the store for you. **Avoid calling Store
  methods directly.**

The Store is plumbing. Almost all your code should be at the Array/Group level.
The top-level API (`zarr.create_array`, `zarr.open_group`, `zarr.from_array`)
handles store construction, async, buffers, and serialization for you.

**Stay at the public API.** If you import from `zarr.core.*`, `zarr.abc.*`, or
`zarr.storage._*`, you're doing it wrong. If you find yourself calling
`store.set()`, `store.get()`, or `store.list_prefix()`, stop — use the
Array/Group API instead.

## Creating Arrays

The most common starting point. You usually pass a path string for the store —
zarr builds the right Store for you (LocalStore for paths, FsspecStore for URLs).

```python
import zarr
import numpy as np

# From shape + dtype (creates empty array)
arr = zarr.create_array(
    "data.zarr",
    shape=(1000, 1000),
    dtype="float32",
    chunks=(100, 100),
)

# From existing data (shape/dtype inferred)
arr = zarr.from_array("data.zarr", data=np.arange(100))

# Remote: just pass the URL
arr = zarr.create_array("s3://bucket/data.zarr", shape=(100,), dtype="f4")

# With compression (defaults are good; override only if needed)
arr = zarr.create_array(
    "data.zarr",
    shape=(1000,),
    dtype="f4",
    compressors=zarr.codecs.ZstdCodec(level=5),
)
```

**Don't mix `data=` with `shape=`/`dtype=` — it raises ValueError:**

```python
# WRONG — raises ValueError
zarr.create_array(store, shape=(3,), dtype="i4", data=np.array([1, 2, 3]))

# RIGHT — use data alone
arr = zarr.from_array(store, data=np.array([1, 2, 3], dtype="i4"))

# RIGHT — or shape+dtype alone, then assign
arr = zarr.create_array(store, shape=(3,), dtype="i4")
arr[:] = np.array([1, 2, 3])
```

## Creating Groups

```python
# Create a new group (preferred for new code)
root = zarr.create_group("experiment.zarr")

# Create nested arrays and groups inside it
data = root.create_array("data", shape=(100,), dtype="f8")
subgroup = root.create_group("experiments")
results = subgroup.create_array("results", shape=(10,), dtype="i4")
```

## Opening Existing Arrays/Groups

```python
# Open array (read-only)
arr = zarr.open_array("data.zarr", mode="r")

# Open group
root = zarr.open_group("data.zarr", mode="r")

# Auto-detect (returns Array OR Group depending on what's there)
obj = zarr.open("data.zarr", mode="r")
```

**Mode semantics** (same across `open`, `open_array`, `open_group`):

| Mode | Behavior |
|------|----------|
| `r` | Read only, must exist |
| `r+` | Read/write, must exist |
| `a` | Read/write, create if missing (default for `open_group`) |
| `w` | Create, overwrite if exists (destructive) |
| `w-` | Create, fail if exists |

## Reading and Writing Data

```python
# Full read
data = arr[:]

# Slice read
subset = arr[0:100, 50:150]

# Write via slice assignment (there is NO .write() method)
arr[0:10] = np.arange(10)
arr[...] = 0          # fill with zeros
arr[0, 0] = 42        # scalar write
```

## Inspecting Groups

```python
# Visual tree
print(root.tree())

# Immediate children (tuple of (name, Array|Group))
for name, child in root.members():
    print(name, type(child).__name__)

# Only arrays, only groups
for name, arr in root.arrays():
    print(name, arr.shape)

for name, grp in root.groups():
    print(name)

# Just the keys
list(root.keys())

# Bracket access
arr = root["data"]
nested = root["experiments/results"]
```

**Note:** Group has no `.items()` method in v3. Use `.members()` for
`(name, child)` pairs, or `.arrays()` / `.groups()` for filtered iteration.

## Resizing Arrays

```python
# N-D: pass a tuple
arr.resize((2000, 2000))

# 1-D: int or tuple both work
arr.resize(2000)
arr.resize((2000,))
```

## Codecs

The default compression (ZstdCodec) is good. Override only if you have a reason.

```python
# No compression
arr = zarr.create_array(store, shape=(100,), dtype="f4", compressors=None)

# Custom compression
arr = zarr.create_array(
    store,
    shape=(1000,),
    dtype="f4",
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
)

# Explicit codec pipeline: filters -> serializer -> compressors
arr = zarr.create_array(
    store,
    shape=(1000,),
    dtype="f4",
    filters=[zarr.codecs.TransposeCodec(order=(0,))],
    serializer=zarr.codecs.BytesCodec(endian="little"),
    compressors=zarr.codecs.ZstdCodec(level=3),
)
```

Available codecs at `zarr.codecs`:
- Compressors: `BloscCodec`, `ZstdCodec`, `GzipCodec`, `Crc32cCodec`
- Serializers: `BytesCodec`, `VLenBytesCodec`, `VLenUTF8Codec`
- Filters: `TransposeCodec`
- Sharding: `ShardingCodec`

**Prefer `ZstdCodec` over `BloscCodec`** — Blosc has a typesize bug in v3 that
silently degrades compression with shuffle (#2171).

## Sharding

For cloud stores with many small chunks, group them into larger storage objects:

```python
arr = zarr.create_array(
    store,
    shape=(100_000, 100),
    dtype="float32",
    chunks=(100, 100),       # logical chunks (access granularity)
    shards=(10_000, 100),    # physical shards (100 chunks per shard)
)
```

Target 10-100+ chunks per shard. Sharding significantly reduces the number of
store operations for remote data.

## Configuration

```python
import zarr

# Global config
zarr.config.set({
    "async.concurrency": 20,           # tune for cloud workloads
    "array.write_empty_chunks": True,  # override v3 default (False)
})

# Per-array config
arr = zarr.create_array(
    store, shape=(100,), dtype="f4",
    config={"write_empty_chunks": True},
)
```

**Default changed in v3:** `write_empty_chunks` is `False` (v2 was `True`).
Fill-value-only chunks are skipped on write. Reads return the fill value
silently — no error. If external tools require all chunks to be present, set
`write_empty_chunks=True`.

## Common Recipes

### Write a numpy array to local zarr

```python
import numpy as np
import zarr

data = np.random.random((1000, 1000)).astype("f4")
arr = zarr.from_array("data.zarr", data=data)
```

### Read a remote zarr

```python
import zarr

arr = zarr.open_array("s3://bucket/data.zarr", mode="r")
chunk = arr[0:100, :]
```

### Create a group with multiple arrays

```python
import zarr

root = zarr.create_group("experiment.zarr")
root.create_array("inputs", shape=(1000, 50), dtype="f4")
root.create_array("labels", shape=(1000,), dtype="i4")
root.attrs["description"] = "training dataset"
```

### Append to an existing array

```python
arr = zarr.open_array("data.zarr", mode="r+")
new_size = arr.shape[0] + new_data.shape[0]
arr.resize((new_size,) + arr.shape[1:])
arr[-new_data.shape[0]:] = new_data
```

### Inspect an unknown zarr

```python
obj = zarr.open("unknown.zarr", mode="r")
if isinstance(obj, zarr.Group):
    print(obj.tree())
else:
    print(f"Array: shape={obj.shape}, dtype={obj.dtype}")
```

## Integration

### xarray

```python
import xarray as xr

# Write
ds.to_zarr("data.zarr", zarr_format=3, consolidated=False)

# Read
ds = xr.open_zarr("data.zarr", consolidated=False)
```

Pass `consolidated=False` if you're writing incrementally or using icechunk.

### icechunk

See `icechunk.md` skill. Short version: `store = session.store`, then use the
usual zarr API on top.

### dask

```python
import dask.array as da

# Write a dask array to zarr (parallel)
dask_arr = da.random.random((10000, 10000), chunks=(1000, 1000))
arr = zarr.create_array("data.zarr", shape=dask_arr.shape,
                         dtype=dask_arr.dtype, chunks=dask_arr.chunksize)
dask_arr.store(arr)

# Read zarr as dask
arr = zarr.open_array("data.zarr", mode="r")
dask_arr = da.from_zarr(arr)
```

## Performance Tips

- **Chunk size**: target 1 MB+ uncompressed per chunk. Too small = filesystem
  overhead; too large = wasted reads when you only need a slice.
- **Sharding**: use for cloud stores with many small chunks. 10-100+ chunks per shard.
- **Consolidated metadata**: `zarr.consolidate_metadata(store)` after building a
  read-heavy dataset — eliminates per-array metadata reads.
- **Async concurrency**: tune `zarr.config.set({'async.concurrency': N})` for
  cloud workloads (default is conservative).
- **Bulk slicing over iteration**: `arr[:]` or `arr[0:1000]` is much faster
  than `[arr[i] for i in range(1000)]`. Row-by-row iteration has significant
  per-access overhead in v3.
- **Default compressor**: `ZstdCodec` is fast and not affected by the Blosc
  typesize bug. Use it as your default.

## Stores (Reference)

Most code should pass path strings, not Store objects. But sometimes you need
explicit Store construction — for non-trivial fsspec configuration, in-memory
testing, or zip files.

```python
# In-memory (for tests)
store = zarr.storage.MemoryStore()

# Local filesystem (when you need explicit construction)
store = zarr.storage.LocalStore("/path/to/data.zarr")

# Remote with fsspec configuration
store = zarr.storage.FsspecStore.from_url(
    "s3://bucket/data.zarr",
    storage_options={"anon": True},
)

# Zip file
store = zarr.storage.ZipStore("/path/to/data.zip", mode="w")
```

Then pass `store` to `zarr.create_array(store, ...)`, `zarr.open_group(store, ...)`,
etc. **Never call `store.set()`, `store.get()`, or `store.list_prefix()` directly**
— these are async and require Buffer types. Always go through Array/Group.

## Known Issues

- **Blosc typesize bug** (#2171, #2766): `BloscCodec` defaults typesize to 1
  byte instead of dtype size, causing 10-20x worse compression when using
  shuffle. Use `ZstdCodec` unless you need Blosc specifically.
- **Performance regression vs v2** (#2710): Large array writes slower due to
  async overhead. Being addressed.
- **FsspecStore requires async filesystem** (#2706): For local files use
  `LocalStore`. For cloud, use `asynchronous=True` on the fsspec filesystem.
- **Thread-safety** (#1435): Group/array initialization not fully thread-safe.
- **F memory order**: v3 always stores in C order. `order='F'` emits a warning.
