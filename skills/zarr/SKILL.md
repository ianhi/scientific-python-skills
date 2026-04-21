---
name: zarr
description: Writing idiomatic zarr v3 code. Use when reading/writing zarr arrays or groups, configuring codecs or sharding, migrating from zarr v2, using cloud stores (S3, GCS), or debugging zarr storage issues.
user-invocable: false
---

# Zarr

> v3 is a breaking rewrite. Store classes, codec classes, and function signatures all changed. Use `compressors=` not `codecs=` on `create_array`. Blosc shuffle is silently broken (typesize=1). `write_empty_chunks` default flipped to False.

## CRITICAL: Do Not Drop to Store Internals

The most common mistake pattern: when debugging or inspecting data, Claude
reaches for low-level store APIs and internal imports instead of using the
zarr group/array interface. This cascades into async errors, type mismatches,
and broken import paths.

```python
# WRONG - over-specified internal imports
from zarr.core.buffer.cpu import Buffer
from zarr.core.buffer import default_buffer_prototype, PROTOTYPE
from zarr.core.buffer.numpy_buffer import NumpyBuffer
from zarr.core.sync import sync
from zarr.storage._common import StorePath
from zarr.abc.store import Store

# WRONG - manual async store calls
asyncio.run(store.list_prefix(""))        # async generator, not awaitable
asyncio.run(store.delete("data/c/0"))     # just use the array API
sync(store.list_prefix(""))               # TypeError: can't await async generator
store.set("key", some_bytes)              # needs await AND Buffer type, not bytes
store.get("key", prototype=_PROTOTYPE)    # don't call store.get() directly

# WRONG - constructing internal objects manually
AsyncGroup(GroupMetadata(), store_path=StorePath(store=store, path="root"))
zarr.api.asynchronous.open_group(store, mode="r")

# RIGHT - stay at the zarr level
root = zarr.open_group(store)             # sync, handles everything
root = zarr.group(store=store)            # create new group
arr = root.create_array("data", shape=(10,), dtype="f8")
arr[:] = np.arange(10)
print(root.tree())                        # inspect structure
print(list(root.members()))               # list contents
data = arr[:]                             # read data
```

**Rule:** If you're importing from `zarr.core.*`, `zarr.abc.*`, or
`zarr.storage._*`, stop and use the top-level zarr API instead.

## v2 -> v3 Migration Quick Reference

If you learned zarr v2 patterns, here are the critical changes. **Most of these are breaking changes that will cause errors.**

| v2 Pattern (WRONG in v3) | v3 Pattern (CORRECT) | Notes |
|--------------------------|----------------------|-------|
| `from zarr import Blosc, Zlib` | `from zarr.codecs import BloscCodec, ZstdCodec, GzipCodec` | Codec classes moved |
| `zarr.Array(store, ...)` | `zarr.create_array(store, ...)` or `zarr.open_array(...)` | Use factory functions |
| `group.array_name` | `group['array_name']` | Dot notation removed |
| `from zarr import DirectoryStore, FSStore` | `from zarr.storage import LocalStore, FsspecStore` | Store classes moved/renamed |
| `compressor=Blosc()` | `compressors=zarr.codecs.BloscCodec()` | Param renamed (note plural) |
| `mapper = fsspec.get_mapper(...); zarr.open(mapper)` | `zarr.open_array('s3://...')` or `FsspecStore.from_url(...)` | FSMap no longer supported |
| `z.resize(1000, 1000)` | `z.resize((1000, 1000))` | Must pass tuple |
| `compressors=[numcodecs.Blosc()]` | `compressors=zarr.codecs.BloscCodec()` | Don't mix numcodecs with v3 |
| `zarr.from_array(data, 'store.zarr')` | `zarr.from_array('store.zarr', data=data)` | Store is first, data is kwarg-only |
| `grp.create_dataset('name', ...)` | `grp.create_array('name', ...)` | Method renamed (deprecated but works) |
| `grp.create_array('x', shape=s, dtype=d, data=arr)` | See "create_array + data" below | Cannot mix shape/dtype with data |
| `node.flat[0]` | `node[0]` | zarr.Array has no .flat attribute |

**Removed features (no v3 equivalent):**
- Object dtype (`dtype='|O'`) and structured dtypes - use separate arrays instead (#2134)
- Dot notation for group members - use `group['member']` bracket syntax

**Default changes (SILENT behavior differences):**
- `write_empty_chunks`: `True` (v2) -> `False` (v3) - sparse arrays now skip writing fill-value chunks
- `zarr_format`: `2` (v2) -> `3` (v3) - use `zarr_format=2` for backwards compatibility

## Codec Pipeline (v3)

```python
# Default: BytesCodec + ZstdCodec (good default, just works)
z = zarr.create_array(store, shape=(1000,), dtype='f4')

# Explicit codec pipeline: use serializer= + compressors=, NOT codecs=
# NOTE: codecs= only works on ShardingCodec, NOT on create_array()
z = zarr.create_array(
    store, shape=(1000,), dtype='f4',
    serializer=zarr.codecs.BytesCodec(endian='little'),
    compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=5,
                                        shuffle=zarr.codecs.BloscShuffle.shuffle),
)

# No compression
z = zarr.create_array(store, shape=(1000,), dtype='f4', compressors=None)
```

## Sharding (v3 only)

```python
# Groups many small chunks into larger storage objects - critical for cloud
z = zarr.create_array(
    store='data.zarr',
    shape=(100_000, 100),
    dtype='float32',
    chunks=(100, 100),       # logical chunks (access granularity)
    shards=(10_000, 100),    # physical shards (100 chunks per shard)
)
```

## Gotchas & Common Mistakes

### create_array: dtype is required in v3
```python
# WRONG - missing dtype
root.create_array("arr", shape=(1,))  # TypeError

# RIGHT
root.create_array("arr", shape=(1,), dtype="i4")
```

### create_array: cannot mix shape/dtype with data=
```python
# WRONG (v2 habit) - passing data= alongside explicit shape/dtype
group.create_array("arr", shape=(3,), dtype="i4", data=np.array([1, 2, 3]))

# RIGHT - two-step: create then assign
arr = group.create_array("arr", shape=(3,), dtype="i4")
arr[:] = np.array([1, 2, 3])

# RIGHT - OR let zarr infer from data (no shape/dtype)
arr = group.create_array("arr", data=np.array([1, 2, 3], dtype="i4"))
```

### store.list_prefix() is an async generator, not an awaitable
```python
# WRONG - all of these fail
sync(store.list_prefix(""))          # TypeError: can't await async generator
asyncio.run(store.list_prefix(""))   # ValueError: coroutine expected
sorted(store.list_prefix(""))        # TypeError: async_generator not iterable
store.list_prefix_sync("")           # AttributeError: does not exist

# RIGHT (if you truly need store-level keys)
async def _keys(store):
    return sorted([k async for k in store.list_prefix("")])
keys = asyncio.run(_keys(store))

# BETTER - stay at zarr level
root = zarr.open_group(store)
print(list(root.members()))
```

### store.set() is async -- calling without await silently drops writes
```python
# WRONG - coroutine created but never awaited (SILENT FAILURE)
session.store.set("zarr.json", meta)
# RuntimeWarning: coroutine 'IcechunkStore.set' was never awaited
# Then commit fails: "no changes made to the session"

# ALSO WRONG - set() requires Buffer, not bytes
await store.set("zarr.json", b'{"zarr_format": 3}')
# TypeError: value must be a Buffer instance

# RIGHT - don't call store.set() directly, use zarr
root = zarr.group(store=session.store)
arr = root.create_array("data", shape=(10,), dtype="f8")
arr[:] = np.arange(10)
```

### IcechunkStore (and other stores) are not iterable
```python
# WRONG
list(store)   # TypeError: 'IcechunkStore' object is not iterable

# RIGHT
root = zarr.open_group(store)
print(root.tree())
```

### zarr.Array has no .flat attribute
```python
# WRONG (numpy habit)
node.flat[0]  # AttributeError

# RIGHT
node[0]
```

### Blosc compression ratio degradation v2 vs v3 (#2171, #2766)
Blosc typesize defaults to 1 byte in v3 (instead of dtype size), causing 10-20x worse compression with shuffle.
```python
# SLOW - shuffle is ineffective because typesize=1 internally
z = zarr.create_array(store, shape=(1000,), dtype='float64',
                      compressors=zarr.codecs.BloscCodec(cname='lz4', clevel=1,
                                                          shuffle=zarr.codecs.BloscShuffle.shuffle))

# FAST - ZstdCodec (default) is not affected by the typesize bug
z = zarr.create_array(store, shape=(1000,), dtype='float64',
                      compressors=zarr.codecs.ZstdCodec(level=3))
```

### write_empty_chunks default changed (#853)
```python
# v2 default: write_empty_chunks=True (stores all chunks)
# v3 default: write_empty_chunks=False (skips fill-value-only chunks)
# Missing chunks return fill_value silently - no error raised

# If external tools need all chunks present:
z = zarr.create_array(store, shape=(100,), dtype='f4',
                      config={'write_empty_chunks': True})
```

### Mode semantics changed (#3062)
```python
# DANGER: mode='a' (default for open_array) was buggy before 3.0.8
# It could DELETE existing data. Always use 3.0.8+

zarr.open_array('data.zarr', mode='r')    # read only
zarr.open_array('data.zarr', mode='r+')   # read/write existing (fails if missing)
zarr.open_array('data.zarr', mode='w-')   # create only (fails if exists)
zarr.open_array('data.zarr', mode='w')    # overwrite (DESTRUCTIVE)
```

### from_array() signature gotcha
```python
# Store is FIRST arg, data is keyword-only
z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))
# WRONG: zarr.from_array(np_array, store='data.zarr')  # TypeError
```

### zarr.codecs.Blosc is NOT numcodecs.Blosc
```python
# These are DIFFERENT types - do not interchange:
from zarr.codecs import BloscCodec, ZstdCodec  # native v3 codecs (preferred)
from zarr.codecs.numcodecs import Blosc         # v3 wrapper class (edge cases only)
from numcodecs import Blosc                     # raw numcodecs class (don't use with v3)
```

### FsspecStore requires async filesystem (#2706)
```python
# WRONG - local filesystem is not async
store = zarr.storage.FsspecStore(fsspec.filesystem("file"), path="/local/path")

# RIGHT - use LocalStore for local files
store = zarr.storage.LocalStore("/local/path")

# RIGHT - for cloud, use asynchronous=True
fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
store = zarr.storage.FsspecStore(fs, path="bucket/data.zarr")
```

### open_array mode is a hidden kwarg
```python
# IDEs will not autocomplete 'mode', but it works:
zarr.open_array('data.zarr', mode='r')  # mode is passed via **kwargs
```

## Known Limitations

- **Blosc shuffle typesize bug** (#2171, #2766): Use ZstdCodec as default compressor to avoid.
- **Performance regression in v3** (#2710): Large array writes slower due to async overhead. Being addressed.
- **fsspec caching broken before fsspec 2025.7.0** (#2988): `simplecache::` protocols did not work with v3 async stores.
- **Thread-safety of group/array init** (#1435): Not fully thread-safe.
- **F memory order**: v3 always stores in C order. `order='F'` emits a warning.
- **moto mock_aws incompatible** with v3 async stores: Use moto server mode for testing.

## Performance Tips

- **Chunk size**: Target 1 MB+ uncompressed per chunk. Too small = filesystem overhead; too large = poor partial-read performance.
- **Sharding**: Use for datasets with many small chunks on cloud stores. Target 10-100+ chunks per shard.
- **Consolidated metadata**: Always use for read-heavy cloud data. Eliminates per-array metadata reads.
- **ZstdCodec over BloscCodec**: ZstdCodec is not affected by the typesize bug and provides good default compression.
- **Async concurrency**: Tune `zarr.config.set({'async.concurrency': N})` for cloud workloads.
- **Avoid row-by-row iteration**: Significantly slower in v3 (#2529). Use bulk slicing.
