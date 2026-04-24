# Zarr v2 -> v3 Migration & Anti-Patterns

Companion to `zarr.md`. This file documents breaking changes between zarr v2
and v3, and specific anti-patterns that recur in Claude-generated code.

If you learned zarr v2 patterns, most of what you know will still work — but
a handful of critical changes will silently break or error in v3.

## Critical: Stay at the Public API

The most pervasive anti-pattern across Claude sessions (145+ occurrences): when
debugging, Claude drops to internal APIs instead of using the zarr group/array
interface. This cascades into async errors, type mismatches, and broken
imports because the internals are unstable.

```python
# WRONG - over-specified internal imports
from zarr.core.buffer.cpu import Buffer
from zarr.core.buffer import default_buffer_prototype, PROTOTYPE
from zarr.core.buffer.numpy_buffer import NumpyBuffer
from zarr.core.sync import sync
from zarr.storage._common import StorePath
from zarr.abc.store import Store

# WRONG - manual async store calls
asyncio.run(store.list_prefix(""))       # async generator, not awaitable
asyncio.run(store.delete("data/c/0"))    # just use the array API
sync(store.list_prefix(""))              # TypeError: can't await async generator
store.set("key", some_bytes)             # needs await AND Buffer type, not bytes
store.get("key", prototype=_PROTOTYPE)   # don't call store.get() directly

# WRONG - constructing internal objects manually
AsyncGroup(GroupMetadata(), store_path=StorePath(store=store, path="root"))
zarr.api.asynchronous.open_group(store, mode="r")

# RIGHT - stay at the zarr level
root = zarr.open_group(store)             # sync, handles everything
root = zarr.create_group(store)           # create new group
arr = root.create_array("data", shape=(10,), dtype="f8")
arr[:] = np.arange(10)
print(root.tree())                        # inspect structure
print(list(root.members()))               # list contents
data = arr[:]                             # read data
```

**Rule:** If you're importing from `zarr.core.*`, `zarr.abc.*`, or
`zarr.storage._*`, stop and use the top-level zarr API instead.

## Breaking Changes Reference

| v2 pattern (WRONG in v3) | v3 pattern (CORRECT) |
|--------------------------|----------------------|
| `from zarr import Blosc, Zlib` | `from zarr.codecs import BloscCodec, ZstdCodec, GzipCodec` |
| `zarr.Array(store, ...)` | `zarr.create_array(store, ...)` or `zarr.open_array(...)` |
| `group.array_name` | `group['array_name']` |
| `from zarr import DirectoryStore, FSStore` | `from zarr.storage import LocalStore, FsspecStore` |
| `compressor=Blosc()` | `compressors=zarr.codecs.BloscCodec()` (plural!) |
| `compressors=[numcodecs.Blosc()]` | `compressors=zarr.codecs.BloscCodec()` (don't mix) |
| `mapper = fsspec.get_mapper(...); zarr.open(mapper)` | `zarr.open_array('s3://...')` or `FsspecStore.from_url(...)` |
| `z.resize(1000, 1000)` | `z.resize((1000, 1000))` (tuple for N-D) |
| `zarr.from_array(data, 'store.zarr')` | `zarr.from_array('store.zarr', data=data)` (store first) |
| `grp.create_dataset('name', ...)` | `grp.create_array('name', ...)` |
| `grp.create_array('x', shape=s, dtype=d, data=arr)` | See "shape+dtype+data" below |
| `group.items()` | `group.members()` or `group.arrays()` / `group.groups()` |
| `node.flat[0]` | `node[0]` (zarr.Array has no `.flat`) |

## Removed Features (No v3 Equivalent)

- **Object dtype** (`dtype='|O'`) and **structured dtypes** — use separate
  arrays instead (#2134)
- **Dot notation** for group members — use `group['member']` bracket syntax
- **`group.items()`** — removed. Use `group.members()` for `(name, child)`
  pairs or `group.arrays()` / `group.groups()` for filtered iteration

## Silent Default Changes

These don't error — they just behave differently. Watch out for:

| Option | v2 default | v3 default | Impact |
|--------|-----------|-----------|--------|
| `write_empty_chunks` | `True` | `False` | Fill-value chunks skipped; reads return fill silently |
| `zarr_format` | 2 | 3 | Use `zarr_format=2` for backwards-compat files |

## Specific Anti-Patterns

### `create_array(data=, shape=, dtype=)` — can't mix

```python
# WRONG — raises ValueError
group.create_array("arr", shape=(3,), dtype="i4", data=np.array([1, 2, 3]))
# ValueError: The data parameter was used, but the shape parameter was also used.

# RIGHT — two-step: create then assign
arr = group.create_array("arr", shape=(3,), dtype="i4")
arr[:] = np.array([1, 2, 3])

# RIGHT — infer from data (no shape/dtype)
arr = zarr.from_array(store, data=np.array([1, 2, 3], dtype="i4"))
```

### `create_array` without `data=` requires `shape` AND `dtype`

```python
# WRONG - missing dtype
root.create_array("arr", shape=(1,))
# ValueError

# RIGHT
root.create_array("arr", shape=(1,), dtype="i4")
```

### `store.list_prefix()` is an async generator, not a coroutine

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

### `store.set()` is async — calling without `await` silently drops writes

```python
# WRONG — coroutine created but never awaited (SILENT FAILURE)
store.set("zarr.json", meta)
# RuntimeWarning: coroutine 'Store.set' was never awaited

# ALSO WRONG — set() requires Buffer, not bytes
await store.set("zarr.json", b'{"zarr_format": 3}')
# TypeError: value must be a Buffer instance

# RIGHT — use zarr API, not raw store
root = zarr.create_group(store)
arr = root.create_array("data", shape=(10,), dtype="f8")
arr[:] = np.arange(10)
```

### Stores are not iterable

```python
# WRONG
list(store)   # TypeError: not iterable

# RIGHT
root = zarr.open_group(store)
print(root.tree())
```

### `zarr.Array` has no `.flat`

```python
# WRONG (numpy habit)
arr.flat[0]   # AttributeError

# RIGHT
arr[0]
```

### Blosc shuffle silently broken in v3 (#2171, #2766)

Blosc typesize defaults to 1 byte in v3 (instead of dtype size), causing
10-20x worse compression with shuffle. `ZstdCodec` is not affected.

```python
# SLOW - shuffle ineffective, compressor thinks data is bytes
zarr.create_array(store, shape=(1000,), dtype='float64',
                  compressors=zarr.codecs.BloscCodec(cname='lz4', clevel=1,
                    shuffle=zarr.codecs.BloscShuffle.shuffle))

# FAST - ZstdCodec works correctly and is the good default
zarr.create_array(store, shape=(1000,), dtype='float64',
                  compressors=zarr.codecs.ZstdCodec(level=3))
```

### `from_array()` signature changed

```python
# WRONG (v2 order)
z = zarr.from_array(np_array, store='data.zarr')  # TypeError

# RIGHT (v3 — store first, data keyword-only)
z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))
```

### `zarr.codecs.Blosc` is NOT `numcodecs.Blosc`

```python
# These are DIFFERENT types — do not interchange
from zarr.codecs import BloscCodec, ZstdCodec  # native v3 (preferred)
from zarr.codecs.numcodecs import Blosc         # v3 wrapper class
from numcodecs import Blosc                     # raw numcodecs (don't use with v3)
```

### FsspecStore needs async filesystem (#2706)

```python
# WRONG - local filesystem is not async
store = zarr.storage.FsspecStore(fsspec.filesystem("file"), path="/local/path")

# RIGHT - use LocalStore for local files
store = zarr.storage.LocalStore("/local/path")

# RIGHT - for cloud, use asynchronous=True
fs = s3fs.S3FileSystem(anon=True, asynchronous=True)
store = zarr.storage.FsspecStore(fs, path="bucket/data.zarr")
```

## `mode='a'` History (#3062)

Before zarr 3.0.8, `mode='a'` on `open_array` was buggy — it could delete
existing data. Always use 3.0.8+ to avoid this footgun.
