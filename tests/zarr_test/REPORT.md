# Zarr Skill File Test Report

**Date**: 2026-02-11
**Zarr version tested**: 3.1.5
**Python version**: 3.12
**Skill file**: `/Users/ian/Documents/dev/scientific-python-skills/skills/zarr.md`

## Summary

7 out of 8 test sections worked correctly on the first attempt using only guidance from the skill file. One section (explicit codec pipeline) failed due to an incorrect parameter name documented in the skill file.

## Sections That Worked Correctly

### 1. Sharded zarr v3 array with ZstdCodec -- PASS
The skill file's guidance on `chunks=`, `shards=`, and `compressors=zarr.codecs.ZstdCodec(level=3)` was accurate. Round-trip data integrity confirmed.

### 2. Group hierarchy with subgroups -- PASS
Creating groups with `create_group()`, creating arrays within subgroups, and accessing via `root["experiment/raw/sensor_data"]` path syntax all worked exactly as documented.

### 3. `from_array` -- PASS
The skill file correctly warns that `data` is keyword-only and `store` is the first positional arg. The example `zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))` worked correctly.

### 5. Consolidated metadata -- PASS
`zarr.consolidate_metadata()`, `zarr.open_consolidated()`, and `zarr.open_group(..., use_consolidated=True)` all worked as documented. Note: zarr emits a warning that consolidated metadata is not yet part of the Zarr format 3 specification -- the skill file does not mention this.

### 6. `zarr.config.set()` context manager -- PASS
The context manager syntax `with zarr.config.set({'array.write_empty_chunks': True})` worked correctly. With `write_empty_chunks=True`, writing all-zero data produced 1 chunk; with the default `False`, it produced 0 chunks. This confirms the skill file's documentation of the `write_empty_chunks` default change.

### 7. Mode semantics -- PASS
All mode behaviors matched the skill file documentation:
- `mode='r'`: Read-only, raises `ValueError` on write attempts
- `mode='r+'`: Read/write on existing, raises `FileNotFoundError` on non-existent
- `mode='a'`: Creates if missing, opens existing if present

### 8. Variable-length string dtype -- PASS
`dtype='string'` worked exactly as documented, creating a `StringDType()` array that round-trips string data correctly.

## Sections That Failed or Had Issues

### 4. Explicit codec pipeline (`codecs=` parameter) -- FAIL on first attempt

**The Problem**: The skill file shows `codecs=` as a valid parameter for `zarr.create_array()` in multiple places:

1. In the "Codec/Compression Patterns (v3)" section:
   ```python
   z = zarr.create_array(
       store, shape=(1000,), dtype='f4',
       codecs=[
           zarr.codecs.BytesCodec(endian='little'),
           zarr.codecs.BloscCodec(cname='zstd', clevel=5, ...),
       ]
   )
   ```

2. In the "Do NOT use compressor=/filters=" section (item #5):
   ```python
   z = zarr.create_array(store, shape=(100,), dtype='f4',
                         codecs=[zarr.codecs.BytesCodec(), zarr.codecs.BloscCodec()])
   ```

3. In the `create_array()` signature, `codecs=` is not listed -- but the examples contradict the signature.

**The Exact Error**:
```
TypeError: create_array() got an unexpected keyword argument 'codecs'
```

**The Correct API**: `zarr.create_array()` uses separate keyword arguments:
- `serializer=` for the bytes-to-bytes serializer (e.g., `BytesCodec`)
- `compressors=` for compression codecs (e.g., `BloscCodec`, `ZstdCodec`)
- `filters=` for array-to-array transforms

The `codecs=` parameter only exists on `ShardingCodec.__init__()`, not on `create_array()`.

**Working version**:
```python
z = zarr.create_array(
    store=store_path,
    shape=(1000,),
    dtype="float32",
    serializer=zarr.codecs.BytesCodec(endian="little"),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5,
                                        shuffle=zarr.codecs.BloscShuffle.shuffle),
)
```

## What Was Missing From the Skill File

1. **Consolidated metadata warning**: When using `zarr.consolidate_metadata()` with zarr format 3, a `ZarrUserWarning` is emitted: "Consolidated metadata is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future." This is worth mentioning in the skill file since it affects interoperability.

2. **Actual `create_array()` signature does not include `codecs=`**: The skill file's signature block for `create_array()` correctly lists `compressors=`, `serializer=`, and `filters=`, but then the code examples throughout the file use the non-existent `codecs=` parameter. These examples are internally inconsistent with the signature shown.

3. **How `serializer=` and `compressors=` interact**: The skill file shows `codecs=` as the way to build a full pipeline but never shows how to combine `serializer=` and `compressors=` as separate keyword arguments.

## Suggestions for Improving the Skill File

1. **Remove all `codecs=` examples from `create_array()` calls.** Replace them with the correct `serializer=` + `compressors=` syntax. The `codecs=` parameter only exists on `ShardingCodec`, not on `create_array()`.

2. **Add an explicit "Full codec pipeline" example** showing the correct way to specify both serializer and compressors:
   ```python
   z = zarr.create_array(
       store, shape=(1000,), dtype='f4',
       serializer=zarr.codecs.BytesCodec(endian='little'),
       compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=5),
   )
   ```

3. **Clarify the `codecs=` parameter scope**: Explicitly note that `codecs=` is only valid inside `ShardingCodec()`, not in `create_array()`. This is a very likely source of confusion.

4. **Add a note about the consolidated metadata warning** for zarr format 3, since it affects whether users should rely on this feature for cross-implementation compatibility.

5. **In the "Do NOT use compressor=/filters=" section (item #5)**, the "RIGHT" example using `codecs=` is actually wrong. It should be updated to use `compressors=` and `serializer=`:
   ```python
   # RIGHT - v3 uses compressors=/serializer=/filters= (NOT codecs=)
   z = zarr.create_array(store, shape=(100,), dtype='f4',
                         serializer=zarr.codecs.BytesCodec(),
                         compressors=zarr.codecs.BloscCodec())
   ```
