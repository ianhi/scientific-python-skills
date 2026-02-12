# Zarr Skill Assessment

**Date**: 2026-02-12
**Zarr version**: 3.1.5
**Python version**: 3.12
**Skill file**: `skills/zarr.md`
**Test script**: `tests/zarr_test/test_zarr_skill.py`

## Verified Correct

34 out of 34 claims verified. Every testable claim in the skill file was confirmed accurate.

- **Codec classes moved to zarr.codecs**: `BloscCodec`, `ZstdCodec`, `GzipCodec` all import from `zarr.codecs`
- **Factory functions**: `zarr.create_array()` works, returns `zarr.Array`
- **Dot notation removed**: `group.myarray` raises `AttributeError`, bracket syntax `group['myarray']` works
- **Store classes renamed**: `LocalStore` and `FsspecStore` import from `zarr.storage`
- **compressors= (plural)**: works on `create_array()` with zarr.codecs codec instances
- **resize requires tuple**: `z.resize((200, 50))` works, `z.resize(300, 50)` raises `TypeError`
- **from_array signature**: store-first, data keyword-only. `zarr.from_array(data, store)` fails as expected
- **create_dataset deprecated**: `group.create_dataset()` still works (with deprecation warning)
- **numcodecs mixing blocked**: `compressors=numcodecs.Blosc()` raises an error
- **codecs= does NOT work on create_array**: `TypeError: create_array() got an unexpected keyword argument 'codecs'` -- confirmed
- **serializer= + compressors= works**: `BytesCodec(endian='little')` + `BloscCodec(cname='zstd', ...)` works correctly
- **compressors=None**: disables compression, works correctly
- **Sharding**: `chunks=` + `shards=` parameters work correctly together
- **Blosc typesize bug**: Blosc+shuffle produces valid data but with degraded compression. Both codecs round-trip correctly.
- **write_empty_chunks=False default**: writing all-zero data to an array with `fill_value=0.0` results in `nchunks_initialized=0`
- **write_empty_chunks=True via config**: `config={"write_empty_chunks": True}` forces chunk storage, `nchunks_initialized=10` as expected
- **Mode 'r'**: read-only, prevents writes
- **Mode 'r+'**: read/write existing, fails on missing store
- **Mode 'w-'**: create only, fails if exists
- **Mode 'w'**: overwrite (destructive), replaces existing data
- **zarr.codecs.BloscCodec != numcodecs.Blosc**: confirmed different types
- **zarr.codecs.numcodecs.Blosc wrapper**: imports successfully
- **LocalStore works for local files**: full read/write round-trip
- **open_array mode is hidden kwarg**: `mode` is NOT in explicit params, passed via `**kwargs`
- **Object dtype not supported**: `dtype="|O"` raises error
- **F memory order warning**: `order="F"` accepted with warning
- **zarr.config.set**: `{"async.concurrency": 10}` works
- **BloscShuffle enum**: has `shuffle` and `noshuffle` members
- **zarr_format defaults to 3**: confirmed `z.metadata.zarr_format == 3`
- **zarr_format=2 for backwards compat**: works, returns `zarr_format == 2`
- **BytesCodec exists**: imports from `zarr.codecs`
- **String path for open_array**: works as FSMap replacement
- **Consolidated metadata**: `consolidate_metadata()` + `open_consolidated()` works
- **FsspecStore async requirement**: `FsspecStore(fsspec.filesystem("file"), ...)` raises `TypeError: Filesystem needs to support async operations.` exactly as documented

## Issues Found

- **Sharding example missing dtype**: The sharding code example on line 53-58 does not include `dtype=`. In zarr 3.1.5, `create_array()` requires either `data=` or `dtype=`. The example will fail with: `ValueError: The data parameter was set to None, but dtype was not specified.` **Fix**: Add `dtype="float32"` to the sharding example.

## Missing Content

- **Consolidated metadata warning**: `zarr.consolidate_metadata()` emits `ZarrUserWarning: Consolidated metadata is currently not part in the Zarr format 3 specification.` The skill file mentions consolidated metadata in performance tips but doesn't warn about this.
- **nbytes_stored is a method, not a property**: `z.nbytes_stored` must be called as `z.nbytes_stored()`. This is a minor API detail but could cause confusion. (Not in skill file, so not an error, just a note.)

## Overall Assessment

The zarr skill file is **highly accurate**. Every claim about v2-to-v3 migration, codec pipeline, mode semantics, store classes, and known limitations verified correctly against zarr 3.1.5. The only concrete issue is a missing `dtype=` parameter in the sharding example code block. The previous report's main finding (incorrect `codecs=` usage on `create_array()`) has been fixed -- the current skill file correctly documents that `codecs=` only works on `ShardingCodec` and that `create_array()` uses `serializer=`/`compressors=`/`filters=`.
