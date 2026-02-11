# Zarr Skill File Review

Reviewed against source at: `repos/zarr-python/src/zarr/` (appears to be latest main branch)

---

## ACCURACY

### Anti-pattern #1: Do NOT import codecs from zarr namespace
**Verdict: Correct.** `zarr.__init__.py` does not export Blosc, Zlib, or Delta. The "RIGHT" imports (`zarr.codecs.BloscCodec`, `ZstdCodec`, `GzipCodec`, `BytesCodec`) all exist in `zarr/codecs/__init__.py`. Good.

### Anti-pattern #2: Do NOT construct Array/Group directly
**Verdict: Mostly correct but slightly misleading.** The claim that `zarr.Array(store, shape=(100,), dtype='f4')` will fail is true -- the Array constructor in v3 takes an `AsyncArray`, not a store+shape. However, saying "constructor signatures completely changed" is vague. It would be more useful to say "Array/Group constructors take internal async objects, not user-facing arguments."

### Anti-pattern #3: Do NOT use dot notation for group members
**Verdict: Correct.** There is no `__getattr__` on the Group class that would enable `group.array_name`. Only `__getitem__` exists.

### Anti-pattern #4: Do NOT use v2 store classes
**Verdict: Correct.** `zarr/storage/__init__.py` exports exactly `LocalStore, MemoryStore, ZipStore, FsspecStore, ObjectStore, LoggingStore, WrapperStore, GpuMemoryStore, StoreLike, StorePath`. The v2 store names do not exist. The skill file omits `GpuMemoryStore`, `LoggingStore`, `WrapperStore` from the "RIGHT" list -- minor but acceptable since those are less commonly used.

### Anti-pattern #5: Do NOT use compressor=/filters= with v3 format
**Verdict: Partially correct, needs nuance.** The `create_array()` function does NOT have `compressor=` or v2-style `filters=` params. But the older `zarr.create()` function DOES still have `compressor=` (for v2 compat). The skill correctly shows `create_array` as the target. However, saying `compressor=Blosc()` is "WRONG" on `create_array` is accurate because `create_array` does not have a `compressor` parameter at all.

**Issue: The `compressors=` parameter can accept a single codec, not just an iterable.** The skill file shows `compressors=zarr.codecs.ZstdCodec(level=3)` (singular) which technically works because the source type is `CompressorsLike` which likely accepts both. Verified in the type alias -- this is fine.

### Anti-pattern #6: Do NOT use FSMap with v3
**Verdict: Correct.** Issue #2706 reference plausible. The code examples look correct.

### Anti-pattern #7: Do NOT use resize with separate args
**Verdict: Correct.** `Array.resize()` takes `new_shape: ShapeLike` -- a single argument. `z.resize(1000, 1000)` would fail because there is only one positional parameter.

### Anti-pattern #8: Do NOT mix v2/v3 codec types
**Verdict: Correct.** Passing numcodecs objects to v3 `compressors=` would fail. The `zarr_format=2` workaround with `compressor=` (singular) is shown correctly as using the old `zarr.create_array` with `zarr_format=2` -- wait, actually `create_array` does NOT have `compressor=` (singular). It has `compressors=` (plural). **This is a bug in the skill file.** The last example uses `compressor=numcodecs.Blosc()` but `create_array` doesn't have that parameter. It would need to use `zarr.create()` (the legacy wrapper) instead, or the example is wrong.

### Anti-pattern #9: Do NOT use v2 dtype strings for v3
**Verdict: Partially accurate.** `'<i4'` might actually work -- zarr v3 can parse numpy dtype strings. The claim that v2 endian prefixes are "WRONG" is overstated. The `string` dtype for vlen strings is correct.

### Anti-pattern #10: Do NOT use removed modules
**Verdict: Correct.** Those modules are not present in the v3 source tree.

### Core API: `zarr.create_array()` signature
**Several issues:**
1. **Missing `name` parameter.** The actual signature has `name: str | None = None` which is not shown in the skill file. This is an important omission -- `name` specifies the path within the store.
2. **`chunks` type is wrong.** Skill file says `ChunkLike = 'auto'` but actual is `chunks: tuple[int, ...] | Literal["auto"] = "auto"`. The type `ChunkLike` does not exist in the codebase.
3. **`zarr_format` default is wrong.** Skill file says `zarr_format: ZarrFormat = None` but the actual `create_array` has `zarr_format: ZarrFormat | None = 3` (defaults to 3). This is different from `open_array` which has `None`. This is a meaningful difference.
4. **Missing `overwrite` parameter.** The actual signature has `overwrite: bool = False`.
5. **Missing `write_data` parameter.** The actual signature has `write_data: bool = True`.

### Core API: `zarr.open_array()` signature
**Issue: `mode` parameter is shown as explicit but it is actually passed via `**kwargs`.** The actual signature is `open_array(store, *, zarr_version, zarr_format, path, storage_options, **kwargs)` -- mode is extracted from kwargs internally (`mode = kwargs.pop("mode", None)`). Showing it as an explicit parameter `mode: AccessModeLiteral = 'a'` is misleading about the actual API surface. It works when passed, but it is a hidden kwarg.

### Core API: `zarr.open_group()` signature
**Correct.** The actual signature matches what is shown, including `use_consolidated`.

### Core API: `zarr.create_group()` signature
**Missing `storage_options` parameter.** The actual has `storage_options: dict[str, Any] | None = None`.

### Batch Creation section
**Issue: `ArrayV3Metadata` is not trivially constructible.** The skill file shows:
```python
from zarr.core.metadata import ArrayV3Metadata as ArrayMetadata
'experiment/raw/images': ArrayMetadata(shape=(1000, 256, 256), dtype='uint16'),
```
But `ArrayV3Metadata.__init__` requires `shape, data_type, chunk_grid, chunk_key_encoding, fill_value, codecs, ...` -- it is NOT a simple `(shape, dtype)` constructor. This example would fail. The `create_hierarchy` function expects metadata objects, which are complex to construct manually. This section is **actively misleading**.

### `from_array` convenience example
**Issue: Wrong call signature in Common Operations.** The skill file shows:
```python
z = zarr.from_array(np_array, store='data.zarr', chunks=(100, 100))
```
But the actual signature is `from_array(store, *, data, ...)` -- store is the first positional argument, data is keyword-only. The correct call would be:
```python
z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))
```
**This is a significant error that would cause immediate failure.**

### Sharding section
**Issue: `target_shard_size_bytes` default is `None`, not 128 MiB.** The skill file says "uses config.array.target_shard_size_bytes = 128 MiB" but the actual config default is `None`. When `None`, auto-sharding falls back to a heuristic of `a_shape // c_shape > 8`. The 128 MiB claim appears fabricated.

### `save_array` convenience
**Correct.** `zarr.save_array('data.zarr', np_array)` matches the actual signature `save_array(store, arr, ...)`.

### Group methods: `list(grp.members())`
**Slightly misleading.** The sync `Group.members()` actually returns a `tuple[tuple[str, AnyArray | Group], ...]`, not a generator. Calling `list()` on it works but is redundant. The docstring even (incorrectly) says "Returns an AsyncGenerator" for the sync version, but the return type annotation says tuple.

### Configuration
**Mostly correct.** The `zarr.config.set()` API exists and the keys shown match the actual config structure. `async.concurrency` defaults to 10 (matches). `threading.max_workers` defaults to `None` (not shown in skill, fine).

### Known Limitations: "F memory order: v3 always stores in C order"
**Needs verification.** The config has `"order": "C"` as default but I see no evidence that F order "emits a warning." This claim may be exaggerated. The `create_array` docs say order is deprecated for v3 in favor of the `config` parameter.

---

## USEFULNESS (1-5 per section)

| Section | Rating | Notes |
|---------|--------|-------|
| CRITICAL: Anti-patterns | 5/5 | The most valuable section. Claude absolutely will generate v2 code without this. Keep and expand. |
| Quick Reference (install/import) | 3/5 | Boilerplate, mildly useful. Could be trimmed. |
| Common Operations | 4/5 | Very useful quick recipes, but contains the `from_array` bug. Fix it. |
| Core API signatures | 3/5 | Useful but several are inaccurate. An inaccurate signature is worse than no signature -- Claude will trust it. Either fix or remove. |
| Store Configuration | 5/5 | Excellent. Cloud storage patterns are exactly what people struggle with. |
| Codec/Compression Patterns | 4/5 | Good coverage. The v3 codec pipeline is confusing and this helps. |
| Sharding | 4/5 | Important v3 feature, well covered. Fix the 128 MiB claim. |
| Consolidated Metadata | 4/5 | Important for cloud use cases. Good. |
| Configuration | 3/5 | Useful but niche. Most users will not touch config. |
| Batch Creation | 2/5 | The example is broken. `ArrayV3Metadata` is not trivially constructible. Remove or fix significantly. |
| Integration (xarray/dask) | 4/5 | Very commonly needed. Could be expanded. |
| Gotchas | 5/5 | Excellent section. Every entry is valuable. |
| Known Limitations | 4/5 | Good reference. Some claims need verification. |
| Performance Tips | 4/5 | Concise and actionable. Good. |

### Sections to cut or reduce to save space:
1. **Batch Creation** -- Remove entirely. The example is broken and `create_hierarchy` is an advanced API that most users will not need. If kept, it needs a complete rewrite.
2. **Core API signatures** -- Either fix them or remove them. Inaccurate signatures actively harm Claude's code generation. Consider showing only the most important 3-4 parameters per function rather than full signatures.
3. **Configuration** -- Could be trimmed to just the context manager pattern and env var format.

### Sections to expand:
1. **Anti-patterns** -- These are the highest-value content. Consider adding:
   - Do NOT use `zarr.create()` for new code (it is legacy, prefer `create_array`)
   - Do NOT use `zarr.open()` when you know the type (use `open_array` or `open_group`)
2. **Integration** -- Add a section on `anndata` / `ome-zarr` / `tifffile` since these are major zarr consumers in practice.
3. **Gotchas** -- Add gotcha about `from_array` store-first signature being different from what you might expect.

---

## MISSING

### Missing Common Patterns

1. **`zarr.create()` vs `zarr.create_array()`**: The skill file uses `create_array` throughout but never explains that `zarr.create()` is the legacy function that still exists. Claude may encounter both in existing code. Explain when to use which.

2. **`zarr.open()` vs `zarr.open_array()` vs `zarr.open_group()`**: `zarr.open()` dispatches based on what it finds. This is important to know.

3. **Async API**: Zarr v3 is fundamentally async-first. The skill never mentions `AsyncArray` or `AsyncGroup` or when to use the async API. For users writing async code (common with cloud storage), this is important.

4. **`Group.require_group()` and `Group.require_groups()`**: These are useful for idempotent hierarchy creation and are mentioned nowhere.

5. **`zarr.zeros()`, `zarr.ones()`, `zarr.full()`, `zarr.empty()`**: These NumPy-like convenience functions exist and are exported but never mentioned. They are commonly used for quick initialization.

6. **`GpuMemoryStore`**: For GPU users, this is important and not mentioned.

7. **`zarr.save()` and `zarr.save_group()`**: The skill only shows `save_array`. `save()` and `save_group()` handle multiple arrays.

8. **Reading v2 data with v3 library**: A very common real-world scenario. How to open a v2 zarr store with the v3 library (it mostly "just works" but `zarr_format=2` can be helpful).

9. **`Group.__iter__()` and `Group.__contains__()`**: Group supports `for name in group:` and `'array_name' in group`. These are useful idioms not mentioned.

### Missing Gotchas

1. **`create_array` defaults to `zarr_format=3` but `open_group`/`open_array` default to `None` (auto-detect).** This is a subtle but important inconsistency that can cause confusion when mixing creation and opening.

2. **`from_array` has store as first positional arg, data as keyword-only.** This is counterintuitive -- most people expect `from_array(data, store)`. This deserves a prominent gotcha.

3. **`open_array` mode is a hidden kwarg.** It is passed through `**kwargs`, not an explicit parameter. This means IDE autocompletion will not show it.

4. **Sync `members()` returns a tuple, not a generator.** Unlike many Python APIs that return iterators, this eagerly loads everything.

5. **`zarr.codecs` namespace includes numcodecs wrappers.** The `zarr.codecs.__init__.py` imports `Blosc`, `Zlib`, `Delta`, etc. from `zarr.codecs.numcodecs`. These are NOT the same as `numcodecs.Blosc` -- they are zarr v3 wrappers. This is extremely confusing and the skill file should warn about it.

---

## SUGGESTIONS

### Critical Fixes (must do)

1. **Fix the `from_array` example in Common Operations.** Change:
   ```python
   z = zarr.from_array(np_array, store='data.zarr', chunks=(100, 100))
   ```
   to:
   ```python
   z = zarr.from_array('data.zarr', data=np_array, chunks=(100, 100))
   ```

2. **Fix anti-pattern #8 v2 example.** Change:
   ```python
   z = zarr.create_array(store, shape=(100,), dtype='f4',
                         compressor=numcodecs.Blosc(), zarr_format=2)
   ```
   to use `zarr.create()` instead of `zarr.create_array()`, since `create_array` does not have a `compressor=` (singular) parameter:
   ```python
   z = zarr.create(shape=(100,), dtype='f4', store=store,
                   compressor=numcodecs.Blosc(), zarr_format=2)
   ```

3. **Fix `create_array` signature.** Add `name` parameter, fix `zarr_format` default to `3` (not `None`), add `overwrite`, remove non-existent `ChunkLike` type.

4. **Fix `open_array` signature.** Remove explicit `mode` parameter (it is a hidden kwarg via `**kwargs`). Or add a note that mode is passed via kwargs.

5. **Remove or completely rewrite the Batch Creation section.** The `ArrayV3Metadata(shape=..., dtype=...)` call will fail. `ArrayV3Metadata` requires many more fields.

6. **Fix the `target_shard_size_bytes` claim.** Default is `None`, not `128 MiB`. When `None`, auto-sharding uses a different heuristic.

### Important Improvements

7. **Add a note about `zarr.create()` being legacy.** Add something like:
   ```
   # NOTE: zarr.create() exists for v2 compatibility but zarr.create_array() is preferred for new code
   ```

8. **Add the `name` parameter to common examples.** When creating arrays within groups or at non-root paths, `name` is used instead of `path`:
   ```python
   z = zarr.create_array(store='data.zarr', name='measurements/temp', shape=(100,), dtype='f4')
   ```

9. **Warn about `zarr.codecs.Blosc` vs `numcodecs.Blosc` confusion.** `zarr.codecs.__init__.py` exports a `Blosc` class (from `zarr.codecs.numcodecs`) that is a v3 wrapper, NOT the same as `numcodecs.Blosc`. This is extremely confusing and deserves a callout:
   ```python
   # CAUTION: These are DIFFERENT classes!
   from zarr.codecs.numcodecs import Blosc as ZarrBlosc   # v3 wrapper
   from numcodecs import Blosc as NumcodecsBlosc           # raw numcodecs
   # For v3 arrays, use zarr.codecs.BloscCodec (native v3)
   # The zarr.codecs.numcodecs wrappers are for edge cases
   ```

10. **Reorder sections.** The current order is good but I would move "Gotchas & Common Mistakes" up to immediately after "Common Operations" since they are closely related. Performance tips could stay at the bottom.

### Minor Polish

11. In the `open_array` section of "Mode semantics changed", the skill claims mode='a' was "buggy before 3.0.8" and "could DELETE existing data." This is a strong claim. Consider softening to "had inconsistent behavior" unless you can verify the exact bug.

12. The "Known Limitations" bullet about `copy/copy_all incomplete` says "Not implemented" -- verified in the source where both `copy()` and `copy_all()` have docstrings saying "Not implemented." This is accurate.

13. The dask integration note about issue #962 seems quite old and may not be relevant for v3. Consider verifying or removing.

14. Anti-pattern #9 about dtype strings: `'<i4'` probably does work via numpy dtype parsing in v3. The warning about endian prefixes is overly strong. The real issue is with `'|O'` (object dtype) which genuinely does not work the same way in v3. Consider refining this to focus on the actual breakage.

---

## OVERALL ASSESSMENT

**Grade: B-**

The skill file is well-structured and covers the most critical territory: preventing Claude from generating v2 code. The anti-patterns section alone makes this file valuable. However, there are several factual errors that would cause Claude to generate broken code (`from_array` signature, `ArrayV3Metadata` constructor, `compressor=` on `create_array` for v2, wrong defaults). These errors are especially damaging in a skill file because Claude will trust them implicitly.

**Top 3 priorities:**
1. Fix the broken code examples (from_array, batch creation, anti-pattern #8)
2. Fix the inaccurate API signatures (create_array, open_array)
3. Add the `zarr.codecs.numcodecs` confusion warning
