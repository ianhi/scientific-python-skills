# NumPy Skill File Review

**Reviewer**: Fresh-eyes agent (verified against `repos/numpy/` source, main branch heading toward 2.5)
**Skill file**: `skills/numpy.md`
**Grade**: **B+**

The skill file is well-structured, covers the most critical breaking changes, and the vast majority of claims are accurate. There are a few errors that need correction to avoid misleading Claude.

---

## Critical Errors (MUST FIX)

### 1. `np.fix` deprecation version is WRONG
**Skill file says (line 180-181)**: "Do NOT use np.fix (pending deprecation in 2.4)"
**Actual**: `np.fix` is a *pending* deprecation in 2.4 (release notes say "pending deprecation"), and was formally deprecated in **2.5** (`.. deprecated:: 2.5` in `numpy/lib/_ufunclike_impl.py:23`).

**Fix**: Change "pending deprecation in 2.4" to something clearer like "pending deprecation in 2.4, formally deprecated in 2.5". Or simply say "deprecated" since the skill file targets 2.4.2 users who should already be using `np.trunc`.

### 2. `a_min`/`a_max` are NOT deprecated
**Skill file says (line 102-106)**: "WRONG - a_min/a_max deprecated (use min=/max= in 2.1+)"
**Actual**: `a_min`/`a_max` are **not deprecated**. The `min=`/`max=` keyword-only parameters were added in 2.1 as Array API compatible alternatives, but `a_min`/`a_max` remain fully supported with no deprecation warning. The current signature is:
```python
def clip(a, a_min=np._NoValue, a_max=np._NoValue, out=None, *, min=np._NoValue, max=np._NoValue, **kwargs)
```
There is no deprecation warning in the source code for `a_min`/`a_max`.

**Fix**: Remove the "WRONG"/"deprecated" label. Instead, note that `min=`/`max=` are the Array API compatible alternatives added in 2.1+, and recommend them for new code, but `a_min`/`a_max` still work fine.

### 3. `np.cross` deprecation timeline is misleading
**Skill file says (line 145-146)**: "deprecated in 2.0, error in 2.5"
**Actual**: 2D vectors were deprecated in 2.0 (confirmed in 2.0.0 release notes). In the current source (heading toward 2.5), `np.cross` now raises `ValueError` for non-3D vectors. The 2.5 error claim is plausible but based on development code -- for 2.4.2 (the stated target version), 2D vectors still produce a DeprecationWarning, not an error. Since the skill file says "Current version: 2.4.2", claiming "error in 2.5" is a forward-looking claim about an unreleased version.

**Fix**: Say "deprecated in 2.0 (2D vectors emit DeprecationWarning), will error in a future version". Or if you want to be specific: "deprecated in 2.0, expected to error in 2.5".

### 4. `linalg.linalg` and `fft.helper` removal version
**Skill file says (line 116-117)**: "removed in 2.4"
**Actual**: These were made private (DeprecationWarning) in 2.0 and **removed** in 2.4. This is confirmed in the 2.4.0 release notes: "Removed numpy.linalg.linalg and numpy.fft.helper". So the skill file is **correct** that they were removed in 2.4. However, the skill file groups them under "np.core renamed to np._core" which is a 2.0 change. The section structure is confusing because it mixes 2.0 changes with 2.4 removals without clearly distinguishing them.

**Fix**: This is actually correct. No fix needed, but consider clarifying the timeline (deprecated 2.0, removed 2.4).

---

## Minor Issues (SHOULD FIX)

### 5. `np.fromstring` claim needs clarification
**Skill file says (line 70)**: "WRONG - removed in 2.3 / np.fromstring(s, dtype) # use np.frombuffer(s, dtype) (binary mode)"
**Actual**: `np.fromstring` itself is NOT removed -- only its **binary mode** (calling without `sep`) now errors. `np.fromstring` with a `sep` argument still works. The source comment says: "The binary mode of fromstring is removed, use frombuffer instead". The skill file's comment "(binary mode)" is correct but easily misread as "fromstring is removed entirely".

**Fix**: Clarify that `np.fromstring` still exists for text parsing with `sep=`, only binary mode is removed.

### 6. `arr.tostring()` claim needs clarification
**Skill file says (line 69)**: "removed in 2.3 / arr.tostring() # use arr.tobytes()"
**Actual**: Confirmed removed in 2.3 release notes. This is correct.

### 7. `bool8`, `int0`, `object0` not in `_expired_attrs_2_0`
**Skill file says (line 15-17)**: These produce `AttributeError` in 2.0.
**Actual**: These are NOT in the `_expired_attrs_2_0.py` file. They may have been removed via a different mechanism (type system changes). The claim that they produce `AttributeError` is likely correct at runtime, but I cannot confirm from the expired attrs list. The `float_`, `complex_`, `string_`, `unicode_` names ARE in the expired list.

**Fix**: No action needed if runtime behavior is correct, but consider verifying at runtime.

### 8. `meshgrid` return type nuance
**Skill file says (line 213)**: "np.meshgrid(x, y) # tuple"
**Actual**: `meshgrid` returns a `tuple` when `copy=True` (the default) via `tuple(x.copy() for x in output)`. When `sparse=True` and `copy=False`, it also returns `tuple(output)`. When `sparse=False` and `copy=False`, it returns the result of `np.broadcast_arrays` which also returns a tuple. So the claim is correct for all paths.

### 9. `product`, `cumproduct`, `msort` are NOT in `_expired_attrs_2_0`
**Skill file says (line 60-62)**: These were "removed in 2.0".
**Actual**: `product`, `cumproduct`, and `msort` are NOT in the `_expired_attrs_2_0.py` list. They were likely removed via `__init__.py` changes (simply not exported anymore) rather than through the expired attrs mechanism. `alltrue` and `sometrue` ARE in the expired list. The claim they were removed in 2.0 may still be correct at runtime even though they use a different removal mechanism.

**Fix**: No action if runtime behavior is correct.

---

## Missing Content (CONSIDER ADDING)

### 10. `np.clip` `min=`/`max=` are keyword-only
The `min=` and `max=` parameters are keyword-only (after `*`), while `a_min`/`a_max` are positional. This is a practical distinction users should know.

### 11. `np.fromstring` still works for text mode
Since the skill file mentions `fromstring` as removed, it should clarify text mode still works:
```python
# Still works (text mode):
np.fromstring("1 2 3", sep=" ", dtype=float)
# Removed (binary mode):
np.fromstring(b"\x01\x02", dtype=np.uint8)  # use np.frombuffer instead
```

### 12. `correction=` parameter in `np.std` and `np.var`
The skill file mentions `mean=` for `np.std` but doesn't mention `correction=`, which is the Array API compatible alternative to `ddof=`. Both were added and are useful to know about.

### 13. `np.strings.slice` signature
The skill file shows `np.strings.slice(arr, 0, 3)` which is correct but the actual signature is unusual: `def slice(a, start=None, stop=np._NoValue, step=None, /)`. When only one positional arg is given after `a`, it becomes `stop` (like Python's `range()`).

---

## Verified Claims (Confirmed Correct)

1. **Removed type aliases** (line 10-28): `np.float_`, `np.complex_`, `np.string_`, `np.unicode_` are in `_expired_attrs_2_0.py`. Replacements are correct.
2. **Removed constant aliases** (line 31-43): `np.Inf`, `np.Infinity`, `np.infty`, `np.NaN`, `np.NINF`, `np.PINF`, `np.NZERO`, `np.PZERO` all confirmed in `_expired_attrs_2_0.py`.
3. **`np.trapezoid`** (line 48/53): Exists with signature `def trapezoid(y, x=None, dx=1.0, axis=-1)`. Correct.
4. **`np.isin`** replaces `np.in1d` (line 49/54): Correct. `in1d` removed in 2.4 per release notes.
5. **`np.alltrue`/`np.sometrue`** (line 58-59): In expired attrs list. Correct.
6. **`np.round_`** (line 63): In expired attrs list. Correct.
7. **`np.asfarray`** (line 64): In expired attrs list. Correct.
8. **`np.find_common_type`** (line 65): In expired attrs list. Correct.
9. **`np.cast`** (line 66): In expired attrs list. Correct.
10. **`copy=False` semantics** (line 73-84): Correctly describes the 2.0 change. `copy=None` means "copy if needed".
11. **`np.quantile` `interpolation=` removed in 2.4** (line 89-93): Confirmed in 2.4.0 release notes.
12. **`np.reshape` `newshape=` removed in 2.4** (line 95-100): Confirmed in 2.4.0 release notes. Actual signature: `def reshape(a, /, shape, order='C', *, copy=None)`.
13. **`np.core` renamed to `np._core`** (line 111-112): Correct.
14. **NEP 50 type promotion** (line 125-141): Correctly describes the change. Python scalars are "weak", NumPy scalars are "strong".
15. **`np.unique` signature** (line 220-223): Confirmed: `def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)`. Correct.
16. **Array API unique functions** (line 228-232): `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` exist.
17. **`cumulative_sum`/`cumulative_prod`** (line 235-238): Signatures confirmed correct.
18. **`np.astype`** (line 241): `def astype(x, dtype, /, *, copy=True, device=None)` confirmed.
19. **`np.isdtype`** (line 244): Exists, confirmed.
20. **`np.unstack`** (line 249): `def unstack(x, /, *, axis=0)` confirmed.
21. **Linalg functions** (line 253-263): All confirmed:
    - `matrix_norm(x, /, *, keepdims=False, ord="fro")` -- correct
    - `vector_norm(x, /, *, axis=None, keepdims=False, ord=2)` -- correct
    - `diagonal(x, /, *, offset=0)` -- correct
    - `trace(x, /, *, offset=0, dtype=None)` -- correct
    - `svdvals(x, /)` -- correct
    - `outer(x1, x2, /)` -- correct
    - `cross(x1, x2, /, *, axis=-1)` -- correct
    - `vecdot(x1, x2, /, *, axis=-1)` -- correct
    - `matrix_transpose(x, /)` -- correct
22. **`arr.mT`** (line 267): Confirmed (matrix transpose attribute).
23. **`np.concat`, `np.permute_dims`** (line 274-275): Both found in `__init__.py` imports. Correct.
24. **StringDType** (line 285-307): `StringDType` class confirmed in `numpy/dtypes.pyi`. `na_object` and `coerce` properties confirmed.
25. **`np.strings` functions** (line 297-301, 309-319): `find`, `upper`, `replace`, `slice` all confirmed in `numpy/_core/strings.py`.
26. **`__array__` protocol** (line 338-345): `copy=` keyword confirmed in test files and type stubs.
27. **`__array_prepare__` removed** (line 349): Confirmed in 2.0.0 release notes.
28. **`__array_wrap__` `return_scalar`** (line 352-353): Confirmed in test files.
29. **`np.reshape` copy= parameter** (line 375-380): Confirmed: `def reshape(a, /, shape, order='C', *, copy=None)`.
30. **`np.std` `mean=` parameter** (line 494): Confirmed in source.
31. **`np.sort` `stable=` keyword** (line 495): Confirmed: `def sort(a, axis=-1, kind=None, order=None, *, stable=None)`.
32. **`ndmax=` for `np.array`** (line 500): Confirmed in `_add_newdocs.py`.
33. **Exceptions in `numpy.exceptions`** (line 198-205): `AxisError`, `ComplexWarning`, `VisibleDeprecationWarning`, `RankWarning`, `DTypePromotionError` all confirmed in `numpy/exceptions.py`.
34. **`np.sum` on generators errors in 2.4** (line 169-178): Confirmed in 2.4.0 release notes.
35. **`gradient` returns tuple** (line 211): Confirmed: `return tuple(outvals)` at line 1395.
36. **`broadcast_arrays` returns tuple** (line 213): Confirmed: `return tuple(result)`.
37. **`atleast_1d` returns tuple** (line 212): Confirmed: `return tuple(res)`.

---

## Summary

| Category | Count |
|----------|-------|
| Critical errors | 3 (a_min/a_max NOT deprecated, np.fix version, np.cross timeline) |
| Minor issues | 3 (fromstring clarification, bool8/int0/object0 verification, section structure) |
| Missing content | 4 suggestions |
| Verified correct | 37 claims |

**Overall**: The skill file is quite good. The most impactful error is the `a_min`/`a_max` deprecation claim -- calling working code "WRONG" will mislead Claude into unnecessarily rewriting valid code. The `np.fix` version mismatch is minor. The `np.cross` timeline could be more precise.
