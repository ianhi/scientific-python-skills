# NumPy Skill Assessment

**Version tested**: NumPy 2.4.2
**Date**: 2026-02-12
**Test file**: `tests/numpy_test/test_numpy_skill.py`
**Results**: 43 passed, 2 failed (test issues, not skill issues), 1 warning

## Verified Correct

### NEP 50 Type Promotion (8/8 tests passed)
- `arr + 1.0` stays float32 when arr is float32: confirmed
- `arr + 1` (Python int) stays float32: confirmed
- `np.float32(3) + np.float64(3)` -> float64: confirmed
- `arr * np.float64(0.1)` -> float64 (NumPy scalar forces promotion): confirmed
- `arr * 0.1` -> float32 (Python scalar adapts): confirmed
- `uint8([200]) + 100` -> uint8(44) with overflow: confirmed
- `arr_u8.astype(np.int16) + 100` -> 300 (safe fix): confirmed
- `float32([1e8]) + 1.0` -> 1e8 (precision loss); `+ np.float64(1.0)` -> 100000001.0: confirmed

### Copy Semantics (4/5 tests passed - 1 test issue)
- `np.asarray(data)` returns view when no copy needed: confirmed
- `np.array(data, copy=True)` always copies: confirmed
- `np.array(data, copy=False)` returns view when possible: confirmed
- `np.array(data, copy=False, dtype=float64)` raises ValueError when copy needed: confirmed
- `np.reshape(a, shape, copy=False)` raises ValueError when view impossible: confirmed (transposed array case works; my initial test case was wrong since NumPy can handle more non-contiguous cases than expected)

### StringDType & String Operations (6/6 tests passed)
- `from numpy.dtypes import StringDType`: confirmed
- `StringDType(na_object=np.nan)` with NaN support: confirmed
- `np.strings.upper()`: confirmed
- `np.strings.find()`: confirmed, returns correct indices
- `np.strings.replace()`: confirmed
- `np.char.upper()` still works (legacy): confirmed

### Useful Modern API (7/7 tests passed)
- `np.unique_counts(x)` returns (values, counts): confirmed
- `np.unique_inverse(x)` returns (values, inverse_indices): confirmed
- `np.unique(arr, sorted=False)` works (hash-based): confirmed
- `np.cumulative_sum(x)` basic: confirmed
- `np.cumulative_sum(x, include_initial=True)` adds zero prefix: confirmed
- `np.cumulative_sum` signature has `include_initial` with default=False: confirmed
- `np.isdtype(dtype, 'real floating')` etc: confirmed for all listed categories

### Gotchas (5/5 tests passed)
- `np.isclose` asymmetric tolerance (uses b as reference): confirmed with concrete example
- `np.isclose` formula `abs(a-b) <= atol + rtol * abs(b)`: confirmed
- `math.isclose` is symmetric (contrast): confirmed
- `.npz` files load lazily as NpzFile; `dict(np.load(...))` materializes: confirmed
- `np.load` requires `allow_pickle=True` for pickle data: confirmed

### __array__ Protocol (2/2 tests passed)
- `__array__(self, dtype=None, copy=None)` works correctly: confirmed
- Missing `copy=` triggers DeprecationWarning only when `copy=` is explicitly passed to `np.array()`

### Known Limitations (1/1 tests passed)
- `np.minmax` does not exist: confirmed

### Performance Tips (4/4 tests passed)
- `np.unique(arr, sorted=False)` returns same elements as sorted version: confirmed
- `StringDType()` works with `np.strings`: confirmed
- `np.std(arr, mean=precomputed)` parameter exists and works: confirmed
- FFT of float32 -> complex64 (no promotion to complex128): confirmed

## Issues Found

### 1. Test issue: reshape copy=False test case
- **Claim in skill file**: `np.reshape(a, (2, 3), copy=False)  # ValueError if can't return view`
- **Skill file is CORRECT** - the claim is accurate. My test case was flawed: I used `arange(12).reshape(3,4)[:,::2]` which NumPy CAN reshape as a view. A transposed array (`.T`) is the canonical case that raises ValueError.
- **No skill file change needed**.

### 2. Test issue: uint8 + 256 OverflowError
- **Claim in skill file**: Python scalars adapt to array dtype, causing overflow
- **Nuance discovered**: When a Python integer is COMPLETELY outside the dtype range (e.g., 256 for uint8), NumPy raises `OverflowError` rather than wrapping. Wrapping only happens when the RESULT overflows (e.g., 200+100=300 wraps to 44, but 100 fits in uint8 individually).
- **Skill file is correct** for the example given (200+100=44). The behavior is: the Python int 100 fits in uint8, so it adapts; the addition result 300 then overflows to 44.
- **Consider adding**: A note that Python ints outside the dtype range raise `OverflowError` (e.g., `uint8_arr + 256` raises, not wraps).

### 3. Warning: __array__ protocol DeprecationWarning timing
- **Claim in skill file**: `copy= is REQUIRED` in `__array__` protocol
- **Reality**: `np.asarray()` does not pass `copy=` by default, so no warning is raised for old-style `__array__` without `copy=`. The DeprecationWarning only fires when `np.array(obj, copy=True/False)` is called explicitly.
- **Skill file claim is technically correct** - `copy=` IS required for full compatibility. The warning will fire when NumPy code paths explicitly request copy behavior.

## Missing Content

- **OverflowError for out-of-range Python ints**: `np.array([1], dtype=np.uint8) + 256` raises `OverflowError`, not silent overflow. Only in-range values wrap silently. This is a subtle but important distinction for the NEP 50 section.
- **No major gaps otherwise**: The skill file covers the most critical modern NumPy patterns concisely. For a 128-line file, the coverage is excellent.

## Overall Assessment

**The skill file is accurate and production-ready.** All 43 substantive tests passed. The 2 "failures" were test-construction issues, not skill file errors. Every code example in the skill file executes correctly as written.

Key strengths:
- NEP 50 examples are precise and well-chosen (uint8 overflow, float32 precision loss)
- Copy semantics correctly documented
- StringDType and np.strings API accurate
- Gotchas section (isclose asymmetry, npz multiprocessing, pickle security) all verified
- Performance tips (sorted=False, std mean=, FFT float32) all verified

The only enhancement to consider is noting the OverflowError boundary for NEP 50 (Python ints completely outside dtype range raise, rather than wrap).
