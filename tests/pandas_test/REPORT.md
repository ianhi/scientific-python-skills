# Pandas Skill Assessment

**Pandas version**: 3.0.0
**NumPy version**: 2.4.2
**Date**: 2026-02-12
**Test results**: 41 passed, 0 failed

## Verified Correct

### Copy-on-Write (CoW)
- **Chained assignment does not modify df** (line 33): Confirmed. `df["col"][df["other"] > 5] = 100` leaves df unchanged. ChainedAssignmentError warning IS emitted (see Issues below).
- **Modifying subset does NOT modify parent** (line 37-39): Confirmed. `subset.iloc[0] = 100` leaves df unchanged, no warning emitted.
- **inplace on extracted column does NOT modify parent** (line 41-43): Confirmed. `col.fillna(0, inplace=True)` leaves df unchanged.
- **Defensive .copy() unnecessary** (line 23): Confirmed. `df[["a", "b"]]` returns independent object.
- **to_numpy() may be read-only** (line 27): Confirmed. `to_numpy()` returns read-only array; `.copy()` makes it writable.
- **Reassign to same variable avoids copy** (line 18-20): Confirmed.
- **Keeping old reference forces copy** (line 46-48): Confirmed. Holding `df_orig` reference means `df2` copy-on-write triggers.
- **copy keyword deprecated on astype** (line 51): Confirmed. FutureWarning emitted when using `copy=True`.
- **Methods return views** (line 10-13): Confirmed via `np.shares_memory()` - reset_index and rename share memory.

### String Dtype
- **Default dtype is "str"** (line 57-58): Confirmed. `pd.Series(["hello", "world"]).dtype` is `str`.
- **Missing values are NaN (float nan)** (line 61-63): Confirmed. `type(ser.iloc[1])` is `float`, value is `nan`.
- **Cannot store non-string - TypeError** (line 66): Confirmed. `ser.iloc[0] = 42` raises TypeError.
- **astype(str) preserves NaN** (line 68-69): Confirmed. Result is `['1.5', nan]`, not `['1.5', 'nan']`.

### Changed Defaults
- **Timestamp unit is 'us'** (line 77): Confirmed. `pd.Timestamp("2024-01-01").unit == "us"`.
- **to_datetime returns datetime64[us]** (line 116): Confirmed.
- **groupby observed=True by default** (line 78): Confirmed. Unobserved categories excluded.
- **axis=None reduces all axes** (line 84-87): Confirmed. `df.sum(axis=None)` returns scalar 10.
- **np.sum(df) reduces all axes** (line 87): Confirmed. `np.sum(df)` returns `int64` scalar 10.
- **Timezone uses zoneinfo** (line 79): Confirmed. `type(ts.tz)` is `zoneinfo.ZoneInfo`.

### Series Indexing
- **__getitem__ is label-based** (line 93-96): Confirmed. `ser[0]` returns label-based value.

### Offset Aliases
- **ME for MonthEnd** (line 104): Confirmed.
- **QE for QuarterEnd** (line 105): Confirmed.
- **Old 'M' alias removed** (implicit): Confirmed. `freq="M"` raises ValueError.
- **MS, QS, YS unchanged** (line 106): Confirmed.
- **PeriodIndex still uses 'M'** (line 107): Confirmed. `pd.period_range("2024", periods=3, freq="M")` works.
- **'min' for minutes** (line 101): Confirmed.

### Datetime
- **as_unit('ns') works** (line 117): Confirmed.

### GroupBy Performance
- **String alias works** (line 125): Confirmed.
- **np.sum works as fallback** (line 128): Confirmed.
- **Named aggregation works** (line 134-138): Confirmed.
- **String alias faster than np.sum**: Confirmed. ~1.4x faster in benchmarks.

### pd.col() Expression Syntax
- **pd.col exists** (line 141): Confirmed.
- **Works in assign with arithmetic and .str accessor** (line 145-148): Confirmed.
- **Works in loc** (line 149): Confirmed.
- **Works in __getitem__ and __setitem__**: Confirmed via additional testing.

### Anti-Joins
- **left_anti merge** (line 155): Confirmed. Returns unmatched rows from left.
- **right_anti merge** (line 156): Confirmed. Returns unmatched rows from right.

### IO Performance
- **PyArrow engine for CSV** (line 163): Confirmed.
- **dtype_backend='pyarrow'** (line 165): Confirmed.
- **Arrow zero-copy types_mapper** (line 167): Confirmed.

### Gotchas
- **0/0 in nullable int produces NA** (line 175-176): Confirmed. Returns `<NA>` (NAType).
- **concat with empty objects dtype** (line 182-185): Confirmed. Result is float64.
- **Day offset preserves wall time across DST** (line 190-191): Confirmed. 01:00 stays 01:00.
- **Timedelta('1D') for fixed 24 hours** (line 192): Confirmed. 01:00 becomes 02:00 (DST).

## Issues Found

### 1. "SILENT" label misleading for chained assignment (line 32)
- **Claim**: `# SILENT - chained assignment does NOT modify df (ChainedAssignmentError)`
- **Reality**: Chained assignment IS NOT silent. It emits a `ChainedAssignmentError` **warning** (which is a `Warning` subclass, not an exception). The warning is visible by default with a multi-line message. The df IS unchanged (correct), but calling it "SILENT" is misleading since users will see a prominent warning.
- **Suggestion**: Change label from "SILENT" to something like "WARNING" or remove the "SILENT" label for this specific case. The other two anti-patterns (subset mutation, inplace on extracted column) are truly silent with no warning emitted.

### 2. Skill doesn't mention inplace=True now returns self (some methods)
- **Claim** (line 51): "inplace=True still works but doesn't help with CoW"
- **Reality**: For `fillna` and `replace`, `inplace=True` now returns `self` instead of `None`. For `drop`, `rename`, `sort_values`, `reset_index`, it still returns `None`. This is a behavioral change that could break code doing `result = df.fillna(0, inplace=True); if result is None: ...`.
- **Suggestion**: Minor issue - the skill's advice to "prefer functional style" sidesteps this. Could add a note.

### 3. Missing: select_dtypes with 'object' now includes string columns
- **Reality**: `df.select_dtypes(include=['object'])` now includes `str` dtype columns with a `Pandas4Warning` deprecation warning. This is a common migration pitfall not mentioned in the skill file.
- **Suggestion**: Add to Gotchas section.

## Missing Content

### Important omissions
- **select_dtypes('object') includes str columns**: With the str dtype change, `select_dtypes(include=['object'])` now matches string columns with a deprecation warning. This will surprise many users migrating from pandas 2.x.
- **DataFrame.from_arrow() / Series.from_arrow()**: New in 3.0, confirmed working. Not mentioned in the skill file, though it's a useful new API.
- **__arrow_c_stream__ protocol**: New in 3.0, works on DataFrames. Not mentioned.
- **.values on str columns returns ArrowStringArray, not ndarray**: `ser.values` returns `ArrowStringArray`, not a numpy array. Only `.to_numpy()` returns ndarray. This breaks code doing `isinstance(ser.values, np.ndarray)`.

### Minor omissions
- The `ChainedAssignmentError` is a `Warning` subclass, not an `Exception`. This is unusual and worth mentioning for users who try `try/except`.
- `Pandas4Warning` is a new warning category for deprecated features being removed in 4.0.

## Overall Assessment

The skill file is **highly accurate**. All 41 tests pass with 0 failures. Every code example produces the documented result. The core content correctly teaches:

- Copy-on-Write semantics and patterns
- String dtype behavior
- Changed defaults (groupby, axis, timestamps, timezones)
- New features (pd.col(), anti-joins)
- Important gotchas (NaN/NA, DST, concat)

The three issues found are minor:
1. "SILENT" label on chained assignment is misleading (it warns, not silent)
2. Missing `select_dtypes('object')` gotcha
3. Missing `.values` behavior change for string columns

**Recommendation**: Production-ready with minor label fix for issue #1. Issues #2 and #3 would strengthen the Gotchas section but are not critical.
