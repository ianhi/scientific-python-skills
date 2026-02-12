# Pandas Skill File Review

**Reviewer**: Fresh-eyes agent
**Source**: `skills/pandas.md` verified against `repos/pandas/` (pandas 3.0.0, commit a2aa6d7)
**Date**: 2026-02-11

## Grade: B+

The skill file is well-structured and covers the most important pandas 3.0 changes. The anti-patterns section is excellent and will prevent Claude from generating broken code. However, there are a few critical errors (utcnow/utcfromtimestamp claimed "removed" when they're deprecated; misleading inplace=True claim) and some minor issues.

---

## Critical Errors (MUST FIX)

### 1. Anti-pattern #14: utcnow/utcfromtimestamp are DEPRECATED, not REMOVED (line 180-189)

The skill file says:
```python
# WRONG - removed
ts = pd.Timestamp.utcnow()
ts = pd.Timestamp.utcfromtimestamp(1700000000)
```

**Actual**: These methods still exist in pandas 3.0 and emit a `Pandas4Warning` (deprecated, will be removed in pandas 4.0). See `pandas/_libs/tslibs/timestamps.pyx:2002-2035`.

**Fix**: Change "removed" to "deprecated in 3.0, will be removed in 4.0" and relabel from WRONG to something like "DEPRECATED" or keep WRONG but add "(deprecated, emits Pandas4Warning)".

### 2. Anti-pattern #18: inplace return value claim is misleading (line 222-233)

The skill file says:
```python
# WRONG - inplace methods now return self, not None
result = df.fillna(0, inplace=True)
if result is None:  # No longer True!
```

**Actual**: This IS confirmed in the whatsnew (line 878-881): "Methods that can operate in-place (replace, fillna, ffill, bfill, interpolate, where, mask, clip) now return the modified DataFrame or Series (self) instead of None when inplace=True".

However, calling this an "anti-pattern" is misleading. The common pattern `df.fillna(0, inplace=True)` still works perfectly - the change only matters if you were checking `result is None` afterward, which is rare. The bigger issue is that **inplace is now pointless under CoW** since `df.fillna(0, inplace=True)` and `df = df.fillna(0)` are functionally equivalent. The anti-pattern framing suggests the method itself is broken when it's just the None-check that changed.

**Fix**: Reframe this. Instead of "Don't check if inplace=True returned None", consider:
- Title: "Avoid inplace=True (it still works but is unnecessary under CoW)"
- Note the return value change as a secondary detail
- The real anti-pattern is relying on inplace=True at all under CoW

### 3. Anti-pattern #3: ChainedAssignmentError description is slightly inaccurate (line 33)

The skill file says: "silently does nothing under CoW, raises ChainedAssignmentError warning"

**Actual**: `ChainedAssignmentError` is a `Warning` subclass (see `pandas/errors/__init__.py:691`), and is emitted via `warnings.warn()`. The assignment does silently not modify the original DataFrame. So the claim is partially right (warns + doesn't mutate), but "raises" is wrong terminology for a warning - it's "emits" or "issues". More importantly, the comment says it "silently does nothing" AND "raises warning" which is contradictory.

**Fix**: Change to something like: "Does NOT modify df; emits ChainedAssignmentError warning"

---

## Minor Issues (SHOULD FIX)

### 4. Anti-pattern #7: Timestamp.value claim is slightly wrong (line 92)

```python
ts = pd.Timestamp("2024-01-01")
ns_val = ts.value  # Now in microseconds! 1000x smaller than before
```

The comment says `.value` is "Now in microseconds" - but `.value` returns the internal integer representation in whatever unit the Timestamp has. Since default is now microseconds, `.value` gives microseconds. The comment is technically correct but the wording "1000x smaller" could confuse - it should say the integer is 1000x smaller because the unit changed from ns to us.

### 5. String dtype section: "string[pyarrow_numpy]" removal claim (line 331)

The skill file says `"string[pyarrow_numpy]"` is "REMOVED in 3.0". The whatsnew confirms: "Enforced deprecation of storage option 'pyarrow_numpy' for StringDtype" (line 1108). However, there's still a test referencing it in `tests/series/test_constructors.py` (in a historical comment). The claim is correct.

### 6. set_option dict syntax (line 641)

```python
pd.set_option({"display.max_rows": 100, "display.max_columns": 50})
```

**Verified**: Confirmed in `pandas/_config/config.py:258-270`. This is correct.

### 7. Rolling.aggregate NamedAgg syntax (line 592)

```python
df.rolling(5).aggregate(avg=("col", "mean"), total=("col", "sum"))
```

This should work via `reconstruct_func` in the rolling aggregate path (`rolling.py:648`). However, this is a DataFrame-level operation only (not Series). The skill file doesn't clarify this.

### 8. select_dtypes for string columns (line 540-545)

```python
df.select_dtypes(include=["object", "string"])
# pandas 3.0 only:
df.select_dtypes(include=["str"])
```

The "cross-version" recommendation includes "object" and "string" but NOT "str". If the goal is cross-version compatibility, this will miss str-dtype columns on 3.0 (where string columns are "str", not "object" or "string"). The recommendation should include all three for true cross-version compatibility.

**Fix**: Change to `df.select_dtypes(include=["object", "string", "str"])` for cross-version.

### 9. Anti-pattern #11: .values on string columns (line 147-154)

```python
# WRONG - returns ArrowStringArray, not numpy array
arr = string_series.values
arr[0] = "new"  # May fail
```

The comment says "returns ArrowStringArray" - this is somewhat misleading. `.values` on a Series with str dtype may return an `ArrowStringArray` or `ObjectArray` depending on whether PyArrow is installed. The key point (may be read-only/immutable) is correct.

---

## Missing Content (CONSIDER ADDING)

### 10. read_iceberg / to_iceberg (NEW in 3.0)

The whatsnew mentions new `read_iceberg` and `DataFrame.to_iceberg` functions for Apache Iceberg table support (line 226). This is a notable new feature not covered.

### 11. Half-year offset classes (NEW in 3.0)

New `HalfYearBegin`, `HalfYearEnd`, `BHalfYearBegin`, `BHalfYearEnd` offset classes were added (line 285).

### 12. Series[int] positional access (anti-pattern #10 scope)

The skill file correctly covers Series integer access now being label-based. However, it could mention the related change: "integer slice on objects with a floating-dtype index is now treated as positional indexing" (line 997).

### 13. infer_string option removal

The `future.infer_string` option context is no longer needed in 3.0 (it's the default). Users migrating from 2.x may still have it in code. Could be worth a mention.

### 14. Deprecated lowercase frequency aliases

Lowercase `d`, `b`, `c`, `w` for Day, BusinessDay, CustomBusinessDay, Week are deprecated in favor of uppercase `D`, `B`, `C`, `W` (lines 932-933). This is a common gotcha not covered.

---

## Verified Claims (Confirmed Correct)

1. **DataFrame.append() removed** - Not found in `pandas/core/frame.py`. Confirmed removed.
2. **DataFrame.applymap() removed** - Not found in `pandas/core/frame.py`. Confirmed removed (line 1066 of whatsnew).
3. **fillna(method=) removed** - `fillna` signature has no `method` parameter. Confirmed (line 1092 whatsnew).
4. **Old offset aliases removed** - M, Q, Y, H, T confirmed removed (lines 958-982 whatsnew).
5. **String dtype default is "str"** - Confirmed by whatsnew section on string dtype (lines 25-78).
6. **Default datetime resolution is microseconds** - Confirmed (lines 407-469 whatsnew).
7. **pd.col() expression syntax** - Confirmed, exists in `pandas/core/col.py`, versionadded 3.0.0.
8. **Anti-joins (left_anti, right_anti)** - Confirmed in source and whatsnew (line 251).
9. **DataFrame.from_arrow / Series.from_arrow** - Confirmed in source (`frame.py:1591`, `series.py:1891`).
10. **NamedAgg with *args, **kwargs** - Confirmed in `pandas/core/groupby/generic.py:110-155`.
11. **observed=True default for groupby** - Confirmed: `frame.py:12354` shows `observed: bool = True`.
12. **DataFrame.first() / DataFrame.last() removed** - Confirmed (whatsnew line 1068).
13. **axis parameter removed from groupby/rolling/resample** - Confirmed (whatsnew line 1079-1080).
14. **errors="ignore" removed from to_datetime/to_numeric** - Confirmed: `DateTimeErrorChoices = Literal["raise", "coerce"]` in `_typing.py:427`.
15. **read_csv: delim_whitespace removed** - Confirmed (whatsnew line 1015).
16. **read_csv: date_parser removed** - Confirmed (whatsnew line 1038).
17. **read_csv: infer_datetime_format removed** - Confirmed (whatsnew line 1041).
18. **PeriodIndex.from_fields()** - Confirmed in `pandas/core/indexes/period.py:384`.
19. **Rolling.first(), Rolling.last(), Rolling.nunique()** - Confirmed in `rolling.py`.
20. **Expanding.pipe()** - Confirmed in `expanding.py`.
21. **CoW is always on** - Confirmed by whatsnew (lines 81-131).
22. **inplace methods return self** - Confirmed (whatsnew line 878-881).
23. **to_excel autofilter parameter** - Confirmed in source.
24. **to_sql delete_rows option** - Confirmed in source.
25. **axis=None reduces over both axes** - Confirmed (whatsnew line 1036).
26. **Day offset is calendar-day** - Confirmed (whatsnew lines 598-631).
27. **NaN vs NA in nullable dtypes** - Confirmed (whatsnew lines 633-714).
28. **set_option accepts dict** - Confirmed in source (`_config/config.py:269`).
29. **performance_warnings option** - Confirmed in `core/config_init.py:447`.
30. **zoneinfo default for timezones** - Confirmed (whatsnew lines 820-845).
31. **Python 3.11+ required** - Confirmed (whatsnew line 737).
32. **Copy keyword deprecated on astype, reindex, rename, merge, align, etc.** - Confirmed (whatsnew lines 890-914).
33. **Series integer indexing always label-based** - Confirmed (whatsnew line 996).
34. **numpy functions no longer auto-replaced in agg/transform** - Confirmed (whatsnew line 1062).

---

## Summary

| Category | Count |
|----------|-------|
| Critical Errors | 3 |
| Minor Issues | 6 |
| Missing Content | 5 |
| Verified Claims | 34 |

The skill file is comprehensive and well-organized. The three critical errors should be fixed before deployment - especially the utcnow/utcfromtimestamp claim ("removed" vs "deprecated") which would cause Claude to tell users the methods don't exist when they still do. The inplace anti-pattern reframing is less urgent but would prevent confusion.
