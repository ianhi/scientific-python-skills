# Pandas Skill File Test Report

**Version tested**: 3.0.0
**Date**: 2026-02-11
**Results**: 102 passed, 7 failed out of 109 tests

## Failures

### 1. 'H' offset alias deprecation status
- **Claim**: Skill file implied 'H' alias was removed
- **Reality**: 'H' is deprecated with Pandas4Warning, NOT removed. Still works in 3.0.
- **Fix**: Changed to clarify it's deprecated (Pandas4Warning), not removed

### 2. Timestamp.as_unit() comparison
- **Claim**: Test compared `ts.as_unit("us").value` with `ts.as_unit("ns").value` expecting 1000x difference
- **Reality**: Test logic was wrong - comparing raw `._value` which already uses different units internally
- **Fix**: Test issue, not skill file issue. The skill file correctly states default is microseconds.

### 3. to_datetime(errors='ignore') removal
- **Claim**: Skill file lists it as removed but doesn't emphasize it clearly
- **Reality**: It's removed, but hits an internal AssertionError (bad error message)
- **Fix**: Skill file is correct, just could be clearer. No change needed - this is a pandas bug (poor error message).

### 4. Copy-on-Write with plain Python assignment
- **Claim**: Test tested `df2 = df; df2.iloc[0,0] = 9` expecting CoW to prevent df mutation
- **Reality**: This is Python aliasing, not CoW. CoW applies to pandas operations, not variable assignment.
- **Fix**: Test issue, not skill file issue. `df2 = df` creates an alias; mutation affects both because they're the same object.

### 5. PeriodIndex.from_fields freq parameter
- **Claim**: Skill file showed `freq="ME"` in PeriodIndex example
- **Reality**: PeriodIndex uses 'M' not 'ME' for monthly frequency (different freq convention)
- **Fix**: Added note that PeriodIndex uses 'M' not 'ME'

### 6. rolling.first() with incomplete windows
- **Claim**: Test expected first value in window
- **Reality**: Returns NaN for initial elements where window is incomplete (min_periods behavior)
- **Fix**: Test issue, not skill file issue. This is correct pandas behavior.

### 7. rolling.last() with incomplete windows
- **Claim**: Same as rolling.first()
- **Reality**: Same as rolling.first() - NaN for incomplete windows
- **Fix**: Test issue, not skill file issue.

## Summary

The skill file is very accurate (93.6% pass rate). Most failures were test script issues (4 out of 7), not skill file problems. The skill file correctly captures:
- Major API removals (DataFrame.append, applymap, fillna(method=))
- Copy-on-Write semantics (with 1 test misunderstanding)
- String dtype changes
- Datetime resolution changes
- Offset alias changes

The two legitimate skill file issues were:
1. 'H' offset alias status (deprecated, not removed)
2. PeriodIndex freq convention (uses 'M' not 'ME')

Both have been fixed.

**Recommendation**: The skill file is production-ready with the applied fixes. The high rate of test issues (vs skill file issues) suggests the skill file is teaching correct pandas 3.0 behavior.
