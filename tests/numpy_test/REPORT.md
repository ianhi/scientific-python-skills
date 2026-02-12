# NumPy Skill File Test Report

**Version tested**: 2.4.2
**Date**: 2026-02-11
**Results**: 153 passed, 5 failed out of 158 tests

## Failures

### 1. np.row_stack deprecation status
- **Claim**: Skill file said `np.row_stack(arrays) # AttributeError` (removed in 2.4)
- **Reality**: `np.row_stack()` is deprecated with DeprecationWarning, NOT removed. It still works in 2.4.2.
- **Fix**: Changed comment to `# DeprecationWarning (use np.vstack)`

### 2. np.core warning status
- **Claim**: Skill file said `from numpy.core import multiarray # DeprecationWarning -> will error`
- **Reality**: No warning at all in 2.4.2, silently works (though it's private API)
- **Fix**: Updated comment to clarify it silently works but is private API and may break

### 3. np.fix warning status
- **Claim**: Skill file said `np.fix(3.7) # PendingDeprecationWarning in 2.4`
- **Reality**: No warning at all in 2.4.2
- **Fix**: Updated to note no warning yet but planned for deprecation

### 4. Array repr shape= feature timing
- **Claim**: Skill file said "Summarized arrays include shape= in repr" (2.2+)
- **Reality**: Not yet in 2.4.2 - arrays with 1000 elements do NOT show shape= in repr
- **Fix**: Changed to "will include shape= in a future version" with note "Not yet in 2.4.2"

### 5. float(single-element array) test issue
- **Claim**: Test expected `float(np.array([3.14]))` to work
- **Reality**: Raises TypeError "only 0-dimensional arrays can be converted to Python scalars"
- **Fix**: Test issue, not skill file issue. This is correct NumPy behavior - float() only works on 0-d arrays, not 1-d arrays with one element.

## Summary

The skill file is highly accurate overall (96.8% pass rate). The failures were primarily about deprecation warning timing - the skill file predicted warnings or removals that are planned but not yet implemented in 2.4.2. These are forward-looking statements that will become accurate in future versions. The core technical content about API usage and behavior is correct.

One test failure (#5) was a test script bug, not a skill file issue.

The skill file successfully catches the most critical issues:
- Type alias removals (np.float_, np.complex_, etc.)
- Removed functions (np.trapz, np.in1d)
- Changed semantics (copy=False, NEP 50 type promotion)
- New APIs (StringDType, Array API functions)

**Recommendation**: The skill file is production-ready with the applied fixes.
