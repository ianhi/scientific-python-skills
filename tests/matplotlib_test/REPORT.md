# Matplotlib Skill File Test Report

**Version tested**: 3.10.8
**Date**: 2026-02-11
**Results**: 97 passed, 3 failed out of 100 tests

## Failures

### 1. matplotlib.cm.get_cmap() removal timing
- **Claim**: Skill file said "removed in 3.9"
- **Reality**: Deprecated in 3.7, still exists in 3.10.8 with DeprecationWarning, will be removed in 3.11
- **Fix**: Changed to "deprecated 3.7, removed in 3.11 (still works in 3.10 with warning)"

### 2. figsize unit tuple feature
- **Claim**: Skill file showed `figsize=(20, 15, 'cm')` as a working pattern
- **Reality**: TypeError in 3.10.8 - this feature doesn't exist yet (tuple with units)
- **Fix**: Added note that this is a future feature, not yet in 3.10. Changed example to standard tuple without units.

### 3. petroff6 and petroff8 styles
- **Claim**: Skill file said petroff6, petroff8, petroff10 all exist
- **Reality**: Only petroff10 exists in 3.10.8
- **Fix**: Removed petroff6/petroff8, added note that only petroff10 exists (others may be in future versions)

## Summary

The skill file is highly accurate (97% pass rate). The three failures were about feature timing:
1. A removal that hasn't happened yet (cm.get_cmap)
2. A feature that doesn't exist yet (figsize units)
3. Styles that don't exist yet (petroff6/8)

The skill file correctly teaches:
- OO vs pyplot interface (critical anti-pattern)
- ContourSet.collections removal
- plot_date deprecation
- Layout management (constrained/compressed/tight)
- Colormap mutation deprecation
- All core APIs (subplots, imshow, colorbar, etc.)
- SubFigures and subplot_mosaic
- Animation patterns
- Performance tips

**Recommendation**: The skill file is production-ready with the applied fixes. The failures were all about predicting future features or slightly wrong removal timings - easy fixes that don't affect the core teaching value.
