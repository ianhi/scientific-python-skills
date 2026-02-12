# Matplotlib Skill File Review

**Reviewer**: Fresh-eyes reviewer (against matplotlib main branch, post-3.10 / pre-3.11)
**File reviewed**: `skills/matplotlib.md`

## Grade: B+

The skill file is well-structured and covers the most important anti-patterns, API signatures, and patterns. The vast majority of claims are accurate. There are a few version number inaccuracies and one misleading deprecation status that should be fixed.

---

## Critical Errors (MUST FIX)

### 1. Anti-pattern #8: `vert=False` deprecation version is wrong

**Skill file says (line 95)**: "Do NOT use `vert=False` on box/violin plots (deprecated 3.10)"

**Actual**: The `vert` parameter deprecation warning fires with version `"3.11"`, not 3.10. In 3.10, `vert` was made keyword-only via `@_api.make_keyword_only("3.10", "vert")`, but the actual deprecation warning in `boxplot()` and `violinplot()` says `_api.warn_deprecated("3.11", ...)`.

Source evidence:
- `lib/matplotlib/axes/_axes.py:4835-4839`: `_api.warn_deprecated("3.11", name="vert: bool", alternative="orientation: {'vertical', 'horizontal'}")`
- `lib/matplotlib/axes/_axes.py:9135-9138`: Same for violinplot.
- `lib/matplotlib/axes/_axes.py:4294`: `@_api.make_keyword_only("3.10", "vert")` -- this only makes it keyword-only, not deprecated.

**Fix**: Change "deprecated 3.10" to "keyword-only in 3.10, deprecated in 3.11".

### 2. Anti-pattern #7: Colormap mutation deprecation version is wrong

**Skill file says (line 83)**: "Do NOT mutate colormaps in-place (pending deprecation)"

**Actual**: The deprecation is no longer "pending" -- it's an active deprecation in 3.11. The source shows `@_api.deprecated("3.11", pending=True, ...)` for `set_bad`, `set_under`, `set_over`, and `set_extremes`. Note: `pending=True` with version `"3.11"` means it's a PendingDeprecationWarning in the current development branch.

Source evidence:
- `lib/matplotlib/colors.py:875-878`: `@_api.deprecated("3.11", pending=True, alternative="cmap.with_extremes(bad=...) or Colormap(bad=...)")`

**Fix**: Update to "deprecated in 3.11 (pending)" or just say "pending deprecation since 3.11" to be precise.

### 3. Anti-pattern #6: `constrained_layout=True` is NOT deprecated

**Skill file says (lines 76-77)**:
```python
# WRONG - deprecated boolean kwargs
fig, ax = plt.subplots(constrained_layout=True)
fig, ax = plt.subplots(tight_layout=True)
```

**Actual**: These are "Discouraged" (with an admonition), NOT deprecated. The source uses `.. admonition:: Discouraged` -- there is no deprecation warning emitted. The `layout=` parameter is preferred, but calling `constrained_layout=True` will work indefinitely without any warning.

Source evidence:
- `lib/matplotlib/figure.py:2518-2530`: Both `tight_layout` and `constrained_layout` have `.. admonition:: Discouraged` notes, NOT deprecation decorators.

**Fix**: Change comment from "deprecated boolean kwargs" to "discouraged boolean kwargs" or "prefer layout= string". Do NOT label these as WRONG since they work fine and produce no warnings.

### 4. Anti-pattern #5: `plot_date()` is removed, not just deprecated

**Skill file says (line 59)**: "Do NOT use `plot_date()` (deprecated 3.9)"

**Actual**: `plot_date()` has been fully **removed** from the codebase. The function does not exist in the current source at all. The `doc/api/next_api_changes/removals/29697-REC.rst` confirms: "The `plot_date` function has now been removed."

Source evidence:
- No `def plot_date` exists anywhere in `lib/matplotlib/axes/_axes.py`
- `doc/api/next_api_changes/removals/29697-REC.rst`: "Use of `plot_date` has been discouraged since Matplotlib 3.5 and deprecated since 3.9. The `plot_date` function has now been removed."

**Fix**: Change "(deprecated 3.9)" to "(removed)" since this is the development branch. The skill targets 3.10.x, and `plot_date` is still present in 3.10.x but deprecated. If targeting 3.10.x specifically, the current wording is OK but should note it will be removed in 3.11. If targeting current/latest, say "removed".

---

## Minor Issues (SHOULD FIX)

### 5. `interpolation_stage` version annotation

**Skill file says (line 219)**: `interpolation_stage=None, # 'data', 'rgba', 'auto' (3.9+)`

**Actual**: The `interpolation_stage` parameter was introduced in matplotlib 3.5/3.6 (fix in 3.6, `set_interpolation_stage` existed by then). The rcParam `image.interpolation_stage` was added in 3.9. The default changed to 'auto' in 3.10. The annotation "(3.9+)" is misleading.

**Fix**: Remove the version annotation or change to "(default 'auto' since 3.10)" since that's the meaningful change.

### 6. `rcParams['image.interpolation']` claim

**Skill file says (line 465)**: `mpl.rcParams['image.interpolation'] = 'auto' # default since 3.10`

**Actual**: Confirmed correct. The default changed from `'antialiased'` to `'auto'` in 3.10 per `doc/api/prev_api_changes/api_changes_3.10.0/behavior.rst`. The value is `'auto'` in the matplotlibrc. This is verified.

### 7. Table from DataFrame anti-pattern wording

**Skill file says (line 431)**: "Do NOT pass rowLabels/colLabels when using a DataFrame"

**Actual**: This is correct but understated. Passing rowLabels or colLabels alongside a DataFrame **raises a ValueError**, not just a warning. The source (`table.py:765,769`) does `raise ValueError("rowLabels cannot be used alongside Pandas DataFrame")`.

**Fix**: Strengthen wording: "Passing rowLabels/colLabels when using a DataFrame raises ValueError."

### 8. `Colorizer` constructor signature

**Skill file says (line 376)**:
```python
colorizer = mpl.colorizer.Colorizer(cmap='RdBu_r', norm=mpl.colors.Normalize(-5, 5))
```

**Actual**: The `Colorizer.__init__` signature is `def __init__(self, cmap=None, norm=None)`. The parameter is named `cmap`, not `colorbar` as the docstring has a typo ("cmap: colorbar.Colorbar or str or None"). The usage in the skill file is correct.

### 9. `Colorizer` docstring parameter type has a typo in source

The source docstring says `cmap: colorbar.Colorbar or str or None` which is a typo (should be `colors.Colormap`). This is a source code issue, not a skill file issue.

---

## Missing Content (CONSIDER ADDING)

### 10. `grouped_bar()` - new in 3.10

The `grouped_bar()` method was added in 3.10 (`lib/matplotlib/axes/_axes.py:3052`). This is a significant new API that Claude would not know about:

```python
ax.grouped_bar([dataset_0, dataset_1], tick_labels=['A', 'B', 'C'],
               labels=['Group 1', 'Group 2'])
```

### 11. `PolyQuadMesh` from `pcolormesh` (since 3.8)

The skill file mentions "pcolormesh returns PolyQuadMesh since 3.8" in the numpy integration section but could be more explicit about the behavioral change.

### 12. `FillBetweenPolyCollection.set_data()` has a `where` parameter

**Skill file says (line 401)**: `coll.set_data(t_new, y1_new, y2_new)`

**Actual**: The `set_data` method also accepts `where=None` keyword argument for conditional filling. The basic usage shown is correct but mentioning `where` would be useful.

### 13. `ecdf` uses `orientation` not `complementary` for direction

The skill file shows `ax.ecdf(data, complementary=True, label='Survival')` which is correct. The `complementary` parameter controls whether to compute 1-CDF vs CDF, and `orientation` controls vertical vs horizontal. Both are present in the source.

---

## Verified Claims (Confirmed Correct)

1. **Anti-pattern #1**: Mixing pyplot and OO interfaces -- correct advice.
2. **Anti-pattern #2**: `matplotlib.cm.get_cmap()` removed in 3.9 -- confirmed via `doc/api/prev_api_changes/api_changes_3.9.0/removals.rst:11`. `plt.get_cmap()` still works -- confirmed it exists in `pyplot.py:2680`.
3. **Anti-pattern #3**: `ContourSet.collections` removed in 3.10 -- confirmed via `doc/api/prev_api_changes/api_changes_3.10.0/removals.rst:97`.
4. **Anti-pattern #4**: Seaborn style names removed in 3.8 -- confirmed via `doc/api/prev_api_changes/api_changes_3.8.0/removals.rst:286`.
5. **Anti-pattern #9**: `labels=` renamed to `tick_labels=` on boxplot in 3.9 -- confirmed via `@_api.rename_parameter("3.9", "labels", "tick_labels")` decorator.
6. **Anti-pattern #10**: `Legend.legendHandles` removed in 3.9 -- confirmed via `doc/api/prev_api_changes/api_changes_3.9.0/removals.rst:147`.
7. **`plt.subplots()` signature** -- verified against `pyplot.py:1707-1717`. All parameters, defaults, and keyword-only markers correct.
8. **`fig.savefig()` signature** -- verified against `figure.py:3353-3362`. All parameters correct including `pad_inches='layout'` (3.8+).
9. **`ax.imshow()` signature** -- verified against `axes/_axes.py:6140-6144`. All parameters correct including `colorizer=None` (3.10+).
10. **`fig.colorbar()` signature** -- verified against `figure.py:1202-1203`. Parameters correct.
11. **`fig.subfigures()` signature** -- verified against `figure.py:1611-1614`. Row-major order change in 3.10 confirmed via `.. versionchanged:: 3.10`.
12. **`plt.subplot_mosaic()` signature** -- verified against `pyplot.py:1918-1934`. All parameters correct.
13. **`FuncAnimation` signature** -- verified against `animation.py:1669-1670`. Parameters correct. GC warning confirmed in source docstring.
14. **`figsize` units** -- `_parse_figsize` accepts `tuple[float, float, Literal["in", "cm", "px"]]` per `figure.pyi:449`.
15. **Petroff styles** -- confirmed `petroff6.mplstyle`, `petroff8.mplstyle`, `petroff10.mplstyle` exist in `mpl-data/stylelib/`.
16. **`align_titles()`** -- confirmed added in 3.9 per `whats_new_3.9.0.rst:162`.
17. **Legend `set_loc()`** -- confirmed added in 3.8 per source `legend.py:659: .. versionadded:: 3.8`.
18. **`label_outer(remove_inner_ticks=True)`** -- confirmed parameter added in 3.8 per source `axes/_base.py:4780: .. versionadded:: 3.8`.
19. **Half violin plots** -- `side` parameter confirmed added in 3.9 per `whats_new_3.9.0.rst:122-134`.
20. **`FillBetweenPolyCollection.set_data()`** -- confirmed exists in `collections.py:1463` and is new in 3.10 per `whats_new_3.10.0.rst:271-277`.
21. **Table accepts DataFrame** -- confirmed in 3.10 per `whats_new_3.10.0.rst:179-182`.
22. **Color with alpha tuple** -- confirmed added in 3.8 per `github_stats_3.8.0.rst:601`.
23. **Normalization classes** -- `BoundaryNorm(boundaries, ncolors, clip=False, *, extend='neither')` signature verified. `CenteredNorm(vcenter=0, halfrange=None, clip=False)` verified. `FuncNorm` verified. `LogNorm` is generated via `make_norm_from_scale` but signature matches.
24. **`set_rasterization_zorder()`** -- confirmed exists in `axes/_base.py:2903`.
25. **Colorizer and ColorizingArtist** -- both confirmed in `colorizer.py:38` and `colorizer.py:696`.
26. **Performance tips** -- all rcParam names verified (`path.simplify`, `path.simplify_threshold`, `agg.path.chunksize`).

---

## Summary

| Category | Count |
|---|---|
| Critical errors | 4 |
| Minor issues | 5 |
| Missing content | 4 |
| Verified claims | 26 |

The skill file is solid overall. The critical fixes are mostly version number corrections (3.10 vs 3.11 for `vert` deprecation, "removed" vs "deprecated" for `plot_date`) and one case where "deprecated" should say "discouraged" (`constrained_layout=True`). These should be fixed because Claude would generate incorrect deprecation warnings or avoid functional code based on wrong information.
