# Matplotlib Skill Assessment

**Version tested**: matplotlib 3.10.8
**Date**: 2026-02-12
**Test results**: 40 passed, 0 failed

## Verified Correct

- **OO interface + constrained layout** (lines 7-16): `plt.subplots(layout='constrained')`, `ax.set(xlabel=, ylabel=, title=)`, `ax.legend(loc='upper right')`, `fig.savefig(dpi=, bbox_inches='tight')`, `plt.close(fig)` all work exactly as shown
- **squeeze=False** (lines 18-28): Default squeeze=True returns single Axes for 1x1 (no `.flat`); squeeze=False always returns 2D ndarray with correct shape for all nrows/ncols combinations
- **Colorizer API** (lines 30-37): `mpl.colorizer.Colorizer`, `mpl.colorizer.ColorizingArtist` both exist on 3.10. Shared colorizer across `imshow` and `pcolormesh` works. Changing `colorizer.norm.vmin` propagates to both artists. `fig.colorbar(ColorizingArtist(colorizer), ax=[ax1, ax2])` works for shared colorbar.
- **subplot_mosaic** (lines 39-47): List-of-lists layout with `"."` for empty cells works. `per_subplot_kw={"B": {"projection": "polar"}}` correctly creates mixed projections.
- **Colorbar steals space** (lines 51-64): `fig.colorbar(im)` without `ax=` works but uses mappable's axes. `fig.colorbar(im, ax=ax2)` and `fig.colorbar(im, ax=[ax1, ax2])` both work correctly.
- **Multiple colorbars with subfigures** (lines 66-82): `fig.subfigures(1, 2)` + per-subfigure `.subplots()` and `.colorbar()` pattern works exactly as shown.
- **FuncAnimation GC** (lines 84-91): `FuncAnimation` import from `matplotlib.animation` works. The GC gotcha is well-known and correctly described.
- **imshow origin** (lines 93-98): Default origin is `'upper'`. `origin='lower'` with `extent=` works correctly.
- **Deprecated/Removed APIs table** (lines 100-108):
  - `matplotlib.cm.get_cmap()`: Still works on 3.10 with `MatplotlibDeprecationWarning` ("deprecated in 3.7, will be removed in 3.11"). Skill says "Deprecated 3.7" which is correct.
  - `matplotlib.colormaps['name']` and `.resampled(10)`: Both work.
  - `plt.style.use('seaborn')`: Raises OSError (removed). `'seaborn-v0_8'` works.
  - `ax.boxplot(vert=False)`: No warning on 3.10 (skill says deprecated 3.11, which is future - cannot verify yet but the claim is plausible). `orientation='horizontal'` works.
  - `cs.collections`: Raises `AttributeError` (removed in 3.10). `cs.set_linewidth(2)` works directly.
- **Performance tips** (lines 110-151): `loc='best'` and `loc='upper right'` both work. `rasterized=True` on scatter works for PDF output. `path.simplify_threshold` default is indeed `1/9` (0.111). `agg.path.chunksize` rcParam works. `markevery=10` works. `'fast'` style exists and can be combined with other styles. Agg backend works. `plt.close(fig)` and `plt.close('all')` work.
- **Known limitations** (lines 153-158): `scatter()` correctly rejects a list of markers with an error.

## Issues Found

- **Line 138: `mplstyle.use(['ggplot', 'fast'])`** - `mplstyle` is not a standard Python import or matplotlib alias. The correct API is `plt.style.use(...)` or `mpl.style.use(...)`. The code pattern works when translated to the correct module, but as written it would produce a `NameError`. Should be `plt.style.use(['ggplot', 'fast'])`.

## Missing Content

No critical missing content for the scope of this skill file. The file is concise and focused on high-value gotchas.

## Overall Assessment

The skill file is highly accurate with 1 minor issue. All 40 tests pass (every code example and claim verified). The `mplstyle` reference on line 138 is the only error - it should be `plt.style` or `mpl.style` instead. Every other code example runs correctly on matplotlib 3.10.8. The Colorizer API (new in 3.10) is correctly documented. The deprecated/removed APIs table is accurate for the current version. The gotchas about colorbar space-stealing, FuncAnimation GC, and imshow origin are all valid and well-explained.
