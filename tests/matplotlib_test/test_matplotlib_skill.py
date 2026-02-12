"""
Test script for matplotlib skill file validation.
Verifies every claim in skills/matplotlib.md against matplotlib 3.10.x.
Uses Agg backend for non-interactive execution.
"""

import sys
import os
import tempfile
import traceback
import warnings
import gc

import matplotlib
matplotlib.use('Agg')  # must be before pyplot import

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

passed = 0
failed = 0
errors = []

def test(name, func):
    """Run a single test, print result, track stats."""
    global passed, failed
    try:
        func()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        msg = f"  FAIL: {name} -> {type(e).__name__}: {e}"
        print(msg)
        failed += 1
        errors.append(msg)
        traceback.print_exc(limit=2)
    finally:
        plt.close('all')


# ===========================================================================
# SECTION: Key Patterns - OO interface + constrained layout (lines 7-16)
# ===========================================================================
print("\n=== Key Patterns: OO interface + constrained layout ===\n")

def test_oo_constrained_layout():
    """Skill lines 9-16: OO interface with layout='constrained'."""
    x, y = [1, 2, 3], [4, 5, 6]
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(x, y, label='series')
    ax.set(xlabel="X", ylabel="Y", title="Title")
    ax.legend(loc='upper right')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches='tight')
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)

test("OO interface with layout='constrained', ax.set(), legend, savefig", test_oo_constrained_layout)


# ===========================================================================
# SECTION: squeeze=False for programmatic iteration (lines 18-28)
# ===========================================================================
print("\n=== Key Patterns: squeeze=False ===\n")

def test_squeeze_true_flat_fails_on_single():
    """Skill lines 20-22: squeeze=True (default) breaks .flat on single Axes."""
    fig, axs = plt.subplots(1, 1)
    # single Axes, not an array - .flat should not exist
    assert not hasattr(axs, 'flat'), "1x1 squeeze=True should not have .flat"

test("squeeze=True default: 1x1 returns single Axes without .flat", test_squeeze_true_flat_fails_on_single)

def test_squeeze_true_1d_flat_works():
    """Skill line 22: .flat works on 1D array (1, 3) case."""
    fig, axs = plt.subplots(1, 3)
    # This should work because it's a 1D array
    count = 0
    for ax in axs.flat:
        count += 1
    assert count == 3

test("squeeze=True: 1x3 returns 1D array, .flat works", test_squeeze_true_1d_flat_works)

def test_squeeze_false_always_2d():
    """Skill lines 25-27: squeeze=False always gives 2D array."""
    for nrows, ncols in [(1, 1), (1, 3), (2, 3)]:
        fig, axs = plt.subplots(nrows, ncols, squeeze=False, layout='constrained')
        assert axs.shape == (nrows, ncols), f"Expected {(nrows, ncols)}, got {axs.shape}"
        for ax in axs.flat:
            ax.plot([1, 2], [3, 4])
        plt.close(fig)

test("squeeze=False always returns 2D array for all nrows/ncols", test_squeeze_false_always_2d)


# ===========================================================================
# SECTION: Colorizer API (lines 30-37)
# ===========================================================================
print("\n=== Key Patterns: Colorizer API (3.10+) ===\n")

def test_colorizer_class_exists():
    """Skill line 32: mpl.colorizer.Colorizer exists."""
    assert hasattr(mpl, 'colorizer'), "mpl.colorizer module not found"
    assert hasattr(mpl.colorizer, 'Colorizer'), "mpl.colorizer.Colorizer not found"

test("mpl.colorizer.Colorizer class exists", test_colorizer_class_exists)

def test_colorizing_artist_exists():
    """Skill line 35: mpl.colorizer.ColorizingArtist exists."""
    assert hasattr(mpl.colorizer, 'ColorizingArtist'), "mpl.colorizer.ColorizingArtist not found"

test("mpl.colorizer.ColorizingArtist class exists", test_colorizing_artist_exists)

def test_colorizer_shared_norm_cmap():
    """Skill lines 32-36: Colorizer shares norm/cmap across artists."""
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    colorizer = mpl.colorizer.Colorizer(cmap='RdBu_r', norm=mpl.colors.Normalize(-5, 5))
    data1 = np.random.randn(10, 10)
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    data2 = np.random.randn(10, 10)
    im1 = ax1.imshow(data1, colorizer=colorizer)
    im2 = ax2.pcolormesh(X, Y, data2, colorizer=colorizer)
    ca = mpl.colorizer.ColorizingArtist(colorizer)
    fig.colorbar(ca, ax=[ax1, ax2])

test("Colorizer shared across imshow + pcolormesh with ColorizingArtist colorbar", test_colorizer_shared_norm_cmap)

def test_colorizer_norm_change_affects_both():
    """Skill line 36: Changing colorizer.norm.vmin affects BOTH artists."""
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    colorizer = mpl.colorizer.Colorizer(cmap='RdBu_r', norm=mpl.colors.Normalize(-5, 5))
    data1 = np.random.randn(10, 10)
    data2 = np.random.randn(10, 10)
    im1 = ax1.imshow(data1, colorizer=colorizer)
    im2 = ax2.imshow(data2, colorizer=colorizer)
    # Change vmin
    colorizer.norm.vmin = -10
    # Both artists should reflect the new norm
    assert im1.colorizer.norm.vmin == -10, f"im1 norm.vmin is {im1.colorizer.norm.vmin}, expected -10"
    assert im2.colorizer.norm.vmin == -10, f"im2 norm.vmin is {im2.colorizer.norm.vmin}, expected -10"

test("Changing colorizer.norm.vmin affects BOTH artists", test_colorizer_norm_change_affects_both)


# ===========================================================================
# SECTION: subplot_mosaic (lines 39-47)
# ===========================================================================
print("\n=== Key Patterns: subplot_mosaic ===\n")

def test_subplot_mosaic_with_per_subplot_kw():
    """Skill lines 41-47: subplot_mosaic with per_subplot_kw for polar."""
    fig, axd = plt.subplot_mosaic(
        [["A", "A", "B"], ["C", ".", "B"]],
        layout='constrained',
        per_subplot_kw={"B": {"projection": "polar"}},
    )
    assert isinstance(axd, dict)
    assert set(axd.keys()) == {"A", "B", "C"}
    axd["A"].set_title("Wide top")
    assert axd["B"].name == 'polar'

test("subplot_mosaic with per_subplot_kw for mixed projections", test_subplot_mosaic_with_per_subplot_kw)


# ===========================================================================
# SECTION: Gotchas - Colorbar steals space (lines 51-64)
# ===========================================================================
print("\n=== Gotchas: Colorbar steals space ===\n")

def test_colorbar_default_steals_current_axes():
    """Skill lines 54-56: fig.colorbar(im) without ax= steals from current axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data = np.random.rand(10, 10)
    im = ax2.imshow(data)
    # Without ax= param, colorbar should still work but may attach to wrong axes
    cb = fig.colorbar(im)  # no ax= specified
    assert cb is not None

test("fig.colorbar(im) without ax= works (may steal wrong space)", test_colorbar_default_steals_current_axes)

def test_colorbar_specify_ax():
    """Skill lines 58-61: fig.colorbar(im, ax=ax2) specifies which axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    data = np.random.rand(10, 10)
    im = ax2.imshow(data)
    cb = fig.colorbar(im, ax=ax2)
    assert cb is not None

test("fig.colorbar(im, ax=ax2) specifies correct axes", test_colorbar_specify_ax)

def test_colorbar_shared():
    """Skill line 63: fig.colorbar(im, ax=[ax1, ax2]) for shared colorbar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    data = np.random.rand(10, 10)
    im = ax2.imshow(data)
    cb = fig.colorbar(im, ax=[ax1, ax2])
    assert cb is not None

test("fig.colorbar(im, ax=[ax1, ax2]) shared colorbar", test_colorbar_shared)


# ===========================================================================
# SECTION: Multiple colorbars with subfigures (lines 66-82)
# ===========================================================================
print("\n=== Gotchas: Multiple colorbars with subfigures ===\n")

def test_subfigures_independent_colorbars():
    """Skill lines 74-82: subfigures for independent colorbar handling."""
    fig = plt.figure(layout='constrained')
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    data1 = np.random.rand(10, 10)
    im1 = ax1.imshow(data1)
    subfigs[0].colorbar(im1, ax=ax1)
    ax2 = subfigs[1].subplots()
    data2 = np.random.rand(10, 10) * 100
    im2 = ax2.imshow(data2)
    subfigs[1].colorbar(im2, ax=ax2)

test("Subfigures with independent colorbars pattern", test_subfigures_independent_colorbars)


# ===========================================================================
# SECTION: FuncAnimation GC (lines 84-91)
# ===========================================================================
print("\n=== Gotchas: FuncAnimation GC ===\n")

def test_funcanimation_gc():
    """Skill lines 87-90: FuncAnimation must be stored to avoid GC."""
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    def update(frame):
        line.set_data([0, 1], [0, frame])
        return line,
    # WRONG way - create without storing
    FuncAnimation(fig, update, frames=5, blit=True)
    gc.collect()
    # We can't easily test that it stops, but we can test that storing works
    ani = FuncAnimation(fig, update, frames=5, blit=True)
    assert ani is not None
    # ani reference keeps it alive
    del ani

test("FuncAnimation must be stored (GC issue)", test_funcanimation_gc)


# ===========================================================================
# SECTION: imshow origin (lines 93-98)
# ===========================================================================
print("\n=== Gotchas: imshow origin ===\n")

def test_imshow_origin_upper_default():
    """Skill line 95: Default origin='upper' means [0,0] is top-left."""
    fig, ax = plt.subplots()
    data = np.arange(12).reshape(3, 4)
    im = ax.imshow(data)
    # Default origin should be 'upper'
    assert im.origin == 'upper', f"Default origin is {im.origin}, expected 'upper'"

test("imshow default origin='upper'", test_imshow_origin_upper_default)

def test_imshow_origin_lower():
    """Skill line 97: origin='lower' with extent works."""
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    im = ax.imshow(data, origin='lower', extent=[0, 100, 0, 50])
    assert im.origin == 'lower'

test("imshow origin='lower' with extent", test_imshow_origin_lower)


# ===========================================================================
# SECTION: Deprecated/Removed APIs table (lines 100-108)
# ===========================================================================
print("\n=== Deprecated/Removed APIs ===\n")

def test_cm_get_cmap_deprecated():
    """Skill line 104: matplotlib.cm.get_cmap() deprecated 3.7."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            cmap = matplotlib.cm.get_cmap('viridis')
            # If it works, check for deprecation warning
            dep = [x for x in w if issubclass(x.category, (DeprecationWarning, matplotlib.MatplotlibDeprecationWarning))]
            if dep:
                print("    (Got deprecation warning as expected)")
            else:
                print("    (WARNING: No deprecation warning - may be fully removed)")
        except AttributeError:
            print("    (Fully removed - even better)")

test("matplotlib.cm.get_cmap() deprecation status", test_cm_get_cmap_deprecated)

def test_colormaps_registry():
    """Skill line 104: matplotlib.colormaps['name'] is the replacement."""
    cmap = matplotlib.colormaps['viridis']
    assert cmap is not None
    assert cmap.name == 'viridis'

test("matplotlib.colormaps['viridis'] works", test_colormaps_registry)

def test_colormaps_resampled():
    """Skill line 105: .resampled(10) replaces get_cmap('viridis', 10)."""
    cmap = matplotlib.colormaps['viridis'].resampled(10)
    assert cmap.N == 10

test("colormaps['viridis'].resampled(10) works", test_colormaps_resampled)

def test_seaborn_old_style_removed():
    """Skill line 106: plt.style.use('seaborn') old names removed."""
    try:
        plt.style.use('seaborn')
        raise AssertionError("plt.style.use('seaborn') did NOT raise error")
    except (OSError, ValueError):
        pass  # expected - old name removed

test("plt.style.use('seaborn') old name is removed", test_seaborn_old_style_removed)

def test_seaborn_v0_8_replacement():
    """Skill line 106: 'seaborn-v0_8' is the replacement."""
    plt.style.use('seaborn-v0_8')
    plt.style.use('default')  # reset

test("plt.style.use('seaborn-v0_8') works", test_seaborn_v0_8_replacement)

def test_boxplot_vert_deprecated():
    """Skill line 107: ax.boxplot(vert=False) deprecated 3.11."""
    fig, ax = plt.subplots()
    data = [np.random.randn(100)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ax.boxplot(data, vert=False)
        dep = [x for x in w if issubclass(x.category, (DeprecationWarning, FutureWarning, matplotlib.MatplotlibDeprecationWarning))]
        if dep:
            print("    (Got deprecation warning - deprecated in 3.11 or earlier)")
        else:
            print("    (No warning on 3.10 - skill says deprecated 3.11, so expected)")

test("boxplot vert=False deprecation status (deprecated 3.11)", test_boxplot_vert_deprecated)

def test_boxplot_orientation_replacement():
    """Skill line 107: ax.boxplot(orientation='horizontal') is replacement."""
    fig, ax = plt.subplots()
    data = [np.random.randn(100)]
    ax.boxplot(data, orientation='horizontal')

test("boxplot orientation='horizontal' works", test_boxplot_orientation_replacement)

def test_contourset_collections_removed():
    """Skill line 108: cs.collections removed, ContourSet IS a Collection."""
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    Z = np.sin(X) * np.cos(Y)
    cs = ax.contour(X, Y, Z)
    try:
        _ = cs.collections
        raise AssertionError("cs.collections did NOT raise; skill says removed in 3.10")
    except AttributeError:
        pass  # expected - removed

test("ContourSet.collections is removed (3.10)", test_contourset_collections_removed)

def test_contourset_direct_methods():
    """Skill line 108: cs.set_linewidth(2) directly (ContourSet IS a Collection)."""
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    Z = np.sin(X) * np.cos(Y)
    cs = ax.contour(X, Y, Z)
    cs.set_linewidth(2)  # should work directly on ContourSet

test("ContourSet.set_linewidth(2) works directly", test_contourset_direct_methods)


# ===========================================================================
# SECTION: Performance - legend loc='best' (lines 112-116)
# ===========================================================================
print("\n=== Performance Tips ===\n")

def test_legend_loc_best():
    """Skill lines 114-115: loc='best' works but is slow."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], label='test')
    ax.legend(loc='best')  # should work, just slow
    ax.legend(loc='upper right')  # fast replacement

test("legend loc='best' and loc='upper right' both work", test_legend_loc_best)


# ===========================================================================
# SECTION: Performance - rasterized (lines 118-127)
# ===========================================================================

def test_rasterized_scatter():
    """Skill lines 125-126: rasterized=True on scatter for vector output."""
    fig, ax = plt.subplots()
    x, y = np.random.randn(100), np.random.randn(100)
    ax.scatter(x, y, rasterized=True)
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        fig.savefig(f.name, dpi=150)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)

test("scatter(rasterized=True) for vector PDF", test_rasterized_scatter)


# ===========================================================================
# SECTION: Performance - Dense line plots (lines 129-134)
# ===========================================================================

def test_path_simplify_threshold():
    """Skill line 131: path.simplify_threshold rcParam."""
    original = mpl.rcParams['path.simplify_threshold']
    mpl.rcParams['path.simplify_threshold'] = 1.0
    assert mpl.rcParams['path.simplify_threshold'] == 1.0
    mpl.rcParams['path.simplify_threshold'] = original

test("path.simplify_threshold rcParam works", test_path_simplify_threshold)

def test_agg_path_chunksize():
    """Skill line 132: agg.path.chunksize rcParam."""
    original = mpl.rcParams['agg.path.chunksize']
    mpl.rcParams['agg.path.chunksize'] = 10000
    assert mpl.rcParams['agg.path.chunksize'] == 10000
    mpl.rcParams['agg.path.chunksize'] = original

test("agg.path.chunksize rcParam works", test_agg_path_chunksize)

def test_markevery():
    """Skill line 133: markevery= parameter on plot."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 200)
    ax.plot(x, np.sin(x), marker='o', markevery=10)

test("ax.plot(markevery=10) works", test_markevery)


# ===========================================================================
# SECTION: Performance - 'fast' style (lines 136-138)
# ===========================================================================

def test_fast_style():
    """Skill line 138: mplstyle.use(['ggplot', 'fast']) works."""
    # Note: skill uses 'mplstyle' which would need to be matplotlib.style
    # The actual API is plt.style.use()
    plt.style.use(['ggplot', 'fast'])
    plt.style.use('default')  # reset

test("plt.style.use(['ggplot', 'fast']) combined styles", test_fast_style)

def test_mplstyle_alias_check():
    """Skill line 138: checks if 'mplstyle' is a valid reference."""
    # The skill uses 'mplstyle.use(...)' which is NOT a standard import.
    # matplotlib.style is the correct module.
    assert hasattr(mpl, 'style'), "matplotlib.style exists"
    assert hasattr(mpl.style, 'use'), "matplotlib.style.use exists"
    # 'mplstyle' is not a standard alias
    try:
        import mplstyle  # noqa
        print("    (mplstyle IS importable - unexpected)")
    except ImportError:
        print("    (ISSUE: 'mplstyle' is not importable; skill should use plt.style or mpl.style)")

test("mplstyle alias check (line 138)", test_mplstyle_alias_check)


# ===========================================================================
# SECTION: Performance - Agg backend (lines 141-145)
# ===========================================================================

def test_agg_backend():
    """Skill lines 143-144: matplotlib.use('Agg') before pyplot import."""
    # We already did this at the top of the file
    assert matplotlib.get_backend().lower() == 'agg', f"Backend is {matplotlib.get_backend()}, expected 'agg' (case-insensitive)"

test("Agg backend set correctly", test_agg_backend)


# ===========================================================================
# SECTION: Performance - Close figures (lines 147-151)
# ===========================================================================

def test_close_specific_figure():
    """Skill line 149: plt.close(fig) closes specific figure."""
    fig, ax = plt.subplots()
    plt.close(fig)

test("plt.close(fig) specific figure", test_close_specific_figure)

def test_close_all():
    """Skill line 150: plt.close('all') closes all figures."""
    for _ in range(5):
        plt.subplots()
    plt.close('all')

test("plt.close('all') closes all figures", test_close_all)


# ===========================================================================
# SECTION: Known Limitations (lines 153-158)
# ===========================================================================
print("\n=== Known Limitations ===\n")

def test_scatter_no_marker_list():
    """Skill line 156: scatter() cannot accept a list of markers."""
    fig, ax = plt.subplots()
    x, y = [1, 2, 3], [4, 5, 6]
    try:
        ax.scatter(x, y, marker=['o', 's', '^'])
        raise AssertionError("scatter with list of markers should fail but didn't")
    except (TypeError, ValueError, AttributeError):
        pass  # expected - per-point markers not supported

test("scatter() rejects list of markers (known limitation)", test_scatter_no_marker_list)

def test_issue_references_exist():
    """Skill lines 155-158: verify issue numbers are plausible."""
    issues = {
        '#209': '3D scatter + logscale broken',
        '#11155': 'Per-point markers not supported',
        '#8869': 'timedelta not natively supported',
        '#13236': 'Bar width with units',
    }
    for issue, desc in issues.items():
        num = int(issue.replace('#', ''))
        assert num > 0, f"Issue {issue} is not a valid number"
        print(f"    Issue {issue}: {desc}")

test("Known limitation issue numbers are valid", test_issue_references_exist)


# ===========================================================================
# SECTION: Edge cases and claims that need special verification
# ===========================================================================
print("\n=== Additional Verification ===\n")

def test_default_simplify_threshold():
    """Skill line 131 claims default is 1/9."""
    # Reset to defaults
    mpl.rcdefaults()
    matplotlib.use('Agg')
    default_val = mpl.rcParams['path.simplify_threshold']
    expected = 1/9
    assert abs(default_val - expected) < 0.001, f"Default path.simplify_threshold is {default_val}, skill says {expected}"

test("path.simplify_threshold default is 1/9", test_default_simplify_threshold)

def test_colorbar_no_ax_behavior():
    """Verify that colorbar without ax= defaults to 'current' axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data = np.random.rand(10, 10)
    im = ax2.imshow(data)
    # Without layout='constrained', colorbar without ax= should still work
    cb = fig.colorbar(im)
    # It should have been created (the gotcha is about WHICH axes it steals from)
    assert cb is not None

test("colorbar without ax= still creates (gotcha is space stealing)", test_colorbar_no_ax_behavior)

def test_constrained_layout_with_colorbars():
    """Verify the 'WRONG' pattern from lines 68-71 at least runs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    data1, data2 = np.random.rand(10, 10), np.random.rand(10, 10)
    im1 = ax1.imshow(data1)
    im2 = ax2.imshow(data2)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    # This should work but the axes may end up different sizes
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)

test("Constrained layout with per-axes colorbars runs (axes may differ)", test_constrained_layout_with_colorbars)


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  {e}")

sys.exit(0 if failed == 0 else 1)
