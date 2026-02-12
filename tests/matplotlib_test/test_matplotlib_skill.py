"""
Test script for matplotlib skill file validation.
Tests claims from skills/matplotlib.md against matplotlib 3.10.x.
Uses Agg backend for non-interactive execution.
"""

import sys
import os
import tempfile
import traceback
import warnings
import datetime

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

# Ensure clean state for each test
def fresh():
    plt.close('all')

# ---------------------------------------------------------------------------
# Section: CRITICAL - Do NOT Use These Outdated Patterns
# ---------------------------------------------------------------------------
print("\n=== SECTION 1: Anti-Patterns (Removed APIs) ===\n")

# Anti-pattern #2: matplotlib.cm.get_cmap() removed in 3.9
def test_cm_get_cmap_removed():
    """Skill claims matplotlib.cm.get_cmap() is removed in 3.9."""
    fresh()
    try:
        cmap = matplotlib.cm.get_cmap('viridis')
        # If we get here, it was NOT removed
        raise AssertionError("matplotlib.cm.get_cmap() did NOT raise an error; skill says removed in 3.9")
    except AttributeError:
        pass  # expected: removed
    except TypeError:
        pass  # also acceptable
test("Anti-pattern #2: matplotlib.cm.get_cmap() is removed", test_cm_get_cmap_removed)

# Anti-pattern #2 replacement: matplotlib.colormaps[]
def test_colormaps_registry():
    """Skill claims matplotlib.colormaps['viridis'] works."""
    fresh()
    cmap = matplotlib.colormaps['viridis']
    assert cmap is not None
    assert cmap.name == 'viridis'
test("Anti-pattern #2 fix: matplotlib.colormaps['viridis'] works", test_colormaps_registry)

# Anti-pattern #2 replacement: .resampled(10)
def test_colormaps_resampled():
    """Skill claims .resampled(N) replaces get_cmap(name, N)."""
    fresh()
    cmap = matplotlib.colormaps['viridis'].resampled(10)
    assert cmap.N == 10
test("Anti-pattern #2 fix: .resampled(10) works", test_colormaps_resampled)

# Anti-pattern #3: ContourSet.collections removed in 3.10
def test_contourset_collections_removed():
    """Skill claims ContourSet.collections is removed in 3.10."""
    fresh()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    Z = np.sin(X) * np.cos(Y)
    cs = ax.contour(X, Y, Z)
    try:
        _ = cs.collections
        raise AssertionError("ContourSet.collections did NOT raise; skill says removed in 3.10")
    except AttributeError:
        pass
    plt.close(fig)
test("Anti-pattern #3: ContourSet.collections is removed", test_contourset_collections_removed)

# Anti-pattern #3 replacement: ContourSet IS a single Collection
def test_contourset_is_collection():
    """Skill claims ContourSet IS a single Collection with set_linewidth()."""
    fresh()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    Z = np.sin(X) * np.cos(Y)
    cs = ax.contour(X, Y, Z)
    cs.set_linewidth(2)  # should work directly
    plt.close(fig)
test("Anti-pattern #3 fix: ContourSet.set_linewidth() works directly", test_contourset_is_collection)

# Anti-pattern #4: old seaborn style names removed
def test_old_seaborn_styles_removed():
    """Skill claims plt.style.use('seaborn') raises error."""
    fresh()
    try:
        plt.style.use('seaborn')
        raise AssertionError("plt.style.use('seaborn') did NOT raise; skill says removed")
    except OSError:
        pass  # expected
    except ValueError:
        pass  # also acceptable
test("Anti-pattern #4: plt.style.use('seaborn') raises error", test_old_seaborn_styles_removed)

def test_old_seaborn_whitegrid_removed():
    """Skill claims 'seaborn-whitegrid' is removed."""
    fresh()
    try:
        plt.style.use('seaborn-whitegrid')
        raise AssertionError("plt.style.use('seaborn-whitegrid') did NOT raise")
    except OSError:
        pass
    except ValueError:
        pass
test("Anti-pattern #4: plt.style.use('seaborn-whitegrid') raises error", test_old_seaborn_whitegrid_removed)

# Anti-pattern #4 replacement: seaborn-v0_8 styles
def test_seaborn_v0_8_styles():
    """Skill claims 'seaborn-v0_8' and 'seaborn-v0_8-whitegrid' work."""
    fresh()
    plt.style.use('seaborn-v0_8')
    plt.style.use('seaborn-v0_8-whitegrid')
    # Reset to default
    plt.style.use('default')
test("Anti-pattern #4 fix: seaborn-v0_8 styles work", test_seaborn_v0_8_styles)

# Anti-pattern #5: plot_date() deprecated 3.9, removed in 3.11
def test_plot_date_status():
    """Skill claims plot_date() is deprecated 3.9, removed 3.11."""
    fresh()
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2024, 1, i+1) for i in range(5)]
    values = [1, 2, 3, 4, 5]
    try:
        ax.plot_date(dates, values)
        # If it still works (we're on 3.10), it should at least warn
        print("    (Note: plot_date() still exists in 3.10, removed in 3.11)")
    except AttributeError:
        print("    (plot_date() confirmed removed)")
    plt.close(fig)
test("Anti-pattern #5: plot_date() status check", test_plot_date_status)

# Anti-pattern #5 replacement: plot() handles datetime natively
def test_plot_handles_datetime():
    """Skill claims plot() handles datetime natively."""
    fresh()
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2024, 1, i+1) for i in range(5)]
    values = [1, 2, 3, 4, 5]
    ax.plot(dates, values)
    plt.close(fig)
test("Anti-pattern #5 fix: ax.plot() handles datetime natively", test_plot_handles_datetime)

# Anti-pattern #8: vert=False deprecated on boxplot (deprecated 3.11)
def test_boxplot_vert_deprecated():
    """Skill claims vert=False on boxplot is deprecated in 3.11."""
    fresh()
    fig, ax = plt.subplots()
    data = [np.random.randn(100)]
    # On 3.10, vert=False should still work (deprecated in 3.11)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ax.boxplot(data, vert=False)
        dep_warnings = [x for x in w if issubclass(x.category, (DeprecationWarning, FutureWarning, matplotlib.MatplotlibDeprecationWarning))]
        if dep_warnings:
            print(f"    (Got deprecation warning as expected: {dep_warnings[0].message})")
        else:
            print("    (Note: No deprecation warning on 3.10; deprecated in 3.11)")
    plt.close(fig)
test("Anti-pattern #8: boxplot vert=False deprecation status", test_boxplot_vert_deprecated)

# Anti-pattern #8 replacement: orientation='horizontal'
def test_boxplot_orientation():
    """Skill claims orientation='horizontal' is the replacement."""
    fresh()
    fig, ax = plt.subplots()
    data = [np.random.randn(100)]
    ax.boxplot(data, orientation='horizontal')
    plt.close(fig)
test("Anti-pattern #8 fix: boxplot orientation='horizontal' works", test_boxplot_orientation)

# Anti-pattern #9: labels= on boxplot deprecated 3.9
def test_boxplot_labels_deprecated():
    """Skill claims labels= is deprecated in 3.9, use tick_labels= instead."""
    fresh()
    fig, ax = plt.subplots()
    data = [np.random.randn(50), np.random.randn(50)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ax.boxplot(data, labels=['A', 'B'])
        dep_warnings = [x for x in w if issubclass(x.category, (DeprecationWarning, FutureWarning, matplotlib.MatplotlibDeprecationWarning))]
        if dep_warnings:
            print(f"    (Got deprecation warning: {dep_warnings[0].message})")
        else:
            print("    (Note: No deprecation warning; may have been already removed or still works silently)")
    plt.close(fig)
test("Anti-pattern #9: boxplot labels= deprecation status", test_boxplot_labels_deprecated)

# Anti-pattern #9 replacement: tick_labels=
def test_boxplot_tick_labels():
    """Skill claims tick_labels= is the replacement."""
    fresh()
    fig, ax = plt.subplots()
    data = [np.random.randn(50), np.random.randn(50)]
    ax.boxplot(data, tick_labels=['A', 'B'])
    plt.close(fig)
test("Anti-pattern #9 fix: boxplot tick_labels= works", test_boxplot_tick_labels)

# Anti-pattern #10: Legend.legendHandles removed 3.9
def test_legend_legendHandles_removed():
    """Skill claims Legend.legendHandles is removed in 3.9."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='test')
    legend = ax.legend()
    try:
        _ = legend.legendHandles
        raise AssertionError("legend.legendHandles did NOT raise; skill says removed in 3.9")
    except AttributeError:
        pass  # expected
    plt.close(fig)
test("Anti-pattern #10: Legend.legendHandles is removed", test_legend_legendHandles_removed)

# Anti-pattern #10 replacement: legend_handles
def test_legend_handles():
    """Skill claims legend.legend_handles is the replacement."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='test')
    legend = ax.legend()
    handles = legend.legend_handles
    assert handles is not None
    assert len(handles) == 1
    plt.close(fig)
test("Anti-pattern #10 fix: legend.legend_handles works", test_legend_handles)


# ---------------------------------------------------------------------------
# Section: OO Interface Patterns (Anti-pattern #1)
# ---------------------------------------------------------------------------
print("\n=== SECTION 2: OO Interface Patterns ===\n")

def test_oo_set_method():
    """Skill claims ax.set(title=, xlabel=, ylabel=) works."""
    fresh()
    fig, ax = plt.subplots()
    ax.set(title="My Title", xlabel="X", ylabel="Y")
    assert ax.get_title() == "My Title"
    assert ax.get_xlabel() == "X"
    assert ax.get_ylabel() == "Y"
    plt.close(fig)
test("OO interface: ax.set() batch method works", test_oo_set_method)

def test_oo_set_extended():
    """Skill claims ax.set() works for xlim, ylim too."""
    fresh()
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 10), ylim=(0, 1))
    assert ax.get_xlim() == (0, 10)
    assert ax.get_ylim() == (0, 1)
    plt.close(fig)
test("OO interface: ax.set(xlim=, ylim=) works", test_oo_set_extended)


# ---------------------------------------------------------------------------
# Section: Core API - plt.subplots()
# ---------------------------------------------------------------------------
print("\n=== SECTION 3: Core API - plt.subplots() ===\n")

def test_subplots_basic():
    """Test basic plt.subplots() signature."""
    fresh()
    fig, ax = plt.subplots()
    assert fig is not None
    assert ax is not None
    plt.close(fig)
test("plt.subplots() basic call", test_subplots_basic)

def test_subplots_nrows_ncols():
    """Test nrows/ncols."""
    fresh()
    fig, axs = plt.subplots(2, 3)
    assert axs.shape == (2, 3)
    plt.close(fig)
test("plt.subplots(2, 3) returns correct shape", test_subplots_nrows_ncols)

def test_subplots_squeeze_false():
    """Skill claims squeeze=False always returns 2D array."""
    fresh()
    fig, axs = plt.subplots(1, 1, squeeze=False)
    assert axs.shape == (1, 1)
    fig2, axs2 = plt.subplots(1, 3, squeeze=False)
    assert axs2.shape == (1, 3)
    plt.close(fig)
    plt.close(fig2)
test("plt.subplots(squeeze=False) always returns 2D array", test_subplots_squeeze_false)

def test_subplots_layout_constrained():
    """Skill claims layout='constrained' works."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    assert fig.get_layout_engine() is not None
    plt.close(fig)
test("plt.subplots(layout='constrained') works", test_subplots_layout_constrained)

def test_subplots_width_height_ratios():
    """Skill claims width_ratios and height_ratios are direct params."""
    fresh()
    fig, axs = plt.subplots(2, 3, width_ratios=[1, 2, 1], height_ratios=[1, 2])
    assert axs.shape == (2, 3)
    plt.close(fig)
test("plt.subplots(width_ratios=, height_ratios=) works", test_subplots_width_height_ratios)

def test_subplots_sharex_sharey():
    """Test sharex/sharey parameters."""
    fresh()
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row')
    assert axs.shape == (2, 2)
    plt.close(fig)
test("plt.subplots(sharex='col', sharey='row') works", test_subplots_sharex_sharey)

def test_subplots_figsize_with_units():
    """Skill claims figsize accepts (w, h, 'cm'|'in'|'px')."""
    fresh()
    fig, ax = plt.subplots(figsize=(20, 15, "cm"))
    # Check that figsize was interpreted (should be roughly 20cm / 2.54 inches)
    w, h = fig.get_size_inches()
    expected_w = 20 / 2.54
    expected_h = 15 / 2.54
    assert abs(w - expected_w) < 0.1, f"Width {w} != expected {expected_w}"
    assert abs(h - expected_h) < 0.1, f"Height {h} != expected {expected_h}"
    plt.close(fig)
test("plt.subplots(figsize=(20, 15, 'cm')) unit support", test_subplots_figsize_with_units)

def test_subplots_subplot_kw_polar():
    """Skill claims subplot_kw={'projection': 'polar'} works."""
    fresh()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    assert ax.name == 'polar'
    plt.close(fig)
test("plt.subplots(subplot_kw={'projection': 'polar'}) works", test_subplots_subplot_kw_polar)


# ---------------------------------------------------------------------------
# Section: Core API - fig.savefig()
# ---------------------------------------------------------------------------
print("\n=== SECTION 4: Core API - fig.savefig() ===\n")

def test_savefig_basic():
    """Test basic savefig."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, dpi=150, bbox_inches='tight')
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)
test("fig.savefig() basic png with dpi and bbox_inches", test_savefig_basic)

def test_savefig_pdf():
    """Test vector format output."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        fig.savefig(f.name)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)
test("fig.savefig() pdf vector output", test_savefig_pdf)

def test_savefig_transparent():
    """Skill claims transparent= is a param."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, transparent=True)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)
test("fig.savefig(transparent=True) works", test_savefig_transparent)

def test_savefig_metadata():
    """Skill claims metadata= dict works."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        fig.savefig(f.name, metadata={'Title': 'Test Plot'})
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)
test("fig.savefig(metadata=) works", test_savefig_metadata)


# ---------------------------------------------------------------------------
# Section: Core API - ax.imshow()
# ---------------------------------------------------------------------------
print("\n=== SECTION 5: Core API - ax.imshow() ===\n")

def test_imshow_basic():
    """Test basic imshow with 2D array."""
    fresh()
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    im = ax.imshow(data, cmap='viridis', origin='lower')
    assert im is not None
    plt.close(fig)
test("ax.imshow() with 2D array", test_imshow_basic)

def test_imshow_rgb():
    """Skill claims (M,N,3) RGB arrays work."""
    fresh()
    fig, ax = plt.subplots()
    rgb = np.random.rand(10, 10, 3)
    im = ax.imshow(rgb)
    assert im is not None
    plt.close(fig)
test("ax.imshow() with RGB (M,N,3) array", test_imshow_rgb)

def test_imshow_rgba():
    """Skill claims (M,N,4) RGBA arrays work."""
    fresh()
    fig, ax = plt.subplots()
    rgba = np.random.rand(10, 10, 4)
    im = ax.imshow(rgba)
    assert im is not None
    plt.close(fig)
test("ax.imshow() with RGBA (M,N,4) array", test_imshow_rgba)

def test_imshow_extent():
    """Skill claims extent=(left, right, bottom, top) works."""
    fresh()
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    im = ax.imshow(data, origin='lower', extent=[0, 100, 0, 50])
    assert im is not None
    plt.close(fig)
test("ax.imshow(extent=) works", test_imshow_extent)

def test_imshow_interpolation_stage():
    """Skill claims interpolation_stage= is available (3.9+)."""
    fresh()
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    im = ax.imshow(data, interpolation_stage='data')
    assert im is not None
    plt.close(fig)
test("ax.imshow(interpolation_stage='data') works (3.9+)", test_imshow_interpolation_stage)

def test_imshow_colorizer():
    """Skill claims colorizer= parameter exists (3.10+)."""
    fresh()
    fig, ax = plt.subplots()
    data = np.random.rand(10, 10)
    try:
        colorizer = mpl.colorizer.Colorizer(cmap='viridis')
        im = ax.imshow(data, colorizer=colorizer)
        assert im is not None
    except AttributeError as e:
        raise AssertionError(f"mpl.colorizer.Colorizer not found: {e}")
    plt.close(fig)
test("ax.imshow(colorizer=) works (3.10+)", test_imshow_colorizer)


# ---------------------------------------------------------------------------
# Section: Core API - fig.colorbar()
# ---------------------------------------------------------------------------
print("\n=== SECTION 6: Core API - fig.colorbar() ===\n")

def test_colorbar_basic():
    """Test basic colorbar creation."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    data = np.random.rand(10, 10)
    im = ax.imshow(data)
    cb = fig.colorbar(im, ax=ax, label='Values')
    assert cb is not None
    plt.close(fig)
test("fig.colorbar(im, ax=ax, label=) works", test_colorbar_basic)

def test_colorbar_orientation():
    """Test colorbar orientation parameter."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    data = np.random.rand(10, 10)
    im = ax.imshow(data)
    cb = fig.colorbar(im, ax=ax, orientation='horizontal')
    assert cb is not None
    plt.close(fig)
test("fig.colorbar(orientation='horizontal') works", test_colorbar_orientation)

def test_colorbar_shared_across_axes():
    """Skill claims ax=[ax1, ax2] for shared colorbar."""
    fresh()
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
    data = np.random.rand(10, 10)
    im = ax2.imshow(data)
    cb = fig.colorbar(im, ax=[ax1, ax2])
    assert cb is not None
    plt.close(fig)
test("fig.colorbar(im, ax=[ax1, ax2]) shared colorbar", test_colorbar_shared_across_axes)


# ---------------------------------------------------------------------------
# Section: Core API - fig.subfigures()
# ---------------------------------------------------------------------------
print("\n=== SECTION 7: Core API - fig.subfigures() ===\n")

def test_subfigures_basic():
    """Skill claims fig.subfigures() returns subfigures."""
    fresh()
    fig = plt.figure(layout='constrained', figsize=(12, 6))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])
    assert len(subfigs) == 2
    plt.close(fig)
test("fig.subfigures(1, 2) returns array of 2 subfigures", test_subfigures_basic)

def test_subfigures_have_subplots():
    """Skill claims each subfigure has .subplots(), .suptitle(), .colorbar()."""
    fresh()
    fig = plt.figure(layout='constrained', figsize=(12, 6))
    subfigs = fig.subfigures(1, 2)
    axs = subfigs[0].subplots(2, 2)
    assert axs.shape == (2, 2)
    subfigs[0].suptitle('Left Panel')
    ax_right = subfigs[1].subplots()
    data = np.random.rand(10, 10)
    im = ax_right.imshow(data)
    subfigs[1].colorbar(im, ax=ax_right)
    subfigs[1].suptitle('Right Panel')
    plt.close(fig)
test("Subfigures have .subplots(), .suptitle(), .colorbar()", test_subfigures_have_subplots)


# ---------------------------------------------------------------------------
# Section: Core API - plt.subplot_mosaic()
# ---------------------------------------------------------------------------
print("\n=== SECTION 8: Core API - plt.subplot_mosaic() ===\n")

def test_subplot_mosaic_list():
    """Skill claims list-of-lists layout works."""
    fresh()
    fig, axd = plt.subplot_mosaic(
        [["A", "A", "B"],
         ["C", ".",  "B"]],
        layout='constrained',
    )
    assert isinstance(axd, dict)
    assert set(axd.keys()) == {"A", "B", "C"}
    plt.close(fig)
test("plt.subplot_mosaic() list-of-lists works", test_subplot_mosaic_list)

def test_subplot_mosaic_string():
    """Skill claims multiline string layout works."""
    fresh()
    fig, axd = plt.subplot_mosaic(
        """
        AAB
        C.B
        """,
        layout='constrained',
        figsize=(10, 6),
    )
    assert isinstance(axd, dict)
    assert "A" in axd and "B" in axd and "C" in axd
    plt.close(fig)
test("plt.subplot_mosaic() string layout works", test_subplot_mosaic_string)

def test_subplot_mosaic_per_subplot_kw():
    """Skill claims per_subplot_kw works for individual projections."""
    fresh()
    fig, axd = plt.subplot_mosaic(
        [["A", "B"]],
        per_subplot_kw={"B": {"projection": "polar"}},
        layout='constrained',
    )
    assert axd["A"].name == 'rectilinear'
    assert axd["B"].name == 'polar'
    plt.close(fig)
test("plt.subplot_mosaic(per_subplot_kw=) works", test_subplot_mosaic_per_subplot_kw)


# ---------------------------------------------------------------------------
# Section: Layout Management
# ---------------------------------------------------------------------------
print("\n=== SECTION 9: Layout Management ===\n")

def test_layout_compressed():
    """Skill claims layout='compressed' works."""
    fresh()
    fig, axs = plt.subplots(2, 2, layout='compressed', sharex=True)
    assert axs.shape == (2, 2)
    plt.close(fig)
test("layout='compressed' works", test_layout_compressed)

def test_layout_tight():
    """Skill claims layout='tight' works."""
    fresh()
    fig, axs = plt.subplots(2, 2, layout='tight')
    assert axs.shape == (2, 2)
    plt.close(fig)
test("layout='tight' works", test_layout_tight)

def test_align_titles():
    """Skill claims fig.align_titles() exists (3.9+)."""
    fresh()
    fig, axs = plt.subplots(1, 3, layout='constrained')
    for ax, title in zip(axs, ['Short', 'A Longer Title', 'X']):
        ax.set_title(title)
    fig.align_titles()
    plt.close(fig)
test("fig.align_titles() works (3.9+)", test_align_titles)

def test_gridspec_programmatic():
    """Skill claims GridSpec with figure= works."""
    fresh()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(layout='constrained', figsize=(10, 6))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    assert ax_main is not None
    assert ax_top is not None
    assert ax_right is not None
    plt.close(fig)
test("GridSpec programmatic layout works", test_gridspec_programmatic)


# ---------------------------------------------------------------------------
# Section: Patterns & Idioms
# ---------------------------------------------------------------------------
print("\n=== SECTION 10: Patterns & Idioms ===\n")

def test_data_parameter():
    """Skill claims data= parameter allows string column access."""
    fresh()
    fig, ax = plt.subplots()
    df_data = {"time": [1, 2, 3], "velocity": [4, 5, 6]}
    ax.plot("time", "velocity", data=df_data)
    plt.close(fig)
test("data= parameter with dict works", test_data_parameter)

def test_color_with_alpha_tuple():
    """Skill claims color=('blue', 0.3) works (3.8+)."""
    fresh()
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), color=('blue', 0.3))
    ax.fill_between(x, 0, np.sin(x), color=('red', 0.2))
    plt.close(fig)
test("color=('blue', 0.3) alpha tuple works (3.8+)", test_color_with_alpha_tuple)

def test_colorizer_shared():
    """Skill claims Colorizer shares norm/cmap across artists (3.10+)."""
    fresh()
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
        colorizer = mpl.colorizer.Colorizer(cmap='RdBu_r', norm=mpl.colors.Normalize(-5, 5))
        data1 = np.random.randn(10, 10)
        X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        data2 = np.random.randn(10, 10)
        im1 = ax1.imshow(data1, colorizer=colorizer)
        im2 = ax2.pcolormesh(X, Y, data2, colorizer=colorizer)
        # Test ColorizingArtist for shared colorbar
        ca = mpl.colorizer.ColorizingArtist(colorizer)
        fig.colorbar(ca, ax=[ax1, ax2])
        plt.close(fig)
    except AttributeError as e:
        raise AssertionError(f"Colorizer API not found: {e}")
test("Colorizer shared across artists (3.10+)", test_colorizer_shared)

def test_petroff_styles():
    """Skill claims petroff6/8/10 accessible styles exist (3.10+)."""
    fresh()
    available = plt.style.available
    for name in ['petroff6', 'petroff8', 'petroff10']:
        assert name in available, f"Style '{name}' not in plt.style.available"
    plt.style.use('default')
test("petroff6/8/10 accessible styles exist (3.10+)", test_petroff_styles)

def test_vectorized_hist_styling():
    """Skill claims vectorized hatch/edgecolor on hist (3.10)."""
    fresh()
    fig, ax = plt.subplots()
    d1, d2, d3 = np.random.randn(100), np.random.randn(100), np.random.randn(100)
    ax.hist([d1, d2, d3],
            hatch=["/", ".", "*"],
            edgecolor=["red", "green", "blue"],
            label=["A", "B", "C"])
    plt.close(fig)
test("Vectorized hist styling (hatch, edgecolor lists) works (3.10)", test_vectorized_hist_styling)

def test_fill_between_set_data():
    """Skill claims FillBetween has .set_data() for in-place update (3.10)."""
    fresh()
    fig, ax = plt.subplots()
    t = np.linspace(0, 1, 50)
    y1 = np.sin(t)
    y2 = np.cos(t)
    coll = ax.fill_between(t, y1, y2)
    # Try in-place update
    t_new = np.linspace(0, 2, 60)
    y1_new = np.sin(t_new)
    y2_new = np.cos(t_new)
    try:
        coll.set_data(t_new, y1_new, y2_new)
    except AttributeError:
        raise AssertionError("FillBetween.set_data() not found; skill says 3.10+")
    plt.close(fig)
test("FillBetween.set_data() in-place update (3.10)", test_fill_between_set_data)

def test_ecdf():
    """Skill claims ax.ecdf() exists (3.8+)."""
    fresh()
    fig, ax = plt.subplots()
    data = np.random.randn(200)
    ax.ecdf(data, label='CDF')
    ax.ecdf(data, complementary=True, label='Survival')
    plt.close(fig)
test("ax.ecdf() works with complementary= (3.8+)", test_ecdf)

def test_half_violin():
    """Skill claims ax.violinplot(side='low'|'high') for half violin (3.9+)."""
    fresh()
    fig, ax = plt.subplots()
    data_left = np.random.randn(100)
    data_right = np.random.randn(100) + 1
    ax.violinplot([data_left], positions=[1], side='low')
    ax.violinplot([data_right], positions=[1], side='high')
    plt.close(fig)
test("Half violin plots with side='low'/'high' (3.9+)", test_half_violin)

def test_legend_set_loc():
    """Skill claims leg.set_loc() works post-creation (3.8+)."""
    fresh()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2], label='test')
    leg = ax.legend()
    try:
        leg.set_loc('upper left')
    except AttributeError:
        raise AssertionError("legend.set_loc() not found; skill says 3.8+")
    plt.close(fig)
test("legend.set_loc() post-creation (3.8+)", test_legend_set_loc)

def test_inset_axes():
    """Skill claims ax.inset_axes() with indicate_inset_zoom works."""
    fresh()
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 200)
    y = np.sin(x)
    ax.plot(x, y)
    axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    axins.plot(x, y)
    axins.set_xlim(4, 6)
    axins.set_ylim(-1, 1)
    ax.indicate_inset_zoom(axins, edgecolor='black')
    plt.close(fig)
test("Inset axes with indicate_inset_zoom works", test_inset_axes)

def test_table_from_dataframe():
    """Skill claims ax.table(cellText=df) accepts DataFrame directly (3.10)."""
    fresh()
    try:
        import pandas as pd
        fig, ax = plt.subplots()
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        table = ax.table(cellText=df, loc='center')
        assert table is not None
        plt.close(fig)
    except ImportError:
        print("    (Skipping: pandas not installed)")
test("ax.table(cellText=df) accepts DataFrame (3.10)", test_table_from_dataframe)


# ---------------------------------------------------------------------------
# Section: Colormap Handling (Anti-pattern #7)
# ---------------------------------------------------------------------------
print("\n=== SECTION 11: Colormap Handling ===\n")

def test_cmap_with_extremes():
    """Skill claims .with_extremes() is the right pattern for bad/under/over."""
    fresh()
    cmap = plt.get_cmap('viridis').with_extremes(bad='red', under='gray', over='white')
    assert cmap is not None
    # Verify extremes were set
    bad_color = cmap.get_bad()
    assert bad_color is not None
test("cmap.with_extremes(bad=, under=, over=) works", test_cmap_with_extremes)

def test_cmap_inplace_mutation_pending_deprecation():
    """Skill claims in-place mutation (set_bad, etc.) has pending deprecation since 3.11."""
    fresh()
    cmap = plt.get_cmap('viridis')
    # On 3.10, this should still work without warning (pending deprecation is for 3.11)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmap.set_bad('red')
        dep_warnings = [x for x in w if issubclass(x.category, (DeprecationWarning, FutureWarning, PendingDeprecationWarning))]
        if dep_warnings:
            print(f"    (Got deprecation warning: {dep_warnings[0].message})")
        else:
            print("    (Note: No warning on 3.10; pending deprecation since 3.11)")
test("cmap.set_bad() deprecation status check", test_cmap_inplace_mutation_pending_deprecation)

def test_plt_get_cmap_still_works():
    """Skill says 'plt.get_cmap() still works but colormaps[] is preferred'."""
    fresh()
    cmap = plt.get_cmap('plasma')
    assert cmap is not None
    assert cmap.name == 'plasma'
test("plt.get_cmap() still works", test_plt_get_cmap_still_works)


# ---------------------------------------------------------------------------
# Section: Normalization Classes
# ---------------------------------------------------------------------------
print("\n=== SECTION 12: Normalization Classes ===\n")

def test_normalize():
    """Test Normalize class."""
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=10, clip=False)
    assert norm(5) == 0.5
test("Normalize class works", test_normalize)

def test_lognorm():
    """Test LogNorm class."""
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=1, vmax=100)
    assert norm is not None
test("LogNorm class works", test_lognorm)

def test_symlognorm():
    """Test SymLogNorm class with linthresh as first positional."""
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=1, linscale=1.0, vmin=-100, vmax=100)
    assert norm is not None
test("SymLogNorm(linthresh=) class works", test_symlognorm)

def test_powernorm():
    """Test PowerNorm with gamma as first positional."""
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=100)
    assert norm is not None
test("PowerNorm(gamma=) class works", test_powernorm)

def test_twoslopenorm():
    """Test TwoSlopeNorm(vcenter=)."""
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vcenter=0, vmin=-10, vmax=50)
    assert norm is not None
    # Test it normalizes correctly
    val = norm(0)  # vcenter should map to 0.5
    assert abs(val - 0.5) < 0.01, f"TwoSlopeNorm(vcenter=0)(0) = {val}, expected 0.5"
test("TwoSlopeNorm(vcenter=, vmin=, vmax=) class works", test_twoslopenorm)

def test_centerednorm():
    """Test CenteredNorm."""
    from matplotlib.colors import CenteredNorm
    norm = CenteredNorm(vcenter=0, halfrange=10, clip=False)
    assert norm is not None
test("CenteredNorm class works", test_centerednorm)

def test_boundarynorm():
    """Test BoundaryNorm."""
    from matplotlib.colors import BoundaryNorm
    bounds = [0, 5, 10, 20, 50, 100]
    norm = BoundaryNorm(bounds, ncolors=256, extend='both')
    assert norm is not None
test("BoundaryNorm class works", test_boundarynorm)

def test_funcnorm():
    """Test FuncNorm with lambda pair."""
    from matplotlib.colors import FuncNorm
    norm = FuncNorm((lambda x: np.sqrt(x), lambda x: x**2), vmin=0, vmax=100)
    assert norm is not None
test("FuncNorm with lambda pair works", test_funcnorm)

def test_asinhnorm():
    """Test AsinhNorm."""
    from matplotlib.colors import AsinhNorm
    norm = AsinhNorm(linear_width=1, vmin=-100, vmax=100)
    assert norm is not None
test("AsinhNorm class works", test_asinhnorm)

def test_nonorm():
    """Test NoNorm."""
    from matplotlib.colors import NoNorm
    norm = NoNorm()
    assert norm is not None
test("NoNorm class works", test_nonorm)


# ---------------------------------------------------------------------------
# Section: Styling
# ---------------------------------------------------------------------------
print("\n=== SECTION 13: Styling ===\n")

def test_style_available():
    """Test plt.style.available returns list."""
    avail = plt.style.available
    assert isinstance(avail, list)
    assert len(avail) > 0
test("plt.style.available returns non-empty list", test_style_available)

def test_style_combine():
    """Skill claims style lists work, last wins."""
    fresh()
    plt.style.use(['dark_background', 'fast'])
    plt.style.use('default')  # reset
test("plt.style.use(['dark_background', 'fast']) combined styles", test_style_combine)

def test_style_context():
    """Skill claims plt.style.context() for temporary style."""
    fresh()
    with plt.style.context('seaborn-v0_8-whitegrid'):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        plt.close(fig)
test("plt.style.context() for temporary style", test_style_context)

def test_rc_context():
    """Skill claims mpl.rc_context() for temporary rcParams."""
    fresh()
    with mpl.rc_context({'font.size': 14, 'lines.linewidth': 2, 'axes.grid': True}):
        fig, ax = plt.subplots(layout='constrained')
        assert mpl.rcParams['font.size'] == 14
        assert mpl.rcParams['lines.linewidth'] == 2
        plt.close(fig)
test("mpl.rc_context() for temporary rcParams", test_rc_context)

def test_key_rcparams():
    """Test setting key rcParams mentioned in skill file."""
    fresh()
    original_vals = {}
    keys = [
        'figure.constrained_layout.use',
        'image.interpolation',
        'image.interpolation_stage',
        'savefig.dpi',
        'savefig.bbox',
    ]
    for k in keys:
        original_vals[k] = mpl.rcParams[k]

    mpl.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['image.interpolation'] = 'auto'
    mpl.rcParams['image.interpolation_stage'] = 'auto'
    mpl.rcParams['savefig.dpi'] = 150
    mpl.rcParams['savefig.bbox'] = 'tight'

    assert mpl.rcParams['figure.constrained_layout.use'] is True
    assert mpl.rcParams['savefig.dpi'] == 150

    # Restore
    for k, v in original_vals.items():
        mpl.rcParams[k] = v
test("Key rcParams from skill file are all valid", test_key_rcparams)


# ---------------------------------------------------------------------------
# Section: Integration - datetime
# ---------------------------------------------------------------------------
print("\n=== SECTION 14: Integration - datetime ===\n")

def test_concise_date_formatter():
    """Skill claims ConciseDateFormatter + AutoDateLocator pattern works."""
    fresh()
    import matplotlib.dates as mdates
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    values = np.random.randn(100).cumsum()
    ax.plot(dates, values)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.close(fig)
test("ConciseDateFormatter + AutoDateLocator pattern", test_concise_date_formatter)

def test_date_formatter():
    """Skill claims DateFormatter and MonthLocator work."""
    fresh()
    import matplotlib.dates as mdates
    fig, ax = plt.subplots()
    dates = [datetime.datetime(2024, m, 1) for m in range(1, 13)]
    values = range(12)
    ax.plot(dates, values)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    plt.close(fig)
test("DateFormatter + MonthLocator + autofmt_xdate", test_date_formatter)


# ---------------------------------------------------------------------------
# Section: Integration - numpy masked arrays
# ---------------------------------------------------------------------------
print("\n=== SECTION 15: Integration - numpy ===\n")

def test_masked_array_pcolormesh():
    """Skill claims masked arrays work with pcolormesh."""
    fresh()
    import numpy.ma as ma
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    data = np.random.rand(10, 10)
    masked = ma.masked_where(data < 0.3, data)
    mesh = ax.pcolormesh(X, Y, masked)
    assert mesh is not None
    plt.close(fig)
test("Masked arrays with pcolormesh work", test_masked_array_pcolormesh)


# ---------------------------------------------------------------------------
# Section: Gotchas & Common Mistakes
# ---------------------------------------------------------------------------
print("\n=== SECTION 16: Gotchas & Common Mistakes ===\n")

def test_squeeze_true_fragile():
    """Skill claims squeeze=True (default) makes 1x1 not iterable."""
    fresh()
    fig, ax = plt.subplots(1, 1)
    # ax should be a single Axes, not an array
    assert not hasattr(ax, 'flat'), "1x1 with squeeze=True should not have .flat"
    plt.close(fig)
test("squeeze=True: 1x1 returns single Axes (not array)", test_squeeze_true_fragile)

def test_squeeze_false_always_2d():
    """Skill claims squeeze=False always returns 2D array."""
    fresh()
    fig, axs = plt.subplots(1, 1, squeeze=False)
    assert hasattr(axs, 'flat')
    assert axs.shape == (1, 1)
    for ax in axs.flat:
        ax.plot([1, 2], [1, 2])
    plt.close(fig)
test("squeeze=False: 1x1 returns 2D array with .flat", test_squeeze_false_always_2d)

def test_label_outer():
    """Skill claims label_outer(remove_inner_ticks=True) works (3.8+)."""
    fresh()
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout='constrained')
    x = np.linspace(0, 1, 10)
    for ax in axs.flat:
        ax.plot(x, x)
    for ax in axs.flat:
        ax.label_outer(remove_inner_ticks=True)
    plt.close(fig)
test("label_outer(remove_inner_ticks=True) works (3.8+)", test_label_outer)


# ---------------------------------------------------------------------------
# Section: Performance Tips
# ---------------------------------------------------------------------------
print("\n=== SECTION 17: Performance Tips ===\n")

def test_fast_style():
    """Skill claims 'fast' style exists."""
    fresh()
    assert 'fast' in plt.style.available
    plt.style.use('fast')
    plt.style.use('default')
test("'fast' style exists in plt.style.available", test_fast_style)

def test_fast_style_combined():
    """Skill claims 'fast' can be combined with other styles."""
    fresh()
    plt.style.use(['ggplot', 'fast'])
    plt.style.use('default')
test("plt.style.use(['ggplot', 'fast']) combined works", test_fast_style_combined)

def test_path_simplify_rcparams():
    """Skill claims path.simplify and path.simplify_threshold rcParams exist."""
    fresh()
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 1.0
    assert mpl.rcParams['path.simplify'] is True
    assert mpl.rcParams['path.simplify_threshold'] == 1.0
    # Reset
    mpl.rcParams['path.simplify_threshold'] = 1/9
test("path.simplify rcParams work", test_path_simplify_rcparams)

def test_agg_chunksize():
    """Skill claims agg.path.chunksize rcParam exists."""
    fresh()
    mpl.rcParams['agg.path.chunksize'] = 10000
    assert mpl.rcParams['agg.path.chunksize'] == 10000
    mpl.rcParams['agg.path.chunksize'] = 0  # reset
test("agg.path.chunksize rcParam works", test_agg_chunksize)

def test_markevery():
    """Skill claims markevery= parameter works."""
    fresh()
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 200)
    ax.plot(x, np.sin(x), marker='o', markevery=10)
    plt.close(fig)
test("ax.plot(markevery=10) works", test_markevery)

def test_rasterized():
    """Skill claims rasterized=True works on scatter."""
    fresh()
    fig, ax = plt.subplots()
    x = np.random.randn(100)
    y = np.random.randn(100)
    ax.scatter(x, y, rasterized=True)
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
        fig.savefig(f.name, dpi=150)
        assert os.path.getsize(f.name) > 0
        os.unlink(f.name)
    plt.close(fig)
test("scatter(rasterized=True) works for vector output", test_rasterized)

def test_rasterization_zorder():
    """Skill claims ax.set_rasterization_zorder() works."""
    fresh()
    fig, ax = plt.subplots()
    ax.scatter(np.random.randn(50), np.random.randn(50), zorder=-1)
    ax.set_rasterization_zorder(0)
    plt.close(fig)
test("ax.set_rasterization_zorder() works", test_rasterization_zorder)

def test_close_figures():
    """Skill claims plt.close(fig) and plt.close('all') work."""
    fig1, _ = plt.subplots()
    fig2, _ = plt.subplots()
    plt.close(fig1)
    plt.close('all')
test("plt.close(fig) and plt.close('all') work", test_close_figures)


# ---------------------------------------------------------------------------
# Section: Additional Claim Validation
# ---------------------------------------------------------------------------
print("\n=== SECTION 18: Additional Claims ===\n")

def test_violinplot_orientation():
    """Skill claims violinplot orientation='horizontal' is replacement for vert=False."""
    fresh()
    fig, ax = plt.subplots()
    data = [np.random.randn(100)]
    ax.violinplot(data, orientation='horizontal')
    plt.close(fig)
test("violinplot(orientation='horizontal') works", test_violinplot_orientation)

def test_contourf_basic():
    """Test filled contour from quick reference."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    Z = np.sin(X) * np.cos(Y)
    cf = ax.contourf(X, Y, Z, levels=20, cmap='RdBu_r')
    fig.colorbar(cf, ax=ax)
    plt.close(fig)
test("Filled contour (contourf) with colorbar works", test_contourf_basic)

def test_scatter_with_colormap():
    """Test scatter with c= and cmap= from quick reference."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    x = np.random.randn(50)
    y = np.random.randn(50)
    values = np.random.randn(50)
    sizes = np.random.rand(50) * 100
    sc = ax.scatter(x, y, c=values, cmap='viridis', s=sizes)
    fig.colorbar(sc, ax=ax, label='Value')
    plt.close(fig)
test("scatter(c=, cmap=, s=) with colorbar works", test_scatter_with_colormap)

def test_bar_chart():
    """Test bar chart from quick reference."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    categories = ['A', 'B', 'C']
    values = [10, 20, 15]
    ax.bar(categories, values)
    plt.close(fig)
test("Bar chart works", test_bar_chart)

def test_hist_basic():
    """Test histogram from quick reference."""
    fresh()
    fig, ax = plt.subplots(layout='constrained')
    data = np.random.randn(1000)
    ax.hist(data, bins=30, edgecolor='black')
    plt.close(fig)
test("Histogram with bins and edgecolor works", test_hist_basic)

def test_rectangle_patch_with_alpha():
    """Skill claims Rectangle facecolor=('green', 0.5) works."""
    fresh()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    rect = Rectangle((0, 0), 1, 1, facecolor=('green', 0.5))
    ax.add_patch(rect)
    plt.close(fig)
test("Rectangle facecolor=('green', 0.5) alpha tuple", test_rectangle_patch_with_alpha)

def test_funcanimation_import():
    """Skill claims FuncAnimation import path."""
    from matplotlib.animation import FuncAnimation
    assert FuncAnimation is not None
test("FuncAnimation import from matplotlib.animation", test_funcanimation_import)

def test_constrained_layout_kwarg_still_works():
    """Skill says constrained_layout=True is 'discouraged' but still works."""
    fresh()
    fig, ax = plt.subplots(constrained_layout=True)
    assert fig.get_layout_engine() is not None
    plt.close(fig)
test("constrained_layout=True (discouraged) still works", test_constrained_layout_kwarg_still_works)

def test_tight_layout_kwarg_still_works():
    """Skill says tight_layout=True is 'discouraged' but still works."""
    fresh()
    fig, ax = plt.subplots(tight_layout=True)
    plt.close(fig)
test("tight_layout=True (discouraged) still works", test_tight_layout_kwarg_still_works)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(f"  {e}")

sys.exit(0 if failed == 0 else 1)
