# Matplotlib Research Summary (v3.10.8)

## 1. Version Info

- **Installed**: 3.10.8 (repo on main branch is dev/3.11+)
- **Key recent releases**: 3.8.0 (Sept 2023), 3.9.0 (May 2024), 3.10.0 (Dec 2024)
- Provides first-party PEP 484 type hints since 3.8
- Preliminary free-threaded Python 3.13 support in 3.10

## 2. Explicit vs Implicit API

### Explicit "Axes" (OO) Interface - RECOMMENDED
```python
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("time")
ax.set(xlabel="time", ylabel="value", title="My Plot")  # batch setter
```

### Implicit "pyplot" Interface - For quick exploration only
```python
plt.plot(x, y)
plt.xlabel("time")  # operates on "current axes"
```

**Key differences** (from `galleries/users_explain/figure/api_interfaces.rst`):
- pyplot getters: `plt.xlabel()` vs Axes: `ax.get_xlabel()`
- pyplot setters: `plt.xlabel("time")` vs Axes: `ax.set_xlabel("time")`
- The OO interface is how matplotlib is actually implemented; pyplot is a thin convenience wrapper
- For anything beyond a single plot, the OO interface avoids ambiguity about which axes/figure is "current"

### Anti-pattern: Mixing interfaces
Claude commonly generates code that mixes `plt.title()` / `plt.xlabel()` with `fig, ax = plt.subplots()`. This is confusing and error-prone. If you have an explicit `ax`, use `ax.set_title()`, etc.

## 3. Layout Management

### Modern approach: `layout=` parameter
```python
# PREFERRED - constrained layout
fig, ax = plt.subplots(layout='constrained')

# For fixed-aspect-ratio axes
fig, ax = plt.subplots(layout='compressed')

# Legacy - still works but less capable
fig, ax = plt.subplots(layout='tight')

# Disable layout engine
fig, ax = plt.subplots(layout='none')
```

**Source: `figure.py:2532`**:
```python
layout : {'constrained', 'compressed', 'tight', 'none', LayoutEngine, None}
```

### Deprecated patterns:
- `fig.tight_layout()` call after plotting - use `layout='tight'` at creation
- `plt.subplots(constrained_layout=True)` - use `layout='constrained'`
- `plt.subplots(tight_layout=True)` - use `layout='tight'`
- `fig.set_constrained_layout(True)` - deprecated since 3.6

### Key detail: constrained layout
- `ConstrainedLayoutEngine.__init__`: `h_pad=None, w_pad=None, hspace=None, wspace=None, rect=(0, 0, 1, 1), compress=False`
- Default padding from rcParams: `figure.constrained_layout.h_pad`, `figure.constrained_layout.w_pad`
- `savefig(fname, bbox_inches="tight", pad_inches="layout")` uses layout engine padding (3.8+)
- In 3.10, compressed layout auto-positions `suptitle` above top row of axes

### SubFigures (stable since 3.10)
```python
fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(2, 1)  # 2 rows, 1 col
axs_top = subfigs[0].subplots(1, 3)
axs_bottom = subfigs[1].subplots(1, 2)
```
- Row-major order since 3.10
- Controllable zorders since 3.9
- `fig.subfigs` attribute for iteration

### subplot_mosaic (stable since 3.7+)
```python
fig, axd = plt.subplot_mosaic(
    [["A", "A", "B"],
     ["C", ".", "B"]],  # "." = empty
    layout='constrained',
    per_subplot_kw={"B": {"projection": "polar"}}
)
axd["A"].plot(x, y)
```
Returns a dict of axes keyed by label, supports per_subplot_kw for mixed projections.

## 4. Modern Patterns (3.8-3.10)

### Batch property setting with `ax.set()`
```python
ax.set(xlabel="X", ylabel="Y", title="Title", xlim=(0, 10), ylim=(0, 1))
```

### The `data` parameter (string-based column access)
```python
ax.plot("time", "velocity", data=df)
ax.scatter("x", "y", c="color_col", s="size_col", data=df)
```

### New `Axes.ecdf()` (3.8+)
```python
ax.ecdf(data)  # empirical CDF, no binning needed
ax.ecdf(data, complementary=True)  # survival function
```

### Colorizer (3.10+) - shared norm/cmap pipeline
```python
import matplotlib as mpl
colorizer = mpl.colorizer.Colorizer(norm=norm, cmap='RdBu')
im1 = ax1.imshow(data1, colorizer=colorizer)
im2 = ax2.imshow(data2, colorizer=colorizer)
# Changing colorizer.cmap or colorizer.vmin affects BOTH images
```

### New accessible color cycles (3.10+)
```python
plt.style.use('petroff10')  # 10-color accessible cycle
```

### `orientation` parameter (3.10+)
```python
ax.boxplot(data, orientation='horizontal')  # replaces vert=False
ax.violinplot(data, orientation='horizontal')  # replaces vert=False
```

### Vectorized `hist` styling (3.10)
```python
ax.hist([d1, d2, d3], hatch=["/", ".", "*"], edgecolor=["r", "g", "b"])
```

### `FillBetweenPolyCollection.set_data()` (3.10)
```python
coll = ax.fill_between(t, y1, y2)
coll.set_data(t, new_y1, new_y2)  # update without recreating
```

### Color format: `(color, alpha)` tuple (3.8+)
```python
ax.plot(x, y, color=('blue', 0.3))
Rectangle((0, 0), 1, 1, facecolor=('red', 0.5))
```

## 5. Anti-Patterns (What Claude Will Get Wrong)

### CRITICAL: `plt.get_cmap()` and `matplotlib.cm.get_cmap()`
```python
# WRONG (removed in 3.9):
cmap = matplotlib.cm.get_cmap('viridis')

# RIGHT:
cmap = matplotlib.colormaps['viridis']
# or
cmap = plt.get_cmap('viridis')  # still works, kept for compat
```

### CRITICAL: `ContourSet.collections` (removed in 3.10)
```python
# WRONG (removed):
cs = ax.contour(X, Y, Z)
for coll in cs.collections:
    coll.set_linewidth(2)

# RIGHT: ContourSet IS a single Collection now
cs = ax.contour(X, Y, Z)
cs.set_linewidth(2)
```

### CRITICAL: Old seaborn style names
```python
# WRONG (removed long ago):
plt.style.use('seaborn')
plt.style.use('seaborn-whitegrid')

# RIGHT:
plt.style.use('seaborn-v0_8')
plt.style.use('seaborn-v0_8-whitegrid')
```

### CRITICAL: `plot_date()` (removed)
```python
# WRONG:
ax.plot_date(dates, values)

# RIGHT:
ax.plot(dates, values)
# If you need timezone: ax.xaxis.axis_date(tz=...)
```

### Layout anti-patterns
```python
# WRONG: calling tight_layout() after the fact
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout()

# RIGHT: set layout at creation time
fig, ax = plt.subplots(layout='constrained')
ax.plot(x, y)
```

### Mixing pyplot and OO
```python
# WRONG (confusing):
fig, ax = plt.subplots()
plt.title("My Title")  # operates on "current axes", fragile
plt.xlabel("X")

# RIGHT:
fig, ax = plt.subplots()
ax.set_title("My Title")
ax.set_xlabel("X")
```

### Old `vert=False` pattern
```python
# DEPRECATED (pending in 3.10, full in 3.11):
ax.boxplot(data, vert=False)

# RIGHT:
ax.boxplot(data, orientation='horizontal')
```

### Colormap mutation (pending deprecation)
```python
# DEPRECATED:
cmap = plt.get_cmap('viridis')
cmap.set_bad('red')
cmap.set_under('gray')

# RIGHT:
cmap = plt.get_cmap('viridis').with_extremes(bad='red', under='gray')
```

### `boxplot` labels parameter
```python
# DEPRECATED (3.9):
ax.boxplot(data, labels=['A', 'B', 'C'])

# RIGHT:
ax.boxplot(data, tick_labels=['A', 'B', 'C'])
```

## 6. Deprecated/Removed APIs (3.8-3.10)

### Removed in 3.9:
- `matplotlib.cm.get_cmap()` -> `matplotlib.colormaps[name]`
- `matplotlib.cm.register_cmap()` -> `matplotlib.colormaps.register()`
- `ContourLabeler` attributes (`labelFontProps`, `labelFontSizeList`, etc.)
- `collections.BrokenBarHCollection` -> use `fill_between`
- `Legend.legendHandles` -> `legend_handles`

### Removed in 3.10:
- `ContourSet.collections` -> ContourSet is now a single Collection
- `ContourSet.antialiased` -> use `get_antialiased()`/`set_antialiased()`
- `ContourSet.tcolors`, `ContourSet.tlinewidths`
- `Tick.set_label`, `Tick.set_label1`, `Tick.set_label2`
- `PdfPages(keep_empty=True)` -> empty PDFs no longer created
- `FigureCanvasBase.switch_backends`
- `RendererAgg.tostring_rgb` -> use `buffer_rgba`
- Various `proj3d` functions (private)

### Deprecated in 3.10 (will be removed in future):
- Positional parameters in plotting functions becoming keyword-only
- `vert=` param on boxplot/violinplot -> `orientation=`
- `boxplot.vertical` rcParam (no replacement)
- `Figure.number` mutation
- `FontProperties` non-standard init patterns
- `matplotlib.validate_backend` -> `matplotlib.rcsetup.validate_backend`
- `matplotlib.sanitize_sequence` -> `matplotlib.cbook.sanitize_sequence`

### Deprecated in 3.9:
- `plot_date()` -> use `plot()` with datetime data
- `boxplot` *labels* -> *tick_labels*
- `rcsetup.interactive_bk`, etc. -> `backend_registry.list_builtin()`

### Pending deprecations (in dev/3.11):
- Colormap in-place mutation: `set_bad()`, `set_under()`, `set_over()` -> `with_extremes()`
- `matplotlib.style.core` module -> use `matplotlib.style` directly
- `ListedColormap(N=...)` parameter

## 7. New Features (3.8-3.10)

### 3.8 (Sept 2023):
- First-party type hints
- `Axes.ecdf()` for empirical CDFs
- `(color, alpha)` tuple color format
- `pad_inches="layout"` for savefig
- `Legend.set_loc()` post-creation
- `rcParams['legend.loc']` accepts float tuples
- `label_outer(remove_inner_ticks=True)`
- `tick_params(labelfontfamily='monospace')`
- `SpinesProxy.set(visible=False)` -> `ax.spines[:].set(visible=False)`
- `MultipleLocator(base=3, offset=0.3)`

### 3.9 (May 2024):
- `Axes.inset_axes` stable (no longer experimental)
- Boxplot `label` parameter for legends
- `violinplot(side='low'|'high')` for half violins
- `stackplot(hatch=[...])` per-layer hatching
- `Figure.align_titles()` for subplot title alignment
- Subfigure zorder control
- `BackendRegistry` class
- `imshow` `interpolation_stage` rcParam

### 3.10 (Dec 2024):
- 'petroff10' accessible color cycle
- "berlin", "managua", "vanimo" dark-mode diverging colormaps
- `Colorizer` class for shared norm/cmap pipelines
- `FillBetweenPolyCollection.set_data()`
- `orientation` param for boxplot/violinplot (replaces `vert`)
- Vectorized hist style params (hatch, edgecolor, facecolor, linewidth, linestyle)
- `InsetIndicator` artist (auto-updating inset indicators)
- `ax.table(df)` accepts pandas DataFrame directly
- SubFigures in row-major order + API considered stable
- 3D: `fill_between`, arcball rotation, `axlim_clip`
- 20-25% faster Axes creation
- Agg renderer limit increased to 2^23 pixels

## 8. Styling

### Available built-in styles:
`bmh`, `classic`, `dark_background`, `fast`, `fivethirtyeight`, `ggplot`, `grayscale`, `petroff10`, `seaborn-v0_8-*` (14 variants), `Solarize_Light2`, `tableau-colorblind10`

### Style usage:
```python
plt.style.use('ggplot')
plt.style.use(['dark_background', 'fast'])  # combine, last wins

# Temporary style
with plt.style.context('seaborn-v0_8-whitegrid'):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### rcParams context manager:
```python
with mpl.rc_context({'font.size': 14, 'lines.linewidth': 2}):
    fig, ax = plt.subplots()
```

### The `fast` style:
```python
mplstyle.use('fast')  # sets path simplification + chunking for speed
mplstyle.use(['dark_background', 'ggplot', 'fast'])  # apply last
```

### Key rcParams:
- `figure.constrained_layout.use` - enable constrained layout by default
- `image.interpolation` - default 'auto' (was 'antialiased' before 3.10)
- `image.interpolation_stage` - 'auto' (new default in 3.10)
- `axes3d.mouserotationstyle` - 'arcball', 'azel', 'trackball', 'sphere' (3.10)
- `axes3d.automargin` - False (exact limits, changed in 3.9)

## 9. Key API Signatures

### `plt.subplots`
```python
plt.subplots(
    nrows: int = 1, ncols: int = 1, *,
    sharex: bool | str = False,  # 'none', 'all', 'row', 'col'
    sharey: bool | str = False,
    squeeze: bool = True,
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    subplot_kw: dict | None = None,
    gridspec_kw: dict | None = None,
    **fig_kw  # including layout=, figsize=, dpi=, etc.
) -> tuple[Figure, Axes | np.ndarray]
```

### `plt.figure`
```python
plt.figure(
    num=None,
    figsize: tuple[float, float] | tuple[float, float, str] | None = None,
    # figsize units: "in", "cm", "px" (3.11+, not in 3.10)
    dpi: float | None = None, *,
    facecolor=None, edgecolor=None,
    frameon: bool = True, FigureClass=Figure,
    clear: bool = False, **kwargs
)
```

### `fig.savefig`
```python
fig.savefig(
    fname, *, transparent=None,
    dpi='figure', format=None, metadata=None,
    bbox_inches=None, pad_inches=0.1,
    facecolor='auto', edgecolor='auto',
    backend=None
)
```

### `ax.imshow`
```python
ax.imshow(
    X, cmap=None, norm=None, *, aspect=None,
    interpolation=None, alpha=None,
    vmin=None, vmax=None, colorizer=None,  # colorizer new in 3.10
    origin=None, extent=None,
    interpolation_stage=None,  # 'data', 'rgba', 'auto'(default)
    filternorm=True, filterrad=4.0,
    resample=None, url=None, **kwargs
)
```

### `fig.colorbar`
```python
fig.colorbar(
    mappable,  # AxesImage, ContourSet, etc.
    cax=None,  # existing axes for colorbar
    ax=None,   # axes to steal space from
    use_gridspec=True, **kwargs
)
```

### `fig.subfigures`
```python
fig.subfigures(
    nrows=1, ncols=1, squeeze=True,
    wspace=None, hspace=None,
    width_ratios=None, height_ratios=None, **kwargs
)
```

### `GridSpec`
```python
GridSpec(nrows, ncols, figure=None,
         left=None, bottom=None, right=None, top=None,
         wspace=None, hspace=None,
         width_ratios=None, height_ratios=None)
```

### `FuncAnimation`
```python
FuncAnimation(
    fig,  # Figure object
    func,  # callable(frame, *fargs) -> iterable_of_artists
    frames=None, init_func=None,
    fargs=None, save_count=None,
    *, cache_frame_data=True, **kwargs
)
# IMPORTANT: Must store reference! ani = FuncAnimation(...)
```

### Normalization classes
```python
# Available in matplotlib.colors:
Normalize(vmin=None, vmax=None, clip=False)
LogNorm(vmin=None, vmax=None, clip=False)
SymLogNorm(linthresh, linscale=1.0, vmin=None, vmax=None, clip=False)
PowerNorm(gamma, vmin=None, vmax=None, clip=False)
TwoSlopeNorm(vcenter, vmin=None, vmax=None)
CenteredNorm(vcenter=0, halfrange=None, clip=False)
BoundaryNorm(boundaries, ncolors, clip=False, extend='neither')
FuncNorm(functions, vmin=None, vmax=None, clip=False)
AsinhNorm(linear_width=1, vmin=None, vmax=None, clip=False)
NoNorm(vmin=None, vmax=None, clip=False)
```

## 10. Performance Tips

### Line simplification (rcParams)
```python
mpl.rcParams['path.simplify'] = True
mpl.rcParams['path.simplify_threshold'] = 1.0  # default 1/9, max 1.0
```

### Agg chunking for large datasets
```python
mpl.rcParams['agg.path.chunksize'] = 10000
```

### Marker subsampling
```python
ax.plot(x, y, markevery=10)  # only draw every 10th marker
```

### Legend placement
```python
# SLOW (default): loc='best' searches for fewest data points
ax.legend(loc='best')

# FAST: specify location explicitly
ax.legend(loc='upper right')
```

### The `fast` style for interactive work
```python
import matplotlib.style as mplstyle
mplstyle.use(['dark_background', 'ggplot', 'fast'])  # fast must be last
```

### Backend selection for scripts
```python
import matplotlib
matplotlib.use('Agg')  # non-interactive, fastest for saving
```

### Rasterization for complex vector outputs
```python
ax.plot(x, y, rasterized=True)  # rasterize individual artists in vector output
fig.savefig('out.pdf', dpi=150)
```

### Axes creation is 20-25% faster in 3.10

## 11. Integration Notes

### Pandas
```python
# Direct DataFrame plotting (uses matplotlib under the hood)
df.plot(kind='line', ax=ax)
df.plot.scatter(x='col1', y='col2', ax=ax)

# Matplotlib's data parameter with DataFrames
ax.plot('date', 'value', data=df)
ax.scatter('x', 'y', c='category', data=df)

# DataFrame as table (3.10+)
ax.table(df, loc='center')
```

### NumPy
- All plotting functions accept numpy arrays natively
- Masked arrays work with `pcolor()` (returns `PolyQuadMesh` since 3.8)
- `imshow` accepts (M,N), (M,N,3), (M,N,4) arrays

### Datetime
- `plot()` handles datetime objects natively (no need for `plot_date`)
- For timezone: `ax.xaxis.axis_date(tz=timezone)`
- `ConciseDateFormatter` for clean date axis labels
- `AutoDateLocator` + `AutoDateFormatter` for automatic formatting

### Colormap access:
```python
import matplotlib as mpl
cmap = mpl.colormaps['viridis']  # returns a copy
mpl.colormaps.register(my_cmap, name='my_cmap')
```
