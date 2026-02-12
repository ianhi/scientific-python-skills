# Matplotlib

> Common pitfalls: colorbar steals space from the wrong axes, `FuncAnimation` silently stops if not stored in a variable, `legend(loc='best')` is slow. Always use OO interface with `layout='constrained'`.

## Key Patterns

### Always use OO interface + constrained layout
```python
fig, ax = plt.subplots(layout='constrained')
ax.plot(x, y, label='series')
ax.set(xlabel="X", ylabel="Y", title="Title")
ax.legend(loc='upper right')  # explicit loc is FAST; loc='best' is SLOW

fig.savefig('plot.png', dpi=150, bbox_inches='tight')
plt.close(fig)  # free memory
```

### squeeze=False for programmatic iteration
```python
# FRAGILE - breaks when nrows=1 or ncols=1 (squeeze=True default)
fig, axs = plt.subplots(1, 3)
for ax in axs.flat:  # .flat fails on single Axes

# SAFE - always get 2D array
fig, axs = plt.subplots(nrows, ncols, squeeze=False, layout='constrained')
for ax in axs.flat:
    ax.plot(x, y)
```

### Colorizer - shared norm/cmap across artists (3.10+)
```python
colorizer = mpl.colorizer.Colorizer(cmap='RdBu_r', norm=mpl.colors.Normalize(-5, 5))
im1 = ax1.imshow(data1, colorizer=colorizer)
im2 = ax2.pcolormesh(X, Y, data2, colorizer=colorizer)
fig.colorbar(mpl.colorizer.ColorizingArtist(colorizer), ax=[ax1, ax2])
# Changing colorizer.norm.vmin affects BOTH artists
```

### subplot_mosaic for complex layouts
```python
fig, axd = plt.subplot_mosaic(
    [["A", "A", "B"], ["C", ".", "B"]],
    layout='constrained',
    per_subplot_kw={"B": {"projection": "polar"}},
)
axd["A"].set_title("Wide top")
```

## Gotchas & Common Mistakes

### Colorbar steals space from wrong axes
```python
# WRONG - colorbar shrinks ax1 when ax2 has the image
fig, (ax1, ax2) = plt.subplots(1, 2)
im = ax2.imshow(data)
fig.colorbar(im)  # steals from "current" axes

# RIGHT - specify which axes
fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
im = ax2.imshow(data)
fig.colorbar(im, ax=ax2)

# For shared colorbar: fig.colorbar(im, ax=[ax1, ax2])
```

### Multiple colorbars with constrained layout
```python
# WRONG - each colorbar shifts layout differently, axes end up different sizes
fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained')
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

# RIGHT - use subfigures for independent colorbar handling
fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(1, 2)
ax1 = subfigs[0].subplots()
im1 = ax1.imshow(data1)
subfigs[0].colorbar(im1, ax=ax1)
ax2 = subfigs[1].subplots()
im2 = ax2.imshow(data2)
subfigs[1].colorbar(im2, ax=ax2)
```

### FuncAnimation garbage collection
```python
# WRONG - animation immediately garbage collected, stops working
FuncAnimation(fig, update, frames=100)

# RIGHT - MUST store reference
ani = FuncAnimation(fig, update, frames=100)
```

### imshow origin confusion
```python
# Default origin='upper' means [0,0] is top-left (image convention)
# Use origin='lower' for scientific data where [0,0] is bottom-left
ax.imshow(data, origin='lower', extent=[xmin, xmax, ymin, ymax])
```

## Deprecated/Removed APIs

| Removed/Deprecated | Modern Replacement | Notes |
|-------------------|-------------------|-------|
| `matplotlib.cm.get_cmap()` | `matplotlib.colormaps['name']` | Deprecated 3.7 |
| `get_cmap('viridis', 10)` | `colormaps['viridis'].resampled(10)` | N-level colormaps |
| `plt.style.use('seaborn')` | `plt.style.use('seaborn-v0_8')` | Old names removed |
| `ax.boxplot(vert=False)` | `ax.boxplot(orientation='horizontal')` | Deprecated 3.11 |
| `cs.collections` | `cs.set_linewidth(2)` directly | ContourSet IS a Collection (3.10) |

## Performance Tips

### legend loc='best' is slow
```python
ax.legend(loc='best')         # SLOW - scans all data to find least overlap
ax.legend(loc='upper right')  # FAST - specify directly
```

### Rasterize heavy artists in vector output
```python
# SLOW - vector PDF with millions of points
ax.scatter(x, y)
fig.savefig('plot.pdf')

# FAST - rasterize scatter, keep axes/labels as vectors
ax.scatter(x, y, rasterized=True)
fig.savefig('plot.pdf', dpi=150)
```

### Dense line plots
```python
mpl.rcParams['path.simplify_threshold'] = 1.0  # default 1/9, aggressive simplification
mpl.rcParams['agg.path.chunksize'] = 10000      # split paths for very large datasets
ax.plot(x, y, markevery=10)                      # render every 10th marker only
```

### Use 'fast' style for interactive work
```python
mplstyle.use(['ggplot', 'fast'])  # fast must be last; enables simplification + chunking
```

### Use Agg backend for non-interactive scripts
```python
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import - fastest for batch rendering
```

### Close figures to free memory
```python
plt.close(fig)     # close specific figure
plt.close('all')   # close all (batch scripts)
```

## Known Limitations

- **3D scatter + logscale broken** (#209): Open since 2012.
- **Per-point markers not supported** (#11155): `scatter()` cannot accept a list of markers.
- **timedelta not natively supported** (#8869): Requires manual conversion.
- **Bar width with units** (#13236): No reasonable default width with date/category x-axis.
