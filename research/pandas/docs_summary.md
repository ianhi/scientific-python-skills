# Pandas 3.0.0 Research Summary

## 1. Version Info

- **Current version**: pandas 3.0.0 (January 21, 2026)
- **Python**: 3.11+
- **NumPy minimum**: 1.26.0
- **PyArrow minimum**: 13.0.0 (optional but strongly recommended - enables default string dtype backing)
- **pytz**: now OPTIONAL (was required). Uses `zoneinfo` from stdlib by default.

## 2. Copy-on-Write (CoW) - THE BIGGEST CHANGE

CoW is now the **default and only mode** in pandas 3.0. There is no opt-out.

### How It Works
- **Any** indexing operation or method returning a new DataFrame/Series **always behaves as a copy**
- Under the hood, pandas uses lazy copies (views) and only copies data when mutation is detected
- The `mode.copy_on_write` option is deprecated and has no effect

### Critical Behavioral Changes

**Chained assignment is DEAD:**
```python
# BROKEN in 3.0 - raises ChainedAssignmentError warning
df["foo"][df["bar"] > 5] = 100  # Does NOT modify df

# CORRECT in 3.0
df.loc[df["bar"] > 5, "foo"] = 100
```

**Column selection never mutates parent:**
```python
df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
subset = df["foo"]
subset.iloc[0] = 100  # Does NOT modify df (subset is independent)
```

**Inplace operations on column selections don't propagate:**
```python
# BROKEN in 3.0
df["foo"].replace(1, 5, inplace=True)  # Does NOT modify df

# CORRECT alternatives:
df.replace({"foo": {1: 5}}, inplace=True)
# or
df["foo"] = df["foo"].replace(1, 5)
```

**NumPy arrays from .to_numpy() are read-only when sharing data:**
```python
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
arr = df.to_numpy()
arr[0, 0] = 100  # ValueError: assignment destination is read-only

# To make writable:
arr = df.to_numpy()
arr.flags.writeable = True  # Circumvents CoW - use with caution
# or
arr = df.to_numpy().copy()  # Safe - independent copy
```

**`copy` keyword deprecated on many methods:**
Deprecated on: `truncate`, `tz_convert`, `tz_localize`, `infer_objects`, `align`, `astype`, `reindex`, `reindex_like`, `set_axis`, `to_period`, `to_timestamp`, `rename`, `transpose`, `swaplevel`, `merge`. The keyword has no effect (CoW handles everything).

**Constructors copy NumPy arrays by default** (pass `copy=False` to avoid).

### Performance Pattern
```python
# SLOW - keeps unnecessary reference, triggers copy on modify
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df2 = df.reset_index(drop=True)
df2.iloc[0, 0] = 100  # Copy triggered because df still exists

# FAST - reassign to same variable, no copy needed
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df = df.reset_index(drop=True)
df.iloc[0, 0] = 100  # No copy needed
```

## 3. String Dtype Changes - SECOND BIGGEST CHANGE

### Default String Dtype
pandas 3.0 uses a dedicated `str` dtype by default instead of `object`:
```python
>>> pd.Series(["a", "b"])
0    a
1    b
dtype: str  # Was dtype: object
```

- Backed by PyArrow if installed, otherwise falls back to NumPy object
- Missing values are always `NaN` (np.nan), NOT `None` or `pd.NA`
- Can only hold strings - setting non-string values raises `TypeError`

### Key Breaking Changes

**Dtype checking:**
```python
# BROKEN in 3.0
ser.dtype == "object"  # Returns False for string columns

# CORRECT in 3.0
ser.dtype == "str"
# or cross-version compatible:
pd.api.types.is_string_dtype(ser.dtype)
```

**None is coerced to NaN:**
```python
ser = pd.Series(["a", "b", None])
print(ser[2])  # nan, NOT None
```

**Setting non-string values raises TypeError:**
```python
ser = pd.Series(["a", "b", "c"])
ser[1] = 2.5  # TypeError in 3.0
```

**`.values` returns ExtensionArray, not ndarray:**
```python
ser = pd.Series(["a", "b"], dtype="str")
type(ser.values)  # ArrowStringArray, NOT numpy.ndarray
ser.to_numpy()    # Use this for numpy array
```

**`astype(str)` now preserves missing values:**
```python
# In pandas 3.0:
pd.Series([1.5, np.nan]).astype("str")
# 0    1.5
# 1    NaN   <-- preserved as NaN, NOT converted to string "nan"
```

**select_dtypes for strings:**
```python
# Cross-version compatible:
df.select_dtypes(include=["object", "string"])
# pandas 3.x only:
df.select_dtypes(include=["str"])
```

### Four StringDtype Variants
- `dtype="str"` - default, NaN semantics, PyArrow-backed if available
- `dtype="string"` - opt-in, pd.NA semantics
- `dtype=pd.ArrowDtype(pa.string())` - full ArrowDtype (returns Arrow types from operations)
- `dtype="string[pyarrow_numpy]"` - REMOVED in 3.0

## 4. Breaking Changes in 3.0 (Comprehensive)

### Datetime/Timedelta Resolution
- Default resolution is now **microseconds** (was nanoseconds)
- `datetime64[us]` is the default when parsing strings
- Integer conversion gives values 1000x smaller (use `dt.as_unit()` before casting)
```python
pd.to_datetime(["2024-03-22"]).dtype  # datetime64[us], NOT datetime64[ns]
```

### Timezone Changes
- `zoneinfo` from stdlib is default (was `pytz`)
- `pytz` is optional, no longer auto-used for string timezone inputs
- Ambiguous/nonexistent times raise `ValueError` (not pytz exceptions)

### Series Integer Indexing
- `Series.__getitem__` and `__setitem__` with integer keys **always treat as labels, never positional**
```python
ser = pd.Series([10, 20, 30], index=[2, 0, 1])
ser[0]  # Returns 20 (label 0), NOT 10 (position 0)
# Use ser.iloc[0] for positional access
```

### Groupby Defaults Changed
- `observed=True` is now the default in `groupby()` (was False)
- `groupby()` no longer has `axis` parameter
- `include_groups=True` removed from `.apply()` and `.Resampler.apply()`

### Offset Aliases Renamed
| Old | New |
|-----|-----|
| `M` | `ME` (MonthEnd) |
| `BM` | `BME` |
| `SM` | `SME` |
| `Q` | `QE` (QuarterEnd) |
| `Y` | `YE` (YearEnd) |
| `A` | removed (use `YE`) |
| `H` | `h` (Hour) |
| `T` | `min` (Minute) |
| `L` | `ms` (Milli) |
| `U` | `us` (Micro) |
| `N` | `ns` (Nano) |
| `BH` | `bh` |

### inplace=True Now Returns self (Not None)
Methods `replace`, `fillna`, `ffill`, `bfill`, `interpolate`, `where`, `mask`, `clip` now return `self` instead of `None` when `inplace=True`. Code that checks `if result is None` will break.

### Day Offset Changed
`pd.offsets.Day` is now calendar-day (preserves time across DST), not fixed 24 hours.

### NaN vs NA in Nullable Dtypes
- By default NaN is treated as equivalent to NA in nullable dtypes
- Arithmetic producing NaN (like 0/0) now produces NA instead
- New option: `pd.options.future.distinguish_nan_and_na`

### concat Changes
- `sort=False` now actually works for DatetimeIndex
- Keys/objs length mismatch raises ValueError
- Empty objects no longer ignored for dtype determination

### axis=None Behavioral Change for DataFrame Reductions
`numpy.sum(df)` / `df.sum(axis=None)` now reduces over BOTH axes (was same as axis=0).

### Integer Slice on Float-dtype Index
Integer slices on objects with float-dtype index are now treated as positional indexing.

### All IO Writer Arguments Keyword-Only
All arguments except the first path-like argument in IO writers are now keyword only.

## 5. Removed APIs (Complete List)

### Major Removals
- `DataFrame.append()` - removed (use `pd.concat()`)
- `Series.append()` - removed (use `pd.concat()`)
- `DataFrame.applymap()` - removed (use `DataFrame.map()`)
- `Styler.applymap()` / `Styler.applymap_index()` - removed (use `Styler.map()`)
- `DataFrame.bool()` / `Series.bool()` - removed
- `DataFrame.first()` / `DataFrame.last()` - removed
- `DataFrame.swapaxes()` / `Series.swapaxes()` - removed
- `Series.ravel()` - removed
- `Series.view()` - removed
- `Series.__int__()` / `Series.__float__()` - removed (use `int(ser.iloc[0])`)
- `Index.format()` - removed
- `Index.is_boolean()`, `.is_integer()`, `.is_floating()`, `.holds_integer()`, `.is_numeric()`, `.is_categorical()`, `.is_object()`, `.is_interval()` - removed
- `pandas.value_counts()` - removed (use `Series.value_counts()`)
- `read_gbq` / `DataFrame.to_gbq()` - removed (use `pandas_gbq`)
- `DataFrameGroupBy.fillna` / `SeriesGroupBy.fillna` / `Resample.fillna` - removed
- `DataFrameGroupBy.grouper` / `SeriesGroupBy.grouper` - removed
- `Grouper.ax`, `.groups`, `.indexer`, `.obj` - removed
- `ArrayManager` - removed
- `pandas.api.types.is_interval` / `pandas.api.types.is_period` - removed (use isinstance)
- `pandas.io.sql.execute` - removed
- `Categorical.to_list()` - removed (use `.tolist()`)
- `Index.sort` - removed (was always TypeError)
- `Series.dt.to_pydatetime` now returns Series (not ndarray)

### Removed Parameters
- `fillna(method=)` - removed (use `ffill()` / `bfill()`)
- `replace(method=, limit=)` - removed
- `read_csv(delim_whitespace=)` - removed (use `sep=r"\s+"`)
- `read_csv(date_parser=)` - removed (use `date_format=`)
- `read_csv(keep_date_col=)` - removed
- `read_csv(infer_datetime_format=)` - removed (strict version is default)
- `read_csv(verbose=)` - removed
- `groupby(axis=)` - removed from ALL groupby, rolling, resample operations
- `Series.apply(convert_dtype=)` - removed
- `DataFrame.align(method=, limit=, fill_axis=, broadcast_axis=)` - removed
- `use_nullable_dtypes` from `read_parquet` - removed
- `resample(kind=)` - removed
- `mode.use_inf_as_na` option - removed
- `future.no_silent_downcasting` option - deprecated (no longer used)
- `Categorical(fastpath=)` - removed
- `PeriodArray(freq=)` - removed (use dtype)
- `DatetimeIndex(closed=, normalize=)` - removed
- `TimedeltaIndex(closed=, unit=)` - removed
- `PeriodIndex(year=, month=, quarter=, day=, hour=, minute=, second=, ordinal=)` - removed (use `from_fields()` / `from_ordinals()`)

### Behavioral Removals
- `errors="ignore"` removed from `to_datetime`, `to_timedelta`, `to_numeric`
- `read_excel`/`read_json`/`read_html`/`read_xml` no longer accept raw strings (must use StringIO/BytesIO)
- `SettingWithCopyWarning` - completely gone (replaced by CoW behavior)
- `.apply()`, `.agg()`, `.transform()` no longer replace numpy/builtin functions with pandas implementations (use string aliases like `"sum"`, `"min"`)
- `concat` combining parsed datetime columns via `parse_dates` - removed
- Passing dict to `SeriesGroupBy.agg` - removed
- Silent downcasting in `where`, `mask`, `fillna` etc. - enforced (no more silent downcasting)
- `DataFrame.stack(future_stack=False)` raises FutureWarning (True is default)
- `Series.agg` using `Series.apply` behavior - removed

## 6. Anti-Patterns (What Claude Will Get Wrong)

### HIGH PRIORITY - Claude will definitely generate these

1. **Using `DataFrame.append()`** - REMOVED. Must use `pd.concat([df1, df2])`.

2. **Using `DataFrame.applymap()`** - REMOVED. Must use `DataFrame.map()`.

3. **Chained assignment** - SILENTLY FAILS:
```python
# Claude will generate this - IT DOES NOTHING in 3.0
df["col"][mask] = value
# Must use: df.loc[mask, "col"] = value
```

4. **Assuming `dtype == "object"` for strings** - Now returns `"str"`.

5. **Using old offset aliases**: `"M"` -> `"ME"`, `"Y"` -> `"YE"`, `"Q"` -> `"QE"`, `"H"` -> `"h"`, `"T"` -> `"min"`

6. **Using `fillna(method="ffill")`** - REMOVED. Use `df.ffill()`.

7. **Assuming nanosecond datetime resolution**:
```python
# Will break: datetime64[ns] is no longer default
ts = pd.Timestamp("2024-01-01")
int(ts.value)  # Different value! Now microseconds by default
```

8. **Assuming `inplace=True` returns None**:
```python
# Old pattern that may break:
result = df.fillna(0, inplace=True)
if result is None:  # No longer True!
    ...
```

9. **Using `groupby(observed=False)` by default** - Default is now `True`.

10. **Using pytz for timezone strings** - Now uses `zoneinfo` by default.

11. **Using `Series[int_key]` for positional access** - Always label-based now.

12. **Using `.values` on string columns expecting numpy array** - Returns ExtensionArray.

13. **Using `read_csv` with raw string data** - Must wrap in `StringIO`.

14. **Using `astype(str)` expecting NaN to become "nan"** - NaN is now preserved as NaN.

15. **Using `np.sum` in `.agg()`** - No longer replaced with pandas `"sum"`. Use string alias.

16. **Defensive `.copy()` calls** - Unnecessary with CoW, wastes memory.

17. **Using `DataFrame.first()` / `.last()`** - REMOVED.

18. **Modifying column subset expecting parent mutation**:
```python
# BROKEN: subset is independent under CoW
col = df["A"]
col.fillna(0, inplace=True)  # Does NOT modify df
# CORRECT:
df["A"] = df["A"].fillna(0)
```

## 7. Arrow Backend

### When to Use PyArrow-backed DataFrames
- String columns automatically use PyArrow if installed (default `str` dtype)
- Use `dtype_backend="pyarrow"` in IO functions for full PyArrow backing
- Use `pd.ArrowDtype(pa_type)` for specific Arrow types (list, map, decimal, etc.)

### New Features
- `DataFrame.from_arrow()` / `Series.from_arrow()` - construct from Arrow PyCapsule Protocol objects
- `__arrow_c_stream__()` - export to Arrow C stream format
- `ArrowDtype` now supports `pyarrow.JsonType`
- Anti-joins: `how="left_anti"` and `how="right_anti"` in merge

### ArrowDtype vs StringDtype
```python
# StringDtype (default for strings) - returns numpy-backed nullable types from operations
pd.Series(["a", "b"], dtype="str")

# Full ArrowDtype - returns Arrow-backed types from all operations
pd.Series(["a", "b"], dtype=pd.ArrowDtype(pa.string()))

# Reading with full pyarrow backend
df = pd.read_csv("file.csv", dtype_backend="pyarrow")
```

### Converting Arrow Tables
```python
# Arrow Table -> pandas DataFrame with ArrowDtype
table = pa.table(...)
df = table.to_pandas(types_mapper=pd.ArrowDtype)
```

### Supported PyArrow Types
All PyArrow types can be used via `pd.ArrowDtype(pa_type)`:
- Numeric: `int8` through `int64`, `uint8` through `uint64`, `float16` through `float64`
- String: `pa.string()`, `pa.large_string()`
- Binary: `pa.binary()`, `pa.large_binary()`
- Temporal: `pa.timestamp()`, `pa.duration()`, `pa.date32()`, `pa.date64()`, `pa.time32()`, `pa.time64()`
- Nested: `pa.list_()`, `pa.large_list()`, `pa.map_()`, `pa.struct()`
- Other: `pa.decimal128()`, `pa.dictionary()`, `pa.bool_()`, `pa.null()`, JSON type

### PyArrow IO Engines
- `read_csv(engine="pyarrow")` - fastest CSV parsing
- `read_json(engine="pyarrow")`
- `read_parquet` - always uses pyarrow or fastparquet
- `read_feather` - always uses pyarrow
- `read_orc` - uses pyarrow

## 8. Key New API Additions

### `pd.col()` - Expression Syntax (NEW in 3.0)
```python
# Instead of lambdas:
df.assign(c=lambda df: df['a'] + df['b'])
# Use:
df.assign(c=pd.col('a') + pd.col('b'))

# Works in assign, loc, getitem/setitem
df.loc[pd.col("speed") > 105]

# Supports all operators and Series methods
pd.col("name").str.upper()
pd.col("val").sum()
```

### `read_iceberg()` / `DataFrame.to_iceberg()` - NEW
Read/write Apache Iceberg tables.

### Anti-joins in merge
```python
pd.merge(df1, df2, how="left_anti", on="key")
pd.merge(df1, df2, how="right_anti", on="key")
```

### `set_option` accepts dict
```python
pd.set_option({"display.max_rows": 100, "display.max_columns": 50})
```

### NamedAgg supports *args and **kwargs
```python
df.groupby("key").agg(
    result=pd.NamedAgg(column="val", aggfunc="mean")
)
```

### GroupBy methods accept `skipna`
`sum`, `mean`, `median`, `prod`, `min`, `max`, `std`, `var`, `sem` now accept `skipna`.

### `fillna(value=None)` now works for non-object dtypes

### New Warning Hierarchy
- `PandasChangeWarning` (base for all upcoming change warnings)
- `Pandas4Warning` -> `PandasDeprecationWarning`
- `Pandas5Warning` -> `PandasPendingDeprecationWarning`
- `PandasFutureWarning`

### Rolling/Expanding additions
- `Rolling.first()`, `Rolling.last()`, `Expanding.first()`, `Expanding.last()`
- `Rolling.nunique()`, `Expanding.nunique()`
- `Rolling.pipe()`, `Expanding.pipe()`
- `Rolling.aggregate()` / `Expanding.aggregate()` accept NamedAgg via **kwargs

### Other Notable Additions
- `DataFrame.iloc` and `Series.iloc` now support boolean masks in `__getitem__`
- `Series.map` accepts `kwargs` to pass to func
- `Series.str.isascii()` - new method
- `Series.str.replace` accepts dict via `pat` parameter
- `Series.str.get_dummies` accepts `dtype` parameter
- `DataFrame.to_csv` / `Series.to_csv` support f-strings for `float_format`
- `DataFrame.to_json` encodes `Decimal` as strings (not floats)
- `DataFrame.to_sql` has `if_exists="delete_rows"` option
- `DataFrame.to_excel` has `autofilter` parameter and `merge_cells="columns"` option
- `DataFrame.cummin/cummax/cumprod/cumsum` have `numeric_only` parameter
- `Styler.to_typst()` - new output format
- `Styler.format_index_names()` - format index/column names

## 9. Migration Patterns (Old -> New)

| Old Pattern | New Pattern |
|---|---|
| `df.append(other)` | `pd.concat([df, other])` |
| `df.applymap(func)` | `df.map(func)` |
| `df["col"][mask] = val` | `df.loc[mask, "col"] = val` |
| `df["col"].replace(a, b, inplace=True)` | `df["col"] = df["col"].replace(a, b)` |
| `ser.dtype == "object"` (for strings) | `pd.api.types.is_string_dtype(ser)` |
| `df.fillna(method="ffill")` | `df.ffill()` |
| `df.fillna(method="bfill")` | `df.bfill()` |
| `pd.date_range(freq="M")` | `pd.date_range(freq="ME")` |
| `pd.date_range(freq="Y")` | `pd.date_range(freq="YE")` |
| `pd.date_range(freq="Q")` | `pd.date_range(freq="QE")` |
| `pd.date_range(freq="H")` | `pd.date_range(freq="h")` |
| `pd.Timedelta("5T")` | `pd.Timedelta("5min")` |
| `pd.Timedelta("5L")` | `pd.Timedelta("5ms")` |
| `df.groupby("a", observed=False)` | Default changed to `observed=True` |
| `ser[0]` (positional on non-RangeIndex) | `ser.iloc[0]` |
| `df.copy()` (defensive) | Not needed with CoW |
| `read_csv(data_string)` | `read_csv(StringIO(data_string))` |
| `np.sum(df)` | `df.sum(axis=None)` (reduces both axes) |
| `df.agg(np.sum)` | `df.agg("sum")` |
| `Series.values` (for numpy) | `Series.to_numpy()` |
| `type(ts.tz)` -> pytz | `type(ts.tz)` -> zoneinfo |
| `Timestamp.utcnow()` | `Timestamp.now("UTC")` |
| `Timestamp.utcfromtimestamp(ts)` | `Timestamp.fromtimestamp(ts, "UTC")` |
| `read_csv(delim_whitespace=True)` | `read_csv(sep=r"\s+")` |
| `read_csv(date_parser=func)` | `read_csv(date_format=fmt)` |
| `errors="ignore"` in to_datetime etc | Use `errors="coerce"` |
| `df.interpolate(method="pad")` | `df.ffill()` |
| `df.groupby(..., axis=1)` | `df.T.groupby(...)` |
| `df.first("3D")` | `df.loc[:df.index[0] + pd.Timedelta("3D")]` |
| `Categorical.to_list()` | `.tolist()` |
| `int(ser)` | `int(ser.iloc[0])` |
| `Index.is_numeric()` | `pd.api.types.is_numeric_dtype(idx)` |

## 10. Performance Tips

### CoW Performance Patterns
- **Reassign to same variable** after operations to avoid unnecessary copies:
  ```python
  df = df.reset_index(drop=True)  # FAST: no copy when modifying df after
  ```
- **Don't keep references** to intermediate DataFrames
- **`drop()` and `rename()` return views** (no copy) under CoW
- **Defensive `.copy()` calls are no longer needed** - remove them for speed
- Methods like `reset_index`, `set_index`, `rename`, `drop`, `assign` return views

### String Performance
- PyArrow-backed strings use less memory and are faster for many operations
- For repeated string values, use `CategoricalDtype` then `.str` methods
- PyArrow string operations are accelerated with native compute functions

### Arrow Backend
- `dtype_backend="pyarrow"` can be significantly faster for IO operations
- Arrow-native merge path for Arrow-backed dtypes (new in 3.0)
- Use `engine="pyarrow"` in `read_csv` for fastest CSV parsing
- `table.to_pandas(types_mapper=pd.ArrowDtype)` for zero-copy conversion

### Groupby
- `skipna` parameter avoids extra filtering steps
- NamedAgg with `*args`/`**kwargs` avoids extra lambdas
- Use string aliases (`"sum"`) instead of numpy functions (`np.sum`) in agg/transform
- `hash-join` optimization in merge (automatic)

### Datetime
- Use `dt.as_unit()` to control resolution before integer conversion
- `date_range` with explicit `unit=` avoids inference overhead
- Default microsecond resolution uses less memory than nanosecond

### General
- `PerformanceWarning` can be globally disabled: `pd.set_option("mode.performance_warnings", False)`
- `numexpr` and `bottleneck` still recommended for large datasets
- `RangeIndex` is now returned more often (from dict constructors, concat, append, etc.) - more memory efficient
- `to_csv(index=False)` has improved performance
- `merge` with sorted non-unique indexes is faster
- `DataFrame.join` with `how="left"/"right"` and `sort=True` is faster

### Avoid These Performance Traps
- **SLOW**: Keeping intermediate DataFrame references alive (forces CoW copies)
- **SLOW**: Using `np.sum(df)` instead of `df.sum()` (no longer auto-optimized)
- **SLOW**: Using `.apply()` with row-wise lambdas when vectorized operations exist
- **SLOW**: String operations on object dtype when PyArrow string dtype is available
