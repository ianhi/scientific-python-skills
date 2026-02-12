# Pandas

> Copy-on-Write is always on — chained assignment and subset mutation silently do nothing. String columns are now `str` dtype, not `object`. Many defaults changed (`groupby observed`, `axis=None`, offset aliases).

## Copy-on-Write (CoW) - Always On

Every indexing operation returns an independent object. Mutation never propagates to parent.

```python
# Methods return views (no copy until mutation):
df2 = df.reset_index(drop=True)  # view
df2 = df.rename(columns={"a": "x"})  # view
```

**Key patterns:**
```python
# FAST - reassign to same variable (no copy on later mutation)
df = df.reset_index(drop=True)
df = df.rename(columns={"old": "new"})
df.iloc[0, 0] = 100  # No copy triggered

# FAST - remove defensive .copy() calls (CoW handles it)
df2 = df[["a", "b"]]       # No copy needed
subset = df.loc[mask]      # No copy needed

# to_numpy() may be read-only under CoW
arr = df.to_numpy().copy()  # guaranteed writable
```

**Anti-patterns (SILENT behavior):**
```python
# SILENT - chained assignment does NOT modify df (ChainedAssignmentError)
df["col"][df["other"] > 5] = 100  # df unchanged!
# RIGHT: df.loc[df["other"] > 5, "col"] = 100

# SILENT - modifying subset does NOT modify parent
subset = df["col"]
subset.iloc[0] = 100  # df unchanged!
# RIGHT: df.loc[df.index[0], "col"] = 100

col = df["A"]
col.fillna(0, inplace=True)  # df unchanged!
# RIGHT: df["A"] = df["A"].fillna(0)

# SLOW - keeping old reference forces copy
df_orig = df
df2 = df.reset_index(drop=True)
df2.iloc[0, 0] = 100  # Copy triggered because df_orig still referenced
```

**Note:** `copy` keyword deprecated on `astype`, `reindex`, `rename`, `merge`, `align`, etc. `inplace=True` still works but doesn't help with CoW - prefer functional style.

## String Dtype - New Default

```python
# Default dtype for strings is now "str" (PyArrow-backed if available)
ser = pd.Series(["hello", "world"])
ser.dtype  # str

# Missing values are NaN (np.nan), NOT None or pd.NA
ser = pd.Series(["a", None, "c"])
pd.isna(ser[1])  # True
type(ser[1])     # float (nan)

# Cannot store non-string values - TypeError
ser[0] = 42  # TypeError: Scalar must be NA or str

# astype(str) preserves NaN (does NOT convert to "nan" string)
pd.Series([1.5, np.nan]).astype("str")  # [1.5, NaN] not [1.5, "nan"]
```

## Changed Defaults (SILENT behavior changes)

| API | Old Default | New Default | Impact |
|-----|-------------|-------------|--------|
| String columns | `dtype=object` | `dtype=str` | Type checks may break |
| `pd.Timestamp()` | `unit="ns"` | `unit="us"` | Integer conversion values differ |
| `groupby()` | `observed=False` | `observed=True` | Fewer rows with categoricals |
| Timezone handling | `pytz` | `zoneinfo` | `isinstance(tz, pytz.timezone)` fails |
| `axis=None` reduction | Same as `axis=0` | Reduces over all axes | **Silent result change** |

**`axis=None` behavior changed:**
```python
df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
df.sum()           # Series(a=3, b=7) - default axis=0, unchanged
df.sum(axis=None)  # 10 - reduces ALL axes (new behavior)
np.sum(df)         # 10 - numpy reductions also reduce all axes now
```

## Series Indexing - Always Label-Based

```python
ser = pd.Series([10, 20, 30], index=[2, 0, 1])
ser[0]       # Returns 20 (label=0), NOT 10 (position=0)
ser.iloc[0]  # 10 (positional)
ser.loc[0]   # 20 (label)
```

## Offset Aliases

Period offsets require "End" suffix: `M` → `ME`, `Q` → `QE`, `Y` → `YE`. Use `min` not `T`.

```python
pd.date_range("2024", periods=12, freq="ME")  # MonthEnd
pd.date_range("2024", periods=4, freq="QE")   # QuarterEnd
# Start offsets unchanged: "MS", "QS", "YS"
# Exception: PeriodIndex still uses 'M' not 'ME'
```

## Datetime - Microseconds Default

```python
ts = pd.Timestamp("2024-01-01")
ts.unit  # "us" (was "ns")
idx = pd.to_datetime(["2024-01-01"])
idx.dtype  # datetime64[us], NOT datetime64[ns]
# Use .as_unit("ns") if you need nanosecond resolution
# Timezones now use zoneinfo (not pytz)
```

## GroupBy Performance

```python
# FAST - string alias uses optimized C path
df.groupby("k").agg("sum")

# SLOW - numpy fallback, no pandas optimization
df.groupby("k").agg(np.sum)

# SLOW - .apply() with row-wise lambdas
df.groupby("k").apply(lambda g: g.sum())

# Multiple aggregations
df.groupby("key").agg(
    total=("val", "sum"),
    average=("val", "mean"),
    count=("val", "count"),
)
```

## pd.col() Expression Syntax (NEW in 3.0)

```python
# Replace lambdas with pd.col() - works in assign, loc, __getitem__, __setitem__
df.assign(
    total=pd.col("price") * pd.col("quantity"),
    name_upper=pd.col("name").str.upper(),
)
df.loc[pd.col("speed") > 105]
```

## Anti-Joins (NEW in 3.0)

```python
unmatched = pd.merge(customers, orders, how="left_anti", on="customer_id")
# Also: how="right_anti"
```

## IO Performance

```python
# FAST - PyArrow engine for CSV
df = pd.read_csv("large.csv", engine="pyarrow")
# FAST - Arrow-native parquet
df = pd.read_parquet("data.parquet", dtype_backend="pyarrow")
# FAST - Arrow zero-copy from pyarrow tables
df = table.to_pandas(types_mapper=pd.ArrowDtype)
```

## Gotchas

### NaN vs NA in nullable dtypes (#32265)
```python
# Arithmetic producing NaN (like 0/0) now produces NA in nullable dtypes
ser = pd.Series([0, 1], dtype="Int64")
result = ser / ser  # [NA, 1] - 0/0 is NA, not NaN
```

### concat with empty objects
```python
# Empty objects are no longer ignored for dtype determination
pd.concat([pd.DataFrame({"a": pd.array([], dtype="int64")}),
           pd.DataFrame({"a": [1.5]})])
# Column "a" dtype is float64 (empty int64 participates in type promotion)
```

### Day offset is now calendar-day
```python
# pd.offsets.Day preserves wall time across DST (NOT fixed 24 hours)
ts = pd.Timestamp("2024-03-10 01:00", tz="US/Eastern")
ts + pd.offsets.Day(1)  # 2024-03-11 01:00 (wall time preserved)
# Use pd.Timedelta("1D") for fixed 24 hours
```

## Known Limitations

- **NA vs NaN distinction** (#32265): Full separation is ongoing work.
- **NamedAgg single-column only** (#29268): No multi-column lambdas.
- **read_parquet loses Arrow list types** (#53011): Use `dtype_backend="pyarrow"`.
- **Pivot overflow** (#26314): Can overflow with int32 on many columns.

## Performance Tips

- **FAST**: Reassign to same variable - avoids copy on later mutation
- **FAST**: Remove `.copy()` calls - CoW handles this
- **FAST**: `engine="pyarrow"` for CSV, `dtype_backend="pyarrow"` for parquet
- **FAST**: `groupby("k").agg("sum")` string alias over `agg(np.sum)`
- **FAST**: PyArrow-backed strings use less memory than object dtype
- **SLOW**: `.apply()` with row-wise lambdas - use vectorized operations
- **SLOW**: Keeping references to intermediate DataFrames forces copies
