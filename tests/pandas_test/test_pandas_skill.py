"""
Comprehensive assessment of skills/pandas.md claims.
Every code example and claim in the skill file is tested here.
"""
import sys
import traceback
import numpy as np
import pandas as pd
import warnings

PASS = []
FAIL = []

def test(name):
    """Decorator that captures test results."""
    def decorator(func):
        try:
            func()
            PASS.append(name)
            print(f"  PASS: {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL: {name} -- {e}")
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  FAIL: {name} -- {type(e).__name__}: {e}")
        return func
    return decorator

print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print()

# ============================================================
# SECTION: Copy-on-Write (CoW) - Always On
# ============================================================
print("=== Copy-on-Write ===")

@test("CoW: chained assignment does not modify df (line 33)")
def _():
    df = pd.DataFrame({"col": [1, 2, 3], "other": [10, 20, 30]})
    original_val = df["col"].iloc[2]
    # Chained assignment should either raise or silently do nothing
    try:
        df["col"][df["other"] > 5] = 100
    except (Warning, ValueError, TypeError, pd.errors.ChainedAssignmentError):
        pass  # CoW may raise ChainedAssignmentError
    assert df["col"].iloc[2] == original_val, f"Expected {original_val}, got {df['col'].iloc[2]}"

@test("CoW: modifying subset does NOT modify parent (line 37-39)")
def _():
    df = pd.DataFrame({"A": [1, 2, 3]})
    subset = df["A"]
    try:
        subset.iloc[0] = 100
    except (Warning, ValueError, TypeError, pd.errors.ChainedAssignmentError):
        pass  # CoW may raise
    assert df["A"].iloc[0] == 1, f"Expected 1, got {df['A'].iloc[0]}"

@test("CoW: inplace on extracted column does NOT modify parent (line 41-43)")
def _():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    col = df["A"]
    try:
        col.fillna(0, inplace=True)
    except (FutureWarning, ValueError, TypeError):
        pass  # May warn or error under CoW
    assert pd.isna(df["A"].iloc[1]), f"Expected NaN, got {df['A'].iloc[1]}"

@test("CoW: .copy() is unnecessary - subset is independent (line 23)")
def _():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = df[["a", "b"]]
    df2.iloc[0, 0] = 999
    assert df.iloc[0, 0] == 1, f"Parent modified unexpectedly: {df.iloc[0, 0]}"

@test("CoW: to_numpy() may be read-only (line 27)")
def _():
    df = pd.DataFrame({"a": [1, 2, 3]})
    arr = df.to_numpy()
    try:
        arr[0, 0] = 999
        writable = True
    except ValueError:
        writable = False
    print(f"    to_numpy() writable: {writable}")
    # The recommended pattern should always work
    arr2 = df.to_numpy().copy()
    arr2[0, 0] = 999
    assert arr2[0, 0] == 999

@test("CoW: reassign to same variable - no copy on mutation (line 18-20)")
def _():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.reset_index(drop=True)
    df.iloc[0, 0] = 100
    assert df.iloc[0, 0] == 100

@test("CoW: keeping old reference forces copy (line 46-48)")
def _():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df_orig = df  # keep reference
    df2 = df.reset_index(drop=True)
    df2.iloc[0, 0] = 100
    assert df_orig.iloc[0, 0] == 1, f"Original modified: {df_orig.iloc[0, 0]}"
    assert df2.iloc[0, 0] == 100

@test("CoW: claim 'copy keyword deprecated' (line 51)")
def _():
    ser = pd.Series([1, 2, 3])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ser.astype(float, copy=True)
        deprecation_warnings = [x for x in w if issubclass(x.category, (FutureWarning, DeprecationWarning))]
        if deprecation_warnings:
            print(f"    Got deprecation warning for copy=: YES")
        else:
            print(f"    No deprecation warning for copy= on astype")
            # Check if copy= parameter still exists at all
            import inspect
            sig = inspect.signature(pd.Series.astype)
            has_copy = "copy" in sig.parameters
            print(f"    copy= parameter exists in signature: {has_copy}")

# ============================================================
# SECTION: String Dtype - New Default
# ============================================================
print("\n=== String Dtype ===")

@test("String dtype: default dtype for strings is 'str' (line 57-58)")
def _():
    ser = pd.Series(["hello", "world"])
    dtype_name = str(ser.dtype)
    assert "str" in dtype_name.lower(), f"Expected str dtype, got {ser.dtype}"

@test("String dtype: missing values are NaN (float nan) (line 61-63)")
def _():
    ser = pd.Series(["a", None, "c"])
    val = ser.iloc[1]
    assert pd.isna(val), f"Expected NaN, got {val}"
    val_type = type(val).__name__
    is_float_nan = isinstance(val, float) and np.isnan(val)
    print(f"    type of missing value: {val_type} (value: {repr(val)})")
    print(f"    is float nan: {is_float_nan}")
    # Skill claims type(ser[1]) is float (nan)
    assert is_float_nan, f"Expected float nan, got {val_type}: {repr(val)}"

@test("String dtype: cannot store non-string - TypeError (line 66)")
def _():
    ser = pd.Series(["a", "b", "c"])
    raised = False
    try:
        ser.iloc[0] = 42
    except TypeError:
        raised = True
    assert raised, "Expected TypeError when assigning int to str Series"

@test("String dtype: astype(str) preserves NaN not 'nan' (line 68-69)")
def _():
    result = pd.Series([1.5, np.nan]).astype("str")
    assert pd.isna(result.iloc[1]), f"Expected NaN, got {repr(result.iloc[1])}"
    assert result.iloc[0] == "1.5", f"Expected '1.5', got {repr(result.iloc[0])}"

# ============================================================
# SECTION: Changed Defaults
# ============================================================
print("\n=== Changed Defaults ===")

@test("Changed default: Timestamp unit is 'us' (line 77)")
def _():
    ts = pd.Timestamp("2024-01-01")
    assert ts.unit == "us", f"Expected 'us', got {ts.unit}"

@test("Changed default: to_datetime returns datetime64[us] (line 116)")
def _():
    idx = pd.to_datetime(["2024-01-01"])
    dtype_str = str(idx.dtype)
    assert "us" in dtype_str, f"Expected datetime64[us], got {idx.dtype}"

@test("Changed default: groupby observed=True by default (line 78)")
def _():
    df = pd.DataFrame({
        "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
        "val": [1, 2, 3]
    })
    result = df.groupby("cat").sum()
    assert "c" not in result.index, f"'c' should be excluded with observed=True. Index: {result.index.tolist()}"
    assert len(result) == 2, f"Expected 2 groups, got {len(result)}"

@test("Changed default: axis=None reduces all axes (line 84-87)")
def _():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result_default = df.sum()
    assert isinstance(result_default, pd.Series), f"Expected Series for default, got {type(result_default)}"
    result_none = df.sum(axis=None)
    assert result_none == 10, f"Expected 10, got {result_none}"

@test("Changed default: np.sum(df) reduces all axes (line 87)")
def _():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = np.sum(df)
    print(f"    np.sum(df) = {result} (type: {type(result).__name__})")
    if isinstance(result, (int, float, np.integer, np.floating)):
        assert result == 10, f"Expected 10, got {result}"
    elif isinstance(result, pd.Series):
        assert False, f"np.sum(df) returned Series (not scalar): {result.to_dict()}"
    else:
        assert False, f"np.sum(df) returned unexpected type {type(result).__name__}: {result}"

@test("Changed default: timezone uses zoneinfo not pytz (line 79)")
def _():
    ts = pd.Timestamp("2024-01-01", tz="US/Eastern")
    tz = ts.tz
    tz_type = type(tz).__module__
    print(f"    Timezone type: {type(tz)} module: {tz_type}")
    assert "zoneinfo" in str(type(tz)).lower() or "zoneinfo" in tz_type, \
        f"Expected zoneinfo, got {type(tz)}"

# ============================================================
# SECTION: Series Indexing - Always Label-Based
# ============================================================
print("\n=== Series Indexing ===")

@test("Series indexing: __getitem__ is label-based (line 93-96)")
def _():
    ser = pd.Series([10, 20, 30], index=[2, 0, 1])
    assert ser[0] == 20, f"Expected 20 (label=0), got {ser[0]}"
    assert ser.iloc[0] == 10, f"Expected 10 (positional), got {ser.iloc[0]}"
    assert ser.loc[0] == 20, f"Expected 20 (label), got {ser.loc[0]}"

# ============================================================
# SECTION: Offset Aliases
# ============================================================
print("\n=== Offset Aliases ===")

@test("Offset aliases: ME for MonthEnd (line 104)")
def _():
    result = pd.date_range("2024", periods=12, freq="ME")
    assert len(result) == 12
    assert result[0].day == 31, f"Expected Jan 31, got day={result[0].day}"

@test("Offset aliases: QE for QuarterEnd (line 105)")
def _():
    result = pd.date_range("2024", periods=4, freq="QE")
    assert len(result) == 4

@test("Offset aliases: old 'M' alias behavior")
def _():
    raised = False
    try:
        pd.date_range("2024", periods=3, freq="M")
    except ValueError:
        raised = True
    print(f"    Using 'M' raised ValueError: {raised}")
    if not raised:
        print("    NOTE: Old 'M' alias still accepted (not yet removed)")

@test("Offset aliases: MS, QS, YS unchanged (line 106)")
def _():
    result = pd.date_range("2024", periods=3, freq="MS")
    assert len(result) == 3
    assert result[0].day == 1

@test("Offset aliases: PeriodIndex still uses 'M' (line 107)")
def _():
    try:
        result = pd.period_range("2024", periods=3, freq="M")
        assert len(result) == 3
        print(f"    PeriodIndex with 'M' works: {result}")
    except ValueError:
        # If 'M' fails, try 'ME'
        result = pd.period_range("2024", periods=3, freq="ME")
        print(f"    PeriodIndex with 'M' failed, 'ME' works: {result}")
        assert False, "Skill says PeriodIndex still uses 'M', but it requires 'ME'"

@test("Offset aliases: 'min' for minutes (line 101)")
def _():
    td = pd.Timedelta("30min")
    assert td.total_seconds() == 30 * 60

# ============================================================
# SECTION: Datetime - Microseconds Default
# ============================================================
print("\n=== Datetime ===")

@test("Datetime: as_unit('ns') works (line 117)")
def _():
    ts = pd.Timestamp("2024-01-01")
    ns_ts = ts.as_unit("ns")
    assert ns_ts.unit == "ns", f"Expected 'ns', got {ns_ts.unit}"

# ============================================================
# SECTION: GroupBy Performance
# ============================================================
print("\n=== GroupBy Performance ===")

@test("GroupBy: string alias works (line 125)")
def _():
    df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    result = df.groupby("k").agg("sum")
    assert result.loc["a", "v"] == 4
    assert result.loc["b", "v"] == 2

@test("GroupBy: np.sum works as fallback (line 128)")
def _():
    df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    result = df.groupby("k").agg(np.sum)
    assert result.loc["a", "v"] == 4

@test("GroupBy: named aggregation works (line 134-138)")
def _():
    df = pd.DataFrame({"key": ["a", "b", "a"], "val": [1, 2, 3]})
    result = df.groupby("key").agg(
        total=("val", "sum"),
        average=("val", "mean"),
        count=("val", "count"),
    )
    assert result.loc["a", "total"] == 4
    assert result.loc["a", "average"] == 2.0
    assert result.loc["a", "count"] == 2

# ============================================================
# SECTION: pd.col() Expression Syntax
# ============================================================
print("\n=== pd.col() Expression Syntax ===")

@test("pd.col(): exists in pandas (line 141)")
def _():
    assert hasattr(pd, "col"), "pd.col does not exist in this version"

@test("pd.col(): works in assign with arithmetic (line 145-148)")
def _():
    assert hasattr(pd, "col"), "pd.col does not exist"
    df = pd.DataFrame({"price": [10, 20], "quantity": [2, 3], "name": ["a", "b"]})
    result = df.assign(
        total=pd.col("price") * pd.col("quantity"),
        name_upper=pd.col("name").str.upper(),
    )
    assert result["total"].tolist() == [20, 60]
    assert result["name_upper"].tolist() == ["A", "B"]

@test("pd.col(): works in loc (line 149)")
def _():
    assert hasattr(pd, "col"), "pd.col does not exist"
    df = pd.DataFrame({"speed": [100, 110, 90]})
    result = df.loc[pd.col("speed") > 105]
    assert len(result) == 1
    assert result.iloc[0]["speed"] == 110

# ============================================================
# SECTION: Anti-Joins
# ============================================================
print("\n=== Anti-Joins ===")

@test("Anti-joins: left_anti merge (line 155)")
def _():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["A", "B", "C"]})
    orders = pd.DataFrame({"customer_id": [1, 3], "order": ["x", "y"]})
    result = pd.merge(customers, orders, how="left_anti", on="customer_id")
    assert len(result) == 1
    assert result.iloc[0]["customer_id"] == 2

@test("Anti-joins: right_anti merge (line 156)")
def _():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["A", "B", "C"]})
    orders = pd.DataFrame({"customer_id": [1, 4], "order": ["x", "y"]})
    result = pd.merge(customers, orders, how="right_anti", on="customer_id")
    assert len(result) == 1
    assert result.iloc[0]["customer_id"] == 4

# ============================================================
# SECTION: IO Performance
# ============================================================
print("\n=== IO Performance ===")

@test("IO: PyArrow engine for CSV (line 163)")
def _():
    import io
    csv_data = "a,b\n1,2\n3,4"
    df = pd.read_csv(io.StringIO(csv_data), engine="pyarrow")
    assert len(df) == 2

@test("IO: dtype_backend='pyarrow' for parquet (line 165)")
def _():
    import tempfile, os
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df.to_parquet(path)
        result = pd.read_parquet(path, dtype_backend="pyarrow")
        assert len(result) == 2
        print(f"    dtypes: {dict(result.dtypes)}")
    finally:
        os.unlink(path)

@test("IO: Arrow zero-copy types_mapper (line 167)")
def _():
    import pyarrow as pa
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    assert len(df) == 3
    print(f"    dtypes: {dict(df.dtypes)}")

# ============================================================
# SECTION: Gotchas
# ============================================================
print("\n=== Gotchas ===")

@test("Gotcha: 0/0 in nullable int produces NA (line 175-176)")
def _():
    ser = pd.Series([0, 1], dtype="Int64")
    result = ser / ser
    print(f"    0/0 result: {result.iloc[0]} (type: {type(result.iloc[0]).__name__})")
    assert pd.isna(result.iloc[0]), f"Expected NA for 0/0, got {result.iloc[0]}"
    assert result.iloc[1] == 1, f"Expected 1 for 1/1, got {result.iloc[1]}"

@test("Gotcha: concat with empty objects dtype (line 182-185)")
def _():
    result = pd.concat([
        pd.DataFrame({"a": pd.array([], dtype="int64")}),
        pd.DataFrame({"a": [1.5]})
    ])
    print(f"    concat result dtype: {result['a'].dtype}")
    assert result["a"].dtype == np.float64, f"Expected float64, got {result['a'].dtype}"

@test("Gotcha: Day offset preserves wall time across DST (line 190-191)")
def _():
    ts = pd.Timestamp("2024-03-10 01:00", tz="US/Eastern")
    result = ts + pd.offsets.Day(1)
    print(f"    Original: {ts}")
    print(f"    + Day(1): {result}")
    assert result.hour == 1, f"Expected hour=1 (wall time preserved), got {result.hour}"
    assert result.day == 11, f"Expected day=11, got {result.day}"

@test("Gotcha: Timedelta('1D') for fixed 24 hours (line 192)")
def _():
    ts = pd.Timestamp("2024-03-10 01:00", tz="US/Eastern")
    result = ts + pd.Timedelta("1D")
    print(f"    Original: {ts}")
    print(f"    + Timedelta('1D'): {result}")
    print(f"    Hour: {result.hour}")
    # With DST spring-forward, 24 actual hours later should be 02:00
    assert result.hour == 2, f"Expected hour=2 (fixed 24h crossing DST), got {result.hour}"

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"PASSED: {len(PASS)}")
print(f"FAILED: {len(FAIL)}")
print("=" * 60)

if FAIL:
    print("\nFAILURES:")
    for name, reason in FAIL:
        print(f"  - {name}: {reason}")

sys.exit(1 if FAIL else 0)
