"""
Test script for pandas skill file validation.
Tests claims from skills/pandas.md against pandas 3.0.0.
Uses simple assert + try/except, NOT pytest.
"""

import sys
import traceback
import warnings

import numpy as np
import pandas as pd

passed = 0
failed = 0
errors = []


def test(name, func):
    """Run a test function, track pass/fail."""
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  PASS: {name}")
    except AssertionError as e:
        failed += 1
        tb = traceback.format_exc()
        errors.append((name, str(e), tb))
        print(f"  FAIL: {name}")
        print(f"        {e}")
    except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
        failed += 1
        tb = traceback.format_exc()
        errors.append((name, f"{type(e).__name__}: {e}", tb))
        print(f"  FAIL: {name}")
        print(f"        {type(e).__name__}: {e}")


# =============================================================================
# SECTION 1: Do NOT Use These Outdated Patterns (Anti-patterns)
# =============================================================================
print("\n=== SECTION 1: Removed APIs / Anti-patterns ===\n")

# --- 1. DataFrame.append() removed ---
def test_append_removed():
    df = pd.DataFrame({"a": [1, 2]})
    try:
        df.append(pd.DataFrame({"a": [3]}))
        assert False, "DataFrame.append() should be removed"
    except AttributeError:
        pass  # Expected: append is removed

test("1. DataFrame.append() removed", test_append_removed)

def test_series_append_removed():
    ser = pd.Series([1, 2, 3])
    try:
        ser.append(pd.Series([4]))
        assert False, "Series.append() should be removed"
    except AttributeError:
        pass  # Expected: append is removed

test("1. Series.append() removed", test_series_append_removed)

def test_concat_replacement():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    result = pd.concat([df1, df2], ignore_index=True)
    assert len(result) == 4
    assert list(result["a"]) == [1, 2, 3, 4]

test("1. pd.concat() replacement works", test_concat_replacement)

# --- 2. DataFrame.applymap() removed ---
def test_applymap_removed():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    try:
        df.applymap(lambda x: x**2)
        assert False, "DataFrame.applymap() should be removed"
    except AttributeError:
        pass  # Expected: applymap is removed

test("2. DataFrame.applymap() removed", test_applymap_removed)

def test_map_replacement():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.map(lambda x: x**2)
    assert result.iloc[0, 0] == 1
    assert result.iloc[1, 1] == 16

test("2. DataFrame.map() replacement works", test_map_replacement)

# --- 3. Chained assignment ---
def test_chained_assignment_no_effect():
    df = pd.DataFrame({"col": [1, 2, 3], "other": [10, 20, 30]})
    original_val = df.loc[0, "col"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df["col"][0] = 42
    except (Warning, ValueError, TypeError):
        pass  # CoW may raise an error, which is also acceptable
    # Under CoW, chained assignment should NOT modify df
    assert df.loc[0, "col"] == original_val, f"Expected {original_val}, got {df.loc[0, 'col']}"

test("3. Chained assignment does not modify df (CoW)", test_chained_assignment_no_effect)

def test_loc_assignment_works():
    df = pd.DataFrame({"col": [1, 2, 3], "other": [10, 20, 30]})
    df.loc[0, "col"] = 42
    assert df.loc[0, "col"] == 42

test("3. df.loc[] assignment works correctly", test_loc_assignment_works)

# --- 4. fillna(method=) removed ---
def test_fillna_method_removed():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    try:
        df.fillna(method="ffill")
        assert False, "fillna(method=) should be removed"
    except TypeError:
        pass  # Expected: method= parameter is removed

test("4. fillna(method=) removed", test_fillna_method_removed)

def test_ffill_bfill_work():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result_f = df.ffill()
    assert result_f["a"].iloc[1] == 1.0
    result_b = df.bfill()
    assert result_b["a"].iloc[1] == 3.0

test("4. df.ffill() and df.bfill() work", test_ffill_bfill_work)

# --- 5. Old offset aliases removed ---
def test_old_month_alias_removed():
    try:
        pd.date_range("2024", periods=12, freq="M")
        assert False, "Old 'M' alias should be removed"
    except ValueError:
        pass  # Expected: old alias removed

test("5. Old 'M' offset alias removed", test_old_month_alias_removed)

def test_old_quarter_alias_removed():
    try:
        pd.date_range("2024", periods=4, freq="Q")
        assert False, "Old 'Q' alias should be removed"
    except ValueError:
        pass  # Expected: old alias removed

test("5. Old 'Q' offset alias removed", test_old_quarter_alias_removed)

def test_old_year_alias_removed():
    try:
        pd.date_range("2024", periods=3, freq="Y")
        assert False, "Old 'Y' alias should be removed"
    except ValueError:
        pass  # Expected: old alias removed

test("5. Old 'Y' offset alias removed", test_old_year_alias_removed)

def test_old_hour_alias_removed():
    try:
        pd.Timedelta("5H")
        assert False, "Old 'H' alias should be removed"
    except ValueError:
        pass  # Expected: old alias removed

test("5. Old 'H' (hour) alias removed", test_old_hour_alias_removed)

def test_old_minute_alias_removed():
    try:
        pd.Timedelta("30T")
        assert False, "Old 'T' (minute) alias should be removed"
    except ValueError:
        pass  # Expected: old alias removed

test("5. Old 'T' (minute) alias removed", test_old_minute_alias_removed)

def test_new_month_end_alias():
    result = pd.date_range("2024", periods=3, freq="ME")
    assert len(result) == 3
    assert result[0].month == 1

test("5. New 'ME' (MonthEnd) alias works", test_new_month_end_alias)

def test_new_quarter_end_alias():
    result = pd.date_range("2024", periods=4, freq="QE")
    assert len(result) == 4

test("5. New 'QE' (QuarterEnd) alias works", test_new_quarter_end_alias)

def test_new_year_end_alias():
    result = pd.date_range("2024", periods=3, freq="YE")
    assert len(result) == 3

test("5. New 'YE' (YearEnd) alias works", test_new_year_end_alias)

def test_new_hour_alias():
    td = pd.Timedelta("5h")
    assert td.total_seconds() == 5 * 3600

test("5. New 'h' (hour) alias works", test_new_hour_alias)

def test_new_minute_alias():
    td = pd.Timedelta("30min")
    assert td.total_seconds() == 30 * 60

test("5. New 'min' (minute) alias works", test_new_minute_alias)

def test_start_offsets_unchanged():
    ms = pd.date_range("2024", periods=3, freq="MS")
    qs = pd.date_range("2024", periods=3, freq="QS")
    ys = pd.date_range("2024", periods=3, freq="YS")
    assert len(ms) == 3
    assert len(qs) == 3
    assert len(ys) == 3

test("5. Start offsets (MS, QS, YS) unchanged", test_start_offsets_unchanged)

# --- 6. String dtype default ---
def test_string_dtype_default():
    ser = pd.Series(["hello", "world"])
    assert str(ser.dtype) == "str", f"Expected 'str', got '{ser.dtype}'"

test("6. Default string dtype is 'str' not 'object'", test_string_dtype_default)

def test_is_string_dtype():
    ser = pd.Series(["a", "b", "c"])
    assert pd.api.types.is_string_dtype(ser)

test("6. pd.api.types.is_string_dtype() works with new str dtype", test_is_string_dtype)

def test_string_dtype_check_equals_str():
    ser = pd.Series(["a", "b"])
    assert ser.dtype == "str"

test("6. ser.dtype == 'str' works", test_string_dtype_check_equals_str)

# --- 7. Datetime resolution - microseconds default ---
def test_timestamp_default_unit():
    ts = pd.Timestamp("2024-01-01")
    assert ts.unit == "us", f"Expected 'us', got '{ts.unit}'"

test("7. Timestamp default resolution is microseconds ('us')", test_timestamp_default_unit)

def test_to_datetime_dtype():
    idx = pd.to_datetime(["2024-01-01", "2024-06-15"])
    dtype_str = str(idx.dtype)
    assert "datetime64[us]" in dtype_str, f"Expected datetime64[us], got {dtype_str}"

test("7. pd.to_datetime() returns datetime64[us]", test_to_datetime_dtype)

def test_as_unit():
    ts = pd.Timestamp("2024-01-01")
    ts_ns = ts.as_unit("ns")
    assert ts_ns.unit == "ns"
    # ns value should be 1000x the us value
    us_val = ts.as_unit("us").value
    ns_val = ts_ns.value
    assert ns_val == us_val * 1000, f"ns={ns_val}, us={us_val}"

test("7. Timestamp.as_unit() works for ns/us conversion", test_as_unit)

# --- 8. CoW subset mutation does not propagate ---
def test_cow_subset_no_propagation():
    df = pd.DataFrame({"col": [1, 2, 3]})
    subset = df["col"]
    subset.iloc[0] = 100
    assert df["col"].iloc[0] == 1, f"Expected 1, got {df['col'].iloc[0]}"

test("8. CoW: subset mutation does not propagate to parent", test_cow_subset_no_propagation)

def test_cow_fillna_inplace_no_propagation():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    col = df["A"]
    col.fillna(0, inplace=True)
    assert pd.isna(df["A"].iloc[1]), f"Expected NaN, got {df['A'].iloc[1]}"

test("8. CoW: fillna(inplace=True) on subset does not modify parent", test_cow_fillna_inplace_no_propagation)

def test_cow_replace_inplace_no_propagation():
    df = pd.DataFrame({"A": [1, 2, 3]})
    col = df["A"]
    col.replace(1, 5, inplace=True)
    assert df["A"].iloc[0] == 1, f"Expected 1, got {df['A'].iloc[0]}"

test("8. CoW: replace(inplace=True) on subset does not modify parent", test_cow_replace_inplace_no_propagation)

def test_assign_back_works():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    df["A"] = df["A"].fillna(0)
    assert df["A"].iloc[1] == 0.0

test("8. Assign-back pattern works correctly", test_assign_back_works)

# --- 9. numpy functions in agg/transform ---
def test_string_alias_vs_numpy_agg():
    df = pd.DataFrame({"a": ["x", "x", "y"], "b": [1, 2, 3]})
    # Both should produce the same result, but string alias is preferred
    result_str = df.groupby("a").agg("sum")
    result_np = df.groupby("a").agg(np.sum)
    assert result_str.loc["x", "b"] == 3
    assert result_np.loc["x", "b"] == 3

test("9. String aliases and numpy functions both work in agg", test_string_alias_vs_numpy_agg)

# --- 10. Series[int] is label-based ---
def test_series_int_label_based():
    ser = pd.Series([10, 20, 30], index=[2, 0, 1])
    assert ser[0] == 20, f"Expected 20 (label=0), got {ser[0]}"
    assert ser.iloc[0] == 10, f"Expected 10 (position=0), got {ser.iloc[0]}"
    assert ser.loc[0] == 20, f"Expected 20 (label=0), got {ser.loc[0]}"

test("10. Series[int] is label-based, iloc for positional", test_series_int_label_based)

# --- 11. .values on string columns ---
def test_string_values_type():
    ser = pd.Series(["a", "b", "c"])
    vals = ser.values
    # Skill says returns ArrowStringArray, not numpy array
    is_plain_numpy = isinstance(vals, np.ndarray)
    # With the new str dtype, .values should NOT return a plain numpy array
    assert not is_plain_numpy, \
        f"Expected non-numpy array type, got numpy ndarray with dtype {vals.dtype}"

test("11. .values on string columns does not return plain ndarray", test_string_values_type)

def test_to_numpy_returns_ndarray():
    ser = pd.Series(["a", "b", "c"])
    arr = ser.to_numpy()
    assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"

test("11. .to_numpy() returns ndarray for string columns", test_to_numpy_returns_ndarray)

# --- 12. Removed read_csv parameters ---
def test_delim_whitespace_removed():
    from io import StringIO
    try:
        pd.read_csv(StringIO("a b\n1 2"), delim_whitespace=True)
        assert False, "delim_whitespace should be removed"
    except TypeError:
        pass  # Expected: parameter removed

test("12. read_csv(delim_whitespace=) removed", test_delim_whitespace_removed)

def test_date_parser_removed():
    from io import StringIO
    try:
        pd.read_csv(StringIO("date\n2024-01-01"), date_parser=lambda x: x)
        assert False, "date_parser should be removed"
    except TypeError:
        pass  # Expected: parameter removed

test("12. read_csv(date_parser=) removed", test_date_parser_removed)

def test_infer_datetime_format_removed():
    from io import StringIO
    try:
        pd.read_csv(StringIO("date\n2024-01-01"), infer_datetime_format=True)
        assert False, "infer_datetime_format should be removed"
    except TypeError:
        pass  # Expected: parameter removed

test("12. read_csv(infer_datetime_format=) removed", test_infer_datetime_format_removed)

def test_sep_whitespace_replacement():
    from io import StringIO
    result = pd.read_csv(StringIO("a b\n1 2"), sep=r"\s+")
    assert list(result.columns) == ["a", "b"]
    assert result["a"].iloc[0] == 1

test("12. sep=r'\\s+' replacement works", test_sep_whitespace_replacement)

# --- 13. errors="ignore" removed ---
def test_to_datetime_errors_ignore_removed():
    ser = pd.Series(["2024-01-01", "not-a-date"])
    try:
        pd.to_datetime(ser, errors="ignore")
        assert False, "errors='ignore' should be removed"
    except ValueError:
        pass  # Expected: errors="ignore" removed

test("13. pd.to_datetime(errors='ignore') removed", test_to_datetime_errors_ignore_removed)

def test_to_numeric_errors_ignore_removed():
    ser = pd.Series(["1", "not-a-number"])
    try:
        pd.to_numeric(ser, errors="ignore")
        assert False, "errors='ignore' should be removed"
    except ValueError:
        pass  # Expected: errors="ignore" removed

test("13. pd.to_numeric(errors='ignore') removed", test_to_numeric_errors_ignore_removed)

def test_errors_coerce_works():
    ser_dt = pd.to_datetime(pd.Series(["2024-01-01", "bad"]), errors="coerce")
    assert pd.isna(ser_dt.iloc[1])
    ser_num = pd.to_numeric(pd.Series(["1", "bad"]), errors="coerce")
    assert pd.isna(ser_num.iloc[1])

test("13. errors='coerce' works correctly", test_errors_coerce_works)

# --- 14. Timestamp.utcnow() / utcfromtimestamp() ---
def test_utcnow_deprecated():
    # Skill says "deprecated in 3.0 (Pandas4Warning), will be removed in 4.0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            ts = pd.Timestamp.utcnow()
            # Check if a deprecation or FutureWarning was raised
            has_warning = any(
                issubclass(x.category, (DeprecationWarning, FutureWarning))
                for x in w
            )
            if not has_warning:
                has_warning = any(
                    "utcnow" in str(x.message).lower() or "deprecated" in str(x.message).lower()
                    for x in w
                )
            assert has_warning, (
                f"Expected deprecation warning for utcnow(), "
                f"got warnings: {[str(x.message) for x in w]}"
            )
        except AttributeError:
            pass  # Already fully removed is also acceptable

test("14. Timestamp.utcnow() deprecated", test_utcnow_deprecated)

def test_timestamp_now_utc():
    ts = pd.Timestamp.now("UTC")
    assert ts.tzinfo is not None, "Expected timezone-aware timestamp"

test("14. Timestamp.now('UTC') replacement works", test_timestamp_now_utc)

def test_fromtimestamp_utc():
    ts = pd.Timestamp.fromtimestamp(1700000000, "UTC")
    assert ts.tzinfo is not None

test("14. Timestamp.fromtimestamp(ts, 'UTC') replacement works", test_fromtimestamp_utc)

# --- 16. DataFrame.first() / DataFrame.last() removed ---
def test_first_removed():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": range(10)}, index=idx)
    try:
        df.first("3D")
        assert False, "DataFrame.first() should be removed"
    except AttributeError:
        pass  # Expected: method removed
    except TypeError:
        pass  # Also acceptable: removed but different error

test("16. DataFrame.first() removed", test_first_removed)

def test_last_removed():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"a": range(10)}, index=idx)
    try:
        df.last("3D")
        assert False, "DataFrame.last() should be removed"
    except AttributeError:
        pass  # Expected: method removed
    except TypeError:
        pass  # Also acceptable: removed but different error

test("16. DataFrame.last() removed", test_last_removed)

# --- 17. groupby(axis=) removed ---
def test_groupby_axis_removed():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    try:
        df.groupby("a", axis=1)
        assert False, "groupby(axis=) should be removed"
    except TypeError:
        pass  # Expected: axis parameter removed

test("17. groupby(axis=) removed", test_groupby_axis_removed)

def test_rolling_axis_removed():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    try:
        df.rolling(2, axis=1)
        assert False, "rolling(axis=) should be removed"
    except TypeError:
        pass  # Expected: axis parameter removed

test("17. rolling(axis=) removed", test_rolling_axis_removed)

# --- 18. inplace=True under CoW ---
def test_fillna_inplace_behavior():
    # Skill says inplace=True returns self (not None) in 3.0
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    result = df.fillna(0, inplace=True)
    # Traditional pandas returns None for inplace; skill says it now "returns self"
    if result is None:
        assert False, (
            "Skill claims inplace=True returns self, but it returned None. "
            "Skill file may be inaccurate on this point."
        )
    else:
        assert result is df, "Expected inplace to return self (the same object)"

test("18. fillna(inplace=True) returns self under CoW", test_fillna_inplace_behavior)


# =============================================================================
# SECTION 2: Quick Reference
# =============================================================================
print("\n=== SECTION 2: Quick Reference ===\n")

def test_string_none_is_nan():
    ser = pd.Series(["a", "b", None])
    assert str(ser.dtype) == "str"
    assert pd.isna(ser[2])

test("QR: String None becomes NaN", test_string_none_is_nan)

def test_cow_basic():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = df
    df2.iloc[0, 0] = 9
    assert df.iloc[0, 0] == 1, f"CoW: expected df unchanged, got {df.iloc[0, 0]}"

test("QR: CoW basic - df unchanged after df2 mutation", test_cow_basic)

def test_pd_col_assign():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    try:
        result = df.assign(c=pd.col("a") + pd.col("b"))
        assert list(result["c"]) == [5, 7, 9]
    except AttributeError:
        assert False, "pd.col() not available"

test("QR: pd.col() expression syntax in assign", test_pd_col_assign)

def test_pd_col_loc():
    df = pd.DataFrame({"a": [1, 10, 3], "b": [4, 5, 6]})
    try:
        result = df.loc[pd.col("a") > 5]
        assert len(result) == 1
        assert result.iloc[0]["a"] == 10
    except AttributeError:
        assert False, "pd.col() not available in loc"

test("QR: pd.col() in df.loc[]", test_pd_col_loc)

def test_anti_join_left():
    df1 = pd.DataFrame({"key": [1, 2, 3], "val1": ["a", "b", "c"]})
    df2 = pd.DataFrame({"key": [2, 3, 4], "val2": ["x", "y", "z"]})
    try:
        result = pd.merge(df1, df2, how="left_anti", on="key")
        assert len(result) == 1
        assert result["key"].iloc[0] == 1
    except ValueError:
        assert False, "left_anti join not available"

test("QR: Anti-join left_anti", test_anti_join_left)

def test_anti_join_right():
    df1 = pd.DataFrame({"key": [1, 2, 3], "val1": ["a", "b", "c"]})
    df2 = pd.DataFrame({"key": [2, 3, 4], "val2": ["x", "y", "z"]})
    try:
        result = pd.merge(df1, df2, how="right_anti", on="key")
        assert len(result) == 1
        assert result["key"].iloc[0] == 4
    except ValueError:
        assert False, "right_anti join not available"

test("QR: Anti-join right_anti", test_anti_join_right)

def test_to_numpy_read_only():
    df = pd.DataFrame({"a": [1, 2, 3]})
    arr = df.to_numpy()
    is_readonly = not arr.flags.writeable
    # Just record the behavior; CoW may or may not make it read-only
    if is_readonly:
        print("        (confirmed: to_numpy() returns read-only array)")
    else:
        print("        (note: to_numpy() returned writable array for int dtype)")

test("QR: to_numpy() CoW behavior check", test_to_numpy_read_only)


# =============================================================================
# SECTION 3: Core API Changes
# =============================================================================
print("\n=== SECTION 3: Core API Changes ===\n")

# --- CoW detailed tests ---
def test_cow_series_subset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    subset = df["a"]
    subset.iloc[0] = 99
    assert df["a"].iloc[0] == 1

test("CoW: Series subset is independent", test_cow_series_subset)

def test_cow_dataframe_subset():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cols = df[["a", "b"]]
    cols.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: DataFrame column subset is independent", test_cow_dataframe_subset)

def test_cow_iloc_row():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    row = df.iloc[0]
    row.iloc[0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: iloc row subset is independent", test_cow_iloc_row)

def test_cow_loc_slice():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    sliced = df.loc[0:1]
    sliced.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: loc slice is independent", test_cow_loc_slice)

def test_cow_reset_index_view():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df.reset_index(drop=True)
    df2.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: reset_index returns view/independent copy", test_cow_reset_index_view)

def test_cow_rename_view():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df.rename(columns={"a": "x"})
    df2.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: rename returns view/independent copy", test_cow_rename_view)

def test_cow_drop_view():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = df.drop(columns=["b"])
    df2.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: drop returns view/independent copy", test_cow_drop_view)

def test_cow_assign_view():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = df.assign(c=1)
    df2.iloc[0, 0] = 99
    assert df.iloc[0, 0] == 1

test("CoW: assign returns view/independent copy of shared columns", test_cow_assign_view)

# --- String dtype detailed ---
def test_string_dtype_backed():
    ser = pd.Series(["hello", "world"])
    assert str(ser.dtype) == "str"

test("String: default dtype is 'str'", test_string_dtype_backed)

def test_string_missing_is_nan():
    ser = pd.Series(["a", None, "c"])
    assert pd.isna(ser.iloc[1])
    # Skill says type(ser[1]) is float (nan)
    val = ser.iloc[1]
    assert isinstance(val, float) and np.isnan(val), f"Expected float nan, got {type(val)}: {val}"

test("String: missing values are NaN (float)", test_string_missing_is_nan)

def test_string_no_non_string_values():
    ser = pd.Series(["a", "b"])
    try:
        ser.iloc[0] = 42
        assert False, "Should raise TypeError for non-string value"
    except TypeError:
        pass  # Expected: non-string value rejected

test("String: cannot store non-string values (TypeError)", test_string_no_non_string_values)

def test_astype_str_preserves_nan():
    result = pd.Series([1.5, np.nan]).astype("str")
    assert pd.isna(result.iloc[1]), f"Expected NaN preserved, got {result.iloc[1]}"

test("String: astype(str) preserves NaN", test_astype_str_preserves_nan)

def test_string_pyarrow_numpy_removed():
    # Skill says "string[pyarrow_numpy]" is REMOVED in 3.0
    try:
        ser = pd.Series(["a", "b"], dtype="string[pyarrow_numpy]")
        # If it doesn't raise, that's a skill file error
        assert False, "'string[pyarrow_numpy]' should be removed in 3.0"
    except TypeError:
        pass  # Expected: removed
    except ValueError:
        pass  # Also acceptable for removed dtype

test("String: 'string[pyarrow_numpy]' dtype removed", test_string_pyarrow_numpy_removed)

# --- Datetime resolution ---
def test_timestamp_unit_us():
    ts = pd.Timestamp("2024-01-01")
    assert ts.unit == "us"

test("Datetime: Timestamp default unit is 'us'", test_timestamp_unit_us)

def test_timezone_zoneinfo():
    import zoneinfo
    ts = pd.Timestamp("2024-01-01", tz="US/Eastern")
    assert isinstance(ts.tzinfo, zoneinfo.ZoneInfo), f"Expected ZoneInfo, got {type(ts.tzinfo)}"

test("Datetime: timezone uses zoneinfo (not pytz)", test_timezone_zoneinfo)


# =============================================================================
# SECTION 4: Patterns & Idioms
# =============================================================================
print("\n=== SECTION 4: Patterns & Idioms ===\n")

# --- pd.col() ---
def test_pd_col_assign_arithmetic():
    df = pd.DataFrame({"price": [10.0, 20.0], "quantity": [2, 3]})
    try:
        result = df.assign(total=pd.col("price") * pd.col("quantity"))
        assert list(result["total"]) == [20.0, 60.0]
    except AttributeError:
        assert False, "pd.col() not available"

test("pd.col(): arithmetic in assign", test_pd_col_assign_arithmetic)

def test_pd_col_str_accessor():
    df = pd.DataFrame({"name": ["hello", "world"]})
    try:
        result = df.assign(name_upper=pd.col("name").str.upper())
        assert list(result["name_upper"]) == ["HELLO", "WORLD"]
    except AttributeError:
        assert False, "pd.col().str accessor not available"

test("pd.col(): .str accessor in assign", test_pd_col_str_accessor)

def test_pd_col_dt_accessor():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-15", "2024-06-20"])})
    try:
        result = df.assign(year=pd.col("date").dt.year)
        assert list(result["year"]) == [2024, 2024]
    except AttributeError:
        assert False, "pd.col().dt accessor not available"

test("pd.col(): .dt accessor in assign", test_pd_col_dt_accessor)

def test_pd_col_fillna():
    df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
    try:
        result = df.assign(filled=pd.col("val").fillna(0))
        assert list(result["filled"]) == [1.0, 0.0, 3.0]
    except AttributeError:
        assert False, "pd.col().fillna() not available"

test("pd.col(): .fillna() method", test_pd_col_fillna)

# --- GroupBy patterns ---
def test_groupby_observed_default():
    cat = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c"])
    df = pd.DataFrame({"cat": cat, "val": [1, 2, 3]})
    result = df.groupby("cat").sum()
    # With observed=True (default), "c" should NOT appear
    assert "c" not in result.index, (
        f"'c' should not be in result with observed=True default: {result.index.tolist()}"
    )

test("GroupBy: observed=True by default", test_groupby_observed_default)

def test_groupby_named_agg():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1, 2, 3]})
    result = df.groupby("key").agg(
        total=("val", "sum"),
        average=("val", "mean"),
        count=("val", "count"),
    )
    assert result.loc["a", "total"] == 3
    assert result.loc["a", "average"] == 1.5
    assert result.loc["a", "count"] == 2

test("GroupBy: NamedAgg tuple syntax", test_groupby_named_agg)

def test_groupby_skipna():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1.0, np.nan, 3.0]})
    result_skip = df.groupby("key").sum(skipna=True)
    result_noskip = df.groupby("key").sum(skipna=False)
    assert result_skip.loc["a", "val"] == 1.0
    assert pd.isna(result_noskip.loc["a", "val"]), (
        f"Expected NaN, got {result_noskip.loc['a', 'val']}"
    )

test("GroupBy: skipna parameter on sum", test_groupby_skipna)

# --- CoW-friendly patterns ---
def test_loc_inplace_mutation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df.loc[df["a"] > 1, "b"] = 99
    assert df.loc[1, "b"] == 99
    assert df.loc[2, "b"] == 99
    assert df.loc[0, "b"] == 10

test("CoW-friendly: loc for in-place mutation", test_loc_inplace_mutation)


# =============================================================================
# SECTION 5: IO Patterns
# =============================================================================
print("\n=== SECTION 5: IO Patterns ===\n")

def test_read_json_stringio():
    from io import StringIO
    # Skill says read_json no longer accepts raw strings
    df = pd.read_json(StringIO('{"a": [1, 2]}'))
    assert list(df["a"]) == [1, 2]

test("IO: read_json with StringIO works", test_read_json_stringio)

def test_read_json_raw_string_fails():
    # Skill says read_json no longer accepts raw strings
    try:
        pd.read_json('{"a": [1, 2]}')
        # If it works, the skill claim might be wrong
        assert False, "read_json should not accept raw strings directly"
    except (ValueError, FileNotFoundError):
        pass  # Expected: raw string not accepted
    except TypeError:
        pass  # Also acceptable

test("IO: read_json with raw string fails", test_read_json_raw_string_fails)

def test_read_csv_stringio():
    from io import StringIO
    df = pd.read_csv(StringIO("a,b\n1,2\n3,4"))
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]

test("IO: read_csv with StringIO", test_read_csv_stringio)


# =============================================================================
# SECTION 6: Integration
# =============================================================================
print("\n=== SECTION 6: Integration ===\n")

# --- PyArrow integration ---
def test_arrow_types_mapper():
    import pyarrow as pa
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    # Should have ArrowDtype
    assert "arrow" in str(df["a"].dtype).lower() or "int64[pyarrow]" in str(df["a"].dtype).lower(), \
        f"Expected ArrowDtype, got {df['a'].dtype}"

test("Arrow: types_mapper=pd.ArrowDtype works", test_arrow_types_mapper)

def test_from_arrow():
    import pyarrow as pa
    table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    try:
        df = pd.DataFrame.from_arrow(table)
        assert len(df) == 3
    except AttributeError:
        assert False, "pd.DataFrame.from_arrow() not available"

test("Arrow: DataFrame.from_arrow() (NEW in 3.0)", test_from_arrow)

def test_series_from_arrow():
    import pyarrow as pa
    arr = pa.array([1, 2, 3])
    try:
        ser = pd.Series.from_arrow(arr)
        assert len(ser) == 3
    except AttributeError:
        assert False, "pd.Series.from_arrow() not available"

test("Arrow: Series.from_arrow() (NEW in 3.0)", test_series_from_arrow)

def test_arrow_c_stream():
    df = pd.DataFrame({"a": [1, 2, 3]})
    try:
        stream = df.__arrow_c_stream__()
        assert stream is not None
    except AttributeError:
        assert False, "__arrow_c_stream__() not available"

test("Arrow: __arrow_c_stream__() protocol", test_arrow_c_stream)

# --- NumPy integration ---
def test_numpy_axis_none():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    # Skill says axis=None reduces over BOTH axes
    result = df.sum(axis=None)
    # Should be a scalar: 1+2+3+4 = 10
    assert result == 10, f"Expected 10, got {result}"

test("NumPy: df.sum(axis=None) reduces over both axes", test_numpy_axis_none)

def test_numpy_sum_per_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.sum(axis=0)
    assert isinstance(result, pd.Series)
    assert result["a"] == 3
    assert result["b"] == 7

test("NumPy: df.sum(axis=0) returns per-column sums", test_numpy_sum_per_column)


# =============================================================================
# SECTION 7: Gotchas & Common Mistakes
# =============================================================================
print("\n=== SECTION 7: Gotchas & Common Mistakes ===\n")

def test_astype_str_preserves_nan_gotcha():
    result = pd.Series([1.5, np.nan]).astype("str")
    assert pd.isna(result.iloc[1])
    # Should NOT be the string "nan"
    val = result.iloc[1]
    if isinstance(val, str):
        assert val != "nan", "Should be NaN, not string 'nan'"

test("Gotcha: astype(str) preserves NaN, not 'nan' string", test_astype_str_preserves_nan_gotcha)

def test_set_nonstring_on_str_dtype():
    ser = pd.Series(["a", "b", "c"])
    try:
        ser.iloc[1] = 2.5
        assert False, "Should raise TypeError"
    except TypeError:
        pass  # Expected: non-string rejected

test("Gotcha: Setting non-string on str dtype raises TypeError", test_set_nonstring_on_str_dtype)

def test_select_dtypes_str():
    df = pd.DataFrame({"s": ["a", "b"], "n": [1, 2]})
    result = df.select_dtypes(include=["str"])
    assert "s" in result.columns
    assert "n" not in result.columns

test("Gotcha: select_dtypes(include=['str']) works", test_select_dtypes_str)

def test_nullable_int_nan_is_na():
    ser = pd.Series([0, 1], dtype="Int64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ser / ser
    # 0/0 should be NA, not NaN
    assert pd.isna(result.iloc[0]), f"Expected NA, got {result.iloc[0]}"

test("Gotcha: 0/0 with nullable Int64 produces NA", test_nullable_int_nan_is_na)

def test_concat_empty_dtype():
    empty = pd.DataFrame({"a": pd.array([], dtype="int64")})
    nonempty = pd.DataFrame({"a": [1.5]})
    result = pd.concat([empty, nonempty])
    assert result["a"].dtype == np.float64, f"Expected float64, got {result['a'].dtype}"

test("Gotcha: concat empty preserves dtype in type promotion", test_concat_empty_dtype)

def test_day_offset_calendar():
    ts = pd.Timestamp("2024-03-10 01:00", tz="US/Eastern")
    result = ts + pd.offsets.Day(1)
    # Wall time should be preserved (01:00 -> 01:00 next day)
    assert result.hour == 1, f"Expected hour=1, got {result.hour}"

test("Gotcha: Day offset preserves wall time across DST", test_day_offset_calendar)

def test_period_index_from_fields():
    # WRONG: positional field args
    try:
        pd.PeriodIndex(year=[2024], month=[1], freq="ME")
        assert False, "Positional field args should be removed"
    except TypeError:
        pass  # Expected: positional field args removed

test("Gotcha: PeriodIndex positional field args removed", test_period_index_from_fields)

def test_period_index_from_fields_works():
    try:
        result = pd.PeriodIndex.from_fields(year=[2024], month=[1], freq="ME")
        assert len(result) == 1
    except AttributeError:
        assert False, "PeriodIndex.from_fields() not available"

test("Gotcha: PeriodIndex.from_fields() works", test_period_index_from_fields_works)

# --- Rolling/Expanding new methods ---
def test_rolling_first():
    ser = pd.Series([np.nan, 1.0, 2.0, np.nan, 4.0])
    try:
        result = ser.rolling(3).first()
        # First non-NaN in window of 3
        # Window [NaN, 1, 2] -> first non-NaN = 1
        assert result.iloc[2] == 1.0, f"Expected 1.0, got {result.iloc[2]}"
    except AttributeError:
        assert False, "rolling().first() not available"

test("Rolling: .first() new method", test_rolling_first)

def test_rolling_last():
    ser = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    try:
        result = ser.rolling(3).last()
        # Window [1, 2, NaN] -> last non-NaN = 2
        assert result.iloc[2] == 2.0, f"Expected 2.0, got {result.iloc[2]}"
    except AttributeError:
        assert False, "rolling().last() not available"

test("Rolling: .last() new method", test_rolling_last)

def test_rolling_nunique():
    ser = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0])
    try:
        result = ser.rolling(3).nunique()
        # Window [1, 1, 2] -> 2 unique values
        assert result.iloc[2] == 2.0, f"Expected 2, got {result.iloc[2]}"
    except AttributeError:
        assert False, "rolling().nunique() not available"

test("Rolling: .nunique() new method", test_rolling_nunique)


# =============================================================================
# SECTION 8: Performance Tips
# =============================================================================
print("\n=== SECTION 8: Performance Tips ===\n")

def test_rangeindex_from_dict():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(df.index, pd.RangeIndex), f"Expected RangeIndex, got {type(df.index)}"

test("Perf: RangeIndex from dict constructor", test_rangeindex_from_dict)

def test_rangeindex_from_concat():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [3, 4]})
    result = pd.concat([df1, df2], ignore_index=True)
    assert isinstance(result.index, pd.RangeIndex), f"Expected RangeIndex, got {type(result.index)}"

test("Perf: RangeIndex from concat(ignore_index=True)", test_rangeindex_from_concat)

def test_set_option_dict():
    # Skill claims: pd.set_option({"display.max_rows": 100, "display.max_columns": 50})
    try:
        pd.set_option({"display.max_rows": 100, "display.max_columns": 50})
        val = pd.get_option("display.max_rows")
        assert val == 100
    except TypeError as e:
        assert False, f"set_option with dict failed: {e}"
    finally:
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")

test("Perf: pd.set_option() with dict", test_set_option_dict)


# =============================================================================
# SECTION 9: Additional claims
# =============================================================================
print("\n=== SECTION 9: Additional Claims ===\n")

def test_to_excel_autofilter():
    # Skill claims: to_excel has autofilter parameter
    import inspect
    sig = inspect.signature(pd.DataFrame.to_excel)
    assert "autofilter" in sig.parameters, (
        f"autofilter not in to_excel params: {list(sig.parameters.keys())}"
    )

test("IO: to_excel has autofilter parameter", test_to_excel_autofilter)

def test_to_sql_has_if_exists():
    # Skill claims: to_sql has if_exists="delete_rows"
    import inspect
    sig = inspect.signature(pd.DataFrame.to_sql)
    params = sig.parameters
    assert "if_exists" in params, "if_exists not in to_sql params"

test("IO: to_sql has if_exists parameter", test_to_sql_has_if_exists)

def test_performance_warnings_option():
    try:
        pd.set_option("mode.performance_warnings", False)
        val = pd.get_option("mode.performance_warnings")
        assert val is False
    except (KeyError, ValueError) as e:
        assert False, f"mode.performance_warnings option failed: {e}"
    finally:
        try:
            pd.reset_option("mode.performance_warnings")
        except (KeyError, ValueError):
            print("        (note: could not reset mode.performance_warnings)")

test("Perf: mode.performance_warnings option", test_performance_warnings_option)

def test_expanding_pipe():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    try:
        result = df["a"].expanding().pipe(lambda x: x.mean())
        assert len(result) == 5
    except AttributeError:
        assert False, "expanding().pipe() not available"

test("Rolling/Expanding: pipe() method", test_expanding_pipe)

def test_named_agg_positional_args():
    # Skill claims: NamedAgg with *args and **kwargs (NEW in 3.0)
    # pd.NamedAgg("val", my_func, 0, 100)
    def between_count(x, low, high):
        return ((x >= low) & (x <= high)).sum()

    df = pd.DataFrame({"key": ["a", "a", "b", "b"], "val": [1, 50, 150, 200]})
    try:
        result = df.groupby("key").agg(
            count_between=pd.NamedAgg("val", between_count, 0, 100),
        )
        assert result.loc["a", "count_between"] == 2
        assert result.loc["b", "count_between"] == 0
    except TypeError as e:
        assert False, f"NamedAgg positional args failed: {e}"

test("GroupBy: NamedAgg with positional args (NEW)", test_named_agg_positional_args)

def test_to_csv_fstring_float_format():
    # Skill claims: to_csv supports f-string float formatting
    from io import StringIO
    df = pd.DataFrame({"a": [1.23456, 2.78901]})
    buf = StringIO()
    df.to_csv(buf, float_format="{:.2f}", index=False)
    content = buf.getvalue()
    assert "1.23" in content, f"Expected '1.23' in output, got: {content}"
    assert "1.23456" not in content, f"Expected truncated float, got: {content}"

test("IO: to_csv f-string float_format", test_to_csv_fstring_float_format)

def test_distinguish_nan_and_na_option():
    # Skill claims: pd.options.future.distinguish_nan_and_na is available
    try:
        val = pd.get_option("future.distinguish_nan_and_na")
        # Just checking the option exists
        print(f"        (option value: {val})")
    except (KeyError, ValueError) as e:
        assert False, f"future.distinguish_nan_and_na option not available: {e}"

test("Gotcha: future.distinguish_nan_and_na option exists", test_distinguish_nan_and_na_option)


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for name, err, tb in errors:
        print(f"\n  --- {name} ---")
        print(f"  Error: {err}")
        # Show last few lines of traceback for context
        tb_lines = tb.strip().splitlines()
        for line in tb_lines[-5:]:
            print(f"    {line}")

sys.exit(0 if failed == 0 else 1)
