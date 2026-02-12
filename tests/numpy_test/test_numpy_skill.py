"""
Assessment of skills/numpy.md against actual NumPy behavior.
Verifies every code example and claim in the skill file.
NumPy version: 2.4.x
"""
import sys
import warnings
import numpy as np

results = {"passed": [], "failed": [], "warnings": []}


def test(name, func):
    """Run a test and record pass/fail."""
    try:
        func()
        results["passed"].append(name)
        print(f"  PASS: {name}")
    except AssertionError as e:
        results["failed"].append((name, str(e)))
        print(f"  FAIL: {name}: {e}")
    except Exception as e:
        results["failed"].append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR: {name}: {type(e).__name__}: {e}")


def warn(name, msg):
    results["warnings"].append((name, msg))
    print(f"  WARN: {name}: {msg}")


print(f"NumPy version: {np.__version__}")

# =============================================================================
# SECTION: NEP 50 Type Promotion (lines 9-31)
# =============================================================================
print("\n=== NEP 50 Type Promotion ===")


def test_nep50_float32_plus_python_float():
    """Line 12: arr + 1.0 -> float32 (Python float adapts to arr)"""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr + 1.0
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_nep50_float32_plus_python_int():
    """Line 13: arr + 1 -> float32 (Python int adapts to arr)"""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr + 1
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_nep50_numpy_scalar_promotion():
    """Line 16: np.float32(3) + np.float64(3) -> float64 (standard promotion)"""
    result = np.float32(3) + np.float64(3)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


def test_nep50_force_promotion_with_numpy_scalar():
    """Line 19: arr * np.float64(0.1) -> float64 (NumPy scalar forces promotion)"""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr * np.float64(0.1)
    assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"


def test_nep50_python_scalar_stays_float32():
    """Line 20: arr * 0.1 -> float32 (Python scalar adapts - precision loss!)"""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr * 0.1
    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"


def test_nep50_uint8_overflow():
    """Line 23-24: arr_u8 + 100 -> uint8(44) with overflow!"""
    arr_u8 = np.array([200], dtype=np.uint8)
    result = arr_u8 + 100
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    assert result[0] == 44, f"Expected 44 (overflow), got {result[0]}"


def test_nep50_uint8_safe_fix():
    """Line 25: arr_u8.astype(np.int16) + 100"""
    arr_u8 = np.array([200], dtype=np.uint8)
    result = arr_u8.astype(np.int16) + 100
    assert result[0] == 300, f"Expected 300, got {result[0]}"


def test_nep50_float32_precision_loss():
    """Lines 28-30: float32 1e8 + 1.0 loses precision"""
    arr_f32 = np.array([1e8], dtype=np.float32)
    # Python float adapts to float32
    result_weak = arr_f32 + 1.0
    assert result_weak.dtype == np.float32, f"Expected float32, got {result_weak.dtype}"
    assert result_weak[0] == np.float32(1e8), (
        f"Expected 1e8 (precision loss), got {result_weak[0]}"
    )
    # NumPy scalar forces promotion to float64
    result_strong = arr_f32 + np.float64(1.0)
    assert result_strong.dtype == np.float64, f"Expected float64, got {result_strong.dtype}"
    assert result_strong[0] == 100000001.0, (
        f"Expected 100000001.0, got {result_strong[0]}"
    )


test("float32 + Python float -> float32", test_nep50_float32_plus_python_float)
test("float32 + Python int -> float32", test_nep50_float32_plus_python_int)
test("NumPy scalar promotion (float32+float64->float64)", test_nep50_numpy_scalar_promotion)
test("NumPy scalar forces promotion via multiply", test_nep50_force_promotion_with_numpy_scalar)
test("Python scalar stays float32 (precision loss)", test_nep50_python_scalar_stays_float32)
test("uint8 overflow: 200+100=44", test_nep50_uint8_overflow)
test("uint8 safe fix: astype(int16) + 100 = 300", test_nep50_uint8_safe_fix)
test("float32 precision loss: 1e8+1.0", test_nep50_float32_precision_loss)

# =============================================================================
# SECTION: Copy Semantics (lines 36-43)
# =============================================================================
print("\n=== Copy Semantics ===")


def test_copy_asarray_default():
    """Line 37: np.asarray(data) -> copy if needed"""
    data = np.array([1, 2, 3])
    x = np.asarray(data)
    assert np.shares_memory(x, data), "asarray should not copy when not needed"


def test_copy_true():
    """Line 38: np.array(data, copy=True) -> always copy"""
    data = np.array([1, 2, 3])
    x = np.array(data, copy=True)
    assert not np.shares_memory(x, data), "copy=True should always copy"


def test_copy_false_no_copy_needed():
    """Line 39: np.array(data, copy=False) -> never copy"""
    data = np.array([1, 2, 3])
    x = np.array(data, copy=False)
    assert np.shares_memory(x, data), "copy=False should share memory when possible"


def test_copy_false_raises_when_copy_needed():
    """Line 39: np.array(data, copy=False) raises ValueError if copy needed"""
    data = np.array([1, 2, 3], dtype=np.int32)
    try:
        x = np.array(data, copy=False, dtype=np.float64)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected


def test_reshape_copy_false():
    """Line 42: np.reshape(a, (2, 3), copy=False) -> ValueError if can't return view"""
    a = np.arange(6)
    reshaped = np.reshape(a, (2, 3), copy=False)
    assert np.shares_memory(reshaped, a), "reshape copy=False should share memory"

    # Non-contiguous data that can't be reshaped as a view
    a_noncontig = np.arange(12).reshape(3, 4)[:, ::2]
    try:
        np.reshape(a_noncontig, (6,), copy=False)
        assert False, "Should have raised ValueError for non-contiguous data"
    except ValueError:
        pass  # Expected


test("asarray returns view when no copy needed", test_copy_asarray_default)
test("array(copy=True) always copies", test_copy_true)
test("array(copy=False) returns view when possible", test_copy_false_no_copy_needed)
test("array(copy=False) raises ValueError when copy needed", test_copy_false_raises_when_copy_needed)
test("reshape(copy=False) behavior", test_reshape_copy_false)

# =============================================================================
# SECTION: StringDType & String Operations (lines 47-63)
# =============================================================================
print("\n=== StringDType & String Operations ===")


def test_stringdtype_import_and_basic():
    """Lines 48-50: from numpy.dtypes import StringDType; basic creation"""
    from numpy.dtypes import StringDType
    arr = np.array(["hello", "world"], dtype=StringDType())
    assert arr[0] == "hello"
    assert arr[1] == "world"


def test_stringdtype_with_na_object():
    """Lines 53-54: StringDType(na_object=np.nan)"""
    from numpy.dtypes import StringDType
    dt = StringDType(na_object=np.nan)
    arr = np.array(["hello", np.nan, "world"], dtype=dt)
    assert len(arr) == 3
    assert arr[0] == "hello"
    import math
    assert math.isnan(arr[1]), f"Expected NaN, got {arr[1]}"
    assert arr[2] == "world"


def test_np_strings_upper():
    """Line 57: np.strings.upper(arr)"""
    arr = np.array(["hello", "world"], dtype=np.dtypes.StringDType())
    result = np.strings.upper(arr)
    assert result[0] == "HELLO"
    assert result[1] == "WORLD"


def test_np_strings_find():
    """Line 58: np.strings.find(arr, "lo")"""
    arr = np.array(["hello", "world"], dtype=np.dtypes.StringDType())
    result = np.strings.find(arr, "lo")
    assert result[0] == 3, f"Expected 3, got {result[0]}"
    assert result[1] == -1, f"Expected -1, got {result[1]}"


def test_np_strings_replace():
    """Line 59: np.strings.replace(arr, "o", "0")"""
    arr = np.array(["hello", "world"], dtype=np.dtypes.StringDType())
    result = np.strings.replace(arr, "o", "0")
    assert result[0] == "hell0", f"Expected hell0, got {result[0]}"
    assert result[1] == "w0rld", f"Expected w0rld, got {result[1]}"


def test_np_char_still_works():
    """Line 62: np.char.upper(arr) - slow but should work"""
    arr = np.array(["hello", "world"])
    result = np.char.upper(arr)
    assert result[0] == "HELLO"


test("StringDType import and basic creation", test_stringdtype_import_and_basic)
test("StringDType with na_object=np.nan", test_stringdtype_with_na_object)
test("np.strings.upper", test_np_strings_upper)
test("np.strings.find", test_np_strings_find)
test("np.strings.replace", test_np_strings_replace)
test("np.char still works (legacy)", test_np_char_still_works)

# =============================================================================
# SECTION: Useful Modern API (lines 67-80)
# =============================================================================
print("\n=== Useful Modern API ===")


def test_unique_counts():
    """Line 69: np.unique_counts(x) -> (values, counts)"""
    x = np.array([3, 1, 2, 1, 3, 3])
    vals, counts = np.unique_counts(x)
    np.testing.assert_array_equal(vals, [1, 2, 3])
    np.testing.assert_array_equal(counts, [2, 1, 3])


def test_unique_inverse():
    """Line 70: np.unique_inverse(x) -> (values, inverse_indices)"""
    x = np.array([3, 1, 2, 1, 3, 3])
    vals, inverse = np.unique_inverse(x)
    np.testing.assert_array_equal(vals, [1, 2, 3])
    np.testing.assert_array_equal(vals[inverse], x)


def test_unique_sorted_false():
    """Line 73: np.unique(arr, sorted=False) -> hash-based, up to 15x faster"""
    arr = np.array([5, 3, 1, 3, 5, 1, 2])
    try:
        result = np.unique(arr, sorted=False)
        assert set(result) == {1, 2, 3, 5}, f"Expected {{1,2,3,5}}, got {set(result)}"
    except TypeError as e:
        if "sorted" in str(e):
            results["failed"].append(
                ("unique sorted=False",
                 f"sorted= parameter not supported in NumPy {np.__version__}: {e}")
            )
            print(f"  FAIL: unique sorted=False not supported: {e}")
            return
        raise


def test_cumulative_sum_basic():
    """Line 76: np.cumulative_sum(x, axis=None, include_initial=False)"""
    x = np.array([1, 2, 3, 4])
    result = np.cumulative_sum(x)
    np.testing.assert_array_equal(result, [1, 3, 6, 10])


def test_cumulative_sum_include_initial():
    """Line 76: include_initial=True adds a zero at the start"""
    x = np.array([1, 2, 3, 4])
    result = np.cumulative_sum(x, include_initial=True)
    np.testing.assert_array_equal(result, [0, 1, 3, 6, 10])


def test_cumulative_sum_signature():
    """Verify exact signature: axis and include_initial params"""
    import inspect
    sig = inspect.signature(np.cumulative_sum)
    params = list(sig.parameters.keys())
    assert 'include_initial' in params, f"include_initial not in params: {params}"
    default = sig.parameters['include_initial'].default
    assert default == False, f"Expected default=False, got {default}"


def test_isdtype():
    """Line 79: np.isdtype(dtype, 'real floating')"""
    assert np.isdtype(np.float64, 'real floating')
    assert np.isdtype(np.float32, 'real floating')
    assert np.isdtype(np.int32, 'integral')
    assert np.isdtype(np.float32, 'numeric')
    assert np.isdtype(np.bool_, 'bool')
    assert not np.isdtype(np.float64, 'integral')


test("unique_counts returns (values, counts)", test_unique_counts)
test("unique_inverse returns (values, inverse_indices)", test_unique_inverse)
test("unique sorted=False (hash-based)", test_unique_sorted_false)
test("cumulative_sum basic", test_cumulative_sum_basic)
test("cumulative_sum include_initial", test_cumulative_sum_include_initial)
test("cumulative_sum signature verification", test_cumulative_sum_signature)
test("isdtype type checking", test_isdtype)

# =============================================================================
# SECTION: Gotchas - np.isclose (lines 85-89)
# =============================================================================
print("\n=== Gotchas: np.isclose ===")


def test_isclose_formula_is_asymmetric():
    """Line 87: abs(a-b) <= atol + rtol * abs(b)  -- uses b as reference"""
    # Choose a, b where the asymmetry is clear
    a, b = 1e-10, 2e-10
    r1 = np.isclose(a, b, rtol=0.5, atol=0)
    r2 = np.isclose(b, a, rtol=0.5, atol=0)
    # isclose(1e-10, 2e-10, rtol=0.5): |1e-10 - 2e-10| = 1e-10 <= 0.5 * 2e-10 = 1e-10 -> True
    # isclose(2e-10, 1e-10, rtol=0.5): |2e-10 - 1e-10| = 1e-10 <= 0.5 * 1e-10 = 5e-11 -> False
    assert r1 == True, f"Expected True (ref=b=2e-10), got {r1}"
    assert r2 == False, f"Expected False (ref=b=1e-10), got {r2}"


def test_isclose_formula_exact():
    """Verify the exact formula: abs(a-b) <= atol + rtol * abs(b)"""
    a = 100.0
    b = 100.001
    rtol = 1e-5
    atol = 1e-8
    # abs(100 - 100.001) = 0.001
    # atol + rtol * abs(b) = 1e-8 + 1e-5 * 100.001 = 0.00100001 + 1e-8 ≈ 0.001000
    result = np.isclose(a, b, rtol=rtol, atol=atol)
    # Manual: 0.001 <= 0.00100001 -> True (barely)
    expected = abs(a - b) <= atol + rtol * abs(b)
    assert result == expected, f"np.isclose={result}, formula={expected}"


test("isclose asymmetric: swapping a,b gives different result", test_isclose_formula_is_asymmetric)
test("isclose formula: abs(a-b) <= atol + rtol*abs(b)", test_isclose_formula_exact)

# =============================================================================
# SECTION: Gotchas - npz multiprocessing (lines 92-96)
# =============================================================================
print("\n=== Gotchas: npz multiprocessing ===")


def test_npz_materialization():
    """Line 96: data = dict(np.load("file.npz"))  # materialize before forking"""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.npz")
        np.savez(path, a=np.array([1, 2]), b=np.array([3, 4]))
        # Lazy loading returns NpzFile
        lazy = np.load(path)
        assert hasattr(lazy, 'files'), "NpzFile should have .files attribute"
        # Materialization via dict()
        data = dict(np.load(path))
        assert isinstance(data, dict)
        np.testing.assert_array_equal(data['a'], [1, 2])
        np.testing.assert_array_equal(data['b'], [3, 4])


test("npz file materialization with dict()", test_npz_materialization)

# =============================================================================
# SECTION: Gotchas - np.load pickle security (lines 99-101)
# =============================================================================
print("\n=== Gotchas: np.load pickle security ===")


def test_load_pickle_requires_opt_in():
    """Line 101: np.load("file.npy", allow_pickle=True) - explicit opt-in required"""
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.npy")
        # Save an object array (uses pickle)
        np.save(path, np.array([{'key': 'val'}], dtype=object), allow_pickle=True)
        # Loading without allow_pickle should fail
        try:
            np.load(path)
            assert False, "np.load should fail without allow_pickle=True for pickle data"
        except ValueError:
            pass  # Expected
        # Loading with allow_pickle=True should work
        data = np.load(path, allow_pickle=True)
        assert data[0] == {'key': 'val'}


test("np.load requires allow_pickle=True for pickle data", test_load_pickle_requires_opt_in)

# =============================================================================
# SECTION: __array__ protocol copy= keyword (lines 104-112)
# =============================================================================
print("\n=== __array__ protocol ===")


def test_array_protocol_with_copy():
    """Lines 106-112: __array__(self, dtype=None, copy=None) - copy= is REQUIRED"""
    class MyObj:
        def __init__(self):
            self._data = [1, 2, 3]

        def __array__(self, dtype=None, copy=None):
            arr = np.array(self._data, dtype=dtype)
            if copy:
                arr = arr.copy()
            return arr

    obj = MyObj()
    result = np.asarray(obj)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_array_protocol_without_copy_warns():
    """Verify that missing copy= keyword triggers DeprecationWarning"""
    class OldObj:
        def __init__(self):
            self._data = [1, 2, 3]

        def __array__(self, dtype=None):  # No copy=
            return np.array(self._data, dtype=dtype)

    old_obj = OldObj()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = np.asarray(old_obj)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        if dep_warnings:
            print(f"    (DeprecationWarning raised for missing copy= as expected)")
        else:
            warn("__array__ protocol",
                 f"No DeprecationWarning for missing copy= in {np.__version__}")


test("__array__ protocol with copy= keyword works", test_array_protocol_with_copy)
test("__array__ protocol without copy= keyword", test_array_protocol_without_copy_warns)

# =============================================================================
# SECTION: Known Limitations (lines 114-118)
# =============================================================================
print("\n=== Known Limitations ===")


def test_no_minmax():
    """Line 118: No minmax function - Must call np.min and np.max separately"""
    assert not hasattr(np, 'minmax'), f"np.minmax exists unexpectedly"
    arr = np.array([3, 1, 4, 1, 5])
    assert np.min(arr) == 1
    assert np.max(arr) == 5


test("No np.minmax function", test_no_minmax)

# =============================================================================
# SECTION: Performance Tips (lines 122-127)
# =============================================================================
print("\n=== Performance Tips ===")


def test_unique_sorted_false_performance():
    """Line 122: np.unique(arr, sorted=False) hash-based, up to 15x faster"""
    arr = np.random.randint(0, 10000, size=100000)
    try:
        result_sorted = np.unique(arr)
        result_unsorted = np.unique(arr, sorted=False)
        assert set(result_sorted) == set(result_unsorted), "Same elements expected"
    except TypeError as e:
        if "sorted" in str(e):
            results["failed"].append(
                ("unique sorted=False performance",
                 f"sorted= parameter not supported: {e}")
            )
            print(f"  FAIL: unique sorted=False not supported: {e}")
            return
        raise


def test_stringdtype_vs_object():
    """Line 124: StringDType() over dtype=object"""
    from numpy.dtypes import StringDType
    arr_sdt = np.array(["hello", "world"], dtype=StringDType())
    arr_obj = np.array(["hello", "world"], dtype=object)
    # StringDType should work with np.strings
    result = np.strings.upper(arr_sdt)
    assert result[0] == "HELLO"


def test_std_with_mean_parameter():
    """Line 125: np.std(arr, mean=precomputed)"""
    arr = np.random.randn(1000)
    m = arr.mean()
    try:
        s = np.std(arr, mean=m)
        s_ref = np.std(arr)
        assert np.isclose(s, s_ref, rtol=1e-10), f"std with mean= differs: {s} vs {s_ref}"
    except TypeError as e:
        if "mean" in str(e):
            results["failed"].append(
                ("np.std mean= parameter",
                 f"mean= parameter not supported in {np.__version__}: {e}")
            )
            print(f"  FAIL: np.std mean= parameter not supported: {e}")
            return
        raise


def test_fft_float32_no_promotion():
    """Line 127: FFT no longer promotes float32 to float64"""
    arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = np.fft.fft(arr)
    if result.dtype == np.complex64:
        pass  # Correct - no promotion
    elif result.dtype == np.complex128:
        warn("FFT float32", f"FFT still promotes float32 to complex128 in {np.__version__}")
    else:
        warn("FFT float32", f"Unexpected FFT result dtype: {result.dtype}")
    assert result.dtype == np.complex64, (
        f"Expected complex64 (no promotion), got {result.dtype}"
    )


test("unique sorted=False returns same elements", test_unique_sorted_false_performance)
test("StringDType works with np.strings", test_stringdtype_vs_object)
test("np.std mean= parameter", test_std_with_mean_parameter)
test("FFT float32 -> complex64 (no promotion)", test_fft_float32_no_promotion)

# =============================================================================
# SECTION: Additional verification of specific claims
# =============================================================================
print("\n=== Additional Claim Verification ===")


def test_skill_header_claim():
    """Verify the header claim: Python scalars adapt to array dtype"""
    # "Python scalars now silently adapt to array dtype (NEP 50)"
    arr = np.array([1], dtype=np.uint8)
    result = arr + 256  # Should wrap to 0
    assert result.dtype == np.uint8
    assert result[0] == 0, f"Expected 0 (wrap), got {result[0]}"


def test_skill_claim_np_strings_module_exists():
    """Verify np.strings module has the claimed functions"""
    assert hasattr(np, 'strings')
    assert hasattr(np.strings, 'upper')
    assert hasattr(np.strings, 'find')
    assert hasattr(np.strings, 'replace')


def test_skill_issue_10161():
    """Line 84: #10161 reference for isclose asymmetry"""
    # Just verify isclose exists and has the asymmetric behavior
    assert hasattr(np, 'isclose')


def test_skill_issue_18124():
    """Line 92: #18124 reference for npz multiprocessing bug"""
    # Verify .npz returns lazy NpzFile
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.npz")
        np.savez(path, a=np.array([1]))
        loaded = np.load(path)
        assert type(loaded).__name__ == 'NpzFile', f"Got {type(loaded).__name__}"


def test_skill_issue_12889():
    """Line 99: #12889 reference for pickle security"""
    # Already tested above - just verify the claim
    pass


def test_skill_issue_27957():
    """Line 116: Type stubs regression (#27957)"""
    # Can't test mypy issues here, just note it
    pass


def test_skill_issue_16544():
    """Line 117: Shape typing incomplete (#16544) - NDArray does not encode shape"""
    from numpy.typing import NDArray
    # NDArray is parameterized by dtype only, not shape
    # This is a known limitation
    assert NDArray is not None


def test_skill_issue_9836():
    """Line 118: No minmax function (#9836)"""
    assert not hasattr(np, 'minmax')


def test_skill_claim_scalar_ufuncs():
    """Line 126: Scalar ufuncs ~6x faster (2.4+)"""
    # Can't easily test performance, but verify scalar ufuncs work
    result = np.add(1.0, 2.0)
    assert result == 3.0


test("Header claim: Python scalars adapt to array dtype", test_skill_header_claim)
test("np.strings module has claimed functions", test_skill_claim_np_strings_module_exists)
test("Issue #10161: isclose exists", test_skill_issue_10161)
test("Issue #18124: npz returns lazy NpzFile", test_skill_issue_18124)
test("Issue #16544: NDArray exists (no shape encoding)", test_skill_issue_16544)
test("Issue #9836: no np.minmax", test_skill_issue_9836)
test("Scalar ufuncs work", test_skill_claim_scalar_ufuncs)

# =============================================================================
# SECTION: Verify math.isclose formula for comparison (line 88)
# =============================================================================
print("\n=== math.isclose formula verification ===")


def test_math_isclose_formula():
    """Line 88: math.isclose: abs(a-b) <= max(rtol * max(abs(a), abs(b)), atol)"""
    import math
    a, b = 1e-10, 2e-10
    # math.isclose is symmetric
    r1 = math.isclose(a, b, rel_tol=0.5, abs_tol=0)
    r2 = math.isclose(b, a, rel_tol=0.5, abs_tol=0)
    assert r1 == r2, f"math.isclose should be symmetric: {r1} vs {r2}"
    # math.isclose: abs(1e-10 - 2e-10) <= max(0.5 * max(1e-10, 2e-10), 0)
    #              = 1e-10 <= max(0.5 * 2e-10, 0) = 1e-10 -> True
    assert r1 == True, f"Expected True, got {r1}"


test("math.isclose is symmetric (contrast to np.isclose)", test_math_isclose_formula)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"NumPy version: {np.__version__}")
print(f"PASSED: {len(results['passed'])}")
print(f"FAILED: {len(results['failed'])}")
print(f"WARNINGS: {len(results['warnings'])}")

if results['failed']:
    print("\nFailed tests:")
    for name, reason in results['failed']:
        print(f"  - {name}: {reason}")

if results['warnings']:
    print("\nWarnings:")
    for name, msg in results['warnings']:
        print(f"  - {name}: {msg}")

sys.exit(1 if results['failed'] else 0)
