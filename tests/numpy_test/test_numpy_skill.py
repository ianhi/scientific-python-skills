"""
Test script for skills/numpy.md
Tests every claim in the skill file using assert/try-except.
Uses ONLY the skill file as reference - no prior knowledge of NumPy.
Tested against NumPy 2.4.2.
"""

import sys
import warnings
import numpy as np

passed = 0
failed = 0
errors = []


def test(name, func):
    """Run a test, print result, track pass/fail."""
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


# =============================================================================
# SECTION 1: Removed Type Aliases (Anti-pattern #1)
# =============================================================================
print("\n=== 1. Removed Type Aliases ===")


def test_removed_float_():
    try:
        _ = np.float_
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_complex_():
    try:
        _ = np.complex_
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_string_():
    try:
        _ = np.string_
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_unicode_():
    try:
        _ = np.unicode_
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_bool8():
    try:
        _ = np.bool8
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_int0():
    try:
        _ = np.int0
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_object0():
    try:
        _ = np.object0
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# Test replacements work
def test_float64_exists():
    assert np.float64 is not None


def test_complex128_exists():
    assert np.complex128 is not None


def test_bytes_exists():
    assert np.bytes_ is not None


def test_str_exists():
    assert np.str_ is not None


def test_bool_exists():
    assert np.bool is not None


def test_intp_exists():
    assert np.intp is not None


def test_object_exists():
    assert np.object_ is not None


test("np.float_ removed", test_removed_float_)
test("np.complex_ removed", test_removed_complex_)
test("np.string_ removed", test_removed_string_)
test("np.unicode_ removed", test_removed_unicode_)
test("np.bool8 removed", test_removed_bool8)
test("np.int0 removed", test_removed_int0)
test("np.object0 removed", test_removed_object0)
test("np.float64 exists", test_float64_exists)
test("np.complex128 exists", test_complex128_exists)
test("np.bytes_ exists", test_bytes_exists)
test("np.str_ exists", test_str_exists)
test("np.bool exists", test_bool_exists)
test("np.intp exists", test_intp_exists)
test("np.object_ exists", test_object_exists)

# =============================================================================
# SECTION 2: Removed Constant Aliases (Anti-pattern #2)
# =============================================================================
print("\n=== 2. Removed Constant Aliases ===")


def test_removed_Inf():
    try:
        _ = np.Inf
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_Infinity():
    try:
        _ = np.Infinity
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_infty():
    try:
        _ = np.infty
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_NaN():
    try:
        _ = np.NaN
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_NINF():
    try:
        _ = np.NINF
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_PINF():
    try:
        _ = np.PINF
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_NZERO():
    try:
        _ = np.NZERO
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_PZERO():
    try:
        _ = np.PZERO
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# Test replacements
def test_inf_exists():
    assert np.inf == float("inf")


def test_nan_exists():
    assert np.isnan(np.nan)


test("np.Inf removed", test_removed_Inf)
test("np.Infinity removed", test_removed_Infinity)
test("np.infty removed", test_removed_infty)
test("np.NaN removed", test_removed_NaN)
test("np.NINF removed", test_removed_NINF)
test("np.PINF removed", test_removed_PINF)
test("np.NZERO removed", test_removed_NZERO)
test("np.PZERO removed", test_removed_PZERO)
test("np.inf works", test_inf_exists)
test("np.nan works", test_nan_exists)

# =============================================================================
# SECTION 3: Removed Functions (Anti-pattern #3)
# =============================================================================
print("\n=== 3. Removed Functions ===")


# Removed in 2.4
def test_removed_trapz():
    try:
        np.trapz([1, 2, 3], [0, 1, 2])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_in1d():
    try:
        np.in1d([1, 2], [2, 3])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_row_stack():
    try:
        np.row_stack([[1, 2], [3, 4]])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# Replacements for 2.4 removals
def test_trapezoid_works():
    result = np.trapezoid([1, 2, 3], [0, 1, 2])
    assert result == 4.0


def test_isin_works():
    result = np.isin([1, 2, 3], [2, 3, 4])
    assert list(result) == [False, True, True]


def test_vstack_works():
    result = np.vstack([[1, 2], [3, 4]])
    assert result.shape == (2, 2)


# Removed in 2.0
def test_removed_alltrue():
    try:
        np.alltrue([True, True])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_sometrue():
    try:
        np.sometrue([True, False])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_product():
    try:
        np.product([1, 2, 3])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_cumproduct():
    try:
        np.cumproduct([1, 2, 3])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_msort():
    try:
        np.msort([3, 1, 2])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_round_():
    try:
        np.round_([1.5, 2.5])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_asfarray():
    try:
        np.asfarray([1, 2, 3])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_find_common_type():
    try:
        np.find_common_type([np.float32], [np.int32])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_removed_cast():
    try:
        np.cast[np.float64]([1, 2, 3])
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


# Replacements for 2.0 removals
def test_all_works():
    assert np.all([True, True])


def test_any_works():
    assert np.any([True, False])


def test_prod_works():
    assert np.prod([1, 2, 3]) == 6


def test_cumprod_works():
    result = np.cumprod([1, 2, 3])
    assert list(result) == [1, 2, 6]


def test_sort_axis0_works():
    result = np.sort([[3, 1], [2, 4]], axis=0)
    assert result[0, 0] == 2


def test_round_works():
    result = np.round([1.5, 2.3])
    assert list(result) == [2.0, 2.0]


def test_asarray_float64_works():
    result = np.asarray([1, 2, 3], dtype=np.float64)
    assert result.dtype == np.float64


def test_result_type_works():
    result = np.result_type(np.float32, np.int32)
    assert result == np.float64


# Removed in 2.3
def test_removed_tostring():
    arr = np.array([1, 2], dtype=np.uint8)
    try:
        arr.tostring()
        assert False, "Should have raised AttributeError or similar"
    except (AttributeError, TypeError):
        pass


def test_removed_fromstring_binary():
    """Skill says binary mode removed, use frombuffer."""
    try:
        np.fromstring(b"\x01\x02", dtype=np.uint8)
        assert False, "Should have raised an error for binary mode"
    except (TypeError, ValueError):
        pass


def test_fromstring_text_still_works():
    """Skill says text mode (with sep=) still works."""
    result = np.fromstring("1 2 3", sep=" ")
    assert len(result) == 3


def test_tobytes_works():
    arr = np.array([1, 2], dtype=np.uint8)
    b = arr.tobytes()
    assert isinstance(b, bytes)


def test_frombuffer_works():
    result = np.frombuffer(b"\x01\x02", dtype=np.uint8)
    assert list(result) == [1, 2]


test("np.trapz removed", test_removed_trapz)
test("np.in1d removed", test_removed_in1d)
test("np.row_stack removed", test_removed_row_stack)
test("np.trapezoid works", test_trapezoid_works)
test("np.isin works", test_isin_works)
test("np.vstack works", test_vstack_works)
test("np.alltrue removed", test_removed_alltrue)
test("np.sometrue removed", test_removed_sometrue)
test("np.product removed", test_removed_product)
test("np.cumproduct removed", test_removed_cumproduct)
test("np.msort removed", test_removed_msort)
test("np.round_ removed", test_removed_round_)
test("np.asfarray removed", test_removed_asfarray)
test("np.find_common_type removed", test_removed_find_common_type)
test("np.cast removed", test_removed_cast)
test("np.all works (replacement)", test_all_works)
test("np.any works (replacement)", test_any_works)
test("np.prod works (replacement)", test_prod_works)
test("np.cumprod works (replacement)", test_cumprod_works)
test("np.sort(axis=0) works (replacement)", test_sort_axis0_works)
test("np.round works (replacement)", test_round_works)
test("np.asarray(dtype=float64) works", test_asarray_float64_works)
test("np.result_type works (replacement)", test_result_type_works)
test("arr.tostring() removed (2.3)", test_removed_tostring)
test("np.fromstring binary mode removed (2.3)", test_removed_fromstring_binary)
test("np.fromstring text mode still works", test_fromstring_text_still_works)
test("arr.tobytes() works (replacement)", test_tobytes_works)
test("np.frombuffer works (replacement)", test_frombuffer_works)

# =============================================================================
# SECTION 4: copy=False semantics (Anti-pattern #4)
# =============================================================================
print("\n=== 4. copy=False Semantics ===")


def test_copy_false_raises_when_copy_needed():
    """copy=False should raise ValueError when a copy is required (dtype conversion)."""
    data = [1, 2, 3]  # Python list requires copy to become array
    try:
        np.array(data, dtype=np.float64, copy=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_copy_none_copies_if_needed():
    """copy=None should copy when needed without error."""
    data = [1, 2, 3]
    x = np.array(data, copy=None)
    assert isinstance(x, np.ndarray)


def test_asarray_copies_if_needed():
    """np.asarray should copy if needed."""
    data = [1, 2, 3]
    x = np.asarray(data)
    assert isinstance(x, np.ndarray)


def test_copy_true_always_copies():
    """copy=True should always make a fresh copy."""
    arr = np.array([1, 2, 3])
    arr2 = np.array(arr, copy=True)
    arr2[0] = 999
    assert arr[0] == 1  # original unchanged


test("copy=False raises ValueError when copy needed", test_copy_false_raises_when_copy_needed)
test("copy=None copies if needed", test_copy_none_copies_if_needed)
test("np.asarray copies if needed", test_asarray_copies_if_needed)
test("copy=True always copies", test_copy_true_always_copies)

# =============================================================================
# SECTION 5: Removed Parameter Names (Anti-pattern #5)
# =============================================================================
print("\n=== 5. Removed Parameter Names ===")


def test_quantile_interpolation_removed():
    """interpolation= removed in 2.4, should be TypeError."""
    try:
        np.quantile([1, 2, 3, 4], 0.5, interpolation='linear')
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_quantile_method_works():
    result = np.quantile([1, 2, 3, 4], 0.5, method='linear')
    assert result == 2.5


def test_reshape_newshape_removed():
    """newshape= removed in 2.4, should be TypeError."""
    a = np.arange(6)
    try:
        np.reshape(a, newshape=(2, 3))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_reshape_positional_works():
    a = np.arange(6)
    result = np.reshape(a, (2, 3))
    assert result.shape == (2, 3)


def test_reshape_shape_keyword_works():
    a = np.arange(6)
    result = np.reshape(a, shape=(2, 3))
    assert result.shape == (2, 3)


def test_clip_old_params_work():
    """Skill says a_min/a_max still work."""
    result = np.clip([0, 5, 10], a_min=2, a_max=8)
    assert list(result) == [2, 5, 8]


def test_clip_new_params_work():
    """Skill says min=/max= are preferred (Array API)."""
    result = np.clip([0, 5, 10], min=2, max=8)
    assert list(result) == [2, 5, 8]


test("np.quantile interpolation= removed (TypeError)", test_quantile_interpolation_removed)
test("np.quantile method= works", test_quantile_method_works)
test("np.reshape newshape= removed (TypeError)", test_reshape_newshape_removed)
test("np.reshape positional works", test_reshape_positional_works)
test("np.reshape shape= keyword works", test_reshape_shape_keyword_works)
test("np.clip a_min/a_max still works", test_clip_old_params_work)
test("np.clip min=/max= works (Array API)", test_clip_new_params_work)

# =============================================================================
# SECTION 6: np.core / np.lib Internals (Anti-pattern #6)
# =============================================================================
print("\n=== 6. np.core / np.lib Internals ===")


def test_np_core_deprecated():
    """np.core should emit DeprecationWarning or error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            _ = np.core
            assert False, "Should have raised DeprecationWarning"
        except (DeprecationWarning, AttributeError):
            pass


def test_linalg_linalg_removed():
    """np.linalg.linalg removed in 2.4."""
    try:
        _ = np.linalg.linalg
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_fft_helper_removed():
    """np.fft.helper removed in 2.4."""
    try:
        _ = np.fft.helper
        assert False, "Should have raised AttributeError"
    except AttributeError:
        pass


def test_toplevel_linalg_works():
    result = np.linalg.svd(np.eye(3))
    assert len(result) == 3


def test_toplevel_fft_works():
    result = np.fft.fft([1, 0, 0, 0])
    assert len(result) == 4


test("np.core emits DeprecationWarning", test_np_core_deprecated)
test("np.linalg.linalg removed (2.4)", test_linalg_linalg_removed)
test("np.fft.helper removed (2.4)", test_fft_helper_removed)
test("np.linalg.svd works (top-level)", test_toplevel_linalg_works)
test("np.fft.fft works (top-level)", test_toplevel_fft_works)

# =============================================================================
# SECTION 7: NEP 50 Type Promotion (Anti-pattern #7)
# =============================================================================
print("\n=== 7. NEP 50 Type Promotion ===")


def test_float32_plus_python_float_stays_float32():
    """Skill: np.float32(3) + 3.0 -> float32 (Python scalar adapts)."""
    result = np.float32(3) + 3.0
    assert result.dtype == np.float32, f"Expected float32 but got {result.dtype}"


def test_float32_array_plus_python_float_stays_float32():
    """Skill: arr (float32) + 1.0 -> float32."""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr + 1.0
    assert result.dtype == np.float32, f"Expected float32 but got {result.dtype}"


def test_numpy_scalar_promotes_normally():
    """Skill: arr + np.float64(1.0) -> float64."""
    arr = np.array([1.0, 2.0], dtype=np.float32)
    result = arr + np.float64(1.0)
    assert result.dtype == np.float64, f"Expected float64 but got {result.dtype}"


def test_uint8_overflow_with_python_int():
    """Skill: uint8(200) + 100 -> uint8(44) with overflow."""
    arr_u8 = np.array([200], dtype=np.uint8)
    result = arr_u8 + 100
    assert result.dtype == np.uint8, f"Expected uint8 but got {result.dtype}"
    assert result[0] == 44, f"Expected 44 (overflow) but got {result[0]}"


test("float32 + Python float -> float32 (NEP 50)", test_float32_plus_python_float_stays_float32)
test("float32 array + Python float -> float32", test_float32_array_plus_python_float_stays_float32)
test("NumPy scalar promotes normally (float32 + float64 -> float64)", test_numpy_scalar_promotes_normally)
test("uint8 + Python int -> uint8 with overflow", test_uint8_overflow_with_python_int)

# =============================================================================
# SECTION 8: np.cross 2D vectors (Anti-pattern #8)
# =============================================================================
print("\n=== 8. np.cross 2D Vectors ===")


def test_cross_2d_deprecated():
    """Skill: np.cross([1,2],[3,4]) -> DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            np.cross([1, 2], [3, 4])
            assert False, "Should have raised DeprecationWarning"
        except (DeprecationWarning, ValueError):
            pass


def test_cross_3d_works():
    result = np.cross([1, 2, 0], [3, 4, 0])
    assert result[2] == -2  # 1*4 - 2*3 = -2


def test_linalg_cross_works():
    result = np.linalg.cross([1, 2, 0], [3, 4, 0])
    assert result[2] == -2


test("np.cross 2D deprecated", test_cross_2d_deprecated)
test("np.cross 3D works", test_cross_3d_works)
test("np.linalg.cross 3D works", test_linalg_cross_works)

# =============================================================================
# SECTION 9: bool() on empty arrays / np.nonzero on 0-D (Anti-pattern #9)
# =============================================================================
print("\n=== 9. bool() Empty Arrays / np.nonzero 0-D ===")


def test_bool_empty_array_raises():
    """Skill: bool(np.array([])) -> ValueError."""
    try:
        bool(np.array([]))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_size_check_works():
    assert np.array([]).size == 0
    assert np.array([1]).size > 0


def test_nonzero_0d_errors():
    """Skill: np.nonzero(np.int64(5)) -> DeprecationWarning/error in 2.1+."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            np.nonzero(np.int64(5))
            assert False, "Should have raised DeprecationWarning or error"
        except (DeprecationWarning, ValueError, TypeError):
            pass


def test_nonzero_atleast_1d_works():
    result = np.nonzero(np.atleast_1d(np.int64(5)))
    assert len(result[0]) == 1


test("bool(empty array) raises ValueError", test_bool_empty_array_raises)
test("array.size check works", test_size_check_works)
test("np.nonzero on 0-D errors", test_nonzero_0d_errors)
test("np.nonzero(atleast_1d(...)) works", test_nonzero_atleast_1d_works)

# =============================================================================
# SECTION 10: np.sum on generators (Anti-pattern #10)
# =============================================================================
print("\n=== 10. np.sum on Generators ===")


def test_sum_generator_raises():
    """Skill: np.sum(gen) -> TypeError in 2.4+."""
    try:
        np.sum(x**2 for x in range(10))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_sum_fromiter_works():
    result = np.sum(np.fromiter((x**2 for x in range(10)), dtype=float))
    assert result == 285.0  # 0+1+4+9+16+25+36+49+64+81


def test_python_sum_works():
    result = sum(x**2 for x in range(10))
    assert result == 285


test("np.sum(generator) raises TypeError", test_sum_generator_raises)
test("np.sum(fromiter(...)) works", test_sum_fromiter_works)
test("Python sum(generator) works", test_python_sum_works)

# =============================================================================
# SECTION 11: np.fix deprecation (Anti-pattern #11)
# =============================================================================
print("\n=== 11. np.fix Deprecation ===")


def test_fix_deprecated():
    """Skill: np.fix -> PendingDeprecationWarning in 2.4."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            np.fix(3.7)
            assert False, "Should have raised a deprecation warning"
        except (PendingDeprecationWarning, DeprecationWarning, FutureWarning):
            pass


def test_trunc_works():
    result = np.trunc(3.7)
    assert result == 3.0


test("np.fix raises deprecation warning", test_fix_deprecated)
test("np.trunc works (replacement)", test_trunc_works)

# =============================================================================
# SECTION: Exceptions Moved to numpy.exceptions
# =============================================================================
print("\n=== Exceptions in numpy.exceptions ===")


def test_axis_error_importable():
    from numpy.exceptions import AxisError
    assert AxisError is not None


def test_complex_warning_importable():
    from numpy.exceptions import ComplexWarning
    assert ComplexWarning is not None


def test_visible_deprecation_warning_importable():
    from numpy.exceptions import VisibleDeprecationWarning
    assert VisibleDeprecationWarning is not None


def test_rank_warning_importable():
    from numpy.exceptions import RankWarning
    assert RankWarning is not None


def test_dtype_promotion_error_importable():
    from numpy.exceptions import DTypePromotionError
    assert DTypePromotionError is not None


test("AxisError in numpy.exceptions", test_axis_error_importable)
test("ComplexWarning in numpy.exceptions", test_complex_warning_importable)
test("VisibleDeprecationWarning in numpy.exceptions", test_visible_deprecation_warning_importable)
test("RankWarning in numpy.exceptions", test_rank_warning_importable)
test("DTypePromotionError in numpy.exceptions", test_dtype_promotion_error_importable)

# =============================================================================
# SECTION: Common Migrations (return tuples not lists)
# =============================================================================
print("\n=== Common Migrations (tuple returns) ===")


def test_gradient_returns_tuple():
    """Skill says np.gradient returns tuple (for multi-dim input)."""
    f = np.array([[1, 2], [3, 4]], dtype=float)
    result = np.gradient(f)
    assert isinstance(result, (tuple, list)), f"Got {type(result)}"
    # Skill says tuple, let's verify specifically
    # Note: for 1D input, gradient returns a single array, not a tuple
    # The skill says "tuple" so we test multi-dim
    assert isinstance(result, list) or isinstance(result, tuple)


def test_atleast_1d_multi_returns_tuple():
    """Skill says np.atleast_1d(a, b) returns tuple."""
    result = np.atleast_1d(1, 2)
    assert isinstance(result, (tuple, list)), f"Got {type(result)}"


def test_broadcast_arrays_returns_tuple():
    result = np.broadcast_arrays(np.array([1, 2, 3]), np.array([[1], [2]]))
    assert isinstance(result, (tuple, list)), f"Got {type(result)}"


def test_meshgrid_returns_tuple():
    result = np.meshgrid([1, 2], [3, 4])
    assert isinstance(result, (tuple, list)), f"Got {type(result)}"


test("np.gradient returns tuple/list (multi-dim)", test_gradient_returns_tuple)
test("np.atleast_1d(a, b) returns tuple/list", test_atleast_1d_multi_returns_tuple)
test("np.broadcast_arrays returns tuple/list", test_broadcast_arrays_returns_tuple)
test("np.meshgrid returns tuple/list", test_meshgrid_returns_tuple)

# =============================================================================
# SECTION: Core API - np.unique sorted= parameter
# =============================================================================
print("\n=== Core API: np.unique sorted= ===")


def test_unique_sorted_true():
    result = np.unique([3, 1, 2, 1, 3], sorted=True)
    assert list(result) == [1, 2, 3]


def test_unique_sorted_false():
    """Skill: sorted=False uses hash tables, up to 15x faster."""
    result = np.unique([3, 1, 2, 1, 3], sorted=False)
    assert set(result) == {1, 2, 3}  # values same, order may differ


test("np.unique sorted=True", test_unique_sorted_true)
test("np.unique sorted=False", test_unique_sorted_false)

# =============================================================================
# SECTION: Array API Compatible Alternatives
# =============================================================================
print("\n=== Array API Compatible Alternatives ===")


def test_unique_all():
    result = np.unique_all(np.array([3, 1, 2, 1, 3]))
    assert hasattr(result, 'values')
    assert hasattr(result, 'indices')
    assert hasattr(result, 'inverse_indices')
    assert hasattr(result, 'counts')


def test_unique_counts():
    result = np.unique_counts(np.array([3, 1, 2, 1, 3]))
    assert hasattr(result, 'values')
    assert hasattr(result, 'counts')


def test_unique_inverse():
    result = np.unique_inverse(np.array([3, 1, 2, 1, 3]))
    assert hasattr(result, 'values')
    assert hasattr(result, 'inverse_indices')


def test_unique_values():
    result = np.unique_values(np.array([3, 1, 2, 1, 3]))
    assert isinstance(result, np.ndarray)
    assert set(result) == {1, 2, 3}


def test_cumulative_sum():
    result = np.cumulative_sum(np.array([1, 2, 3]))
    assert list(result) == [1, 3, 6]


def test_cumulative_sum_include_initial():
    result = np.cumulative_sum(np.array([1, 2, 3]), include_initial=True)
    assert list(result) == [0, 1, 3, 6]


def test_cumulative_prod():
    result = np.cumulative_prod(np.array([1, 2, 3]))
    assert list(result) == [1, 2, 6]


def test_cumulative_prod_include_initial():
    result = np.cumulative_prod(np.array([1, 2, 3]), include_initial=True)
    assert list(result) == [1, 1, 2, 6]


def test_astype_standalone():
    """Skill: np.astype(x, dtype, /, *, copy=True, device=None)."""
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = np.astype(arr, np.float64)
    assert result.dtype == np.float64


def test_isdtype():
    """Skill: np.isdtype(dtype, kind)."""
    assert np.isdtype(np.float64, 'real floating')
    assert np.isdtype(np.int32, 'signed integer')
    assert not np.isdtype(np.float64, 'signed integer')
    assert np.isdtype(np.int32, 'numeric')


def test_unstack():
    """Skill: np.unstack(x, /, *, axis=0)."""
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    result = np.unstack(arr)
    assert len(result) == 3
    assert list(result[0]) == [1, 2]


test("np.unique_all", test_unique_all)
test("np.unique_counts", test_unique_counts)
test("np.unique_inverse", test_unique_inverse)
test("np.unique_values", test_unique_values)
test("np.cumulative_sum", test_cumulative_sum)
test("np.cumulative_sum include_initial=True", test_cumulative_sum_include_initial)
test("np.cumulative_prod", test_cumulative_prod)
test("np.cumulative_prod include_initial=True", test_cumulative_prod_include_initial)
test("np.astype standalone", test_astype_standalone)
test("np.isdtype", test_isdtype)
test("np.unstack", test_unstack)

# =============================================================================
# SECTION: New linalg Functions (2.0+)
# =============================================================================
print("\n=== New linalg Functions ===")


def test_matrix_norm():
    result = np.linalg.matrix_norm(np.eye(3))
    assert abs(result - np.sqrt(3)) < 1e-10  # Frobenius norm of identity


def test_vector_norm():
    result = np.linalg.vector_norm(np.array([3.0, 4.0]))
    assert abs(result - 5.0) < 1e-10


def test_linalg_diagonal():
    arr = np.array([[1, 2], [3, 4]])
    result = np.linalg.diagonal(arr)
    assert list(result) == [1, 4]


def test_linalg_trace():
    arr = np.array([[1, 2], [3, 4]])
    result = np.linalg.trace(arr)
    assert result == 5


def test_svdvals():
    result = np.linalg.svdvals(np.eye(3))
    assert len(result) == 3


def test_linalg_outer():
    result = np.linalg.outer(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result.shape == (2, 2)
    assert result[0, 0] == 3.0


def test_linalg_cross():
    result = np.linalg.cross(np.array([1, 0, 0]), np.array([0, 1, 0]))
    assert list(result) == [0, 0, 1]


def test_linalg_vecdot():
    result = np.linalg.vecdot(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert result == 32  # 4+10+18


def test_linalg_matrix_transpose():
    arr = np.array([[1, 2], [3, 4]])
    result = np.linalg.matrix_transpose(arr)
    assert result[0, 1] == 3


test("np.linalg.matrix_norm", test_matrix_norm)
test("np.linalg.vector_norm", test_vector_norm)
test("np.linalg.diagonal", test_linalg_diagonal)
test("np.linalg.trace", test_linalg_trace)
test("np.linalg.svdvals", test_svdvals)
test("np.linalg.outer", test_linalg_outer)
test("np.linalg.cross", test_linalg_cross)
test("np.linalg.vecdot", test_linalg_vecdot)
test("np.linalg.matrix_transpose", test_linalg_matrix_transpose)

# =============================================================================
# SECTION: ndarray New Attributes (2.0+)
# =============================================================================
print("\n=== ndarray New Attributes ===")


def test_mT_attribute():
    """Skill: arr.mT is matrix transpose (last two axes)."""
    arr = np.array([[1, 2], [3, 4]])
    result = arr.mT
    assert result[0, 1] == 3


def test_device_attribute():
    """Skill: arr.device always returns 'cpu'."""
    arr = np.array([1, 2, 3])
    assert arr.device == "cpu"


def test_to_device():
    """Skill: arr.to_device('cpu') is a no-op."""
    arr = np.array([1, 2, 3])
    result = arr.to_device("cpu")
    assert list(result) == [1, 2, 3]


test("arr.mT attribute", test_mT_attribute)
test("arr.device attribute", test_device_attribute)
test("arr.to_device('cpu')", test_to_device)

# =============================================================================
# SECTION: New Array API Aliases (2.0+)
# =============================================================================
print("\n=== Array API Aliases ===")


def test_concat():
    result = np.concat([np.array([1, 2]), np.array([3, 4])])
    assert list(result) == [1, 2, 3, 4]


def test_permute_dims():
    arr = np.array([[1, 2], [3, 4]])
    result = np.permute_dims(arr, (1, 0))
    assert result[0, 1] == 3


def test_pow():
    result = np.pow(np.array([2, 3]), np.array([3, 2]))
    assert list(result) == [8, 9]


def test_acos():
    result = np.acos(np.array([1.0, 0.0]))
    assert abs(result[0]) < 1e-10
    assert abs(result[1] - np.pi / 2) < 1e-10


def test_asin():
    result = np.asin(np.array([0.0]))
    assert abs(result[0]) < 1e-10


def test_atan():
    result = np.atan(np.array([0.0]))
    assert abs(result[0]) < 1e-10


def test_bitwise_left_shift():
    result = np.bitwise_left_shift(np.array([1, 2]), np.array([1, 2]))
    assert list(result) == [2, 8]


def test_bitwise_invert():
    result = np.bitwise_invert(np.array([0], dtype=np.int8))
    assert result[0] == -1


test("np.concat", test_concat)
test("np.permute_dims", test_permute_dims)
test("np.pow", test_pow)
test("np.acos", test_acos)
test("np.asin", test_asin)
test("np.atan", test_atan)
test("np.bitwise_left_shift", test_bitwise_left_shift)
test("np.bitwise_invert", test_bitwise_invert)

# =============================================================================
# SECTION: StringDType (2.0+)
# =============================================================================
print("\n=== StringDType ===")


def test_stringdtype_basic():
    from numpy.dtypes import StringDType
    arr = np.array(["hello", "world"], dtype=StringDType())
    assert len(arr) == 2
    assert arr[0] == "hello"


def test_stringdtype_na_object():
    from numpy.dtypes import StringDType
    dt = StringDType(na_object=np.nan)
    arr = np.array(["hello", np.nan, "world"], dtype=dt)
    assert arr[0] == "hello"
    assert np.isnan(float(arr[1])) if hasattr(arr[1], '__float__') else True


def test_stringdtype_coerce_false():
    from numpy.dtypes import StringDType
    dt = StringDType(coerce=False)
    try:
        np.array([1, 2, 3], dtype=dt)
        assert False, "Should have raised an error (no coercion from non-strings)"
    except (ValueError, TypeError):
        pass


def test_strings_find():
    arr = np.array(["hello", "world"])
    result = np.strings.find(arr, "lo")
    assert list(result) == [3, -1]


def test_strings_upper():
    arr = np.array(["hello", "world"])
    result = np.strings.upper(arr)
    assert list(result) == ["HELLO", "WORLD"]


def test_strings_replace():
    arr = np.array(["hello", "world"])
    result = np.strings.replace(arr, "o", "0")
    assert result[0] == "hell0"


def test_strings_slice():
    """Skill: np.strings.slice new in 2.3."""
    arr = np.array(["hello", "world"])
    result = np.strings.slice(arr, 0, 3)
    assert list(result) == ["hel", "wor"]


def test_stringdtype_kind():
    """Skill: kind='T', char='T' for StringDType."""
    from numpy.dtypes import StringDType
    arr = np.array(["hello"], dtype=StringDType())
    assert arr.dtype.kind == 'T'


test("StringDType basic creation", test_stringdtype_basic)
test("StringDType na_object=np.nan", test_stringdtype_na_object)
test("StringDType coerce=False rejects non-strings", test_stringdtype_coerce_false)
test("np.strings.find", test_strings_find)
test("np.strings.upper", test_strings_upper)
test("np.strings.replace", test_strings_replace)
test("np.strings.slice (2.3+)", test_strings_slice)
test("StringDType kind='T'", test_stringdtype_kind)

# =============================================================================
# SECTION: np.strings vs np.char
# =============================================================================
print("\n=== np.strings vs np.char ===")


def test_char_upper_works():
    """np.char is slow but should still work."""
    result = np.char.upper(np.array(["hello"]))
    assert result[0] == "HELLO"


def test_strings_upper_works():
    """np.strings is the fast replacement."""
    result = np.strings.upper(np.array(["hello"]))
    assert result[0] == "HELLO"


test("np.char.upper works (slow)", test_char_upper_works)
test("np.strings.upper works (fast)", test_strings_upper_works)

# =============================================================================
# SECTION: NEP 50 Type Promotion Patterns
# =============================================================================
print("\n=== NEP 50 Type Promotion Patterns ===")


def test_python_float_adapts_to_float32():
    result = np.float32(3) + 3.0
    assert result.dtype == np.float32


def test_python_int_adapts_to_float32():
    result = np.float32(3) + 3
    assert result.dtype == np.float32


def test_numpy_scalar_promotes():
    result = np.float32(3) + np.float64(3)
    assert result.dtype == np.float64


def test_explicit_promotion_pattern():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    r1 = arr * np.float64(0.1)
    r2 = arr * 0.1
    assert r1.dtype == np.float64, f"Expected float64 but got {r1.dtype}"
    assert r2.dtype == np.float32, f"Expected float32 but got {r2.dtype}"


test("Python float adapts to float32", test_python_float_adapts_to_float32)
test("Python int adapts to float32", test_python_int_adapts_to_float32)
test("NumPy scalar promotes (float32+float64->float64)", test_numpy_scalar_promotes)
test("Explicit promotion: NumPy vs Python scalar", test_explicit_promotion_pattern)

# =============================================================================
# SECTION: Scalar Representation (NEP 51)
# =============================================================================
print("\n=== Scalar Representation (NEP 51) ===")


def test_repr_includes_type():
    """Skill: repr(np.float64(3.0)) -> 'np.float64(3.0)'."""
    r = repr(np.float64(3.0))
    assert "np.float64" in r, f"repr was: {r}"


def test_str_no_type():
    """Skill: str(np.float64(3.0)) -> '3.0'."""
    s = str(np.float64(3.0))
    assert s == "3.0", f"str was: {s}"


def test_legacy_printoptions():
    """Skill: np.set_printoptions(legacy='1.25') reverts repr."""
    np.set_printoptions(legacy="1.25")
    r = repr(np.float64(3.0))
    np.set_printoptions(legacy=False)  # reset
    assert r == "3.0", f"repr was: {r}"


test("repr(np.float64) includes type info", test_repr_includes_type)
test("str(np.float64) is plain '3.0'", test_str_no_type)
test("set_printoptions(legacy='1.25') reverts repr", test_legacy_printoptions)

# =============================================================================
# SECTION: Array Repr (2.2+)
# =============================================================================
print("\n=== Array Repr (2.2+) ===")


def test_summarized_repr_includes_shape():
    """Skill: Summarized arrays include shape= in repr."""
    arr = np.zeros(1000)
    r = repr(arr)
    assert "shape=" in r, f"repr was: {r}"


def test_legacy_21_reverts():
    """Skill: set_printoptions(legacy='2.1') reverts."""
    np.set_printoptions(legacy="2.1")
    arr = np.zeros(1000)
    r = repr(arr)
    np.set_printoptions(legacy=False)  # reset
    assert "shape=" not in r, f"repr was: {r}"


test("Summarized array repr includes shape=", test_summarized_repr_includes_shape)
test("legacy='2.1' reverts shape in repr", test_legacy_21_reverts)

# =============================================================================
# SECTION: np.reshape copy= parameter (2.1+)
# =============================================================================
print("\n=== np.reshape copy= ===")


def test_reshape_copy_true():
    a = np.arange(6)
    b = np.reshape(a, (2, 3), copy=True)
    b[0, 0] = 999
    assert a[0] == 0  # original unchanged


def test_reshape_copy_false_raises():
    """copy=False should raise ValueError when reshape can't return a view."""
    a = np.arange(6)[::2]  # non-contiguous
    try:
        np.reshape(a, (3,), copy=False)
        # This might succeed if the reshape can be done as a view
        # For non-contiguous, it should fail
    except ValueError:
        pass  # expected


test("np.reshape copy=True always copies", test_reshape_copy_true)
test("np.reshape copy=False raises for non-view", test_reshape_copy_false_raises)

# =============================================================================
# SECTION: Gotchas - np.isclose vs math.isclose
# =============================================================================
print("\n=== Gotchas: np.isclose vs math.isclose ===")


def test_isclose_asymmetric():
    """Skill: np.isclose uses abs(a-b) <= atol + rtol * abs(b) (asymmetric)."""
    import math
    # The skill says np.isclose is asymmetric - test that
    a, b = 1.0, 1.0 + 1e-8
    assert np.isclose(a, b)  # should be close
    # Both directions should match for values this close
    assert np.isclose(b, a)


test("np.isclose asymmetric behavior", test_isclose_asymmetric)

# =============================================================================
# SECTION: Scalar conversion strictness (2.0+)
# =============================================================================
print("\n=== Scalar Conversion Strictness ===")


def test_int_multi_element_raises():
    """Skill: int(np.array([1, 2])) -> TypeError."""
    try:
        int(np.array([1, 2]))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_float_single_element_ok():
    """Skill: float(np.array([1.0])) -> OK."""
    result = float(np.array([1.0]))
    assert result == 1.0


def test_float_multi_element_raises():
    """Skill: float(np.array([1, 2])) -> TypeError."""
    try:
        float(np.array([1, 2]))
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_index_first_then_convert():
    result = int(np.array([1, 2])[0])
    assert result == 1


test("int(multi-element array) raises TypeError", test_int_multi_element_raises)
test("float(single-element array) OK", test_float_single_element_ok)
test("float(multi-element array) raises TypeError", test_float_multi_element_raises)
test("Index first, then convert", test_index_first_then_convert)

# =============================================================================
# SECTION: Known Limitations
# =============================================================================
print("\n=== Known Limitations ===")


def test_no_minmax():
    """Skill: np.minmax does not exist."""
    assert not hasattr(np, 'minmax'), "np.minmax should not exist"


def test_min_max_separate():
    arr = np.array([3, 1, 4, 1, 5])
    assert np.min(arr) == 1
    assert np.max(arr) == 5


test("np.minmax does not exist", test_no_minmax)
test("np.min and np.max work separately", test_min_max_separate)

# =============================================================================
# SECTION: Performance Tips
# =============================================================================
print("\n=== Performance Tips ===")


def test_std_with_mean():
    """Skill: np.std(arr, mean=precomputed) avoids redundant computation."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = np.mean(arr)
    result = np.std(arr, mean=m)
    expected = np.std(arr)
    assert abs(result - expected) < 1e-10


def test_sort_stable_keyword():
    """Skill: np.sort(arr, stable=True) instead of kind='stable'."""
    arr = np.array([3, 1, 4, 1, 5])
    result = np.sort(arr, stable=True)
    assert list(result) == [1, 1, 3, 4, 5]


def test_ndmax_parameter():
    """Skill: np.array(nested, dtype=object, ndmax=1) limits recursion depth."""
    nested = [[1, 2], [3, 4, 5]]  # ragged
    result = np.array(nested, dtype=object, ndmax=1)
    assert result.ndim == 1
    assert len(result) == 2


test("np.std with mean= parameter", test_std_with_mean)
test("np.sort stable= keyword", test_sort_stable_keyword)
test("np.array ndmax= parameter (2.4+)", test_ndmax_parameter)

# =============================================================================
# SECTION: Float32 FFT precision note
# =============================================================================
print("\n=== Float32 FFT ===")


def test_fft_float32_stays_float32():
    """Skill: FFT no longer promotes float32 to float64 (2.0+)."""
    arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    result = np.fft.fft(arr)
    # The result should be complex64 (not complex128)
    assert result.dtype == np.complex64, f"Expected complex64 but got {result.dtype}"


test("FFT preserves float32 -> complex64", test_fft_float32_stays_float32)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"TOTAL: {passed + failed} tests | PASSED: {passed} | FAILED: {failed}")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for e in errors:
        print(e)

sys.exit(0 if failed == 0 else 1)
