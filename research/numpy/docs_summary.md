# NumPy Research Summary for Skill File

## 1. Version Info

- **Current installed version**: NumPy 2.4.2
- **Latest in repo (main branch)**: developing toward 2.5.0
- **Python support**: 2.4.x supports Python 3.11-3.14
- **NumPy 2.0** was the first major release since 2006 (June 2024)

## 2. Breaking Changes in NumPy 2.0+

### 2a. Type Promotion (NEP 50) - CRITICAL

The single biggest behavioral change. NumPy 2.0 removed value-based type promotion.

**Old (1.x)**: `np.result_type(np.int8, 1)` -> `np.int8` (value 1 fits in int8)
**Old (1.x)**: `np.result_type(np.int8, 255)` -> `np.int16` (value 255 doesn't fit)
**New (2.0+)**: Python scalars are "weakly typed" - they adapt to the NumPy dtype:

```python
# NumPy 2.0+: Python scalar adapts to array's dtype
np.float32(3) + 3.   # returns float32 (was float64 in 1.x!)
np.array([1, 2, 3], dtype=np.uint8) + 1  # returns uint8
np.array([1, 2, 3], dtype=np.float32) + 2.  # returns float32

# But higher-precision NumPy scalars are NOT ignored:
np.array([3], dtype=np.float32) + np.float64(3)  # returns float64
```

**Key consequence**: Lower precision results when mixing Python scalars with NumPy float32. Possible integer overflows.

### 2b. Copy Keyword Semantics Change

`np.array()` and `np.asarray()` `copy` parameter changed:
- `copy=None` (default) - copy only if needed (same as old `copy=False`)
- `copy=True` - always copy
- `copy=False` - **NEVER copy; raises ValueError if copy needed** (changed meaning!)

**Anti-pattern**: `np.array(x, copy=False)` in old code meant "copy if needed". Now it means "never copy". Replace with `np.asarray(x)`.

### 2c. Windows Default Integer

Default integer on 64-bit Windows changed from int32 to int64 (matching other platforms). Now `np.intp` everywhere.

### 2d. Scalar Representation (NEP 51)

Scalars now print with type info: `np.float64(3.0)` instead of just `3.0`.
Use `np.set_printoptions(legacy="1.25")` for old behavior.

### 2e. Array Repr Change (2.2)

Summarized arrays now include `shape=` in their repr. Use `legacy='2.1'` print option to revert.

### 2f. Namespace Cleanup (~100 removals)

`np.core` renamed to `np._core` (private). `np.lib` mostly empty. About 100 members removed from main namespace.

### 2g. Return Type Changes (2.0)

- `np.gradient()` returns tuple (was list)
- `np.atleast_1d`, `np.atleast_2d`, `np.atleast_3d`, `np.broadcast_arrays`, `np.meshgrid`, `np.ogrid`, `np.histogramdd` return tuples (were lists)
- `np.any` and `np.all` return booleans for object arrays (previously returned one of the arguments)

### 2h. String Truthiness Change (2.0)

Strings are now True when non-empty, False when empty. Previously `"0"` cast to bool was False (treated as integer). Now `"0"` is True because it's non-empty.

### 2i. Complex Sign Change (2.0)

Complex sign is now `z / |z|` (Array API standard), not based on real part.

### 2j. np.round Always Returns Copy (2.4)

`np.round` now always returns a copy. Previously returned a view for integer inputs with `decimals >= 0`.

## 3. Removed/Expired APIs (CRITICAL for anti-patterns)

### Removed in 2.0 (errors immediately):

| Old | Replacement |
|-----|-------------|
| `np.float_` | `np.float64` |
| `np.complex_` | `np.complex128` |
| `np.string_` | `np.bytes_` |
| `np.unicode_` | `np.str_` |
| `np.Inf`, `np.Infinity`, `np.infty` | `np.inf` |
| `np.NaN` | `np.nan` |
| `np.NINF` | `-np.inf` |
| `np.PINF` | `np.inf` |
| `np.NZERO` | `-0.0` |
| `np.PZERO` | `0.0` |
| `np.cfloat` | `np.complex128` |
| `np.longfloat` | `np.longdouble` |
| `np.singlecomplex` | `np.complex64` |
| `np.longcomplex` | `np.clongdouble` |
| `np.clongfloat` | `np.clongdouble` |
| `np.mat` | `np.asmatrix` |
| `np.source` | `inspect.getsource` |
| `np.who` | IDEs / `locals()` |
| `np.lookfor` | Search NumPy docs directly |
| `np.cast` | `np.asarray(arr, dtype=dtype)` |
| `np.safe_eval` | `ast.literal_eval` |
| `np.find_common_type` | `np.promote_types` or `np.result_type` |
| `np.round_` | `np.round` |
| `np.nbytes` | `np.dtype(<dtype>).itemsize` |
| `np.asfarray` | `np.asarray(x, dtype=float)` |
| `np.alltrue` | `np.all` |
| `np.sometrue` | `np.any` |
| `np.product` | `np.prod` |
| `np.cumproduct` | `np.cumprod` |
| `np.msort` | `np.sort(a, axis=0)` |
| `np.geterrobj` / `np.seterrobj` | `np.errstate` context manager |
| `np.set_string_function` | `np.set_printoptions` |
| `np.fastCopyAndTranspose` | `arr.T.copy()` |
| `np.core.*` | `np.*` (main namespace) |
| `numpy.array_api` submodule | main `numpy` namespace or `array-api-strict` |
| dtype aliases: `int0`, `uint0`, `void0`, `object0`, `str0`, `bytes0`, `bool8` | removed |
| `ndarray.newbyteorder` | `arr.view(arr.dtype.newbyteorder(order))` |
| `ndarray.ptp` | `np.ptp(arr, ...)` |
| `ndarray.setitem` | `arr[index] = value` |
| `np.compat` | removed (Python 2 support) |
| `np.disp` | use your own print function |
| `np.DataSource` | `np.lib.npyio.DataSource` |
| `np.add_docstring` | `np.lib.add_docstring` |
| `np.add_newdoc` | `np.lib.add_newdoc` |
| `np.byte_bounds` | `np.lib.array_utils.byte_bounds` |
| `np.compare_chararrays` | `np.char.compare_chararrays` |
| `np.format_parser` | `np.rec.format_parser` |
| `np.recfromcsv` | `np.genfromtxt` with comma delimiter |
| `np.recfromtxt` | `np.genfromtxt` |
| `np.issctype` | `issubclass(rep, np.generic)` |
| `np.issubsctype` | `np.issubdtype` |
| `np.issubclass_` | `issubclass` builtin |
| `np.obj2sctype` | `np.dtype(obj).type` |
| `np.sctype2char` | `np.dtype(obj).char` |
| `np.sctypes` | Access dtypes explicitly |
| `np.maximum_sctype` | Use specific dtype |
| `np.deprecate` | `warnings.warn(DeprecationWarning)` |
| `np.deprecate_with_doc` | `warnings.warn(DeprecationWarning)` |
| `np.set_numeric_ops` | `PyUFunc_ReplaceLoopBySignature` or `__array_ufunc__` |
| `np.tracemalloc_domain` | `np.lib.tracemalloc_domain` |
| `__array_prepare__` | `__array_ufunc__` or `__array_wrap__` |
| Exceptions from main namespace | `from numpy.exceptions import ...` |

### Removed in 2.1:

| Old | Replacement |
|-----|-------------|
| `np.nonzero` on scalars/0-D arrays | errors |
| `np.set_string_function` | fully removed |

### Removed in 2.2:

| Old | Replacement |
|-----|-------------|
| `bool(np.array([]))` | errors; use `arr.size > 0` |
| NEP 50 promotion state settings | removed (`_set_promotion_state`, `_get_promotion_state`) |

### Removed in 2.3:

| Old | Replacement |
|-----|-------------|
| `np.tostring()` / `ndarray.tostring()` | `.tobytes()` |
| binary mode of `fromstring` | `np.frombuffer` |
| `np.conjugate` on non-numeric types | errors |
| `np.bincount(minlength=None)` | use `minlength=0` |
| Converting `np.inexact`/`np.floating` to dtype | errors |
| Converting `np.complex`/`np.integer`/etc to dtype | errors |
| `round()` on complex scalars | use `np.round` or `scalar.round` |
| `np.sum(generator)` | use `np.sum(np.fromiter(gen))` |

### Removed in 2.4:

| Old | Replacement |
|-----|-------------|
| `np.trapz` | `np.trapezoid` |
| `np.in1d` | `np.isin` |
| `np.row_stack` | `np.vstack` |
| `np.linalg.linalg` | `np.linalg` |
| `np.fft.helper` | `np.fft` |
| `quantile/percentile` `interpolation=` param | `method=` param |
| `np.reshape` `newshape=` param | positional or `shape=` |
| `np.save` `fix_imports=` param | removed |
| `np.ndindex.ndincr()` | `next(ndindex)` |
| `np.array2string` `style=` param | removed |
| `np.corrcoef` `bias=`/`ddof=` params | removed (had no effect) |
| `np.MachAr` | removed |
| ndarray conversion to scalar with ndim > 0 | TypeError |

### Removed in 2.5 (upcoming):

| Old | Replacement |
|-----|-------------|
| `np.cross` with 2D vectors | use 3D vectors only |
| `np.chararray` re-export | `np.char.chararray` (also deprecated) |
| dtype alias `'a'` | use `'S'` |
| `np.distutils` | removed entirely |
| `np.finfo(None)` | TypeError |
| `np._core.numerictypes.maximum_sctype` | removed |
| `np.get_array_wrap` | removed |
| `bincount` with non-integer inputs | TypeError |
| `np.lib.math` | removed |
| `_add_newdoc_ufunc` | `ufunc.__doc__ = newdoc` |

### Pending Deprecation (works now but will break):

- `np.fix` -> use `np.trunc` instead (2.4 pending deprecation, 2.5 deprecated)
- Setting `ndarray.shape` in-place -> use `np.reshape()` (2.4 pending deprecation)
- Setting `ndarray.strides` -> use `np.lib.stride_tricks.as_strided` (2.4 deprecation)
- `np.char.chararray` -> use ndarray with string dtype (2.5 deprecation)
- `np.typename` -> use `np.dtype.name` (2.5 deprecation)
- Positional `out` arg to `np.maximum`/`np.minimum` -> use `out=` keyword (2.4 deprecation)
- `np.testing.assert_warns` / `np.testing.suppress_warnings` -> use pytest (2.4 deprecation)
- Resizing numpy array in-place -> use `np.resize` (2.5 deprecation)

## 4. New Features

### 4a. StringDType (2.0+)

Variable-width UTF-8 string dtype. Import: `from numpy.dtypes import StringDType`

```python
from numpy.dtypes import StringDType

arr = np.array(["hello", "world"], dtype=StringDType())

# With missing data support (NaN-like sentinel):
dt = StringDType(na_object=np.nan)
arr = np.array(["hello", np.nan, "world"], dtype=dt)
# NaN-like sentinels propagate through string ops:
arr + arr  # ['hellohello', nan, 'worldworld']
# NaN-like sentinels sort to end:
np.sort(arr)  # ['hello', 'world', nan]

# With None sentinel:
dt = StringDType(na_object=None)
arr = np.array(["hello", None, "world"], dtype=dt)
# None sentinel raises on string ops like sort (not NaN-like)

# Disable coercion (strict validation):
dt = StringDType(coerce=False)
np.array([1, 2], dtype=dt)  # ValueError

# Casting to fixed-width requires explicit size:
arr.astype("U10")  # OK
arr.astype(np.str_)  # ERROR - must specify size

# Empty arrays filled with empty strings:
np.empty(3, dtype=StringDType())  # ['', '', '']
```

Key properties:
- Kind: `'T'`, char: `'T'`
- New user-defined dtype (not legacy)
- Data stored outside main array buffer (metadata in buffer)
- Supports MaskedArray fill_value (2.4+, default `'N/A'`)
- Generic type in 2.3+: `StringDType[None]` for typing
- `coerce=True` by default (non-strings converted via `str()`)

### 4b. numpy.strings Namespace (2.0+)

New ufunc-based string operations. Prefer over `np.char`:

```python
np.strings.find(arr, "pattern")  # faster than np.char.find
np.strings.rfind(arr, "pattern")
np.strings.slice(arr, start, stop)  # new in 2.3
```

### 4c. Array API Standard Functions (2.0+)

New aliases and functions for Array API compatibility:

**Trig aliases**: `np.acos`, `np.acosh`, `np.asin`, `np.asinh`, `np.atan`, `np.atanh`, `np.atan2`

**Bitwise aliases**: `np.bitwise_left_shift`, `np.bitwise_invert`, `np.bitwise_right_shift`

**Other aliases**: `np.concat` (for concatenate), `np.permute_dims` (for transpose), `np.pow`

**New functions**:
- `np.unstack(array, axis=0)` (2.1) - inverse of `np.stack`
- `np.matrix_transpose(x)` (2.0) - also available as `ndarray.mT`
- `np.vecdot(x1, x2, axis=-1)` (2.0) - vector dot product
- `np.matvec(x1, x2)` (2.2) - matrix-vector product
- `np.vecmat(x1, x2)` (2.2) - vector-matrix product (conjugates complex left input)
- `np.cumulative_sum(x, *, axis, include_initial=False)` (2.1) - like cumsum with optional initial
- `np.cumulative_prod(x, *, axis, include_initial=False)` (2.1)
- `np.unique_all`, `np.unique_counts`, `np.unique_inverse`, `np.unique_values` (2.0)
- `np.isdtype(dtype, kind)` (2.0) - classify dtypes by kind string
- `np.astype(arr, dtype)` (2.0) - standalone function
- `np.bitwise_count(x)` (2.0) - count 1-bits

**Array attributes and methods**:
- `ndarray.mT` - matrix transpose (swaps last two axes)
- `ndarray.device` - returns `"cpu"`
- `ndarray.to_device(device)` - only `"cpu"` supported

**Array creation keyword**: `device="cpu"` on `np.asarray`, `np.arange`, `np.empty`, `np.zeros`, `np.ones`, `np.full`, `np.linspace`, `np.eye`, and `*_like` variants.

**`__array_api_version__`**: `"2024.12"` as of NumPy 2.3

**Note on vecdot/vecmat conjugation**: `np.vecdot` and `np.vecmat` take the complex conjugate of the left-hand vector (physics convention). `np.matmul` never conjugates.

### 4d. New linalg Functions (2.0+)

- `np.linalg.matrix_norm(x)` - matrix norm
- `np.linalg.vector_norm(x)` - vector norm
- `np.linalg.diagonal(x)` / `np.linalg.trace(x)` - different default axes than np.diagonal/trace
- `np.linalg.svdvals(x)` - singular values (= `np.linalg.svd(x, compute_uv=False)`)
- `np.linalg.outer(x1, x2)` - outer product (1D only, unlike np.outer)
- `np.linalg.cross(x1, x2)` - cross product (3D only, unlike np.cross)
- `np.linalg.vecdot` - alias for `np.vecdot`
- `np.linalg.matrix_transpose` - alias for `np.matrix_transpose`
- `np.linalg.tensordot` - alias for `np.tensordot`
- `np.linalg.matmul` - alias for `np.matmul`

### 4e. New Parameters on Existing Functions

| Function | New Parameter | Version | Notes |
|----------|--------------|---------|-------|
| `np.var` / `np.std` | `correction=` | 2.0 | Array API alternative to `ddof=` (cannot use both) |
| `np.var` / `np.std` | `mean=` | 2.0 | Pass precalculated mean to avoid recomputation |
| `np.sort` / `np.argsort` | `stable=` | 2.0 | Boolean flag for stable sort |
| `np.linalg.cholesky` | `upper=` | 2.0 | Upper triangular decomposition |
| `np.linalg.pinv` | `rtol=` | 2.0 | Replacing `rcond` (plan to deprecate rcond) |
| `np.linalg.matrix_rank` | `rtol=` | 2.0 | Alternative to tol |
| `np.quantile` / `np.percentile` | `weights=` | 2.0 | Only with `method="inverted_cdf"` |
| `np.reshape` | `shape=`, `copy=` | 2.1 | Named shape param and copy control |
| `np.clip` | `min=`, `max=` | 2.1 | Replacing `a_min`/`a_max` |
| `np.asanyarray` | `copy=`, `device=` | 2.1 | Matching `np.asarray` |
| `np.unique` | `sorted=` | 2.3 | Default True; False uses hash-table for speed |
| `np.array` | `ndmax=` | 2.4 | Max dimensions from nested sequences |
| `np.astype` | `casting='same_value'` | 2.4 | Checks round-trip value safety |
| `np.size` | multiple axes | 2.4 | Accept tuple of axes |
| `np.trim_zeros` | sequence `axis` | 2.4 | Trim along multiple axes |
| `np.pad` | dict `pad_width` | 2.4 | Per-axis padding as dict |
| `np.testing.assert_allclose` | `strict=` | 2.0 | Disables broadcasting, checks dtype match |

### 4f. np.bool (2.0+)

`np.bool` is now the canonical name for NumPy's boolean dtype (was Python's `bool` until 1.24). `np.bool_` is now an alias to it.

### 4g. New long/ulong (2.0+)

`np.long` and `np.ulong` - C `long` and `unsigned long` types. (Previously `np.long` was alias for Python int.)

### 4h. ufunc out=... (2.3+)

Use `out=...` (Ellipsis) to ensure ufuncs return arrays instead of scalars for 0-D inputs:
```python
np.add(np.float64(1), np.float64(2), out=...)  # returns 0-D array, not scalar
```

### 4i. Structural Pattern Matching (2.5)

`np.ndarray` now supports PEP 634 structural pattern matching with `match`/`case`.

### 4j. ufunc where= Warning (2.4)

Ufuncs called with `where=` but without `out=` now emit a warning. Locations where mask is False have undefined values. Suppress by passing `out=None` explicitly.

## 5. Anti-Patterns (What Claude Will Get Wrong)

### CRITICAL Anti-Patterns:

1. **Using removed aliases** (`np.float_`, `np.complex_`, `np.string_`, `np.unicode_`, `np.Inf`, `np.NaN`, etc.)
2. **Using `np.trapz`** instead of `np.trapezoid`
3. **Using `np.in1d`** instead of `np.isin`
4. **Using `np.row_stack`** instead of `np.vstack`
5. **Using `copy=False` to mean "copy if needed"** - now means "never copy"
6. **Using `np.core.anything`** - it's `np._core` now (private)
7. **Using `np.array_api`** - removed; use main namespace
8. **Expecting `np.float32(3) + 3.` to return float64** - returns float32 now (NEP 50)
9. **Using `np.find_common_type`** - removed, use `np.result_type` or `np.promote_types`
10. **Using quantile/percentile `interpolation=` parameter** - use `method=`
11. **Using `np.reshape` `newshape=` parameter** - use positional or `shape=`
12. **Using `np.cross` with 2D vectors** - deprecated in 2.0, error in 2.5
13. **Using `np.linalg.linalg` or `np.fft.helper`** - removed in 2.4
14. **Using `np.fix` instead of `np.trunc`** - pending deprecation in 2.4, deprecated in 2.5
15. **Using `np.tostring`** - removed in 2.3, use `.tobytes()`
16. **Using `bool(np.array([]))`** - errors in 2.2+, use `arr.size > 0`
17. **Using `np.nonzero` on scalars/0-D arrays** - errors in 2.1+
18. **Using `np.sum` on generators** - errors in 2.4+, use `np.fromiter` first
19. **Using exceptions from main namespace** - must use `from numpy.exceptions import ...`
20. **Using `np.msort`** - removed, use `np.sort(a, axis=0)`

## 6. Key API Signature Changes

### np.array / np.asarray copy= semantics:

```python
# copy=None: copy if needed (DEFAULT, same as old copy=False)
# copy=True: always copy
# copy=False: NEVER copy, raises ValueError if copy needed
np.asarray(x)          # preferred for "copy if needed"
np.array(x, copy=True) # explicit copy
```

### __array__ protocol change:

```python
# Custom classes must now accept copy= keyword:
def __array__(self, dtype=None, copy=None):
    ...
```

### np.unique new signature (2.3+):

```python
np.unique(ar, return_index=False, return_inverse=False,
          return_counts=False, axis=None, *, equal_nan=True,
          sorted=True)  # sorted parameter new in 2.3
```

### __array_wrap__ signature change (2.0):

```python
# Must accept all 3 args:
def __array_wrap__(self, arr, context=None, return_scalar=False):
    ...
```

### np.linalg.solve broadcasting change (2.0):

`np.linalg.solve(a, b)` with b having 1 fewer dimensions is now Array API compliant. Old behavior: `np.linalg.solve(a, b[..., None])[..., 0]`.

### np.linalg.lstsq rcond default change (2.0):

Now uses machine precision * max(M, N). Old: just machine precision. Use `rcond=-1` for old behavior.

### concatenate axis=None casting (2.3):

`np.concatenate()` with `axis=None` uses `same-kind` casting by default, not `unsafe`.

## 7. Migration Patterns

### Ruff auto-fix:

```toml
[tool.ruff.lint]
select = ["NPY201"]
```

Then: `ruff check path/to/code/ --select NPY201`

### Common migrations:

```python
# Type aliases
np.float_        -> np.float64
np.complex_      -> np.complex128
np.string_       -> np.bytes_
np.unicode_      -> np.str_

# Constants
np.Inf / np.Infinity / np.infty -> np.inf
np.NaN           -> np.nan
np.NINF          -> -np.inf
np.PINF          -> np.inf
np.NZERO         -> -0.0
np.PZERO         -> 0.0

# Functions
np.trapz(y, x)   -> np.trapezoid(y, x)
np.in1d(a, b)    -> np.isin(a, b)
np.row_stack(arrays) -> np.vstack(arrays)
np.product(a)    -> np.prod(a)
np.alltrue(a)    -> np.all(a)
np.sometrue(a)   -> np.any(a)
np.cumproduct(a) -> np.cumprod(a)
np.msort(a)      -> np.sort(a, axis=0)
np.round_(a)     -> np.round(a)
np.asfarray(a)   -> np.asarray(a, dtype=np.float64)
np.cast[dtype](a)-> np.asarray(a, dtype=dtype)
np.nbytes[dtype] -> np.dtype(dtype).itemsize
.tostring()      -> .tobytes()
np.fromstring(s, dtype) [binary] -> np.frombuffer(s, dtype)
np.fix(x)        -> np.trunc(x)

# Type promotion
np.find_common_type(types, scalars) -> np.result_type(*types, *scalars)

# Exceptions moved:
from numpy.exceptions import AxisError       # not from numpy
from numpy.exceptions import ComplexWarning
from numpy.exceptions import VisibleDeprecationWarning
from numpy.exceptions import DTypePromotionError

# Quantile/percentile
np.percentile(a, 50, interpolation='linear') -> np.percentile(a, 50, method='linear')

# Reshape
np.reshape(a, newshape=(3, 4)) -> np.reshape(a, shape=(3, 4))  # or positional

# isdtype (replaces ad-hoc issubdtype checks)
np.isdtype(arr.dtype, 'real floating')  # instead of np.issubdtype(arr.dtype, np.floating)
```

### Version-conditional code:

```python
if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
    from numpy.exceptions import AxisError
else:
    from numpy import AxisError
```

## 8. Performance Tips

1. **StringDType vs object dtype**: Use `StringDType()` for string arrays instead of `dtype=object`. Variable-width, UTF-8, faster string operations via ufuncs.

2. **np.strings vs np.char**: Use `np.strings` namespace - operations are ufuncs and significantly faster.

3. **np.unique sorted=False (2.3+)**: `np.unique(arr, sorted=False)` uses hash tables, up to 15x faster for large arrays. Also faster for strings (2.4+) and complex dtypes (2.4+).

4. **var/std with precalculated mean**: `np.std(arr, mean=precomputed_mean)` avoids double mean calculation.

5. **FFT precision**: 2.0+ does native float32/longdouble FFT. No longer always computes in float64. Use float64 input explicitly if you need double precision.

6. **Sorting with SIMD**: sort/argsort significantly faster in 2.0+ due to Intel x86-simd-sort and Google Highway libraries.

7. **np.matmul with non-contiguous (2.3+)**: Now copies non-contiguous inputs to use BLAS, faster than element-wise fallback.

8. **Scalar calculations (2.4+)**: Scalar ufuncs are ~6x faster (reduced overhead from factor 19 to factor 3 vs math module).

9. **np.unique for strings (2.4+)**: Hash-based unique is ~15x faster for large string arrays.

10. **np.ndindex (2.4+)**: Rewritten with `itertools.product`, ~5x faster for large iteration spaces.

11. **np.searchsorted (2.5)**: Batched binary search up to 20x faster for many keys.

12. **OpenMP sort (2.3+)**: When built with `-Denable_openmp=true`, sort/argsort use parallel threads (up to 3.5x speedup on x86 with AVX2/AVX-512). Control via `OMP_NUM_THREADS`.

13. **macOS Accelerate (2.0+)**: On macOS >= 14, NumPy uses Accelerate for linear algebra (up to 10x faster).

## 9. Integration Notes

### With pandas:
- pandas >= 2.1 supports NumPy 2.0
- StringDType can interop with pandas ArrowStringDtype
- NEP 50 changes can affect pandas operations mixing Python scalars with NumPy arrays

### With scipy:
- scipy >= 1.13 supports NumPy 2.0
- copy=False semantics affect scipy's internal np.asarray calls

### With other libraries:
- **Binary incompatibility**: Libraries built against NumPy 1.x won't work at runtime with NumPy 2.0. They need to rebuild.
- **ABI**: It IS possible to build against NumPy 2.0 and have it work with both 1.x and 2.0 at runtime.
- Libraries using `np.core` will get DeprecationWarning (2.0) and must switch to main namespace.
- `__array_prepare__` removed - must use `__array_ufunc__` or `__array_wrap__`.
- `__array_wrap__` signature changed to `(self, arr, context=None, return_scalar=False)`.

### Free-threaded Python (3.13+):
- NumPy 2.1+ has preliminary support
- Read-only shared access to ndarrays is safe
- Mutating same array in multiple threads is NOT safe (no locking on array object)
- Object arrays need special care (no GIL protection)
- `np.errstate` and `np.printoptions` are thread-safe (context vars since 2.0/2.1)

### DLPack:
- NumPy 2.1+ supports DLPack v1
- Older DLPack versions will be deprecated

### isdtype kind strings (Array API):
```python
np.isdtype(np.float64, 'real floating')      # True
np.isdtype(np.int32, 'signed integer')        # True
np.isdtype(np.uint8, 'unsigned integer')      # True
np.isdtype(np.int64, 'integral')              # True
np.isdtype(np.complex128, 'complex floating') # True
np.isdtype(np.float32, 'numeric')             # True
np.isdtype(np.bool_, 'bool')                  # True
```

## 10. Source Files Reference

Key source files read for this summary (from repos/numpy/):
- `doc/source/numpy_2_0_migration_guide.rst` - Full migration guide
- `doc/source/release/2.0.0-notes.rst` through `2.4.2-notes.rst` - All release notes
- `doc/release/upcoming_changes/*.rst` - Changes coming in 2.5
- `numpy/_expired_attrs_2_0.py` - Complete list of expired attributes with messages
- `numpy/dtypes.pyi` - Type stubs for all DTypes including StringDType
- `doc/source/user/basics.strings.rst` - StringDType documentation
- `doc/neps/nep-0050-scalar-promotion.rst` - NEP 50 specification
- `numpy/lib/_arraysetops_impl.py` - np.unique signature with sorted= parameter
- `numpy/_core/numerictypes.py` - isdtype implementation
