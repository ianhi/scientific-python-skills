# NumPy - Claude Code Skill

> Skill for writing idiomatic NumPy code. Current version: 2.4.2 (Python 3.11+)

## NEP 50 Type Promotion (SILENT Behavioral Change)

Python scalars now adapt to array dtype instead of promoting. This can silently produce wrong results.

```python
# Python scalars are "weak" - adapt to array dtype:
arr = np.array([1.0, 2.0], dtype=np.float32)
result = arr + 1.0    # float32 (Python float adapts to arr)
result = arr + 1      # float32 (Python int adapts to arr)

# NumPy scalars are "strong" - normal promotion:
np.float32(3) + np.float64(3)  # float64 (standard promotion)

# Force promotion with NumPy scalar:
result = arr * np.float64(0.1)  # float64 (NumPy scalar forces promotion)
result = arr * 0.1              # float32 (Python scalar adapts - precision loss!)

# DANGER: integer overflow is silent:
arr_u8 = np.array([200], dtype=np.uint8)
arr_u8 + 100   # uint8(44) with overflow! Python int adapts to uint8
# Safe: arr_u8.astype(np.int16) + 100

# DANGER: float32 accumulation loses precision:
arr_f32 = np.array([1e8], dtype=np.float32)
arr_f32 + 1.0   # float32: 1e8 (1.0 completely lost!) Python float adapts to float32
# Safe: arr_f32 + np.float64(1.0)  # float64: 100000001.0 (correct)
```

## Copy Semantics

```python
# copy= parameter controls when data is copied:
x = np.asarray(data)             # copy if needed (preferred)
x = np.array(data, copy=True)   # always copy
x = np.array(data, copy=False)  # never copy (raises ValueError if copy needed)

# Reshape with copy control:
np.reshape(a, (2, 3), copy=False)  # ValueError if can't return view
```

## StringDType & String Operations

```python
from numpy.dtypes import StringDType

arr = np.array(["hello", "world"], dtype=StringDType())

# With missing data support:
dt = StringDType(na_object=np.nan)
arr = np.array(["hello", np.nan, "world"], dtype=dt)

# FAST - ufunc-based string operations via np.strings:
np.strings.upper(arr)
np.strings.find(arr, "lo")
np.strings.replace(arr, "o", "0")

# SLOW - avoid np.char (element-wise Python calls):
np.char.upper(arr)
```

## Useful Modern API

```python
# Cleaner unique decompositions:
np.unique_counts(x)    # (values, counts)
np.unique_inverse(x)   # (values, inverse_indices)

# FAST unsorted unique (2.3+):
np.unique(arr, sorted=False)  # hash-based, up to 15x faster

# Cumulative operations:
np.cumulative_sum(x, axis=None, include_initial=False)

# Type checking:
np.isdtype(dtype, 'real floating')  # 'numeric', 'integral', 'bool', etc.
```

## Gotchas & Common Mistakes

### np.isclose vs math.isclose (#10161)
```python
# np.isclose uses ASYMMETRIC relative tolerance:
# np.isclose: abs(a-b) <= atol + rtol * abs(b)     (uses b as reference)
# math.isclose: abs(a-b) <= max(rtol * max(abs(a), abs(b)), atol)
# For symmetric comparison: np.isclose(a, b) & np.isclose(b, a)
```

### Multiprocessing with .npz files (#18124)
```python
# BUG: .npz files use ZipFile which is not fork-safe
# Reading .npz in multiprocessing workers can corrupt data
data = dict(np.load("file.npz"))  # materialize before forking
```

### np.load pickle security (#12889)
```python
data = np.load("file.npy", allow_pickle=True)  # explicit opt-in required
```

### __array__ protocol requires copy= keyword
```python
class MyObj:
    def __array__(self, dtype=None, copy=None):  # copy= is REQUIRED
        arr = np.array(self._data, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr
```

## Known Limitations

- **Type stubs regression** (#27957): mypy issues between 2.1.3 and 2.2.0. Use pyright.
- **Shape typing incomplete** (#16544): `NDArray` does not encode shape.
- **No minmax function** (#9836): Must call `np.min` and `np.max` separately.

## Performance Tips

- **FAST: `np.unique(arr, sorted=False)`** - Hash-based, up to 15x faster.
- **FAST: `np.strings.*`** over `np.char.*` - Ufunc-based, significantly faster.
- **FAST: `StringDType()`** over `dtype=object` - Variable-width UTF-8, ufunc support.
- **FAST: `np.std(arr, mean=precomputed)`** - Avoid redundant mean computation.
- **FAST: Scalar ufuncs** (2.4+) - ~6x faster than 2.3 for scalar operations.
- **SLOW: Float32 FFT** - FFT no longer promotes float32 to float64. Use `arr.astype(np.float64)` if you need double precision.
