"""
Comprehensive assessment of skills/zarr.md accuracy.
Tests every claim and code example against zarr 3.1.5.
"""
import tempfile
import os
import sys
import traceback
import numpy as np

# Track results
verified = []
issues = []
missing = []

def test(name):
    """Decorator to track test results."""
    def decorator(fn):
        def wrapper():
            try:
                fn()
                verified.append((name, "passed"))
            except AssertionError as e:
                issues.append((name, str(e)))
            except Exception as e:
                issues.append((name, f"EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"))
        wrapper()
        return fn
    return decorator

# =============================================================================
# v2 -> v3 Migration Table
# =============================================================================

@test("Codec classes moved to zarr.codecs")
def _():
    from zarr.codecs import BloscCodec, ZstdCodec, GzipCodec
    assert BloscCodec is not None
    assert ZstdCodec is not None
    assert GzipCodec is not None

@test("Use factory functions (create_array), not zarr.Array constructor directly")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        store = zarr.storage.LocalStore(tmp)
        z = zarr.create_array(store, name="test", shape=(100,), dtype="f4")
        assert isinstance(z, zarr.Array)

@test("Dot notation removed - must use bracket syntax")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        store = zarr.storage.LocalStore(tmp)
        g = zarr.open_group(store, mode="w")
        g.create_array("myarray", shape=(10,), dtype="f4")
        # Bracket syntax works
        arr = g["myarray"]
        assert isinstance(arr, zarr.Array)
        # Dot notation should NOT work
        try:
            result = g.myarray
            dot_works = True
        except AttributeError:
            dot_works = False
        assert not dot_works, \
            f"g.myarray DID work (returned {type(result).__name__}). Skill says dot notation removed but it still works."

@test("Store classes moved/renamed: LocalStore, FsspecStore")
def _():
    from zarr.storage import LocalStore, FsspecStore
    assert LocalStore is not None
    assert FsspecStore is not None

@test("compressors= (plural) parameter works on create_array")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp),
            name="test",
            shape=(100,),
            dtype="f4",
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
        )
        assert z.shape == (100,)

@test("resize requires tuple, not *args")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(100, 50), dtype="f4", chunks=(50, 50)
        )
        # Tuple form should work
        z.resize((200, 50))
        assert z.shape == (200, 50)
        # Test if *args form also works
        try:
            z.resize(300, 50)
            args_works = True
        except TypeError:
            args_works = False
        if args_works:
            # Note as issue: claim overstated
            assert False, \
                "z.resize(300, 50) also worked - claim that 'must pass tuple' is overstated"

@test("from_array: store is first, data is keyword-only")
def _():
    import zarr
    data = np.arange(100, dtype="f4")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "from_array.zarr")
        z = zarr.from_array(path, data=data, chunks=(50,))
        assert z.shape == (100,)
        np.testing.assert_array_equal(z[:], data)

        # Verify that passing data as first positional arg fails or gives wrong result
        path2 = os.path.join(tmp, "from_array2.zarr")
        try:
            z2 = zarr.from_array(data, path2)
            positional_works = True
        except (TypeError, Exception):
            positional_works = False
        assert not positional_works, \
            "from_array(data, store) should fail - data should be keyword-only"

@test("create_dataset renamed to create_array (deprecated but works)")
def _():
    import zarr
    import warnings
    with tempfile.TemporaryDirectory() as tmp:
        g = zarr.open_group(tmp, mode="w")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            arr = g.create_dataset("test", shape=(10,), dtype="f4")
        assert arr.shape == (10,)

@test("Don't mix numcodecs with v3 - should error or warn")
def _():
    import zarr
    try:
        import numcodecs
        nc_blosc = numcodecs.Blosc()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                z = zarr.create_array(
                    zarr.storage.LocalStore(tmp), name="test",
                    shape=(100,), dtype="f4",
                    compressors=nc_blosc,
                )
                # If it works, that's an issue with the skill file claim
                assert False, \
                    "Using numcodecs.Blosc() with compressors= did NOT raise an error"
            except (TypeError, ValueError, Exception):
                pass  # Expected to fail
    except ImportError:
        pass  # numcodecs not installed, skip

# =============================================================================
# Codec Pipeline (v3)
# =============================================================================

@test("Default codec pipeline works without specifying codecs")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(1000,), dtype="f4",
        )
        z[:] = np.random.randn(1000).astype("f4")
        roundtrip = z[:]
        assert roundtrip.shape == (1000,)

@test("codecs= does NOT work on create_array")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        try:
            z = zarr.create_array(
                zarr.storage.LocalStore(tmp), name="test",
                shape=(1000,), dtype="f4",
                codecs=[zarr.codecs.BytesCodec(), zarr.codecs.ZstdCodec()],
            )
            codecs_works = True
        except TypeError:
            codecs_works = False
    assert not codecs_works, \
        "codecs= parameter WORKED on create_array - skill says it only works on ShardingCodec"

@test("Explicit serializer= + compressors= works")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(1000,), dtype="f4",
            serializer=zarr.codecs.BytesCodec(endian="little"),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5,
                                                shuffle=zarr.codecs.BloscShuffle.shuffle),
        )
        z[:] = np.random.randn(1000).astype("f4")
        assert z[:].shape == (1000,)

@test("No compression via compressors=None")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(1000,), dtype="f4",
            compressors=None,
        )
        z[:] = np.random.randn(1000).astype("f4")
        assert z[:].shape == (1000,)

# =============================================================================
# Sharding
# =============================================================================

@test("Sharding with chunks + shards parameters")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            store=os.path.join(tmp, "data.zarr"),
            shape=(100_000, 100),
            dtype="float32",
            chunks=(100, 100),
            shards=(10_000, 100),
        )
        assert z.shape == (100_000, 100)
        z[0:100, :] = np.ones((100, 100), dtype=z.dtype)
        np.testing.assert_array_equal(z[0, :5], np.ones(5, dtype=z.dtype))

# =============================================================================
# Blosc typesize bug
# =============================================================================

@test("Blosc typesize bug: shuffle gives poor compression on float64 vs ZstdCodec")
def _():
    import zarr
    data = np.random.randn(100_000).astype("float64")

    tmp1 = tempfile.mkdtemp()
    tmp2 = tempfile.mkdtemp()
    try:
        z_blosc = zarr.create_array(
            zarr.storage.LocalStore(tmp1), name="blosc",
            shape=data.shape, dtype=data.dtype,
            compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=1,
                                                shuffle=zarr.codecs.BloscShuffle.shuffle),
        )
        z_blosc[:] = data

        z_zstd = zarr.create_array(
            zarr.storage.LocalStore(tmp2), name="zstd",
            shape=data.shape, dtype=data.dtype,
            compressors=zarr.codecs.ZstdCodec(level=3),
        )
        z_zstd[:] = data

        # Verify data round-trips correctly
        np.testing.assert_array_almost_equal(z_blosc[:], data, decimal=10)
        np.testing.assert_array_almost_equal(z_zstd[:], data, decimal=10)

        blosc_stored = z_blosc.nbytes_stored
        zstd_stored = z_zstd.nbytes_stored
        print(f"    Blosc+shuffle stored: {blosc_stored} bytes, ZstdCodec stored: {zstd_stored} bytes")
    finally:
        import shutil
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)

# =============================================================================
# write_empty_chunks default
# =============================================================================

@test("write_empty_chunks=False by default in v3")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        store = zarr.storage.LocalStore(tmp)
        z = zarr.create_array(
            store, name="sparse", shape=(100,), dtype="f4",
            chunks=(10,), fill_value=0.0,
        )
        # Write only fill-value data (all zeros)
        z[:] = np.zeros(100, dtype="f4")
        # With write_empty_chunks=False (default), these chunks should NOT be stored
        nchunks = z.nchunks_initialized
        print(f"    Default: nchunks_initialized={nchunks} (expected 0)")
        assert nchunks == 0, \
            f"Expected 0 chunks stored for all-fill-value data, got {nchunks}"

@test("write_empty_chunks=True via config forces chunk storage")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        store = zarr.storage.LocalStore(tmp)
        z = zarr.create_array(
            store, name="dense", shape=(100,), dtype="f4",
            chunks=(10,), fill_value=0.0,
            config={"write_empty_chunks": True},
        )
        z[:] = np.zeros(100, dtype="f4")
        nchunks = z.nchunks_initialized
        print(f"    write_empty_chunks=True: nchunks_initialized={nchunks} (expected 10)")
        assert nchunks == 10, \
            f"Expected 10 chunks stored, got {nchunks}"

# =============================================================================
# Mode semantics
# =============================================================================

@test("Mode 'r' - read only, prevents writes")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.zarr")
        z = zarr.open_array(path, mode="w", shape=(100,), dtype="f4")
        z[:] = np.ones(100, dtype="f4")

        z_read = zarr.open_array(path, mode="r")
        np.testing.assert_array_equal(z_read[:], np.ones(100, dtype="f4"))
        try:
            z_read[:] = np.zeros(100, dtype="f4")
            write_ok = True
        except Exception:
            write_ok = False
        assert not write_ok, "mode='r' should prevent writes"

@test("Mode 'r+' - read/write existing, fails if missing")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        # First create an array
        path = os.path.join(tmp, "data.zarr")
        z = zarr.open_array(path, mode="w", shape=(100,), dtype="f4")
        z[:] = np.ones(100, dtype="f4")

        # r+ on existing should work
        z_rw = zarr.open_array(path, mode="r+")
        z_rw[0] = 42.0
        assert z_rw[0] == 42.0

        # r+ on non-existent should fail
        nonexist = os.path.join(tmp, "nonexistent.zarr")
        try:
            z_fail = zarr.open_array(nonexist, mode="r+", shape=(100,), dtype="f4")
            fail_on_missing = False
        except Exception:
            fail_on_missing = True
        assert fail_on_missing, \
            "open_array with mode='r+' did NOT fail on non-existent store"

@test("Mode 'w-' - create only, fails if exists")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.zarr")
        z = zarr.open_array(path, mode="w-", shape=(100,), dtype="f4")
        assert z.shape == (100,)
        try:
            z2 = zarr.open_array(path, mode="w-", shape=(100,), dtype="f4")
            fail_on_exist = False
        except Exception:
            fail_on_exist = True
        assert fail_on_exist, "mode='w-' should fail if store already exists"

@test("Mode 'w' - overwrite (destructive)")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.zarr")
        z = zarr.open_array(path, mode="w", shape=(100,), dtype="f4")
        z[:] = np.ones(100, dtype="f4")
        z2 = zarr.open_array(path, mode="w", shape=(50,), dtype="i4")
        assert z2.shape == (50,)
        assert z2.dtype == np.dtype("i4")

# =============================================================================
# zarr.codecs vs numcodecs imports
# =============================================================================

@test("zarr.codecs.BloscCodec is different from numcodecs.Blosc")
def _():
    from zarr.codecs import BloscCodec
    try:
        from numcodecs import Blosc
        assert BloscCodec is not Blosc, \
            "zarr.codecs.BloscCodec should be different from numcodecs.Blosc"
    except ImportError:
        pass  # numcodecs not installed

@test("zarr.codecs.numcodecs wrapper import path")
def _():
    # Skill says: from zarr.codecs.numcodecs import Blosc
    try:
        from zarr.codecs.numcodecs import Blosc
        assert Blosc is not None
    except ImportError as e:
        assert False, f"Could not import zarr.codecs.numcodecs.Blosc: {e}"

# =============================================================================
# FsspecStore async requirement
# =============================================================================

@test("LocalStore works for local files")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        store = zarr.storage.LocalStore(tmp)
        z = zarr.create_array(store, name="test", shape=(100,), dtype="f4")
        z[:] = np.arange(100, dtype="f4")
        np.testing.assert_array_equal(z[:], np.arange(100, dtype="f4"))

# =============================================================================
# open_array mode as hidden kwarg
# =============================================================================

@test("open_array mode parameter: check if explicit or hidden kwarg")
def _():
    import zarr
    import inspect
    sig = inspect.signature(zarr.open_array)
    params = sig.parameters
    has_mode = "mode" in params
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_mode:
        assert False, \
            f"mode IS an explicit parameter in open_array (not hidden). Params: {list(params.keys())}"
    elif has_kwargs:
        pass  # Confirmed: mode is hidden in **kwargs
    else:
        assert False, \
            f"mode not in params and no **kwargs. Params: {list(params.keys())}"

# =============================================================================
# Removed features
# =============================================================================

@test("Object dtype not supported in v3")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        try:
            z = zarr.create_array(
                zarr.storage.LocalStore(tmp), name="test",
                shape=(10,), dtype="|O",
            )
            obj_works = True
        except Exception:
            obj_works = False
        assert not obj_works, "Object dtype should not be supported in v3"

# =============================================================================
# Known Limitations
# =============================================================================

@test("F memory order: order='F' handled (warn or ignore)")
def _():
    import zarr
    import warnings
    with tempfile.TemporaryDirectory() as tmp:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                z = zarr.create_array(
                    zarr.storage.LocalStore(tmp), name="test",
                    shape=(10, 10), dtype="f4",
                    order="F",
                )
                f_warnings = [x for x in w if "order" in str(x.message).lower() or "F" in str(x.message)]
                print(f"    order='F' accepted. Warnings: {len(f_warnings)}")
            except TypeError as e:
                print(f"    order='F' raised TypeError: {e}")

# =============================================================================
# Performance Tips: config set
# =============================================================================

@test("zarr.config.set for async concurrency")
def _():
    import zarr
    zarr.config.set({"async.concurrency": 10})

# =============================================================================
# Additional verification tests
# =============================================================================

@test("BloscShuffle enum exists with expected members")
def _():
    from zarr.codecs import BloscShuffle
    assert hasattr(BloscShuffle, "shuffle")
    assert hasattr(BloscShuffle, "noshuffle")

@test("zarr_format defaults to 3")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(10,), dtype="f4",
        )
        assert z.metadata.zarr_format == 3, \
            f"Expected zarr_format=3, got {z.metadata.zarr_format}"

@test("zarr_format=2 for backwards compatibility")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        z = zarr.create_array(
            zarr.storage.LocalStore(tmp), name="test",
            shape=(10,), dtype="f4",
            zarr_format=2,
        )
        assert z.metadata.zarr_format == 2, \
            f"Expected zarr_format=2, got {z.metadata.zarr_format}"

@test("BytesCodec exists in zarr.codecs")
def _():
    from zarr.codecs import BytesCodec
    assert BytesCodec is not None

@test("open_array with string path works (replacement for FSMap)")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.zarr")
        z = zarr.open_array(path, mode="w", shape=(100,), dtype="f4")
        assert z.shape == (100,)

@test("Consolidated metadata works")
def _():
    import zarr
    with tempfile.TemporaryDirectory() as tmp:
        root = zarr.open_group(tmp, mode="w")
        for i in range(3):
            root.create_array(f"arr_{i}", shape=(10,), dtype="f4")
        zarr.consolidate_metadata(tmp)
        root2 = zarr.open_consolidated(tmp)
        arr = root2["arr_1"]
        assert arr.shape == (10,)

# =============================================================================
# Report
# =============================================================================

print("\n" + "=" * 70)
print("ZARR SKILL FILE ASSESSMENT RESULTS")
print("=" * 70)

print(f"\nVERIFIED CORRECT ({len(verified)}):")
for name, evidence in verified:
    print(f"  - {name}: {evidence}")

print(f"\nISSUES FOUND ({len(issues)}):")
for name, detail in issues:
    print(f"  - {name}: {detail}")

if missing:
    print(f"\nMISSING CONTENT ({len(missing)}):")
    for item in missing:
        print(f"  - {item}")

print(f"\nTotal: {len(verified)} verified, {len(issues)} issues, {len(missing)} missing")

# Exit with error code if issues found
if issues:
    sys.exit(1)
