"""
Comprehensive assessment of skills/icechunk.md
Tests every code example and claim against icechunk 1.1.18, zarr 3.1.5, xarray 2026.1.0.
"""

import numpy as np
import traceback
import sys

PASS = 0
FAIL = 0
ISSUES = []

def section(name):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

def ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS: {msg}")

def fail(msg, detail=""):
    global FAIL
    FAIL += 1
    ISSUES.append(msg)
    print(f"  FAIL: {msg}")
    if detail:
        print(f"        {detail}")

def warn(msg):
    print(f"  WARN: {msg}")

# Helper: create array with data (zarr 3.1.5 doesn't allow both shape and data)
def create_with_data(store, name, data):
    """Create a zarr array with data - use data= only, not shape+data."""
    return zarr.create_array(store, name=name, data=np.asarray(data))

# ============================================================================
section("1: Imports and in_memory_storage")
# ============================================================================
try:
    import icechunk
    import zarr
    import xarray as xr
    from icechunk.xarray import to_icechunk
    from icechunk.dask import store_dask
    ok("All imports work: icechunk, zarr, xr, to_icechunk, store_dask")
except Exception as e:
    fail(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
section("2: Repository.open_or_create (idempotent)")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)
    ok("open_or_create works on fresh storage")
except Exception as e:
    fail(f"open_or_create: {e}")
    traceback.print_exc()

# ============================================================================
section("3: Repository.create fails if exists")
# ============================================================================
try:
    storage2 = icechunk.in_memory_storage()
    repo2 = icechunk.Repository.create(storage2)
    ok("Repository.create works on fresh storage")
except Exception as e:
    fail(f"Repository.create: {e}")
    traceback.print_exc()

# ============================================================================
section("4: writable_session / readonly_session")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    ok(f"writable_session('main') returns {type(session).__name__}")

    arr = create_with_data(session.store, "test", [1.0, 2.0, 3.0, 4.0, 5.0])
    snap = session.commit("test data")
    ok(f"commit returns snapshot ID: {snap[:16]}...")

    ro = repo.readonly_session(branch="main")
    ok(f"readonly_session(branch='main') returns {type(ro).__name__}")

    arr = zarr.open_array(ro.store, path="test")
    assert list(arr[:]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    ok("Data round-trips through writable/readonly sessions")
except Exception as e:
    fail(f"writable/readonly session: {e}")
    traceback.print_exc()

# ============================================================================
section("5: xarray to_icechunk write + commit + open_zarr read")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds = xr.Dataset({
        "temp": (["x", "y"], np.random.rand(10, 20)),
        "pressure": (["x", "y"], np.random.rand(10, 20)),
    }, coords={"x": np.arange(10), "y": np.arange(20)})

    session = repo.writable_session("main")
    to_icechunk(ds, session, mode="w")
    snap = session.commit("Initial write")
    ok("to_icechunk(ds, session, mode='w') + commit works")

    ro = repo.readonly_session(branch="main")
    ds_read = xr.open_zarr(ro.store, consolidated=False)
    xr.testing.assert_equal(ds, ds_read)
    ok("xr.open_zarr(session.store, consolidated=False) round-trips correctly")
except Exception as e:
    fail(f"xarray to_icechunk: {e}")
    traceback.print_exc()

# ============================================================================
section("6: to_icechunk append_dim")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds1 = xr.Dataset({
        "temp": (["time", "x"], np.random.rand(3, 5)),
    }, coords={"time": [0, 1, 2], "x": np.arange(5)})

    session = repo.writable_session("main")
    to_icechunk(ds1, session, mode="w")
    session.commit("Initial")

    ds2 = xr.Dataset({
        "temp": (["time", "x"], np.random.rand(2, 5)),
    }, coords={"time": [3, 4], "x": np.arange(5)})

    session = repo.writable_session("main")
    to_icechunk(ds2, session, append_dim="time")
    session.commit("Append timestep")

    ro = repo.readonly_session(branch="main")
    ds_full = xr.open_zarr(ro.store, consolidated=False)
    assert ds_full.sizes["time"] == 5, f"Expected time=5, got {ds_full.sizes['time']}"
    ok(f"append_dim='time' works: time dimension is {ds_full.sizes['time']}")
except Exception as e:
    fail(f"append_dim: {e}")
    traceback.print_exc()

# ============================================================================
section("7: readonly_session with tag")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "val", [1.0])
    snap1 = session.commit("v1")

    repo.create_tag("v1.0", snapshot_id=snap1)
    ok("create_tag works")

    session = repo.writable_session("main")
    arr = zarr.open_array(session.store, path="val")
    arr[:] = [2.0]
    session.commit("v2")

    ro_tag = repo.readonly_session(tag="v1.0")
    val = zarr.open_array(ro_tag.store, path="val")[0]
    assert val == 1.0, f"Expected 1.0 from tag, got {val}"
    ok("readonly_session(tag='v1.0') returns data from tagged snapshot")

    ro_main = repo.readonly_session(branch="main")
    val_main = zarr.open_array(ro_main.store, path="val")[0]
    assert val_main == 2.0, f"Expected 2.0 on main, got {val_main}"
    ok("Time travel via tag returns correct historical data")
except Exception as e:
    fail(f"Tag read: {e}")
    traceback.print_exc()

# ============================================================================
section("8: Transaction context manager")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    # Skill: with repo.transaction("main", message="Auto-commit") as store:
    #     zarr.create_array(store, name="data", shape=(10,), dtype=float)
    with repo.transaction("main", message="Auto-commit") as store:
        zarr.create_array(store, name="data", shape=(10,), dtype=float)
    ok("transaction context manager runs without error")

    ro = repo.readonly_session(branch="main")
    arr = zarr.open_array(ro.store, path="data")
    assert arr.shape == (10,)
    ok("Transaction auto-committed: data visible after exiting context")
except Exception as e:
    fail(f"Transaction: {e}")
    traceback.print_exc()

# ============================================================================
section("9: Branching - lookup_branch, create_branch")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "x", [10.0, 20.0, 30.0])
    snap = session.commit("Initial")

    main_snap = repo.lookup_branch("main")
    assert main_snap == snap, "lookup_branch should return latest snapshot"
    ok(f"lookup_branch('main') returns correct snapshot")

    repo.create_branch("experiment", snapshot_id=main_snap)
    ok("create_branch works")

    session_exp = repo.writable_session("experiment")
    arr = zarr.open_array(session_exp.store, path="x")
    arr[:] = [100.0, 200.0, 300.0]
    session_exp.commit("Experiment change")

    ro_main = repo.readonly_session(branch="main")
    ro_exp = repo.readonly_session(branch="experiment")
    assert list(zarr.open_array(ro_main.store, path="x")[:]) == [10.0, 20.0, 30.0]
    assert list(zarr.open_array(ro_exp.store, path="x")[:]) == [100.0, 200.0, 300.0]
    ok("Branches are independent after create_branch")
except Exception as e:
    fail(f"Branching: {e}")
    traceback.print_exc()

# ============================================================================
section("10: Session becomes read-only after commit")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "a", [1.0, 2.0])
    session.commit("First")

    is_ro = session.read_only
    ok(f"session.read_only = {is_ro} after commit")

    # Try to write again on the same session - should fail
    try:
        arr = zarr.open_array(session.store, path="a")
        arr[:] = [99.0, 99.0]
        session.commit("Should fail")
        fail("Writing after commit should raise an error but didn't")
    except Exception as e2:
        ok(f"Writing after commit raises: {type(e2).__name__}: {e2}")

    session2 = repo.writable_session("main")
    arr2 = zarr.open_array(session2.store, path="a")
    arr2[:] = [99.0, 99.0]
    session2.commit("Second write via new session")

    ro = repo.readonly_session(branch="main")
    assert list(zarr.open_array(ro.store, path="a")[:]) == [99.0, 99.0]
    ok("New writable session works for subsequent writes")
except Exception as e:
    fail(f"Session lifecycle: {e}")
    traceback.print_exc()

# ============================================================================
section("11: consolidated=False is mandatory")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds = xr.Dataset({"v": (["x"], np.arange(5, dtype=float))})
    session = repo.writable_session("main")
    to_icechunk(ds, session, mode="w")
    session.commit("data")

    ro = repo.readonly_session(branch="main")
    ds_ok = xr.open_zarr(ro.store, consolidated=False)
    ok("consolidated=False works fine")

    # Test with consolidated=True
    ro2 = repo.readonly_session(branch="main")
    try:
        ds_bad = xr.open_zarr(ro2.store, consolidated=True)
        warn("consolidated=True did NOT raise an error - may silently work or misbehave")
    except Exception as e2:
        ok(f"consolidated=True raises: {type(e2).__name__}: {e2}")

    # Test with default (no consolidated arg)
    ro3 = repo.readonly_session(branch="main")
    try:
        ds_def = xr.open_zarr(ro3.store)
        warn(f"No consolidated arg: xr.open_zarr works (default behavior). Got dims: {dict(ds_def.sizes)}")
    except Exception as e3:
        ok(f"No consolidated arg raises: {type(e3).__name__}: {e3}")
except Exception as e:
    fail(f"consolidated test: {e}")
    traceback.print_exc()

# ============================================================================
section("12: Multiple groups with mode='a'")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds1 = xr.Dataset({"a": (["x"], [1.0, 2.0])})
    ds2 = xr.Dataset({"b": (["x"], [3.0, 4.0])})

    session = repo.writable_session("main")
    ds1.to_zarr(session.store, group="A", zarr_format=3, consolidated=False, mode="w")
    ds2.to_zarr(session.store, group="B", zarr_format=3, consolidated=False, mode="a")
    session.commit("Two groups")

    ro = repo.readonly_session(branch="main")
    ds_a = xr.open_zarr(ro.store, group="A", consolidated=False)
    ds_b = xr.open_zarr(ro.store, group="B", consolidated=False)
    assert list(ds_a["a"].values) == [1.0, 2.0]
    assert list(ds_b["b"].values) == [3.0, 4.0]
    ok("Multiple groups with mode='a' works correctly")

    # Test what happens if you use mode="w" for second group
    storage2 = icechunk.in_memory_storage()
    repo2 = icechunk.Repository.open_or_create(storage2)
    session2 = repo2.writable_session("main")
    ds1.to_zarr(session2.store, group="A", zarr_format=3, consolidated=False, mode="w")
    ds2.to_zarr(session2.store, group="B", zarr_format=3, consolidated=False, mode="w")
    session2.commit("Two groups with mode=w")

    ro2 = repo2.readonly_session(branch="main")
    try:
        ds_a2 = xr.open_zarr(ro2.store, group="A", consolidated=False)
        warn(f"mode='w' for second group did NOT destroy first group. Group A has: {list(ds_a2.data_vars)}")
    except Exception as e2:
        ok(f"mode='w' for second group destroys first: {e2}")
except Exception as e:
    fail(f"Multiple groups: {e}")
    traceback.print_exc()

# ============================================================================
section("13: to_icechunk takes session, not store")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds = xr.Dataset({"v": (["x"], [1.0, 2.0, 3.0])})
    session = repo.writable_session("main")

    try:
        to_icechunk(ds, session.store, mode="w")
        fail("to_icechunk(ds, session.store) should fail but didn't")
    except (TypeError, AttributeError) as e2:
        ok(f"to_icechunk(ds, session.store) raises: {type(e2).__name__}: {e2}")

    to_icechunk(ds, session, mode="w")
    session.commit("Correct usage")
    ok("to_icechunk(ds, session, mode='w') works correctly")
except Exception as e:
    fail(f"to_icechunk session vs store: {e}")
    traceback.print_exc()

# ============================================================================
section("14: Conflict resolution - ConflictError + BasicConflictSolver")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "v", [1.0, 2.0, 3.0])
    session.commit("Initial")

    sa = repo.writable_session("main")
    sb = repo.writable_session("main")

    arr_a = zarr.open_array(sa.store, path="v")
    arr_a[:] = [10.0, 20.0, 30.0]
    sa.commit("A update")

    arr_b = zarr.open_array(sb.store, path="v")
    arr_b[:] = [100.0, 200.0, 300.0]

    try:
        sb.commit("B update")
        fail("Expected ConflictError but commit succeeded")
    except icechunk.ConflictError:
        ok("ConflictError raised on conflicting commit")

        sb.rebase(icechunk.BasicConflictSolver(
            on_chunk_conflict=icechunk.VersionSelection.UseOurs
        ))
        sb.commit("Resolved - B wins")
        ok("rebase + BasicConflictSolver + VersionSelection.UseOurs works")

        ro = repo.readonly_session(branch="main")
        vals = list(zarr.open_array(ro.store, path="v")[:])
        assert vals == [100.0, 200.0, 300.0], f"Expected B's values, got {vals}"
        ok("UseOurs keeps session B's values")
except Exception as e:
    fail(f"Conflict resolution: {e}")
    traceback.print_exc()

# ============================================================================
section("15: ConflictDetector (rebase_with= on commit)")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    # Use larger array with bigger chunks to ensure different elements are in different chunks
    session = repo.writable_session("main")
    data = np.zeros(200)
    create_with_data(session.store, "v", data)
    session.commit("Initial")

    sa = repo.writable_session("main")
    sb = repo.writable_session("main")

    arr_a = zarr.open_array(sa.store, path="v")
    arr_a[0] = 10.0
    sa.commit("A writes element 0")

    arr_b = zarr.open_array(sb.store, path="v")
    arr_b[199] = 30.0

    try:
        sb.commit("B writes element 199", rebase_with=icechunk.ConflictDetector())
        ok("ConflictDetector auto-rebases non-conflicting writes")
    except Exception as e2:
        warn(f"ConflictDetector raised: {type(e2).__name__}: {e2} (chunks may overlap)")
except Exception as e:
    fail(f"ConflictDetector: {e}")
    traceback.print_exc()

# ============================================================================
section("16: NoChangesToCommitError")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "x", [1.0, 2.0])
    session.commit("Setup")

    session2 = repo.writable_session("main")
    try:
        session2.commit("No changes")
        fail("Expected error on empty commit but succeeded")
    except Exception as e2:
        ename = type(e2).__name__
        ok(f"Empty commit raises: {ename}: {e2}")
        if hasattr(icechunk, 'NoChangesToCommitError'):
            ok("icechunk.NoChangesToCommitError exists as an attribute")
        else:
            warn("icechunk.NoChangesToCommitError not found as module attribute - skill claims it exists")
except Exception as e:
    fail(f"NoChangesToCommitError: {e}")
    traceback.print_exc()

# ============================================================================
section("17: Dask writes (fork, store_dask, merge)")
# ============================================================================
try:
    import dask.array as da

    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    # Skill step 1: Initialize array structure first
    # Skill uses path= but zarr.create_array uses name= in zarr 3
    session = repo.writable_session("main")
    zarr_arr = zarr.create_array(session.store, name="data",
        shape=(100, 100), chunks=(10, 10), dtype="f8")
    session.commit("Initialize array")

    # Skill step 2: Fork session for distributed writes
    session = repo.writable_session("main")
    fork = session.fork()
    zarr_arr = zarr.open_array(fork.store, path="data")

    # Skill step 3: Write dask array through fork
    dask_arr = da.random.random((100, 100), chunks=(20, 20))
    remote_session = store_dask(sources=[dask_arr], targets=[zarr_arr])

    # Skill step 4: Merge and commit
    session.merge(remote_session)
    session.commit("Wrote dask array")
    ok("Dask fork/store_dask/merge/commit pattern works")

    ro = repo.readonly_session(branch="main")
    arr = zarr.open_array(ro.store, path="data")
    assert arr.shape == (100, 100)
    assert np.all(arr[:] >= 0) and np.all(arr[:] <= 1)
    ok("Dask-written data is correct")
except Exception as e:
    fail(f"Dask writes: {e}")
    traceback.print_exc()

# ============================================================================
section("18: to_zarr works for non-dask data")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds = xr.Dataset({"v": (["x"], np.arange(5, dtype=float))})
    session = repo.writable_session("main")
    ds.to_zarr(session.store, zarr_format=3, consolidated=False)
    session.commit("to_zarr write")

    ro = repo.readonly_session(branch="main")
    ds_read = xr.open_zarr(ro.store, consolidated=False)
    xr.testing.assert_equal(ds, ds_read)
    ok("to_zarr works for non-dask data with zarr_format=3, consolidated=False")
except Exception as e:
    fail(f"to_zarr non-dask: {e}")
    traceback.print_exc()

# ============================================================================
section("19: RepositoryConfig (unsafe mode for S3-compatible stores)")
# ============================================================================
try:
    if hasattr(icechunk, 'RepositoryConfig'):
        ok("icechunk.RepositoryConfig exists")
    else:
        fail("icechunk.RepositoryConfig does not exist")

    if hasattr(icechunk, 'StorageSettings'):
        ok("icechunk.StorageSettings exists")
    else:
        fail("icechunk.StorageSettings does not exist")

    try:
        cfg = icechunk.RepositoryConfig()
        ok(f"RepositoryConfig() instantiates")
    except Exception as e2:
        warn(f"RepositoryConfig() failed: {e2}")
except Exception as e:
    fail(f"RepositoryConfig: {e}")
    traceback.print_exc()

# ============================================================================
section("20: Verify skill file API claims exist")
# ============================================================================
apis_to_check = [
    ("icechunk", "in_memory_storage"),
    ("icechunk", "Repository"),
    ("icechunk", "ConflictError"),
    ("icechunk", "BasicConflictSolver"),
    ("icechunk", "VersionSelection"),
    ("icechunk", "ConflictDetector"),
]

for mod_name, attr_name in apis_to_check:
    if hasattr(icechunk, attr_name):
        ok(f"{mod_name}.{attr_name} exists")
    else:
        fail(f"{mod_name}.{attr_name} does NOT exist")

try:
    vs = icechunk.VersionSelection.UseOurs
    ok(f"VersionSelection.UseOurs = {vs}")
except AttributeError:
    fail("VersionSelection.UseOurs does not exist")

# ============================================================================
section("21: Verify RepositoryConfig/StorageSettings skill example")
# ============================================================================
try:
    # The skill shows:
    # config = icechunk.RepositoryConfig(
    #     storage=icechunk.StorageSettings(
    #         unsafe_use_conditional_update=False,
    #         unsafe_use_conditional_create=False,
    #     )
    # )
    try:
        ss = icechunk.StorageSettings(
            unsafe_use_conditional_update=False,
            unsafe_use_conditional_create=False,
        )
        ok(f"StorageSettings with unsafe flags works")
    except TypeError as te:
        fail(f"StorageSettings constructor: {te}")
    except Exception as e2:
        fail(f"StorageSettings: {type(e2).__name__}: {e2}")

    try:
        cfg = icechunk.RepositoryConfig(
            storage=icechunk.StorageSettings(
                unsafe_use_conditional_update=False,
                unsafe_use_conditional_create=False,
            )
        )
        ok(f"RepositoryConfig(storage=StorageSettings(...)) works")
    except TypeError as te:
        fail(f"RepositoryConfig(storage=StorageSettings(...)): {te}")
    except Exception as e2:
        fail(f"RepositoryConfig: {type(e2).__name__}: {e2}")
except Exception as e:
    fail(f"Config verification: {e}")
    traceback.print_exc()

# ============================================================================
section("22: Transaction yields store (not session)")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    with repo.transaction("main", message="Type check") as obj:
        obj_type = type(obj).__name__
        print(f"  transaction yields: {obj_type} ({type(obj)})")
        if "Store" in obj_type or "store" in obj_type.lower():
            ok(f"Transaction yields a store-like object: {obj_type}")
        elif "Session" in obj_type or "session" in obj_type.lower():
            fail(f"Transaction yields a Session (skill says store): {obj_type}")
        else:
            warn(f"Transaction yields unexpected type: {obj_type}")

        zarr.create_array(obj, name="check", shape=(2,), dtype="f8")

    ro = repo.readonly_session(branch="main")
    arr = zarr.open_array(ro.store, path="check")
    assert arr.shape == (2,)
    ok("Data from transaction is committed and readable")
except Exception as e:
    fail(f"Transaction type check: {e}")
    traceback.print_exc()

# ============================================================================
section("23: to_icechunk with dask-backed data (no manual fork needed)")
# ============================================================================
try:
    import dask.array as da

    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    # First write non-dask data to establish the repo
    ds_init = xr.Dataset({
        "temp": (["x", "y"], np.random.rand(10, 20)),
    }, coords={"x": np.arange(10), "y": np.arange(20)})
    session = repo.writable_session("main")
    to_icechunk(ds_init, session, mode="w")
    session.commit("Initial non-dask write")

    # Now overwrite with dask-backed data
    ds_dask = xr.Dataset({
        "temp": (["x", "y"], da.random.random((10, 20), chunks=(5, 10))),
    }, coords={"x": np.arange(10), "y": np.arange(20)})

    session = repo.writable_session("main")
    to_icechunk(ds_dask, session, mode="w")
    session.commit("Dask via to_icechunk")
    ok("to_icechunk handles dask-backed data internally (no manual fork needed)")

    ro = repo.readonly_session(branch="main")
    ds_read = xr.open_zarr(ro.store, consolidated=False)
    assert ds_read.sizes == {"x": 10, "y": 20}
    ok("Dask-written xarray data reads back correctly")
except Exception as e:
    fail(f"to_icechunk with dask: {e}")
    traceback.print_exc()

# ============================================================================
section("24: repo.list_branches() / repo.list_tags() / lookup_tag()")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "x", [1.0, 2.0])
    snap = session.commit("Initial")

    repo.create_branch("dev", snapshot_id=snap)
    repo.create_tag("v1", snapshot_id=snap)

    if hasattr(repo, 'list_branches'):
        branches = repo.list_branches()
        ok(f"list_branches() = {branches}")
    else:
        warn("repo.list_branches() does not exist")

    if hasattr(repo, 'list_tags'):
        tags = repo.list_tags()
        ok(f"list_tags() = {tags}")
    else:
        warn("repo.list_tags() does not exist")

    if hasattr(repo, 'lookup_tag'):
        t = repo.lookup_tag("v1")
        assert t == snap
        ok(f"lookup_tag('v1') returns correct snapshot")
    else:
        warn("repo.lookup_tag() does not exist")
except Exception as e:
    fail(f"list_branches/list_tags: {e}")
    traceback.print_exc()

# ============================================================================
section("25: repo.ancestry()")
# ============================================================================
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "x", [1.0, 2.0])
    session.commit("First")

    session = repo.writable_session("main")
    arr = zarr.open_array(session.store, path="x")
    arr[:] = [3.0, 4.0]
    session.commit("Second")

    if hasattr(repo, 'ancestry'):
        ancestors = list(repo.ancestry(branch="main"))
        ok(f"repo.ancestry() works, {len(ancestors)} ancestors found")
        for a in ancestors:
            print(f"    {a.id[:16]}... : {a.message}")
    else:
        warn("repo.ancestry() does not exist (not in skill file anyway)")
except Exception as e:
    warn(f"ancestry: {e}")

# ============================================================================
section("26: Skill file Dask example uses path= vs name= (zarr v3)")
# ============================================================================
# The skill file shows:
#   zarr_arr = zarr.create_array(session.store, path="data", ...)
# But zarr v3 API uses name= not path=. Let's verify.
try:
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)
    session = repo.writable_session("main")

    try:
        zarr.create_array(session.store, path="data", shape=(10,), dtype="f8")
        fail("zarr.create_array with path= should fail (zarr v3 uses name=)")
    except TypeError as e2:
        ok(f"zarr.create_array rejects path=: {e2}")
        warn("Skill file Dask example uses path= but zarr v3 requires name=")
except Exception as e:
    fail(f"path= vs name= check: {e}")
    traceback.print_exc()

# ============================================================================
section("27: to_icechunk dask with mode='w' on empty repo")
# ============================================================================
# Test if to_icechunk with dask data works on a totally fresh empty repo
try:
    import dask.array as da

    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    ds_dask = xr.Dataset({
        "temp": (["x", "y"], da.random.random((10, 20), chunks=(5, 10))),
    }, coords={"x": np.arange(10), "y": np.arange(20)})

    session = repo.writable_session("main")
    try:
        to_icechunk(ds_dask, session, mode="w")
        session.commit("Fresh dask write")
        ok("to_icechunk with dask on fresh/empty repo works")
    except Exception as e2:
        warn(f"to_icechunk with dask on fresh/empty repo fails: {type(e2).__name__}: {e2}")
        warn("Workaround: write non-dask data first, or use non-dask initial commit")
except Exception as e:
    fail(f"Dask empty repo test: {e}")
    traceback.print_exc()

# ============================================================================
section("28: Skill says 'commit raises NoChangesToCommitError' - verify name")
# ============================================================================
try:
    # The skill file mentions this error by name. Let's check what actually gets raised.
    storage = icechunk.in_memory_storage()
    repo = icechunk.Repository.open_or_create(storage)

    session = repo.writable_session("main")
    create_with_data(session.store, "x", [1.0])
    session.commit("Setup")

    session2 = repo.writable_session("main")
    try:
        session2.commit("Empty")
    except Exception as e2:
        actual_name = type(e2).__name__
        actual_module = type(e2).__module__ if hasattr(type(e2), '__module__') else 'unknown'
        ok(f"Empty commit error type: {actual_module}.{actual_name}")

        # Check if it's actually called NoChangesToCommitError
        if actual_name == "NoChangesToCommitError":
            ok("Error name matches skill file claim: NoChangesToCommitError")
        else:
            warn(f"Skill file says 'NoChangesToCommitError' but actual error is '{actual_name}'")
except Exception as e:
    fail(f"Error name check: {e}")
    traceback.print_exc()


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY: {PASS} passed, {FAIL} failed")
print(f"{'='*70}")
if ISSUES:
    print("  Issues found:")
    for issue in ISSUES:
        print(f"    - {issue}")
print()
