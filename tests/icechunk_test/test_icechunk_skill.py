"""
Test script that exercises icechunk features using ONLY the skill file as a guide.
Each section maps to a specific feature documented in skills/icechunk.md.
"""

import numpy as np
import traceback

# ============================================================================
# Section 1: Create a repository using in-memory storage
# ============================================================================
print("=" * 70)
print("SECTION 1: Create a repository using in-memory storage")
print("=" * 70)

try:
    import icechunk
    import zarr

    # From skill: icechunk.in_memory_storage() for testing
    storage = icechunk.in_memory_storage()
    # From skill: Repository.create(storage)
    repo = icechunk.Repository.create(storage)

    # From skill: repo.writable_session("main")
    session = repo.writable_session("main")

    # From skill: zarr.create_group(session.store)
    root = zarr.create_group(session.store)
    arr = root.create_array("temperature", shape=(100, 100), chunks=(10, 10), dtype="f8")
    arr[:] = np.random.rand(100, 100)

    # From skill: session.commit(message)
    snapshot_id = session.commit("Create temperature array")
    print(f"  PASS: Created repo and committed. Snapshot ID: {snapshot_id[:16]}...")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 2: Branching workflow
# ============================================================================
print()
print("=" * 70)
print("SECTION 2: Branching workflow")
print("=" * 70)

try:
    # From skill: repo.lookup_branch("main") returns snapshot_id
    main_snap = repo.lookup_branch("main")
    # From skill: repo.create_branch(name, snapshot_id)
    repo.create_branch("dev", snapshot_id=main_snap)
    print(f"  Created 'dev' branch from main snapshot: {main_snap[:16]}...")

    # Write data on dev branch
    session_dev = repo.writable_session("dev")
    store_dev = session_dev.store
    dev_arr = zarr.open_array(store_dev, path="temperature")
    dev_arr[0, 0] = -999.0
    session_dev.commit("Set sentinel value on dev")
    print("  Committed change on dev branch")

    # Write different data on main branch
    session_main = repo.writable_session("main")
    store_main = session_main.store
    main_arr = zarr.open_array(store_main, path="temperature")
    main_arr[0, 0] = 42.0
    session_main.commit("Set different value on main")
    print("  Committed change on main branch")

    # Read back from each branch and verify they differ
    ro_dev = repo.readonly_session(branch="dev")
    ro_main = repo.readonly_session(branch="main")
    val_dev = zarr.open_array(ro_dev.store, path="temperature")[0, 0]
    val_main = zarr.open_array(ro_main.store, path="temperature")[0, 0]

    assert val_dev == -999.0, f"Expected -999.0 on dev, got {val_dev}"
    assert val_main == 42.0, f"Expected 42.0 on main, got {val_main}"
    print(f"  PASS: dev[0,0]={val_dev}, main[0,0]={val_main} - branches diverge correctly")

    # From skill: repo.list_branches()
    branches = repo.list_branches()
    print(f"  Branches: {branches}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 3: Tagging
# ============================================================================
print()
print("=" * 70)
print("SECTION 3: Tagging")
print("=" * 70)

try:
    # Tag the initial commit (the very first snapshot_id from section 1)
    # From skill: repo.create_tag(name, snapshot_id)
    repo.create_tag("v1.0-initial", snapshot_id=snapshot_id)
    print(f"  Created tag 'v1.0-initial' pointing to {snapshot_id[:16]}...")

    # From skill: repo.readonly_session(tag="v1.0-release")
    tagged_session = repo.readonly_session(tag="v1.0-initial")
    tagged_arr = zarr.open_array(tagged_session.store, path="temperature")
    val_tagged = tagged_arr[0, 0]

    # The original data was random, but it was NOT -999 or 42 (those were written later)
    assert val_tagged != -999.0 and val_tagged != 42.0, \
        f"Expected original random value, got {val_tagged}"
    print(f"  PASS: Tagged snapshot has original value {val_tagged} (not -999 or 42)")

    # From skill: repo.list_tags()
    tags = repo.list_tags()
    print(f"  Tags: {tags}")

    # From skill: repo.lookup_tag(name) returns snapshot_id
    tag_snap = repo.lookup_tag("v1.0-initial")
    assert tag_snap == snapshot_id, "Tag snapshot ID mismatch"
    print(f"  PASS: lookup_tag returned correct snapshot ID")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 4: History browsing
# ============================================================================
print()
print("=" * 70)
print("SECTION 4: History browsing (ancestry)")
print("=" * 70)

try:
    # From skill: repo.ancestry(branch="main") -> Iterator[SnapshotInfo]
    # From skill: ancestor.id, ancestor.message, ancestor.written_at
    print("  Ancestry of main branch:")
    for ancestor in repo.ancestry(branch="main"):
        print(f"    {ancestor.id[:16]}... : {ancestor.message} (at {ancestor.written_at})")
    print("  PASS: Ancestry listing works")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 5: Conflict resolution
# ============================================================================
print()
print("=" * 70)
print("SECTION 5: Conflict resolution")
print("=" * 70)

try:
    # Create two writable sessions on the same branch
    session_a = repo.writable_session("main")
    session_b = repo.writable_session("main")

    # Write to overlapping regions
    arr_a = zarr.open_array(session_a.store, path="temperature")
    arr_a[0, 0] = 100.0

    arr_b = zarr.open_array(session_b.store, path="temperature")
    arr_b[0, 0] = 200.0

    # Commit session A first
    session_a.commit("Session A update")
    print("  Session A committed successfully")

    # Attempt to commit session B - should get ConflictError
    # From skill: except icechunk.ConflictError
    try:
        session_b.commit("Session B update")
        print("  WARNING: Session B committed without conflict (unexpected)")
    except icechunk.ConflictError as ce:
        print(f"  Got expected ConflictError: {ce}")

        # From skill: session.rebase(icechunk.BasicConflictSolver(...))
        session_b.rebase(icechunk.BasicConflictSolver(
            on_chunk_conflict=icechunk.VersionSelection.UseOurs
        ))
        session_b.commit("Resolved conflicts, session B wins")
        print("  PASS: Rebased with BasicConflictSolver and committed")

        # Verify session B's value won
        ro = repo.readonly_session(branch="main")
        val = zarr.open_array(ro.store, path="temperature")[0, 0]
        assert val == 200.0, f"Expected 200.0 (ours), got {val}"
        print(f"  PASS: After rebase with UseOurs, value is {val}")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 6: Transaction context manager
# ============================================================================
print()
print("=" * 70)
print("SECTION 6: Transaction context manager")
print("=" * 70)

try:
    # From skill: repo.transaction(branch, message=) -> ContextManager[IcechunkStore]
    # From skill: with repo.transaction("main", message="Auto-commit") as store:
    #   zarr.create_array(store, "data", shape=(10,), dtype=float)
    # NOTE: The skill file example passes "data" as positional arg to zarr.create_array,
    # but zarr 3 requires name= as keyword arg. Testing with corrected syntax:
    with repo.transaction("main", message="Auto-commit via transaction") as store:
        zarr.create_array(store, name="new_data", shape=(10,), dtype=float)
        # Skill says this auto-commits on exit

    # Verify the data was committed by reading from a readonly session
    ro = repo.readonly_session(branch="main")
    new_arr = zarr.open_array(ro.store, path="new_data")
    print(f"  new_data shape: {new_arr.shape}, dtype: {new_arr.dtype}")
    print("  PASS: Transaction context manager auto-committed")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 7: Xarray integration
# ============================================================================
print()
print("=" * 70)
print("SECTION 7: Xarray integration (round-trip)")
print("=" * 70)

try:
    import xarray as xr
    from icechunk.xarray import to_icechunk

    # Create a fresh repo for xarray test to avoid conflicts with zarr-created data
    xr_storage = icechunk.in_memory_storage()
    xr_repo = icechunk.Repository.create(xr_storage)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "temperature": (["x", "y"], np.random.rand(10, 20).astype("float64")),
            "pressure": (["x", "y"], np.random.rand(10, 20).astype("float64")),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(20),
        },
    )
    print(f"  Original dataset: {ds}")

    # From skill: to_icechunk(ds, session, mode="w")
    session = xr_repo.writable_session("main")
    to_icechunk(ds, session, mode="w")
    xr_snap = session.commit("Write xarray dataset")
    print(f"  Committed xarray dataset. Snapshot: {xr_snap[:16]}...")

    # From skill: xr.open_zarr(session.store, consolidated=False)
    ro_session = xr_repo.readonly_session(branch="main")
    ds_read = xr.open_zarr(ro_session.store, consolidated=False)
    print(f"  Read-back dataset: {ds_read}")

    # Verify round-trip
    xr.testing.assert_equal(ds, ds_read)
    print("  PASS: Xarray round-trip verified (assert_equal passed)")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Section 8: Session lifecycle (read-only after commit)
# ============================================================================
print()
print("=" * 70)
print("SECTION 8: Session lifecycle (read-only after commit)")
print("=" * 70)

try:
    # From skill: "session is read-only after commit"
    lifecycle_storage = icechunk.in_memory_storage()
    lifecycle_repo = icechunk.Repository.create(lifecycle_storage)

    session = lifecycle_repo.writable_session("main")
    root = zarr.create_group(session.store)
    arr = root.create_array("data", shape=(5,), dtype="f8")
    arr[:] = [1.0, 2.0, 3.0, 4.0, 5.0]
    snap1 = session.commit("First write")
    print(f"  First commit: {snap1[:16]}...")

    # From skill: session.read_only property
    print(f"  Session read_only after commit: {session.read_only}")

    # Try to write again on the same session - should fail
    try:
        arr2 = zarr.open_array(session.store, path="data")
        arr2[:] = [10.0, 20.0, 30.0, 40.0, 50.0]
        session.commit("Second write on same session")
        print("  WARNING: Writing on committed session unexpectedly succeeded")
    except Exception as e2:
        print(f"  Expected error after reusing committed session: {type(e2).__name__}: {e2}")

    # From skill: Create a new writable session for subsequent writes
    session2 = lifecycle_repo.writable_session("main")
    arr3 = zarr.open_array(session2.store, path="data")
    arr3[:] = [10.0, 20.0, 30.0, 40.0, 50.0]
    snap2 = session2.commit("Second write with new session")
    print(f"  Second commit (new session): {snap2[:16]}...")

    # Verify the new data
    ro = lifecycle_repo.readonly_session(branch="main")
    final_arr = zarr.open_array(ro.store, path="data")
    assert list(final_arr[:]) == [10.0, 20.0, 30.0, 40.0, 50.0], \
        f"Expected [10,20,30,40,50], got {list(final_arr[:])}"
    print("  PASS: Session lifecycle works correctly")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 70)
print("ALL SECTIONS COMPLETE")
print("=" * 70)
