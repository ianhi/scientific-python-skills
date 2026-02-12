# Icechunk

> Transactional storage engine for Zarr with git-like version control. Data is lost without `commit()`. Sessions become read-only after commit. Use `to_icechunk` (not `to_zarr`) for dask-backed data. Always pass `consolidated=False` when reading.

## Common Operations (Copy-Paste Ready)

```python
import icechunk
import zarr
import xarray as xr
from icechunk.xarray import to_icechunk  # for xarray writes
from icechunk.dask import store_dask      # for dask array writes

# --- Create/open repo ---
storage = icechunk.in_memory_storage()  # for testing
repo = icechunk.Repository.open_or_create(storage)  # idempotent (preferred)

# --- Write with xarray ---
session = repo.writable_session("main")
to_icechunk(ds, session, mode="w")
snapshot_id = session.commit("Initial write")

# --- Read with xarray ---
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, consolidated=False)

# --- Append data ---
session = repo.writable_session("main")
to_icechunk(ds_new, session, append_dim="time")
session.commit("Append timestep")

# --- Time travel ---
session = repo.readonly_session(tag="v1.0")
ds_old = xr.open_zarr(session.store, consolidated=False)

# --- Branching ---
main_snap = repo.lookup_branch("main")
repo.create_branch("experiment", snapshot_id=main_snap)
session = repo.writable_session("experiment")

# --- Tagging ---
snapshot_id = session.commit("Release v2.0")
repo.create_tag("v2.0-release", snapshot_id=snapshot_id)

# --- Transaction context manager (auto-commits) ---
with repo.transaction("main", message="Auto-commit") as store:
    zarr.create_array(store, name="data", shape=(10,), dtype=float)
```

## Dask/Distributed Writes

```python
from icechunk.dask import store_dask

# 1. Initialize array structure first
session = repo.writable_session("main")
zarr_arr = zarr.create_array(session.store, path="data",
    shape=(100, 100), chunks=(10, 10), dtype="f8")
session.commit("Initialize array")

# 2. Fork session for distributed writes
session = repo.writable_session("main")
fork = session.fork()
zarr_arr = zarr.open_array(fork.store, path="data")

# 3. Write dask array through fork
dask_arr = da.random.random((100, 100), chunks=(20, 20))
remote_session = store_dask(sources=[dask_arr], targets=[zarr_arr])

# 4. Merge and commit
session.merge(remote_session)
session.commit("Wrote dask array")
```

## Common Mistakes & Gotchas

### Always commit after writes (#619)
Data is invisible to other sessions until committed.
```python
session = repo.writable_session("main")
to_icechunk(ds, session, mode="w")
session.commit("Write dataset")  # Required - data is lost without this
```

### Session becomes read-only after commit
Create a new writable session for subsequent writes:
```python
session = repo.writable_session("main")
to_icechunk(ds1, session, mode="w")
session.commit("First write")
session = repo.writable_session("main")  # New session required
to_icechunk(ds2, session, append_dim="time")
session.commit("Second write")
```

### to_icechunk with uncommitted changes + dask raises ValueError
If the session has uncommitted changes AND the dataset contains dask arrays, `to_icechunk` raises `ValueError`. Commit or discard changes first.

### Use to_icechunk for dask-backed data
`to_zarr` works for non-distributed data. It breaks with dask because writable sessions cannot be pickled.
```python
# Dask-backed: use to_icechunk (handles forking internally)
ds_dask = ds.chunk({"time": 1})
to_icechunk(ds_dask, session, mode="w")

# Non-dask: to_zarr is fine
ds.to_zarr(session.store, zarr_format=3, consolidated=False)
```

### consolidated=False is mandatory (#962)
Icechunk manages metadata internally. `consolidated=True` writes a key that hides new groups.
```python
ds = xr.open_zarr(session.store, consolidated=False)  # Always use False
```

### Multiple groups require mode="a" (#176)
```python
ds1.to_zarr(session.store, group="A", zarr_format=3, consolidated=False, mode="w")
ds2.to_zarr(session.store, group="B", zarr_format=3, consolidated=False, mode="a")  # "a" not "w"
```

### Use open_or_create for idempotency (#665)
```python
repo = icechunk.Repository.create(storage)           # Fails if exists
repo = icechunk.Repository.open_or_create(storage)   # Safe - creates or opens
```

### to_icechunk takes a session, not a store
```python
to_icechunk(ds, session, mode="w")  # NOT session.store
```

### Virtual chunks require explicit authorization (#1032, #1545)
Opening a repo with virtual chunks without `authorize_virtual_chunk_access` gives cryptic `AccessDenied` errors at read time.
```python
repo = icechunk.Repository.open(
    storage=storage,
    authorize_virtual_chunk_access=credentials  # Required
)
```

### S3-compatible stores may need unsafe mode (#743)
Object stores without conditional write support fail on `Repository.create`:
```python
config = icechunk.RepositoryConfig(
    storage=icechunk.StorageSettings(
        unsafe_use_conditional_update=False,
        unsafe_use_conditional_create=False,
    )
)
repo = icechunk.Repository.create(storage, config=config)
```

### SyncError in async contexts (#1228)
Zarr 3 uses async internally. Inside an existing event loop (Flask, some Jupyter setups), you get `SyncError: Calling sync() from within a running loop`. Use `nest_asyncio` or restructure.

### Conflict resolution
```python
try:
    session.commit("Update data")
except icechunk.ConflictError:
    session.rebase(icechunk.BasicConflictSolver(
        on_chunk_conflict=icechunk.VersionSelection.UseOurs
    ))
    session.commit("Resolved conflicts")

# Or auto-rebase on commit:
session.commit("Update", rebase_with=icechunk.ConflictDetector())
```

### Known fixed issues (upgrade if encountering)

| Issue | Symptom | Fixed In |
|-------|---------|----------|
| #1087 | Ceph backends segfault on commit | Recent |
| #1181 | Dask deserialization errors with virtual chunks | >= 1.1.4 |
| #1463 | S3 virtual references with spaces cause NoSuchKey | Recent |

## Known Limitations

- **Local filesystem storage**: Not safe for concurrent commits; not on Windows (#665)
- **Zarr sharding on writes**: Must set `zarr.config.set({"async.concurrency": 1})`
- **write_empty_chunks** (#1458): Not yet supported in `to_icechunk`
- **Branch merging**: Not supported (only rebase on same branch)
- **Tags**: Cannot be recreated once deleted (unlike Git)
- **Rebase**: Only resolves chunk conflicts; metadata conflicts need manual resolution
- **GCS**: Bearer token refresh not yet supported
- **Object stores without conditional writes**: Lose ACID consistency with concurrent writers (#743)
