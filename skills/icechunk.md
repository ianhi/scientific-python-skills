# Icechunk - Claude Code Skill

> Skill for writing idiomatic icechunk code. Version: 1.1.x (stable, production-ready)

Icechunk is a transactional storage engine for Zarr with git-like version control. It adds ACID transactions, branching, tagging, and snapshot history to multidimensional array data in cloud object storage.

## CRITICAL: Do NOT Use These Outdated/Broken Patterns

### 1. Forgetting to commit after writes (#619)

The #1 most common mistake. Data is invisible to other sessions until committed.

```python
# WRONG - data written but never committed, invisible to readers
session = repo.writable_session("main")
ds.to_zarr(session.store, zarr_format=3, consolidated=False)
# session goes out of scope, data is lost

# RIGHT - always commit
session = repo.writable_session("main")
ds.to_zarr(session.store, zarr_format=3, consolidated=False)
session.commit("Write dataset")
```

### 2. Using to_zarr with distributed/dask writers

`to_zarr` works fine for non-distributed (in-memory) data. It only breaks with dask/distributed because writable sessions cannot be pickled.

```python
# WRONG - to_zarr cannot handle distributed Dask writers (session is not picklable)
ds_dask = ds.chunk({"time": 1})  # dask-backed
ds_dask.to_zarr(session.store, zarr_format=3, consolidated=False)  # FAILS

# RIGHT - use to_icechunk for dask-backed data (handles forking internally)
from icechunk.xarray import to_icechunk
to_icechunk(ds_dask, session, mode="w")
session.commit("Write dataset")

# ALSO FINE - to_zarr with non-dask data
ds.to_zarr(session.store, zarr_format=3, consolidated=False)
session.commit("Write dataset")
```

### 3. Reusing a session after commit

```python
# WRONG - session is read-only after commit
session = repo.writable_session("main")
to_icechunk(ds1, session, mode="w")
session.commit("First write")
to_icechunk(ds2, session, append_dim="time")  # FAILS

# RIGHT - create a new writable session
session = repo.writable_session("main")
to_icechunk(ds1, session, mode="w")
session.commit("First write")
session = repo.writable_session("main")  # new session
to_icechunk(ds2, session, append_dim="time")
session.commit("Second write")
```

### 4. Using consolidated=True with xarray (#962)

```python
# WRONG - consolidated metadata breaks Icechunk's metadata management
ds = xr.open_zarr(session.store, consolidated=True)
# Also WRONG - consolidated metadata on write hides new groups
ds.to_zarr(session.store, consolidated=True)

# RIGHT - Icechunk handles metadata consolidation internally
ds = xr.open_zarr(session.store, consolidated=False)
```

### 5. Using Repository.create on existing repo (#665)

```python
# WRONG - fails with "repositories can only be created in clean prefixes"
repo = icechunk.Repository.create(storage)  # prefix already has a repo

# RIGHT - use open for existing repos
repo = icechunk.Repository.open(storage)

# BEST - use open_or_create (creates if not exists, opens if exists)
repo = icechunk.Repository.open_or_create(storage)
```

### 6. Passing store instead of session to to_icechunk

```python
# WRONG - to_icechunk takes a session, not a store
to_icechunk(ds, session.store, mode="w")

# RIGHT
to_icechunk(ds, session, mode="w")
```

### 7. Using old async-first API (pre-1.0)

The pre-1.0 API used `IcechunkStore.create(StorageConfig.filesystem(...), mode="w")` directly. In 1.0+, use `Repository.create(storage)` then `repo.writable_session("main").store`.

## Quick Reference

### Installation & Import

```python
# pip install icechunk
import icechunk
import zarr
import xarray as xr
from icechunk.xarray import to_icechunk  # for xarray writes
from icechunk.dask import store_dask      # for dask array writes
```

### Common Operations (Copy-Paste Ready)

```python
# --- Create new repo ---
storage = icechunk.in_memory_storage()  # for testing
repo = icechunk.Repository.create(storage)

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

# --- Transaction context manager (auto-commits) ---
with repo.transaction("main", message="Auto-commit") as store:
    zarr.create_array(store, name="data", shape=(10,), dtype=float)
```

## Core API

### Repository

```python
# Create / Open
Repository.create(storage, config=None, authorize_virtual_chunk_access=None) -> Repository
Repository.open(storage, config=None, authorize_virtual_chunk_access=None) -> Repository
Repository.open_or_create(storage, config=None, authorize_virtual_chunk_access=None) -> Repository

# Sessions
repo.writable_session(branch: str) -> Session
repo.readonly_session(branch=None, snapshot_id=None, tag=None) -> Session
repo.transaction(branch: str, *, message: str) -> ContextManager[IcechunkStore]

# Version control
repo.ancestry(branch=None, snapshot_id=None, tag=None) -> Iterator[SnapshotInfo]
repo.list_branches() -> set[str]
repo.lookup_branch(name: str) -> str  # returns snapshot_id
repo.create_branch(name: str, snapshot_id: str)
repo.delete_branch(name: str)
repo.reset_branch(name: str, snapshot_id: str)  # destructive!
repo.list_tags() -> set[str]
repo.lookup_tag(name: str) -> str  # returns snapshot_id
repo.create_tag(name: str, snapshot_id: str)
repo.delete_tag(name: str)  # cannot recreate once deleted

# Maintenance
repo.expire_snapshots(older_than: datetime) -> set[str]
repo.garbage_collect(delete_object_older_than: datetime, *, dry_run=False) -> GCSummary
repo.save_config()  # saves current config; use repo.reopen(config=new_config) to change config first

# Config
Repository.fetch_config(storage) -> RepositoryConfig  # static method
```

### Session

```python
session.commit(message: str, metadata=None, rebase_with=None) -> str  # snapshot_id
# Raises NoChangesToCommitError if no changes (use allow_empty=True to override)
session.amend(message: str, metadata=None) -> str  # amend/overwrite previous commit
session.status() -> Diff
session.discard_changes()
session.rebase(rebase_with: ConflictSolver)
session.fork() -> ForkSession  # for distributed writes
session.merge(*sessions)       # merge forked sessions back
session.flush(message: str, metadata=None) -> str  # snapshot without updating branch

# Properties
session.store -> IcechunkStore
session.read_only -> bool
session.snapshot_id -> str
session.branch -> str
session.has_uncommitted_changes -> bool

# Note: all methods have _async counterparts (e.g., commit_async, open_async, flush_async)
```

### Storage Backends

```python
# S3
storage = icechunk.s3_storage(
    bucket="my-bucket", prefix="my-prefix",
    from_env=True  # use EC2/env credentials
)
# S3 with explicit creds
storage = icechunk.s3_storage(
    bucket="...", prefix="...",
    region="us-east-1", access_key_id="...", secret_access_key="..."
)
# S3-compatible (MinIO, Ceph)
storage = icechunk.s3_storage(
    bucket="...", prefix="...",
    endpoint_url="http://localhost:9000",
    allow_http=True, force_path_style=True, from_env=True
)

# GCS
storage = icechunk.gcs_storage(bucket="...", prefix="...", from_env=True)

# Azure
storage = icechunk.azure_storage(account="...", container="...", prefix="...", from_env=True)

# Local filesystem (NOT safe for concurrent commits, NOT supported on Windows)
storage = icechunk.local_filesystem_storage("/path/to/repo")

# In-memory (testing only)
storage = icechunk.in_memory_storage()

# Cloudflare R2
storage = icechunk.r2_storage(bucket="...", prefix="...",
    access_key_id="...", secret_access_key="...", account_id="...")

# Tigris (special consistency guarantees)
storage = icechunk.tigris_storage(...)
```

## Patterns & Idioms

### Write with Zarr Directly

```python
session = repo.writable_session("main")
root = zarr.create_group(session.store)
arr = root.create_array("temperature", shape=(100, 100), chunks=(10, 10), dtype="f8")
arr[:] = data
session.commit("Create temperature array")
```

### Branching Workflow

```python
# Create feature branch from current main
main_snap = repo.lookup_branch("main")
repo.create_branch("experiment", snapshot_id=main_snap)

# Work on branch
session = repo.writable_session("experiment")
# ... modify data ...
session.commit("Experimental changes")

# Read main (unaffected by experiment)
session = repo.readonly_session(branch="main")
```

### Tagging Releases

```python
snapshot_id = session.commit("Production release v2.0")
repo.create_tag("v2.0-release", snapshot_id=snapshot_id)

# Later: read the exact tagged version
session = repo.readonly_session(tag="v2.0-release")
ds = xr.open_zarr(session.store, consolidated=False)
```

### Browsing History

```python
for ancestor in repo.ancestry(branch="main"):
    print(f"{ancestor.id}: {ancestor.message} at {ancestor.written_at}")
```

### Conflict Resolution

```python
session = repo.writable_session("main")
# ... make changes ...
try:
    session.commit("Update data")
except icechunk.ConflictError:
    # Auto-resolve: keep our changes on conflict
    session.rebase(icechunk.BasicConflictSolver(
        on_chunk_conflict=icechunk.VersionSelection.UseOurs
    ))
    session.commit("Resolved conflicts, updated data")

# Or auto-rebase on commit:
session.commit("Update", rebase_with=icechunk.ConflictDetector())
```

**Conflict resolution classes:**
- `ConflictDetector()` - detect conflicts, fail if any found
- `BasicConflictSolver(on_chunk_conflict=VersionSelection.UseOurs)` - auto-resolve
- `BasicConflictSolver(on_chunk_conflict=VersionSelection.UseTheirs)` - take other's changes

### Check Uncommitted Changes

```python
session = repo.writable_session("main")
# ... make changes ...
diff = session.status()  # see what changed
session.discard_changes()  # or throw away changes
```

## Integration

### With Xarray

**Requires**: xarray >= 2024.10.0, zarr 3

```python
from icechunk.xarray import to_icechunk

# Write new dataset
session = repo.writable_session("main")
to_icechunk(ds, session, mode="w")
session.commit("Write dataset")

# Append along dimension
session = repo.writable_session("main")
to_icechunk(ds_new, session, append_dim="time")
session.commit("Append time")

# Region write (update a slice)
session = repo.writable_session("main")
to_icechunk(ds_slice, session, region={"time": slice(0, 10)})
session.commit("Update region")

# Read
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, consolidated=False)
```

### With Dask Arrays

**Requires**: dask >= 2025.2.0

```python
import dask.array as da
from icechunk.dask import store_dask

# 1. Initialize the array structure first
session = repo.writable_session("main")
zarr_arr = zarr.create_array(
    session.store, path="data",
    shape=(100, 100), chunks=(10, 10), dtype="f8"
)
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

### With Dask + Xarray (Distributed)

```python
import distributed
from icechunk.xarray import to_icechunk

client = distributed.Client()

session = repo.writable_session("main")
ds = xr.tutorial.open_dataset("rasm", chunks={"time": 1})
to_icechunk(ds, session, mode="w")  # handles distributed writes internally
session.commit("Distributed write")

client.shutdown()
```

### With VirtualiZarr (Virtual Chunks)

```python
from virtualizarr import open_virtual_dataset

# 1. Create virtual datasets from existing files
virtual_ds = xr.concat([
    open_virtual_dataset(url, indexes={})
    for url in file_urls
], dim="time")

# 2. Configure virtual chunk containers
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix="s3://source-bucket",
        store=icechunk.s3_store(region="us-east-1"),
    )
)

# 3. Authorize credentials for reading virtual chunks
credentials = icechunk.containers_credentials({
    "s3://source-bucket": icechunk.s3_credentials(
        access_key_id="...", secret_access_key="..."
    )
})

# 4. Create repo with virtual chunk support
repo = icechunk.Repository.create(
    storage=storage, config=config,
    authorize_virtual_chunk_access=credentials
)

# 5. Write virtual references
session = repo.writable_session("main")
to_icechunk(virtual_ds, session, mode="w")
session.commit("Virtual references")

# 6. Opening later requires re-authorizing virtual access
repo = icechunk.Repository.open(
    storage=storage,
    authorize_virtual_chunk_access=credentials  # REQUIRED
)
```

## Gotchas & Common Mistakes

### Session becomes read-only after commit
A `WritableSession` transitions to read-only after `commit()`. You must create a new writable session for subsequent writes. This is by design for transaction safety.

### to_icechunk with uncommitted changes + dask data raises ValueError
If the session has uncommitted changes AND the dataset contains dask arrays, `to_icechunk` raises `ValueError("Calling to_icechunk is not allowed on a Session with uncommitted changes. Please commit first.")`. Commit or discard changes before calling `to_icechunk` with dask-backed data.

### consolidated=False is mandatory for reads (#962)
Icechunk manages metadata internally. Using `consolidated=True` (xarray's default for `open_consolidated`) causes Zarr to write a `consolidated_metadata` key that can hide new groups. Icechunk now errors on this, but always pass `consolidated=False`.

### Virtual chunks require explicit authorization (#1032, #1545)
Opening a repo with virtual chunks without `authorize_virtual_chunk_access` will give cryptic `AccessDenied` errors at read time, not at open time. The error message references the chunk key, not the missing credentials. Always pass credentials when opening repos with virtual references.

### S3-compatible stores may need unsafe mode (#743)
Object stores without conditional write support (If-None-Match/If-Match headers) will fail on `Repository.create`. Use unsafe storage settings at the cost of losing concurrent write safety:
```python
config = icechunk.RepositoryConfig(
    storage=icechunk.StorageSettings(
        unsafe_use_conditional_update=False,
        unsafe_use_conditional_create=False,
    )
)
repo = icechunk.Repository.create(storage, config=config)
```

### Ceph Object Gateway double-quoting ETag bug (#1087)
Ceph backends may segfault on commit due to ETag header double-quoting. Fixed in recent versions -- upgrade icechunk if hitting this.

### SyncError in async contexts (Flask, Jupyter) (#1228)
Zarr 3 uses async internally. If you're inside an existing event loop (Flask, some Jupyter setups), you'll get `SyncError: Calling sync() from within a running loop`. Use `nest_asyncio` or restructure to avoid nested loops.

### Multiple groups require mode="a" (#176)
When writing multiple groups to the same store with `to_zarr`:
```python
# First group
ds1.to_zarr(session.store, group="A", zarr_format=3, consolidated=False, mode="w")
# Subsequent groups need mode="a"
ds2.to_zarr(session.store, group="B", zarr_format=3, consolidated=False, mode="a")
```

### Dask deserialization errors with virtual chunks (#1181)
Distributed Dask with virtual-chunk-based stores could fail with `ValueError: Failed to deserialize store from bytes`. Fixed in icechunk >= 1.1.4. Upgrade if encountering this.

### S3 virtual references with spaces in keys (#1463)
Object keys containing spaces get URL-encoded, causing `NoSuchKey` errors. Fixed in recent versions.

### from_env=True reads ~/.aws/credentials (#1032 comments)
When using `from_env=True`, icechunk uses the standard AWS SDK credential provider chain, including `~/.aws/credentials`. Stale or wrong entries there will cause `SignatureDoesNotMatch` errors even if you think you're passing explicit credentials elsewhere.

### write_empty_chunks not yet supported in to_icechunk (#1458)
Unlike xarray's `to_zarr`, `to_icechunk` does not yet have a `write_empty_chunks` parameter. Chunks containing only the fill value will not be written. Workaround: use a sentinel fill value and re-mask.

## Configuration

```python
config = icechunk.RepositoryConfig.default()

# Compression (for metadata files)
config.compression = icechunk.CompressionConfig(
    level=3, algorithm=icechunk.CompressionAlgorithm.Zstd
)

# Caching
config.caching = icechunk.CachingConfig(
    num_snapshot_nodes=500000,
    num_chunk_refs=100,
    num_bytes_chunks=100_000_000,  # 100MB chunk cache
)

# Storage concurrency
config.storage = icechunk.StorageSettings(
    concurrency=icechunk.StorageConcurrencySettings(
        max_concurrent_requests_for_object=10,
        ideal_concurrent_request_size=1_000_000,
    ),
)

# Inline small chunks into manifests
config.inline_chunk_threshold_bytes = 4096

# Apply config at creation
repo = icechunk.Repository.create(storage, config=config)
# Or update config on existing repo
repo = repo.reopen(config=new_config)
repo.save_config()  # persist to storage
```

## Known Limitations

- **Local filesystem storage**: Not safe for concurrent commits; not supported on Windows (#665)
- **Zarr sharding on writes**: Must set `zarr.config.set({"async.concurrency": 1})`
- **Sparse arrays**: Not currently supported
- **Branch merging**: Not supported (only rebase sessions on the same branch)
- **Tags**: Cannot be recreated once deleted (unlike Git)
- **Rebase**: Only resolves chunk conflicts; metadata conflicts must be resolved manually
- **GCS**: Config updating potentially unsafe (etag vs generation); bearer token refresh not yet supported
- **Object stores without conditional writes**: Lose ACID consistency with concurrent writers (#743)

## Logging & Debugging

```python
import os
os.environ["ICECHUNK_LOG"] = "icechunk=debug"
# Or at runtime:
icechunk.set_logs_filter("debug,icechunk=trace")
# Print installed versions and config:
icechunk.print_debug_info()
```

## Maintenance

```python
from datetime import datetime, timedelta

# Expire snapshots older than 30 days
expiry = datetime.now() - timedelta(days=30)
repo.expire_snapshots(older_than=expiry)

# Garbage collect unreferenced chunks (dry_run=True to preview)
summary = repo.garbage_collect(delete_object_older_than=expiry, dry_run=True)
# Then actually delete:
summary = repo.garbage_collect(delete_object_older_than=expiry)
```
