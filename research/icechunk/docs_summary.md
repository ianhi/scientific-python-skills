## ICECHUNK RESEARCH SUMMARY

### What is Icechunk

Icechunk is an open-source (Apache 2.0), transactional storage engine for multidimensional array (tensor) data built on top of Zarr. It adds git-like version control semantics and ACID transaction support to Zarr, enabling safe concurrent reads and writes to array data in cloud object storage. Key innovation: Icechunk treats Zarr as a database format rather than just a file format.

**Core Value Propositions:**
- **Safety**: Version history allows recovery from corrupted/deleted data via commits or tags
- **Consistency**: ACID transactions ensure readers never see partial writes from concurrent writers
- **Reproducibility**: Immutable snapshots and tags provide permanent references to specific dataset versions
- **Performance**: Cloud-optimized design with efficient storage through chunk deduplication

**Current Version**: 1.1.11 (stable, production-ready as of July 2025)
**Backwards Compatibility**: Data written by Icechunk 1.0+ will be readable by all future versions

---

### Core Concepts

#### Repository
- Central entity in Icechunk; a self-contained Zarr store containing arrays/groups
- Exists in cloud storage (S3, GCS, Azure, etc.) or local filesystem
- Single repository accessible concurrently by multiple users (unlike Git's local/remote split)
- Scoped to related arrays requiring consistent transactional updates

#### Session
Two types:
1. **ReadOnlySession**: Access committed data snapshots (immutable)
   - Created via `repo.readonly_session(snapshot_id=...)`, `repo.readonly_session(tag=...)`, or `repo.readonly_session(branch="name")`
   - Cannot execute writes
   - Always sees a consistent snapshot

2. **WritableSession**: Stage changes before committing
   - Created via `repo.writable_session(branch="name")`
   - Only creatable from branch tips
   - Maintains a scratch space of uncommitted changes
   - Other users see previous committed version until commit succeeds
   - Becomes read-only after successful commit (must create new session for next write)

3. **ForkSession**: Serializable writable session for distributed computing
   - Created via `session.fork()` (only from sessions with no uncommitted changes)
   - Can be pickled and sent to remote workers
   - Must be merged back with `session.merge()`

#### Store (IcechunkStore)
- Zarr-compatible key-value store interface
- Obtained from session via `session.store`
- Abstracts away Icechunk internal format; Zarr libraries interact with it normally
- Translates Zarr keys to Icechunk's internal file layout

#### Snapshot
- Immutable record of repository state at a point in time
- Created by committing a transaction
- Has unique ID, parent snapshot ID, commit message, timestamp
- Other users can read previous snapshots while new writes happen

#### Transactions
- Group of related read/write operations treated as atomic unit
- All-or-nothing semantics: succeed completely or fail completely
- Implemented via writable sessions:
  ```python
  session = repo.writable_session("main")
  # ... make changes via session.store ...
  snapshot_id = session.commit("message")
  ```
- Or via context manager:
  ```python
  with repo.transaction("main", message="...") as store:
      # changes auto-committed on exit
  ```

#### Branches
- Mutable references to snapshots
- Multiple branches can co-exist
- Default branch is `main`
- Common pattern: `dev`, `stage`, `prod` branches
- Operations: create, list, lookup (get snapshot), reset, delete
- Reset is destructive (overwrites history)

#### Tags
- Immutable references to snapshots
- Used for releases/important versions
- Once deleted, cannot be recreated (unlike Git)
- Operations: create, list, lookup, delete

#### Chunk References / Virtual Chunks
- "Pointers" to chunks in other file formats (HDF5, NetCDF, GRIB, TIFF, etc.)
- Leverage Kerchunk/VirtualiZarr ecosystem
- Can reference multiple external sources in same repository
- Icechunk tracks references alongside native chunks
- Requires explicit authorization in Icechunk 1.0+:
  1. Define virtual chunk containers when creating repo (url_prefix → storage config)
  2. Pass `authorize_virtual_chunk_access` dict when opening repo (url_prefix → credentials)

---

### ACID Transactions

**Atomicity**: All operations in a transaction succeed together or none do (no partial commits)
**Consistency**: Data transitions from valid state to valid state
**Isolation**: Readers see committed snapshots; concurrent writers don't interfere
**Durability**: Committed data is permanent (inherited from underlying storage)

---

### Python API Map

#### Top-Level Exports

**Repository Management:**
- `Repository.create(storage, config=None, authorize_virtual_chunk_access=None, spec_version=None)` → Repository
- `Repository.open(storage, config=None, authorize_virtual_chunk_access=None)` → Repository
- `Repository.open_async()` → async Repository
- `Repository.create_async()` → async Repository

**Session Creation:**
- `Repository.writable_session(branch)` → Session
- `Repository.readonly_session(branch=None, snapshot_id=None, tag=None)` → Session
- `Repository.transaction(branch, message)` → context manager yielding IcechunkStore

**Version Control:**
- `Repository.ancestry(branch=None, snapshot_id=None, tag=None)` → Iterator[SnapshotInfo]
- `Repository.list_branches()` → list[str]
- `Repository.lookup_branch(name)` → str (snapshot_id)
- `Repository.create_branch(name, snapshot_id)`
- `Repository.delete_branch(name)`
- `Repository.reset_branch(name, snapshot_id, from_snapshot_id=None)` (from_snapshot_id is safety check)
- `Repository.list_tags()` → list[str]
- `Repository.lookup_tag(name)` → str (snapshot_id)
- `Repository.create_tag(name, snapshot_id)`
- `Repository.delete_tag(name)`

**Maintenance:**
- `Repository.expire_snapshots(older_than: datetime)` → dict (expiration results)
- `Repository.garbage_collect(expiry_time: datetime)` → dict (GC results)
- `Repository.fetch_config(storage)` → RepositoryConfig (static method to fetch saved config)
- `Repository.authorized_virtual_container_prefixes()` → list[str]

**Session Methods:**
- `Session.commit(message, metadata=None, rebase_with=None)` → str (snapshot_id)
- `Session.status()` → Diff (overview of uncommitted changes)
- `Session.discard_changes()`
- `Session.rebase(rebase_with: ConflictSolver)` (handle conflicts)
- `Session.fork()` → ForkSession
- `Session.merge(*sessions)` (merge other sessions into this one)
- `Session.flush()` (create snapshot without updating branch)
- Properties: `.store`, `.read_only`, `.mode`, `.snapshot_id`, `.branch`, `.has_uncommitted_changes`

**ForkSession Methods:**
- Serializable version of Session for distributed computing
- Can be pickled and sent to remote workers
- Methods similar to Session but designed for parallel writes

#### Conflict Resolution

When two sessions write to same branch concurrently:
- First commit succeeds
- Second commit gets `ConflictError` with parent snapshot mismatch

Resolution via `session.rebase(solver)`:
- `ConflictDetector()`: Detect conflicts, fail if found
- `BasicConflictSolver(on_chunk_conflict=VersionSelection.UseOurs/UseTheirs)`: Auto-resolve

Can also use `session.commit(rebase_with=ConflictDetector())` for auto-retry.

---

### Storage Configuration Patterns

#### S3 / S3-Compatible
```python
# From environment (EC2, etc.)
storage = icechunk.s3_storage(
    bucket="my-bucket", prefix="my-prefix", from_env=True
)

# Explicit credentials
storage = icechunk.s3_storage(
    bucket="my-bucket", prefix="my-prefix",
    region="us-east-1", access_key_id="...", secret_access_key="...",
    session_token=None, endpoint_url=None, allow_http=False
)

# Anonymous
storage = icechunk.s3_storage(
    bucket="my-bucket", prefix="my-prefix", 
    region="us-east-1", anonymous=True
)

# Refreshable credentials
def get_credentials() -> S3StaticCredentials:
    return icechunk.S3StaticCredentials(
        access_key_id="...", secret_access_key="...",
        expires_after=...
    )
storage = icechunk.s3_storage(
    bucket="...", prefix="...", get_credentials=get_credentials
)
```

**S3-Compatible Services:**
- Minio: Use `s3_storage()` with custom endpoint, region, allow_http=True, force_path_style=True
- Cloudflare R2: Use `icechunk.r2_storage(bucket, prefix, access_key_id, secret_access_key, account_id)`
- Tigris: Use `icechunk.tigris_storage()` (special implementation for consistency guarantees)

#### Google Cloud Storage
```python
# From environment
storage = icechunk.gcs_storage(bucket="...", prefix="...", from_env=True)

# Service account file
storage = icechunk.gcs_storage(
    bucket="...", prefix="...",
    service_account_file="/path/to/service-account.json"
)

# Service account key dict
storage = icechunk.gcs_storage(
    bucket="...", prefix="...",
    service_account_key={...}
)

# Application default credentials
storage = icechunk.gcs_storage(
    bucket="...", prefix="...",
    application_credentials="/path/to/adc.json"
)

# Bearer token
storage = icechunk.gcs_storage(
    bucket="...", prefix="...",
    bearer_token="..."
)

# Refreshable credentials
def get_credentials() -> GcsBearerCredential:
    return icechunk.GcsBearerCredential(
        bearer="...", expires_after=...
    )
storage = icechunk.gcs_storage(
    bucket="...", prefix="...",
    get_credentials=get_credentials
)
```

**GCS Limitations:**
- Config updating potentially unsafe (generation vs etag issue)
- No bearer token auth refresh yet (limited to service accounts)

#### Azure Blob Storage
```python
# From environment
storage = icechunk.azure_storage(
    account="...", container="...", prefix="...", from_env=True
)

# Explicit credentials
storage = icechunk.azure_storage(
    account_name="...", container="...", prefix="...",
    account_key="...", access_token=None, sas_token=None, bearer_token=None
)
```

#### Local Filesystem
```python
storage = icechunk.local_filesystem_storage("/path/to/repo")
```
**Warning**: Not safe for concurrent commits; not supported on Windows

#### HTTP (Read-Only)
```python
storage = icechunk.http_storage(
    "https://example.com/path/to/repo",
    opts={"timeout": "30s", "connect_timeout": "5s"}
)
```
**Limitations**: Read-only; server must serve static files

#### In-Memory (Testing)
```python
storage = icechunk.in_memory_storage()
```

---

### Version Control Operations Patterns

```python
# Create and commit
session = repo.writable_session("main")
root = zarr.create_group(session.store)
root.create_array("data", shape=(10, 10), chunks=(5, 5), dtype=float)
snapshot_id = session.commit("Create dataset")

# Time travel to previous snapshot
session = repo.readonly_session(snapshot_id=snapshot_id)
data = zarr.open_group(session.store, mode="r")

# Create branch from snapshot
repo.create_branch("dev", snapshot_id=snapshot_id)
session = repo.writable_session("dev")
# ... make changes ...
session.commit("Work in progress")

# List history
for ancestor in repo.ancestry(branch="main"):
    print(f"{ancestor.id}: {ancestor.message} at {ancestor.written_at}")

# Create immutable tag
repo.create_tag("v1.0-release", snapshot_id=snapshot_id)
session = repo.readonly_session(tag="v1.0-release")

# Reset branch (destructive)
repo.reset_branch("dev", new_snapshot_id)

# Handle conflicts
try:
    session.commit("Update data")
except icechunk.ConflictError:
    session.rebase(icechunk.BasicConflictSolver(
        on_chunk_conflict=icechunk.VersionSelection.UseOurs
    ))
    session.commit("Resolved conflicts")
```

---

### Xarray Integration Patterns

**Required**: Xarray >= 2025.1.1, Zarr 3

```python
import xarray as xr
from icechunk.xarray import to_icechunk

# Write new dataset
ds = xr.Dataset({...})
session = repo.writable_session("main")
to_icechunk(ds, session, mode="w")
session.commit("Write dataset")

# Append to existing
session = repo.writable_session("main")
to_icechunk(ds, session, append_dim="time")
session.commit("Append time dimension")

# Read with Xarray
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, consolidated=False)

# Read at specific version
session = repo.readonly_session(tag="v1.0")
ds = xr.open_zarr(session.store, consolidated=False)
```

**Key differences from `to_zarr`:**
- `to_icechunk()` is required for distributed writes (not `to_zarr`)
- Don't use consolidated=True (Icechunk organizes metadata efficiently)
- Use zarr_format=3

---

### Dask Integration Patterns

**With Dask Arrays:**
```python
import dask.array as da
from icechunk.dask import store_dask

# Create/initialize store first
session = repo.writable_session("main")
zarr_array = zarr.create_array(
    session.store, path="array",
    shape=(100, 100), chunks=(10, 10), dtype="f8"
)
session.commit("Initialize")

# Fork session for distributed writes
session = repo.writable_session("main")
fork = session.fork()
zarr_array = zarr.open_array(fork.store, path="array")

# Write dask array
dask_array = da.random.random((100, 100), chunks=(20, 20))
remote_session = store_dask(sources=[dask_array], targets=[zarr_array])

# Merge and commit
session.merge(remote_session)
session.commit("Wrote dask array")
```

**With Dask + Xarray:**
```python
import distributed
from icechunk.xarray import to_icechunk

client = distributed.Client()

session = repo.writable_session("main")
ds = xr.tutorial.open_dataset("rasm", chunks={"time": 1})

# Use to_icechunk for distributed writes (not to_zarr)
to_icechunk(ds, session, mode="w")
session.commit("Wrote xarray with dask")

client.shutdown()
```

**Requirements:**
- Dask >= 2025.2.0
- Use `zarr.config.set({"async.concurrency": 1})` for zarr sharding on writes

---

### Async API Patterns

```python
import asyncio
from icechunk import Repository

async def get_branches(storage):
    repo = await Repository.open_async(storage)
    return await repo.list_branches_async()

# Run async code
branches = asyncio.run(get_branches(storage))
```

Methods with `_async` suffix available on Repository, Session, and Store for async operations. Sync interface is recommended for interactive data science; async is for backend services managing multiple repos.

---

### Virtual Chunks / References Patterns

**Creating virtual dataset (with VirtualiZarr):**
```python
from virtualizarr import open_virtual_dataset
import xarray as xr

# Create virtual datasets from existing files
virtual_ds = xr.concat([
    open_virtual_dataset(url, indexes={})
    for url in file_urls
], dim="time")

# Define virtual chunk containers
s3_store_config = icechunk.s3_store(region="us-east-1")
container = icechunk.VirtualChunkContainer(
    url_prefix="s3://source-bucket",
    store_config=s3_store_config
)

config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(container)

# Define credentials for reading virtual chunks
credentials = icechunk.containers_credentials({
    "s3://source-bucket": icechunk.s3_credentials(
        access_key_id="...", secret_access_key="..."
    )
})

# Create repo with virtual chunk support
repo = icechunk.Repository.create(
    storage=...,
    config=config,
    authorize_virtual_chunk_access=credentials
)

# Write virtual dataset to icechunk
session = repo.writable_session("main")
to_icechunk(virtual_ds, session, mode="w")
session.commit("Virtual references to source data")

# Opening repo requires authorizing virtual chunk access
credentials = icechunk.containers_credentials({
    "s3://source-bucket": icechunk.s3_credentials(...)
})
repo = icechunk.Repository.open(
    storage=...,
    authorize_virtual_chunk_access=credentials
)
```

**Virtual Chunk Containers (Icechunk 1.0+):**
- Require explicit declaration (no defaults)
- Each container has url_prefix (e.g., "s3://bucket") and storage config
- Icechunk refuses to read chunks outside authorized containers (security)
- Containers can use different object stores in same repo

**Supported Virtual Chunk Sources**: S3, GCS, HTTP/HTTPS, local filesystem (Azure on roadmap)

---

### Configuration Patterns

```python
import icechunk

# Get default config
config = icechunk.RepositoryConfig.default()

# Or fetch from existing repo
config = icechunk.Repository.fetch_config(storage)

# Configure compression
config.compression = icechunk.CompressionConfig(
    level=3,
    algorithm=icechunk.CompressionAlgorithm.Zstd
)

# Configure caching
config.caching = icechunk.CachingConfig(
    num_snapshot_nodes=500000,  # default increased in 1.1.7
    num_chunk_refs=100,
    num_transaction_changes=100,
    num_bytes_attributes=10_000,
    num_bytes_chunks=100_000_000,  # increased in 1.1.7
)

# Configure storage settings
config.storage = icechunk.StorageSettings(
    concurrency=icechunk.StorageConcurrencySettings(
        max_concurrent_requests_for_object=10,
        ideal_concurrent_request_size=1_000_000,
    ),
    storage_class="STANDARD",
    metadata_storage_class="STANDARD_IA",
    chunks_storage_class="STANDARD_IA",
)

# Configure inline chunk threshold
config.inline_chunk_threshold_bytes = 4096

# Configure concurrency
config.get_partial_values_concurrency = 8
config.max_concurrent_requests = 100

# Create/open repo with custom config
repo = icechunk.Repository.create(storage, config=config)

# Save modified config to repo
repo.save_config(config)
```

**Config Categories:**
- **inline_chunk_threshold_bytes**: When to inline small chunks into manifest
- **get_partial_values_concurrency**: Concurrent requests for partial reads
- **max_concurrent_requests**: Total concurrent requests allowed
- **compression**: Zstd compression for metadata files
- **caching**: In-memory caches for snapshots, chunks, attributes
- **storage**: Concurrency, retry, and storage class settings
- **virtual_chunk_containers**: Access to external chunk sources

---

### Migration from Plain Zarr

**Key considerations:**
1. Icechunk slightly higher cold-start latency
2. Less transparent on-disk format
3. Distributed writes more complex to coordinate

**Migration approach:**
```python
# Create Icechunk repo
storage = icechunk.s3_storage(bucket="...", prefix="...")
repo = icechunk.Repository.create(storage)

# Read existing Zarr data
old_zarr_group = zarr.open_group("s3://old-bucket/zarr-data", mode="r")

# Write to Icechunk
session = repo.writable_session("main")
for name, array in old_zarr_group.arrays():
    new_array = zarr.create_array(
        session.store, path=name,
        shape=array.shape, chunks=array.chunks, dtype=array.dtype
    )
    new_array[:] = array[:]

session.commit("Migrated from old Zarr store")
```

**Parallel write migration:**
Use ForkSession pattern for large datasets to distribute copy work.

---

### Known Limitations & Issues

**Storage Backend Limitations:**
- Local filesystem: Not safe for concurrent commits; Windows not supported
- GCS: Config updating potentially unsafe (generation issue)
- Object stores without conditional writes (e.g., JASMIN): Lose consistency with concurrent writers
- HTTP storage: Read-only

**Format Limitations:**
- Zarr sharding: Set `zarr.config.set({"async.concurrency": 1})` on writes
- Sparse arrays: Not currently supported
- Windows: Local filesystem storage not supported (workaround: WSL or cloud backend)

**API Limitations:**
- Rebase: Currently only resolves chunk conflicts; metadata conflicts must be resolved manually
- Merging branches: Not supported (only rebase sessions on same branch)
- Tags: Cannot be recreated once deleted (unlike Git)

**GCS Specific:**
- Bearer token auth refresh not yet supported
- Config updates use etag instead of generation (can be unsafe)

---

### Maintenance Operations

```python
from datetime import datetime, timedelta

# Expire old snapshots (keep only recent commits)
expiry_time = datetime.now() - timedelta(days=30)
results = repo.expire_snapshots(older_than=expiry_time)

# Run garbage collection (cleanup unreferenced chunks)
results = repo.garbage_collect(expiry_time)

# Both have async versions
results = await repo.expire_snapshots_async(expiry_time)
results = await repo.garbage_collect_async(expiry_time)
```

---

### Logging & Debugging

```python
import icechunk
import os

# Via environment variable
os.environ["ICECHUNK_LOG"] = "icechunk=debug"

# Or via API
icechunk.set_logs_filter("debug,icechunk=trace")

# Print debug info (versions, packages)
icechunk.print_debug_info()
```

---

### Key Classes & Types

**Core:**
- `Repository`: Entry point for all operations
- `Session`: Writable session for staging changes
- `ForkSession`: Serializable session for distributed writes
- `IcechunkStore`: Zarr-compatible key-value store
- `SnapshotInfo`: Metadata about a snapshot (id, message, parent, timestamp)

**Configuration:**
- `RepositoryConfig`: Repository behavior settings
- `CachingConfig`: In-memory caching settings
- `CompressionConfig`: Metadata compression settings
- `StorageSettings`: Storage backend settings
- `RepositoryConfig.virtual_chunk_containers`: Virtual chunk definitions

**Credentials:**
- `S3StaticCredentials`, `S3Credentials`, etc.
- `GcsStaticCredentials`, `GcsBearerCredential`, etc.
- `AzureStaticCredentials`, `AzureCredentials`
- `containers_credentials()`: Multi-container credential mapping

**Conflict Resolution:**
- `ConflictDetector`: Detect but don't auto-resolve conflicts
- `BasicConflictSolver`: Auto-resolve with `VersionSelection.UseOurs/UseTheirs`
- `ConflictError`: Raised when commit conflicts with other session
- `RebaseFailedError`: Raised when rebase encounters unresolvable conflicts

**Storage:**
- `Storage`: Opaque storage configuration object (created via s3_storage(), gcs_storage(), etc.)
- `ObjectStoreConfig`: Configuration for individual object stores

**Version Control:**
- `UpdateType`: Enum for repository update types
- `Diff`: Overview of changed chunks, arrays, metadata

---

### Real-World Workflow Example

```python
import icechunk
import zarr
import xarray as xr
from icechunk.xarray import to_icechunk

# 1. Create repo
storage = icechunk.s3_storage(
    bucket="my-data", prefix="climate-model", from_env=True
)
repo = icechunk.Repository.create(storage)

# 2. Initial write
session = repo.writable_session("main")
ds = xr.Dataset({
    "temperature": (["time", "lat", "lon"], initial_temp_data),
    "pressure": (["time", "lat", "lon"], initial_press_data),
})
to_icechunk(ds, session, mode="w")
v1_snapshot = session.commit("Initial forecast")

# 3. Create release tag
repo.create_tag("release-2025-01", snapshot_id=v1_snapshot)

# 4. Continue work on main
session = repo.writable_session("main")
ds_update = xr.Dataset({
    "temperature": (["time", "lat", "lon"], new_temp_data),
})
to_icechunk(ds_update, session, append_dim="time")
session.commit("Append new forecast timestep")

# 5. Create experimental branch
current_main = repo.lookup_branch("main")
repo.create_branch("experimental", snapshot_id=current_main)
session = repo.writable_session("experimental")
# ... make experimental changes ...
session.commit("Experimental model run")

# 6. Reader accessing specific version
session = repo.readonly_session(tag="release-2025-01")
ds_release = xr.open_zarr(session.store, consolidated=False)
# Guaranteed to read v1 even if main has changed 100 times

# 7. Parallel append with distributed workers
from concurrent.futures import ProcessPoolExecutor
from icechunk.xarray import to_icechunk

def write_timestep(itime, fork_session):
    ds = xr.Dataset({
        "temperature": (["lat", "lon"], temp_data[itime]),
    })
    to_icechunk(ds, fork_session, region={"time": slice(itime, itime+1)})
    return fork_session

session = repo.writable_session("main")
fork = session.fork()

with ProcessPoolExecutor() as executor:
    futures = [
        executor.submit(write_timestep, i, fork)
        for i in range(24)
    ]
    remote_sessions = [f.result() for f in futures]

for remote_session in remote_sessions:
    session.merge(remote_session)

session.commit("Appended 24 hours of forecast data")

# 8. Cleanup old versions
from datetime import datetime, timedelta
expiry = datetime.now() - timedelta(days=365)
repo.expire_snapshots(older_than=expiry)
repo.garbage_collect(expiry)
```

---

### Performance Considerations

- **Snapshot caching**: Default cache increased to 500k groups/arrays in v1.1.7
- **Chunk caching**: Default 100MB in-memory cache for chunk data
- **Concurrency**: Configurable `max_concurrent_requests` and `get_partial_values_concurrency`
- **Storage class**: Can specify different classes for metadata vs chunks (STANDARD vs STANDARD_IA)
- **list_dir/list_prefix**: Significantly faster in v1.1.7 for repos with thousands of groups
- **Inline chunks**: Small chunks (<4KB default) inlined into manifests for efficiency
- **Virtual chunks**: Stored alongside native chunks with no performance penalty

---

**End of Research Summary**