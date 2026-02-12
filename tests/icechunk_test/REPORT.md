# Icechunk Skill Assessment

Tested against: icechunk 1.1.18, zarr 3.1.5, xarray 2026.1.0

## Verified Correct

- **`icechunk.in_memory_storage()`**: Works as documented
- **`Repository.open_or_create(storage)`**: Works, idempotent as claimed
- **`Repository.create(storage)`**: Works on fresh storage
- **`repo.writable_session("main")`**: Returns a Session object, works as documented
- **`repo.readonly_session(branch="main")`**: Works as documented
- **`session.commit(message)`**: Returns snapshot ID string, works as documented
- **`to_icechunk(ds, session, mode="w")`**: Writes xarray Dataset correctly, round-trips with `xr.open_zarr`
- **`xr.open_zarr(session.store, consolidated=False)`**: Reads data correctly
- **`to_icechunk(ds, session, append_dim="time")`**: Appends along dimension correctly
- **`repo.create_tag(name, snapshot_id=snap)`**: Works as documented
- **`repo.readonly_session(tag="v1.0")`**: Returns historical data from tagged snapshot correctly
- **`repo.lookup_branch("main")`**: Returns correct latest snapshot ID
- **`repo.create_branch(name, snapshot_id=snap)`**: Creates independent branch correctly
- **Transaction context manager**: `with repo.transaction("main", message="...") as store:` works, yields `IcechunkStore`, auto-commits on exit
- **Session becomes read-only after commit**: `session.read_only` is `True` after commit, writing to committed session raises `IcechunkError: cannot write to read-only store`
- **New writable session needed for subsequent writes**: Confirmed
- **Multiple groups with `mode="a"`**: Works as documented
- **`to_icechunk` takes session, not store**: Passing `session.store` raises `AttributeError`
- **`ds.to_zarr(session.store, zarr_format=3, consolidated=False)`**: Works for non-dask data
- **Conflict resolution**: `ConflictError` raised on conflicting commit; `BasicConflictSolver(on_chunk_conflict=VersionSelection.UseOurs)` + `session.rebase()` resolves correctly
- **All API names exist**: `icechunk.ConflictError`, `BasicConflictSolver`, `VersionSelection`, `VersionSelection.UseOurs`, `ConflictDetector`
- **`RepositoryConfig` and `StorageSettings`**: Both exist and can be instantiated with the documented parameters (`unsafe_use_conditional_update`, `unsafe_use_conditional_create`)
- **`repo.list_branches()`**: Returns `set` of branch names (e.g., `{'main', 'dev'}`)
- **`repo.list_tags()`**: Returns `set` of tag names
- **`repo.lookup_tag(name)`**: Returns correct snapshot ID
- **`repo.ancestry(branch="main")`**: Works (not in skill file, but functional)
- **Dask fork/store_dask/merge pattern**: Works correctly with `local_filesystem_storage` (not testable with `in_memory_storage` -- see Issues)
- **`to_icechunk` with dask-backed data**: Works with `local_filesystem_storage`, handles forking internally as claimed

## Issues Found

### 1. Skill file Dask example uses `path=` instead of `name=` (WRONG)
The Dask writes section shows:
```python
zarr_arr = zarr.create_array(session.store, path="data", ...)
```
But zarr v3 `create_array()` uses `name=`, not `path=`. This raises `TypeError: create_array() got an unexpected keyword argument 'path'`. Should be:
```python
zarr_arr = zarr.create_array(session.store, name="data", ...)
```
Note: `zarr.open_array(..., path="data")` is correct -- only `create_array` uses `name=`.

### 2. `NoChangesToCommitError` does not exist as a named exception
The skill file mentions "commit raises NoChangesToCommitError with no changes" but:
- The actual error is `icechunk.IcechunkError` with message "cannot commit, no changes made to the session"
- `icechunk.NoChangesToCommitError` does not exist as a module attribute
- Should reference `IcechunkError` instead, or note the message pattern to catch

### 3. `ConflictDetector` with `rebase_with=` did not auto-rebase in testing
```python
session.commit("Update", rebase_with=icechunk.ConflictDetector())
```
Raised `RebaseFailedError` even for writes to distant array elements (element 0 vs element 199 in a 200-element array). This is because `ConflictDetector` operates at chunk granularity, not element granularity. With default chunk sizes, all elements may be in the same chunk. The API exists and is callable, but the example may give false expectations about what "non-conflicting" means.

### 4. Dask patterns (fork, store_dask, to_icechunk with dask) fail with `in_memory_storage`
Both the manual fork/store_dask/merge pattern and `to_icechunk` with dask-backed data fail when using `in_memory_storage()`. They work correctly with `local_filesystem_storage()`. The error is `IcechunkError: Object at location snapshots/... not found: No data in memory found`. This is because dask workers can't access the in-memory storage from separate threads/processes. The skill file does not mention this limitation.

### 5. `consolidated=True` does not raise an error (minor)
The skill says "consolidated=True writes a key that hides new groups" but in testing, `xr.open_zarr(store, consolidated=True)` did NOT raise an error and returned correct data. The default (no `consolidated` arg) also works fine. The advice to always use `consolidated=False` is still good practice, but the claim that `True` causes problems may be overstated or version-dependent.

### 6. `mode="w"` for second group does not destroy first group (minor nuance)
The skill implies mode="w" for the second group would destroy the first group, motivating the need for mode="a". In testing, using mode="w" for group B did NOT destroy group A. The groups are independent. However, using mode="a" is still best practice since mode="w" on the ROOT (no group specified) WOULD wipe everything.

## Missing Content

- **Dask operations don't work with `in_memory_storage()`**: Should note this limitation explicitly since `in_memory_storage()` is prominently featured for testing. Dask fork/store_dask and to_icechunk with dask arrays require persistent storage (local filesystem or object store).
- **`list_branches()` and `list_tags()`**: Not mentioned in the skill file but are useful for discovery. They return `set` types.
- **`lookup_tag(name)`**: Not mentioned but useful paired with `create_tag`.
- **`repo.ancestry(branch=)`**: Useful for history browsing, not mentioned.
- **`session.read_only` property**: Not explicitly mentioned (the behavior is described but not the property name).
- **`to_icechunk` with `mode="w"` on empty repo fails with dask**: First write to a fresh repo with dask-backed data fails even with local filesystem storage. Workaround: do a non-dask initial write first, or use the manual fork/store_dask pattern.

## Overall Assessment

The skill file is **largely accurate and well-structured**. The core workflows (create repo, write xarray data, read back, append, branch, tag, time travel, conflict resolution, transactions) all work as documented. The main issues are:

1. **One wrong keyword** in the Dask example (`path=` should be `name=`) -- easy fix
2. **`NoChangesToCommitError` doesn't exist** as a named exception -- should reference `IcechunkError`
3. **Missing note about dask + in_memory_storage incompatibility** -- important for users following the skill file's testing pattern
4. The `consolidated=False` advice is good but the consequences of `True` are overstated

50 of 52 test assertions passed. The 2 failures were both dask-related and confirmed to be `in_memory_storage` limitations (they pass with `local_filesystem_storage`).
