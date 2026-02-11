# Icechunk Skill File Review

Reviewed: `skills/icechunk.md` (566 lines)
Source verified against: icechunk-python source at `repos/icechunk/`

---

## ACCURACY

### Function Signature Errors

**1. `repo.save_config(config=None)` -- WRONG signature**

The skill file (line 178) says:
```python
repo.save_config(config=None)
```

Actual signature in `repository.py` line 437:
```python
def save_config(self) -> None:
```

`save_config()` takes NO arguments. It saves the repo's current config to storage. You modify the config via `repo.reopen(config=...)` or by setting properties on `repo.config` (though the source notes changes to the returned config object "won't be impacted"). The skill file's signature would cause a `TypeError`. This needs to be fixed.

**2. `session.flush()` -- WRONG signature**

The skill file (line 193) says:
```python
session.flush()  # snapshot without updating branch
```

Actual signature in `session.py` line 508:
```python
def flush(self, message: str, metadata: dict[str, Any] | None = None) -> str:
```

`flush()` requires a `message` argument. The skill file omits this and would cause a `TypeError` if someone copies it.

**3. `repo.list_branches()` returns `set[str]`, not `list[str]`**

Skill file (line 170): `repo.list_branches() -> list[str]`
Actual (repository.py line 787): `def list_branches(self) -> set[str]`

Same for `list_tags()` -- returns `set[str]` not `list[str]`.

**4. `repo.expire_snapshots()` returns `set[str]`, not `dict`**

Skill file (line 177): `repo.expire_snapshots(older_than: datetime) -> dict`
Actual (repository.py line 1381): returns `set[str]` (set of expired snapshot IDs).

Also, `expire_snapshots` has additional keyword arguments `delete_expired_branches` and `delete_expired_tags` that are not documented.

**5. `repo.garbage_collect()` signature is wrong**

Skill file (line 178): `repo.garbage_collect(expiry_time: datetime) -> dict`
Actual (repository.py line 1539):
```python
def garbage_collect(self, delete_object_older_than: datetime.datetime, *, dry_run: bool = False, ...) -> GCSummary
```

The parameter name is `delete_object_older_than`, not `expiry_time`. The return type is `GCSummary`, not `dict`. There are additional important kwargs like `dry_run`.

**6. `repo.transaction()` parameter ordering**

Skill file (line 161):
```python
repo.transaction(branch: str, message: str) -> ContextManager[IcechunkStore]
```

Actual (repository.py line 1338):
```python
def transaction(self, branch: str, *, message: str, metadata: dict[str, Any] | None = None,
                rebase_with: ConflictSolver | None = None, rebase_tries: int = 1_000) -> Iterator[IcechunkStore]:
```

The `message` is keyword-only (after `*`). The skill file's quick-reference example at line 145 uses positional-style: `repo.transaction("main", message="Auto-commit")` which is actually correct, but the signature shown at line 161 suggests `message` is positional.

**7. `session.commit()` is missing `rebase_tries` and `allow_empty` params**

Skill file (line 187):
```python
session.commit(message: str, metadata=None, rebase_with=None) -> str
```

Actual (session.py line 306):
```python
def commit(self, message: str, metadata: dict[str, Any] | None = None,
           rebase_with: ConflictSolver | None = None, rebase_tries: int = 1_000,
           allow_empty: bool = False) -> str
```

Missing `rebase_tries` and `allow_empty`.

**8. `Repository.create()` is missing `spec_version` parameter**

Skill file (line 155):
```python
Repository.create(storage, config=None, authorize_virtual_chunk_access=None) -> Repository
```

Actual (repository.py line 33):
```python
def create(cls, storage: Storage, config: RepositoryConfig | None = None,
           authorize_virtual_chunk_access: dict[str, AnyCredential | None] | None = None,
           spec_version: int | None = None) -> Self
```

**9. `authorize_virtual_chunk_access` type is wrong in VirtualiZarr example**

The skill file (lines 415-419) shows:
```python
credentials = icechunk.containers_credentials({
    "s3://source-bucket": icechunk.s3_credentials(
        access_key_id="...", secret_access_key="..."
    )
})
```

This is actually correct for the `containers_credentials` helper. However, the type accepted by `Repository.create/open` for `authorize_virtual_chunk_access` is `dict[str, AnyCredential | None]`. The `containers_credentials()` function wraps inner credential types into `Credentials.S3(...)` etc, producing the right type. The example works.

**10. `RepositoryConfig` construction in the unsafe storage section is wrong**

Skill file (lines 453-459):
```python
config = icechunk.RepositoryConfig(
    storage=icechunk.StorageSettings(
        unsafe_use_conditional_update=False,
        unsafe_use_conditional_create=False,
    )
)
```

This is actually correct per the `RepositoryConfig.__init__` signature in the .pyi file which takes `storage: StorageSettings | None`. Good.

**11. VirtualiZarr `config.set_virtual_chunk_container` call uses wrong argument name**

Skill file (line 408):
```python
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        url_prefix="s3://source-bucket",
        store=icechunk.s3_store(region="us-east-1"),
    )
)
```

The `VirtualChunkContainer.__init__` (pyi line 243) takes positional args `(url_prefix: str, store: AnyObjectStoreConfig)`. Using keyword args `url_prefix=` and `store=` should work since they appear in the signature. This is fine.

### Anti-Pattern Accuracy

**Anti-pattern #1 (forgetting to commit):** Correct. Data is indeed lost if not committed.

**Anti-pattern #2 (to_zarr vs to_icechunk for distributed):** Correct. Writable sessions cannot be pickled; `to_icechunk` internally handles forking. However, `to_zarr` works fine for non-distributed (in-memory) writes. The skill should clarify this -- `to_zarr` is not universally "WRONG", only wrong with dask/distributed.

**Anti-pattern #3 (reusing session after commit):** Correct. Session becomes read-only after commit per source code.

**Anti-pattern #4 (consolidated=True):** Correct and important.

**Anti-pattern #5 (Repository.create on existing):** Correct. The "try open, except create" pattern works but the skill should mention `Repository.open_or_create()` which exists in the actual API (repository.py line 214) and is the idiomatic solution to this exact problem.

**Anti-pattern #6 (store vs session to to_icechunk):** Correct. `to_icechunk` takes `Session`, not `IcechunkStore`.

**Anti-pattern #7 (old async API):** Correct. Useful for migration context.

### Code Snippet Errors

**Maintenance section (lines 557-565):** The `repo.garbage_collect(expiry)` call would not work because the parameter is named `delete_object_older_than`, not a positional arg named `expiry`. Need to use keyword: `repo.garbage_collect(delete_object_older_than=expiry)`.

Wait, checking again -- `garbage_collect(self, delete_object_older_than: datetime.datetime, ...)` -- it IS a positional argument. So `repo.garbage_collect(expiry)` would work positionally. This is fine.

**Browsing History snippet (line 285-287):**
```python
for ancestor in repo.ancestry(branch="main"):
    print(f"{ancestor.id}: {ancestor.message} at {ancestor.written_at}")
```

Verified against `SnapshotInfo` properties: `id`, `message`, `written_at` all exist. Correct.

**Dask example (lines 356-376):** The `store_dask` function returns `ForkSession`, not `remote_session`. The skill writes:
```python
remote_session = store_dask(sources=[dask_arr], targets=[zarr_arr])
session.merge(remote_session)
```

But `store_dask` returns a `ForkSession` and `session.merge()` expects `ForkSession` arguments. This is correct.

However, the Dask example opens the zarr array with `fork.store` but `fork` is a `ForkSession` -- that's correct since `ForkSession.store` returns an `IcechunkStore`.

Wait, but `store_dask` itself internally creates its own fork via the `session_merge_reduction` path. It doesn't take a `session` argument at all -- it takes `sources` and `targets` (zarr arrays). The zarr arrays already contain a reference to the store. So the example's flow of creating a fork, opening an array on it, then passing to `store_dask` is slightly misleading but technically could work since `store_dask` just calls `dask.array.store` on the targets. The fork session would be embedded in the zarr array's store.

Actually, re-reading more carefully: `store_dask` calls `dask.array.store` which writes to the zarr arrays. The zarr arrays' underlying store has the session embedded. Then `session_merge_reduction` extracts and merges the sessions from the stored result arrays. The returned `ForkSession` is then merged back into the original session. This pattern should work.

---

## USEFULNESS

### Section-by-Section Ratings

| Section | Lines | Rating (1-5) | Notes |
|---------|-------|------|-------|
| CRITICAL anti-patterns | 7-104 | **5** | Highest value section. These are exactly the mistakes Claude would make. |
| Quick Reference | 106-147 | **4** | Good copy-paste recipes. The transaction example is useful. |
| Core API - Repository | 149-182 | **3** | Useful but several signatures are wrong (see above). Needs fixing. |
| Core API - Session | 184-201 | **3** | Missing important methods. Some signatures wrong. |
| Storage Backends | 203-241 | **4** | Good coverage. Practical and copy-paste ready. |
| Patterns - Zarr Direct | 243-253 | **3** | Basic but needed for non-xarray users. |
| Patterns - Branching | 255-269 | **4** | Important workflow pattern. |
| Patterns - Tagging | 271-280 | **3** | Simple, correct. |
| Patterns - History | 282-287 | **3** | Simple, correct. |
| Patterns - Conflict Resolution | 289-311 | **5** | Critical and hard to figure out from docs alone. |
| Patterns - Check Changes | 313-319 | **2** | Trivial. Could be folded into Session API section. |
| Integration - Xarray | 321-348 | **5** | Core use case. Well done. |
| Integration - Dask Arrays | 350-376 | **4** | Important but complex. Could use more explanation. |
| Integration - Dask+Xarray | 378-392 | **4** | Key pattern for the ecosystem. |
| Integration - VirtualiZarr | 394-437 | **4** | Complex but important. Good coverage. |
| Gotchas section | 439-487 | **5** | Extremely valuable. These save hours of debugging. |
| Configuration | 489-521 | **3** | Useful reference but some details are slightly off. |
| Known Limitations | 523-533 | **4** | Good to have upfront. Saves time. |
| Performance Tips | 535-541 | **2** | Vague. Most tips lack concrete numbers or code. |
| Logging & Debugging | 543-552 | **3** | Short and useful. |
| Maintenance | 554-565 | **3** | Correct pattern. |

### Sections to Cut (to save space)

1. **Performance Tips (lines 535-541):** Too vague to be actionable. "Increase X for Y workloads" without numbers. Either make it concrete or cut it.
2. **Check Uncommitted Changes (lines 313-319):** 7 lines for `session.status()` and `session.discard_changes()`. Fold into Session API reference.
3. **Tagging Releases (lines 271-280):** Mostly redundant with the API reference. Could be a one-liner.

### Sections to Expand

1. **Anti-patterns section** -- should mention `open_or_create()` for #5.
2. **Conflict Resolution** -- add example of `RebaseFailedError` handling.
3. **to_icechunk gotchas** -- the `to_icechunk` function raises `ValueError` if the session has uncommitted changes AND the data is dask-backed. This is a critical gotcha not mentioned.

---

## MISSING

### Missing API Methods/Features

1. **`Repository.open_or_create()`** -- This is a major omission. It's the idiomatic way to handle "create if not exists, open if exists" and the skill file's anti-pattern #5 shows a try/except workaround that this method eliminates entirely.

2. **`Repository.exists(storage)`** -- Static method to check if a repo exists. Useful for conditional logic.

3. **`session.amend()`** -- Method to amend/overwrite the previous commit (like `git commit --amend`). Present in session.py lines 416-460.

4. **`session.move(from_path, to_path)`** -- Method to move/rename nodes. Uses `repo.rearrange_session()`.

5. **`repo.rearrange_session(branch)`** -- Special session type that only allows `move` operations. Not documented at all.

6. **`repo.reopen(config=..., authorize_virtual_chunk_access=...)`** -- Reopens repo with new config/credentials. The skill says `repo.save_config(config)` which is wrong; the actual pattern is `repo.reopen(config=new_config)` then `repo.save_config()`.

7. **`repo.diff(from_branch=..., to_branch=...)`** -- Compute diff between two versions. This is a useful inspection method not covered.

8. **`repo.ops_log()`** -- Returns iterator of repository operation updates.

9. **`repo.metadata` / `repo.set_metadata()` / `repo.update_metadata()`** -- Repository-level metadata (distinct from commit metadata). Not mentioned.

10. **`repo.set_default_commit_metadata()`** -- Set metadata that gets attached to all commits.

11. **`http_storage()` and `redirect_storage()`** -- Storage backends for read-only HTTP access. Not covered. `http_storage` is useful for reading publicly hosted repositories.

12. **`session.all_virtual_chunk_locations()`** -- List all virtual chunk locations. Useful for debugging.

13. **`repo.chunk_storage_stats()`** -- Replaces deprecated `total_chunks_storage()`. Returns detailed storage breakdown.

14. **Async variants** -- The entire API has `_async` variants for every method. The skill file does not mention this at all. At minimum, a note that "every method has an `_async` counterpart" would be useful.

### Missing Gotchas

1. **`to_icechunk` with uncommitted changes + dask data raises ValueError.** From xarray.py line 321-323: if the session has uncommitted changes AND the dataset contains dask arrays, `to_icechunk` raises `ValueError("Calling to_icechunk is not allowed on a Session with uncommitted changes. Please commit first.")`. This is a non-obvious restriction.

2. **ForkSession cannot commit.** `ForkSession.commit()` raises `TypeError`. You must merge back into the parent session first. The skill's Dask example implies this but doesn't state it explicitly.

3. **Writable sessions cannot be pickled.** `Session.__getstate__` raises `ValueError` for writable sessions. Only read-only sessions and `ForkSession` can be serialized. This is critical for distributed computing patterns.

4. **`commit()` with `rebase_with` will retry up to `rebase_tries` times (default 1000).** This is important for understanding concurrent write behavior.

5. **xarray version requirement:** `to_icechunk` requires xarray >= 2024.10.0 (xarray.py line 34). The skill says >= 2025.1.1 which may be too strict.

6. **dask version requirement:** `store_dask` requires dask >= 2025.2.0 (dask.py line 57). Not mentioned in the skill.

7. **`NoChangesToCommitError`** -- Committing with no changes raises this (unless `allow_empty=True`). Not mentioned.

---

## SUGGESTIONS

### Critical Fixes (Must Do)

1. **Fix `save_config` signature.** Remove `config=None` parameter. Show the correct pattern:
   ```python
   # To update config:
   repo = repo.reopen(config=new_config)
   repo.save_config()  # persist to storage
   ```

2. **Fix `flush()` signature.** Add required `message` parameter:
   ```python
   session.flush(message: str, metadata=None) -> str
   ```

3. **Fix `list_branches()` and `list_tags()` return types** from `list[str]` to `set[str]`.

4. **Fix `expire_snapshots()` return type** from `dict` to `set[str]`.

5. **Fix `garbage_collect()` return type** from `dict` to `GCSummary`.

6. **Add `Repository.open_or_create()`** to the API reference and replace the try/except anti-pattern #5 workaround with it:
   ```python
   # BEST - use open_or_create
   repo = icechunk.Repository.open_or_create(storage)
   ```

### High-Value Additions

7. **Add a note about async variants.** A single line in the Core API section: "All methods have `_async` counterparts (e.g., `commit_async`, `open_async`)."

8. **Clarify anti-pattern #2:** `to_zarr` is fine for non-distributed writes. Only wrong with dask/distributed. Reword:
   ```
   ### 2. Using to_zarr with distributed/dask writers
   # WRONG - to_zarr cannot handle distributed Dask writers (session is not picklable)
   # For in-memory (non-dask) data, to_zarr works fine.
   ```

9. **Add `to_icechunk` gotcha about uncommitted changes with dask data.**

10. **Add xarray and dask minimum version requirements** to the Integration sections.

11. **Add `session.amend()` to Session API reference.** It's a useful git-like feature.

### Ordering Changes

12. **Move anti-pattern #5 (`Repository.create` on existing) higher** -- or better, merge it with the Quick Reference section since `open_or_create` eliminates it entirely.

13. **Move "Gotchas & Common Mistakes" section BEFORE "Integration"** -- developers hit gotchas during integration work, so seeing them first is more useful.

14. **Move "Known Limitations" to the top**, right after anti-patterns. Developers need to know what they can NOT do before they start writing code.

### Style/Clarity

15. **The Configuration section (lines 489-521) builds config using `RepositoryConfig.default()` then sets properties.** This works but the actual `__init__` accepts all these as constructor arguments. Show both patterns:
    ```python
    # Pattern 1: Constructor
    config = icechunk.RepositoryConfig(
        inline_chunk_threshold_bytes=4096,
        caching=icechunk.CachingConfig(num_bytes_chunks=100_000_000),
    )
    # Pattern 2: Default + modify
    config = icechunk.RepositoryConfig.default()
    config.inline_chunk_threshold_bytes = 4096
    ```

16. **The `store_dask` import is shown as `from icechunk.dask import store_dask`** but `store_dask` is not re-exported from the top-level `icechunk` package (it's not in `__all__`). The import path shown is correct; just noting this is consistent.

17. **The Dask direct-write example (lines 356-376) is the most complex pattern in the file.** It would benefit from inline comments explaining the fork/merge lifecycle. Currently it has numbered steps but a brief "why" for each step would help Claude understand the distributed write model.

### Things to Remove

18. **Remove anti-pattern #7 (old async API)** or move it to the very bottom. Claude is unlikely to generate pre-1.0 code and this wastes prime space at the top of the file. If kept, it should be in a "Historical" or "Migration" section.

19. **Remove the Performance Tips section** unless it gets concrete code examples and numbers. "Increase X for Y" advice is not actionable for an LLM.

---

## SUMMARY

**Overall quality: 7/10.** The structure is good: anti-patterns first, then quick reference, then details. The anti-patterns section is the best part and genuinely useful. The xarray and conflict resolution sections are well done.

**Main weaknesses:**
- Multiple incorrect function signatures in the Core API section (save_config, flush, return types)
- Missing `open_or_create()` which is a first-class API method solving exactly the problem anti-pattern #5 describes
- No mention of async API variants
- Several important Session methods missing (amend, move)
- Version requirements for xarray/dask integrations not documented

**If I had to cut to 400 lines:** Remove anti-pattern #7 (old API), Performance Tips, Check Uncommitted Changes, and Tagging Releases. Compress the Core API signatures into a more compact table format.

**If I had to expand to 700 lines:** Add `open_or_create`, async note, `amend`, `move`/`rearrange_session`, `repo.diff()`, version requirements, the uncommitted-changes-with-dask gotcha, `NoChangesToCommitError`, and concrete configuration examples with numbers.
