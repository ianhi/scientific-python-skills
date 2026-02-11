# Icechunk Skill File Test Report

**Skill file**: `skills/icechunk.md`
**Test script**: `tests/icechunk_test/test_icechunk_skill.py`
**Date**: 2026-02-11
**icechunk version**: >= 1.1.18 (installed via pyproject.toml)
**zarr version**: >= 3.1.5

## Summary

8 sections were tested. 7 worked on first attempt using the skill file examples directly. 1 had a code error in the skill file that required correction.

## Examples That Worked Correctly

### Section 1: Create a repository with in-memory storage
The skill file examples for `icechunk.in_memory_storage()`, `Repository.create(storage)`, `repo.writable_session("main")`, `zarr.create_group(session.store)`, `root.create_array(...)`, and `session.commit(message)` all worked exactly as documented.

### Section 2: Branching workflow
`repo.lookup_branch("main")`, `repo.create_branch("dev", snapshot_id=...)`, writing to separate branches, and `repo.readonly_session(branch=...)` all worked correctly. `repo.list_branches()` returned the expected `{'main', 'dev'}` set.

### Section 3: Tagging
`repo.create_tag("v1.0-initial", snapshot_id=...)`, `repo.readonly_session(tag="v1.0-initial")`, `repo.list_tags()`, and `repo.lookup_tag(name)` all worked exactly as documented. Time-travel to a tagged snapshot correctly returned the original data.

### Section 4: History browsing
`repo.ancestry(branch="main")` returned an iterator of `SnapshotInfo` objects. The properties `ancestor.id`, `ancestor.message`, and `ancestor.written_at` all worked as documented. The ancestry correctly included the initial "Repository initialized" commit that icechunk creates automatically.

### Section 5: Conflict resolution
Creating two writable sessions on the same branch, committing the first, then attempting to commit the second correctly raised `icechunk.ConflictError`. Calling `session.rebase(icechunk.BasicConflictSolver(on_chunk_conflict=icechunk.VersionSelection.UseOurs))` followed by `session.commit(...)` worked exactly as documented. The "UseOurs" resolution correctly preserved the second session's value.

### Section 7: Xarray integration
`to_icechunk(ds, session, mode="w")` and `xr.open_zarr(session.store, consolidated=False)` worked correctly. The round-trip was verified with `xr.testing.assert_equal`. Note: the read-back dataset uses dask-backed arrays (lazy loading), which is expected behavior.

### Section 8: Session lifecycle
After `session.commit(...)`, `session.read_only` correctly returned `True`. Attempting to write to the committed session's store raised `IcechunkError: cannot write to read-only store`. Creating a new `repo.writable_session("main")` and writing through it worked correctly.

## Examples That Were Wrong or Misleading

### Section 6: Transaction context manager - INCORRECT code example

**Skill file code (Quick Reference section):**
```python
with repo.transaction("main", message="Auto-commit") as store:
    zarr.create_array(store, "data", shape=(10,), dtype=float)
```

**Error:**
```
TypeError: create_array() takes 1 positional argument but 2 positional arguments (and 2 keyword-only arguments) were given
```

**Root cause:** In zarr 3, `zarr.create_array()` accepts `store` as the only positional argument. The array name/path must be passed as a keyword argument: `name="data"`.

**Correct code:**
```python
with repo.transaction("main", message="Auto-commit") as store:
    zarr.create_array(store, name="data", shape=(10,), dtype=float)
```

This same error pattern appears in the "Write with Zarr Directly" example under Patterns & Idioms, where `root.create_array("temperature", shape=..., ...)` is used. However, when called on a `zarr.Group` object (as opposed to `zarr.create_array` at module level), the first positional argument IS the name, so `root.create_array("temperature", ...)` works correctly. The inconsistency is that `zarr.create_array(store, "name", ...)` does NOT work the same way as `group.create_array("name", ...)`.

**Recommendation:** Fix the transaction example to use `name=` keyword argument:
```python
with repo.transaction("main", message="Auto-commit") as store:
    zarr.create_array(store, name="data", shape=(10,), dtype=float)
```

## What Was Missing From the Skill File

1. **No mention of the automatic "Repository initialized" commit.** When you create a repository, icechunk automatically creates an initial snapshot with the message "Repository initialized". This shows up in ancestry listings and affects snapshot count expectations. The skill file should mention this.

2. **The `zarr.open_array(store, path=...)` pattern is not documented.** The skill file shows how to create arrays and groups, but does not show how to open an existing array from a store by path. This is essential for branching workflows (open array on a different branch to read/modify it).

3. **`session.store` behavior after commit is not fully explained.** The skill file says "session is read-only after commit" but does not clarify that `session.store` also becomes read-only (writes to the store raise `IcechunkError`). It would help to be explicit that the store object inherits the session's read-only state.

4. **No explicit examples of `zarr.open_array` or `zarr.open_group` for reading via store.** The "Read with xarray" pattern is documented, but reading individual zarr arrays from a store (without xarray) is not shown.

## Suggestions for Improving the Skill File

1. **Fix the transaction context manager example** to use `name=` keyword argument for `zarr.create_array()`. This is the only code error found.

2. **Add a note about the auto-created initial snapshot** ("Repository initialized") so users are not surprised when they see it in ancestry listings.

3. **Add a "Read with Zarr Directly" section** showing `zarr.open_array(session.store, path="temperature")` and `zarr.open_group(session.store)` patterns, parallel to the existing "Write with Zarr Directly" section.

4. **Clarify that `session.store` becomes read-only after commit** in the gotchas section, not just the session itself.

5. **Add a note that `xr.open_zarr` returns dask-backed (lazy) arrays** by default. This is standard xarray/zarr behavior but worth noting since users may expect eager arrays when working with small in-memory datasets.

6. **Consider adding `zarr_format=3` to the `to_zarr` examples.** The skill file mentions `zarr_format=3` in the anti-patterns section but some `to_zarr` calls in examples omit it. Since icechunk requires zarr v3 format, this should be consistent.
