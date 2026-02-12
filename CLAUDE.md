# Scientific Python Skills Project

## Goal
Create Claude Code skills (.md files in `skills/`) that teach Claude to write idiomatic code for the scientific Python ecosystem.

## Structure
- `skills/` - Final skill files (the deliverables)
- `skills/TEMPLATE.md` - Template for new skill files
- `repos/` - Local shallow clones of upstream repos (for reading docs + source, gitignored)
- `research/` - Intermediate research data per library
  - `{lib}/docs_summary.md` - Comprehensive docs + source analysis
  - `{lib}/topic_summary.json` - GitHub issues clustered by topic
  - `{lib}/review.md` - Reviewer feedback on skill file accuracy
- `scripts/` - Shared tooling for fetching/analyzing GitHub data
- `tests/` - Test scripts and reports validating skill files
  - `{lib}_test/test_{lib}_skill.py` - Test script guided ONLY by skill file
  - `{lib}_test/REPORT.md` - What worked, what failed, suggestions

## Target Libraries
### Phase 1 (Complete)
- zarr (zarr-developers/zarr-python) - v3.1.5
- xarray (pydata/xarray) - v2026.1.0
- icechunk (earth-mover/icechunk) - v1.1.18

### Phase 2 (Planned)
- numpy, scipy, pandas, polars, matplotlib
- Integration skill (zarr + xarray + icechunk together)

## Skill File Requirements
- **Max 30KB** per skill file
- **Lead with modern patterns** - teach the correct, idiomatic way to use the library FIRST
- **Don't over-focus on removed APIs** - only warn about removals if they are recent AND Claude is likely to generate the old pattern. Ancient removals that Claude doesn't know about anyway are noise.
- **Encourage good behavior over disallowing bad behavior** - show the right way to do things rather than exhaustively listing every wrong way
- Code over prose - working snippets over explanations
- Include function signatures for key APIs, verified against actual source code
- Link issue numbers for known bugs/gotchas
- Structure: Modern Patterns -> Core API -> Migration Notes (compact) -> Integration -> Gotchas -> Known Limitations -> Performance Tips
- Consolidate removed/deprecated API lists into compact markdown tables, NOT verbose code blocks for each item
- Label performance-only issues as "SLOW"/"FAST", NOT "WRONG"/"RIGHT" (Claude will refuse to generate functional "WRONG" code)

## Process per Library

### Phase 1: Research
1. Clone repo: `git clone --depth 1 <url> repos/<name>`
2. Read docs + source via Explore agents (parallel per lib)
3. Fetch issues/PRs via `scripts/fetch_issues.sh <owner/repo> research/<lib>`
4. Cluster by topic via `uv run python scripts/summarize_topics.py research/<lib>`
5. Save research to `research/{lib}/docs_summary.md`

### Phase 2: Write
6. Synthesize into single skill file using `skills/TEMPLATE.md`
7. Use general-purpose agents (parallel per lib) with research + source access

### Phase 3: Review
8. Launch fresh-eyes reviewer agents that read ONLY the skill file + source code
9. Reviewer writes `research/{lib}/review.md` grading accuracy and usefulness
10. Apply critical fixes from review (wrong signatures, broken examples, misleading claims)

### Phase 4: Test
11. Launch test agents that read ONLY the skill file, write a script, and execute it
12. Test agent writes `tests/{lib}_test/REPORT.md` documenting what worked/failed
13. Apply fixes found by testing

## Lessons Learned

### Agent Coordination
- **Explore agents can't write files** - they output text. Extract research content yourself rather than spawning more agents to parse JSON output.
- **Background agents can't get interactive Bash approval** - write scripts yourself first as reusable tools in `scripts/`, then give agents the exact command to run.
- **Parallel agents are effective** - run research, writing, reviewing, and testing agents in parallel (one per library). 3 agents working simultaneously is the sweet spot.

### Skill File Quality
- **Lead with modern patterns, not anti-patterns.** Teaching the correct way is more valuable than exhaustively listing wrong ways. Show good code first, migration notes second.
- **Only warn about recent removals Claude is likely to generate.** If an API was removed years ago and Claude doesn't know about it, a warning adds no value. Focus on changes from the version Claude's training data covered.
- **Consolidate deprecation lists into compact tables.** A 2-column table (old → new) is better than 10 separate code blocks with WRONG/RIGHT pairs.
- **Anti-patterns still matter for high-impact behavioral changes** (e.g., zarr v2→v3 patterns, pandas CoW semantics, NumPy copy=False). These deserve detailed explanation because the OLD code silently does the wrong thing.
- **Every code example must be verified against source.** Inaccurate signatures are WORSE than no signatures - Claude trusts the skill file implicitly. Common errors:
  - Wrong parameter names (e.g., `expiry_time` vs `delete_object_older_than`)
  - Wrong return types (e.g., `list[str]` vs `set[str]`)
  - Wrong defaults (e.g., `zarr_format=None` vs `zarr_format=3`)
  - Non-existent parameters (e.g., `codecs=` on `create_array()` - must use `serializer=` + `compressors=`)
  - Positional vs keyword-only arguments (e.g., `name=` is keyword-only in `zarr.create_array()`)
- **Test the skill file by having an agent use ONLY the skill file** to write and execute real code. This catches errors that review misses.
- **Performance issues need different labels than correctness issues.** Using "WRONG" for slow-but-correct code (e.g., groupby without flox) makes Claude refuse to generate functional code. Use "SLOW"/"FAST" instead.

### Content Priorities
- **Issue comments from maintainers** are the most valuable research source - they reveal real user pain points and correct solutions.
- **Migration guides** (e.g., v2 to v3) should be a first-class research artifact, not just extracted from scattered issues.
- **Don't include basic content Claude already knows** (e.g., `import xarray as xr`, basic plotting). Every line should teach something non-obvious.
- **StackOverflow/Discourse would complement GitHub issues** - issues skew toward bugs/features, forums capture "how do I" confusion.

### Specific Library Notes
- **zarr v3**: `codecs=` only works on `ShardingCodec`, NOT on `create_array()`. Use `serializer=`, `compressors=`, `filters=` separately. `from_array()` has store-first, data keyword-only signature.
- **xarray**: `ds.dims` is being deprecated for dict access - use `ds.sizes`. `use_cftime` kwarg is deprecated - use `CFDatetimeCoder`. Consolidated metadata emits warning with zarr v3.
- **icechunk**: `save_config()` takes no args. `flush()` requires `message`. Use `open_or_create()` instead of try/except for repo creation. All methods have `_async` counterparts.

## Environment
- Use `uv run python` for Python commands
- Use `npx` for Node-based tools
- Never use bare `python` or `pip`
- Packages: zarr, xarray, icechunk, numpy, dask[complete], h5netcdf, h5py
