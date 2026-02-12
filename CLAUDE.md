# Scientific Python Skills Project

## Goal
Create Claude Code skills (.md files in `skills/`) that teach Claude to write idiomatic code for the scientific Python ecosystem.

## Structure
- `skills/` - Final skill files (the deliverables)
- `skills/TEMPLATE.md` - Template for new skill files
- `repos/` - Local shallow clones of upstream repos (for reading docs + source, gitignored)
- `research/{lib}/` - Intermediate research data (docs_summary.md, topic_summary.json, review.md)
- `scripts/` - Shared tooling for fetching/analyzing GitHub data
- `tests/{lib}_test/` - Test scripts (test_{lib}_skill.py) and reports (REPORT.md)
- `PROJECT.md` - Project planning, phases, and status tracking

## Skill File Requirements
- **Max 30KB** per skill file, target 4-10KB
- **Focus on subtle errors and performance traps** - only get into details when there is real risk of silent errors or performance detriments
- **Lead with modern patterns** - teach the correct, idiomatic way first
- **Don't over-focus on removed APIs** - only warn about removals Claude is likely to generate
- Code over prose - working snippets over explanations
- Verify all code examples against actual source
- Link issue numbers for known bugs/gotchas
- Structure: Modern Patterns -> Migration Notes (compact table) -> Gotchas & Common Mistakes -> Known Limitations -> Performance Tips
- Label performance issues as "SLOW"/"FAST", NOT "WRONG"/"RIGHT"
- **Keep skill files lean** - cut API signatures, store configuration, integration examples, and anything Claude can derive from general knowledge

## Writing & Improving Skills

### Content Guidelines
- **Anti-patterns matter most for silent behavioral changes** (e.g., zarr v2→v3, pandas CoW, NumPy NEP 50). Detailed explanation is warranted when old code silently does the wrong thing.
- **Consolidate deprecation lists into compact tables.** A 2-column table (old → new) beats 10 separate WRONG/RIGHT code blocks.
- **Don't include basic content Claude already knows** (e.g., `import xarray as xr`, basic plotting). Every line should teach something non-obvious.
- **Issue comments from maintainers** are the most valuable research source - they reveal real user pain points and correct solutions.

### Verification
- **Every code example must be verified against source.** Inaccurate signatures are worse than no signatures. Common errors: wrong param names, wrong return types, wrong defaults, non-existent params, positional vs keyword-only args.
- **Test the skill file by having an agent use ONLY the skill file** to write and execute real code. This catches errors that review misses.

## Environment
- Use `uv run python` for Python commands
- Use `npx` for Node-based tools
- Never use bare `python` or `pip`
- Packages: zarr, xarray, icechunk, numpy, pandas, matplotlib, dask[complete], h5netcdf, h5py
