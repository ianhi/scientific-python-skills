# Prompt: Next Batch Team Lead

Paste this into a fresh Claude Code session from the `scientific-python-skills` directory.

---

You are the team lead for the next batch of Claude Code skill files for the scientific Python ecosystem.

## Context

Read `CLAUDE.md` first - it contains the full process, lessons learned, and project structure from Phase 1 (zarr, xarray, icechunk). Also read `README.md` for the public-facing docs. Look at the existing skill files in `skills/` and test reports in `tests/` to understand the quality bar and format.

## Your Job

Create a **team** to develop skill files for these libraries in parallel:

1. **numpy** (numpy/numpy) - Focus on: modern NumPy (2.0+), new dtypes, array API standard, copy semantics changes, deprecations Claude gets wrong
2. **pandas** (pandas-dev/pandas) - Focus on: Copy-on-Write (default in 3.0), modern string dtype, arrow backend, deprecations (inplace, append), anti-patterns
3. **matplotlib** (matplotlib/matplotlib) - Focus on: implicit vs explicit API (pyplot vs OO), modern style, constrained_layout, subfigures, the patterns Claude consistently gets wrong

We can do scipy, polars, and the integration skill in a follow-up batch.

## Team Structure

Use `TeamCreate` to set up a team. Then create tasks and spawn teammates:

- **You (team lead)**: Coordinate, create tasks, review outputs, apply fixes, commit
- **3 researcher agents** (one per library, Explore type): Read docs + source, produce `research/{lib}/docs_summary.md`
- **3 writer agents** (one per library, general-purpose): Synthesize research into skill files
- **3 reviewer agents** (one per library, general-purpose): Fresh-eyes review against source code
- **3 test agents** (one per library, general-purpose): Write + execute test scripts guided ONLY by the skill file

Don't run all phases at once. Run them in waves:
1. First wave: Clone repos + fetch issues (you do this via background Bash) + launch researchers (Explore agents)
2. Second wave: Once research is saved, launch writers
3. Third wave: Once skills are written, launch reviewers
4. Fourth wave: Once fixes are applied, launch testers
5. Apply test-driven fixes, commit, push

## Critical Lessons from Phase 1 (read CLAUDE.md for full details)

- **Explore agents can't write files.** You must extract their research output yourself.
- **Background agents can't approve Bash interactively.** Pre-build scripts or run data-fetching yourself.
- **3 parallel agents** is the sweet spot per wave.
- **Anti-patterns are the #1 most valuable content.** Claude WILL generate outdated code without them.
- **Every code example must be verified against source.** Wrong signatures are worse than none.
- **Label slow-but-correct as "SLOW"/"FAST"**, not "WRONG"/"RIGHT".
- **Test by having agents use ONLY the skill file** - this catches errors review misses.
- **Don't include basics Claude already knows** (imports, simple operations).

## Existing Infrastructure

Scripts are ready to use:
```bash
# Clone repos
git clone --depth 1 https://github.com/numpy/numpy repos/numpy
git clone --depth 1 https://github.com/pandas-dev/pandas repos/pandas
git clone --depth 1 https://github.com/matplotlib/matplotlib repos/matplotlib

# Fetch issues (background, takes a few minutes each)
./scripts/fetch_issues.sh numpy/numpy research/numpy
./scripts/fetch_issues.sh pandas-dev/pandas research/pandas
./scripts/fetch_issues.sh matplotlib/matplotlib research/matplotlib

# Topic clustering
uv run python scripts/summarize_topics.py research/numpy
uv run python scripts/summarize_topics.py research/pandas
uv run python scripts/summarize_topics.py research/matplotlib
```

Template: `skills/TEMPLATE.md`
Environment: `uv run python` for all Python. Packages may need to be added via `uv add`.

## Definition of Done

- [ ] 3 skill files in `skills/` (numpy.md, pandas.md, matplotlib.md), each under 30KB
- [ ] Research artifacts in `research/{lib}/` (docs_summary.md, topic_summary.json, review.md)
- [ ] Test scripts and reports in `tests/{lib}_test/`
- [ ] All test reports show >90% pass rate
- [ ] Committed and pushed to origin/main
- [ ] CLAUDE.md updated with any new lessons learned
