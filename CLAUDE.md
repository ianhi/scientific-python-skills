# Scientific Python Skills Project

## Goal
Create Claude Code skills (.md files in `skills/`) that teach Claude to write idiomatic code for the scientific Python ecosystem.

## Structure
- `skills/` - Final skill files (the deliverables)
- `repos/` - Local shallow clones of upstream repos (for reading docs + source)
- `research/` - Intermediate research data per library (issues, summaries, topic analysis)
- `scripts/` - Shared tooling for fetching/analyzing GitHub data

## Target Libraries (Phase 1)
- zarr (zarr-developers/zarr-python)
- xarray (pydata/xarray)
- icechunk (earth-mover/icechunk)

## Skill File Requirements
- **Max 30KB** per skill file
- Lead with anti-patterns ("DO NOT" section)
- Code over prose - working snippets over explanations
- Include function signatures for key APIs
- Link issue numbers for known bugs/gotchas
- Structure: Anti-patterns -> Quick Reference -> API Details -> Gotchas -> Known Limitations

## Process per Library
1. Read docs + source -> understand public API surface
2. Fetch issues/PRs via `scripts/fetch_issues.sh`
3. Rank by engagement, fetch top 100 details
4. Cluster by topic via `scripts/summarize_topics.py`
5. Launch parallel research agents by topic
6. Synthesize into single skill file
7. Verify every code example
8. Trim to <30KB

## Environment
- Use `uv run python` for Python commands
- Use `npx` for Node-based tools
- Never use bare `python` or `pip`
