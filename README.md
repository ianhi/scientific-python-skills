# Scientific Python Skills for Claude Code

> **WARNING: These skills are in active development and have not been thoroughly tested. Code examples may contain errors. Use at your own risk and always verify generated code against official documentation.**

Claude Code [skill files](https://code.claude.com/docs/en/skills) that teach Claude to write idiomatic, up-to-date code for scientific Python libraries. Each skill focuses on preventing common mistakes, especially around major version migrations (e.g., zarr v2 to v3).

## Available Skills

| Library | Version Targeted | Focus | Size |
|---------|-----------------|-------|------|
| [zarr](skills/zarr/SKILL.md) | 3.1.5 | v2-to-v3 migration anti-patterns, codecs, sharding, cloud stores | ~20KB |
| [xarray](skills/xarray/SKILL.md) | 2026.1.0 | apply_ufunc, zarr v3 encoding, dask integration, DataTree | ~24KB |
| [icechunk](skills/icechunk/SKILL.md) | 1.1.18 | Transactional storage, version control, session lifecycle, conflict resolution | ~19KB |
| [numpy](skills/numpy/SKILL.md) | 2.x | NEP 50 type promotion, StringDType, copy semantics | ~8KB |
| [pandas](skills/pandas/SKILL.md) | 3.x | Copy-on-Write, string dtype, changed defaults | ~12KB |
| [matplotlib](skills/matplotlib/SKILL.md) | 3.x | OO interface, colorbar pitfalls, animation, layout | ~8KB |

## How to Use

Each skill lives in its own directory as `SKILL.md`. Skills are automatically loaded by Claude Code when relevant — no explicit invocation needed.

### Option 1: Add to Your Project (Recommended)

Copy the skill directories you need into your project's `.claude/skills/` directory:

```bash
# From your project root
mkdir -p .claude/skills

# Copy individual skills (repeat for each library you need)
cp -r /path/to/scientific-python-skills/skills/zarr .claude/skills/
cp -r /path/to/scientific-python-skills/skills/xarray .claude/skills/
cp -r /path/to/scientific-python-skills/skills/icechunk .claude/skills/
```

Or fetch directly from GitHub:

```bash
mkdir -p .claude/skills/zarr
curl -o .claude/skills/zarr/SKILL.md \
  https://raw.githubusercontent.com/ianhi/scientific-python-skills/main/skills/zarr/SKILL.md
```

Skills placed in `.claude/skills/` are automatically discovered by Claude Code. Commit them to version control so your whole team benefits.

### Option 2: Add Globally (All Projects)

```bash
mkdir -p ~/.claude/skills/zarr ~/.claude/skills/xarray ~/.claude/skills/icechunk

curl -o ~/.claude/skills/zarr/SKILL.md \
  https://raw.githubusercontent.com/ianhi/scientific-python-skills/main/skills/zarr/SKILL.md
curl -o ~/.claude/skills/xarray/SKILL.md \
  https://raw.githubusercontent.com/ianhi/scientific-python-skills/main/skills/xarray/SKILL.md
curl -o ~/.claude/skills/icechunk/SKILL.md \
  https://raw.githubusercontent.com/ianhi/scientific-python-skills/main/skills/icechunk/SKILL.md
```

## What's in a Skill File?

Each `SKILL.md` follows the [Agent Skills](https://agentskills.io) open standard with YAML frontmatter and markdown content:

```yaml
---
name: zarr
description: Writing idiomatic zarr v3 code. Use when ...
user-invocable: false
---
```

The `description` tells Claude when to load the skill automatically. `user-invocable: false` keeps it out of the `/` command menu — these are background reference skills, not commands.

Content structure:

1. **Anti-patterns** (most valuable) - "DO NOT do this, DO this instead" with code examples
2. **Quick Reference** - Copy-paste ready common operations
3. **Gotchas** - Subtle bugs with GitHub issue references
4. **Known Limitations** - What doesn't work yet
5. **Performance Tips** - Actionable optimization advice

Skills are kept under 30KB to fit within context windows while maximizing signal.

## How Skills Were Built

Each skill was developed through a multi-phase process:

1. **Source analysis** - Read library docs and source code from shallow clones
2. **Issue mining** - Fetched all GitHub issues/PRs, ranked by engagement, analyzed top 100 in detail
3. **Topic clustering** - Grouped issues by theme to identify common pain points
4. **Skill drafting** - Synthesized research into structured skill files
5. **Peer review** - Fresh-eyes reviewer agents checked every code example against source
6. **Testing** - Agents wrote and executed test scripts guided only by the skill files

The `research/` directory contains intermediate artifacts (docs summaries, topic clusters, reviews) and `scripts/` has the tooling used.

## Contributing

### Adding a New Skill

1. Clone the repo and target library:
   ```bash
   git clone --depth 1 https://github.com/<org>/<repo> repos/<repo>
   ```

2. Fetch and analyze issues:
   ```bash
   ./scripts/fetch_issues.sh <org/repo> research/<library>
   uv run python scripts/summarize_topics.py research/<library>
   ```

3. Write the skill file following `skills/TEMPLATE/TEMPLATE.md`, placing it at `skills/<library>/SKILL.md`

4. Verify every code example actually runs

5. Keep it under 30KB

### Improving Existing Skills

- Found a wrong code example? Open an issue or PR with the exact error
- Missing an important pattern? PRs welcome
- Test reports in `tests/` document known issues

## Project Structure

```
skills/                   # The deliverables - skill files for Claude Code
  {lib}/
    SKILL.md              # Skill file (name, description frontmatter + content)
  TEMPLATE/
    TEMPLATE.md           # Template for new skills (not named SKILL.md, not auto-discovered)
research/                 # Research artifacts per library
  {lib}/
    docs_summary.md       # Docs + source analysis
    topic_summary.json    # Issues clustered by topic
    review.md             # Reviewer feedback
scripts/                  # Shared tooling
  fetch_issues.sh         # Fetch all issues via gh API
  summarize_topics.py     # Cluster issues by topic
tests/                    # Test scripts and reports
  {lib}_test/
    test_{lib}_skill.py   # Test script guided only by skill file
    REPORT.md             # What worked, what didn't
```

## License

MIT
