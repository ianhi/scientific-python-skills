#!/usr/bin/env bash
# Fetch all issues and PRs from a GitHub repo using gh API pagination
# Usage: ./fetch_issues.sh <owner/repo> <output_dir>
# Example: ./fetch_issues.sh zarr-developers/zarr-python research/zarr

set -euo pipefail

REPO="${1:?Usage: fetch_issues.sh <owner/repo> <output_dir>}"
OUTDIR="${2:?Usage: fetch_issues.sh <owner/repo> <output_dir>}"

mkdir -p "$OUTDIR"

echo "=== Fetching issues for $REPO ==="

# Fetch all issues (includes PRs) with key metadata
echo "Fetching all issues..."
gh api --paginate "repos/$REPO/issues?state=all&per_page=100&sort=comments&direction=desc" \
  --jq '.[] | {
    number: .number,
    title: .title,
    state: .state,
    is_pr: (.pull_request != null),
    comments: .comments,
    reactions_total: .reactions.total_count,
    labels: [.labels[].name],
    created_at: .created_at,
    updated_at: .updated_at,
    user: .user.login,
    body_length: (.body // "" | length)
  }' > "$OUTDIR/all_issues_raw.jsonl" 2>/dev/null

TOTAL=$(wc -l < "$OUTDIR/all_issues_raw.jsonl" | tr -d ' ')
echo "Fetched $TOTAL items total"

# Score and rank: recency * comments * reactions
# Score = comments * 2 + reactions * 3 + (1 if updated in last year)
echo "Ranking issues by engagement..."
cat "$OUTDIR/all_issues_raw.jsonl" | python3 -c "
import json, sys
from datetime import datetime, timezone

items = [json.loads(line) for line in sys.stdin]

cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
for item in items:
    updated = datetime.fromisoformat(item['updated_at'].replace('Z', '+00:00'))
    recency_bonus = 5 if updated > cutoff else 0
    item['score'] = item['comments'] * 2 + item['reactions_total'] * 3 + recency_bonus

items.sort(key=lambda x: x['score'], reverse=True)

for item in items:
    print(json.dumps(item))
" > "$OUTDIR/ranked_issues.jsonl"

# Extract top 100 issue numbers for detailed fetching
head -100 "$OUTDIR/ranked_issues.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    item = json.loads(line)
    print(item['number'])
" > "$OUTDIR/top_100_numbers.txt"

echo "Top 100 high-engagement issues identified"

# Fetch detailed comments for top issues (batched)
echo "Fetching comments for top issues..."
mkdir -p "$OUTDIR/comments"

while IFS= read -r num; do
  if [ ! -f "$OUTDIR/comments/${num}.json" ]; then
    gh api "repos/$REPO/issues/${num}/comments?per_page=100" \
      --jq '[.[] | {user: .user.login, body: .body, created_at: .created_at, reactions: .reactions.total_count}]' \
      > "$OUTDIR/comments/${num}.json" 2>/dev/null || true
    # Small delay to avoid rate limiting
    sleep 0.1
  fi
done < "$OUTDIR/top_100_numbers.txt"

echo "Fetching issue bodies for top issues..."
mkdir -p "$OUTDIR/bodies"

while IFS= read -r num; do
  if [ ! -f "$OUTDIR/bodies/${num}.json" ]; then
    gh api "repos/$REPO/issues/${num}" \
      --jq '{number: .number, title: .title, body: .body, state: .state, labels: [.labels[].name], user: .user.login}' \
      > "$OUTDIR/bodies/${num}.json" 2>/dev/null || true
    sleep 0.1
  fi
done < "$OUTDIR/top_100_numbers.txt"

# Generate summary stats
echo ""
echo "=== Summary for $REPO ==="
echo "Total issues/PRs: $TOTAL"

ISSUES_ONLY=$(grep '"is_pr":false' "$OUTDIR/all_issues_raw.jsonl" | wc -l | tr -d ' ')
PRS_ONLY=$(grep '"is_pr":true' "$OUTDIR/all_issues_raw.jsonl" | wc -l | tr -d ' ')
echo "Issues: $ISSUES_ONLY"
echo "PRs: $PRS_ONLY"

echo ""
echo "Top 10 by engagement:"
head -10 "$OUTDIR/ranked_issues.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    item = json.loads(line)
    kind = 'PR' if item['is_pr'] else 'Issue'
    print(f\"  #{item['number']} ({kind}, score={item['score']}, comments={item['comments']}): {item['title'][:80]}\")
"

echo ""
echo "=== Done. Output in $OUTDIR ==="
