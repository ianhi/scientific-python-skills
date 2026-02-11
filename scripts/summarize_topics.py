#!/usr/bin/env python3
"""Summarize top issues by topic clusters for a given research directory.

Usage: python summarize_topics.py <research_dir>
Example: python summarize_topics.py research/zarr

Reads ranked_issues.jsonl and produces topic_summary.json with issues
grouped by detected topic keywords.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

TOPIC_KEYWORDS = {
    "storage/stores": ["store", "storage", "s3", "gcs", "azure", "fsspec", "filesystem", "local", "remote", "zip"],
    "codecs/compression": ["codec", "compress", "blosc", "zstd", "lz4", "gzip", "filter", "shuffle", "endian"],
    "arrays/chunks": ["chunk", "array", "shape", "dtype", "resize", "append", "slice", "indexing", "dimension"],
    "groups/hierarchy": ["group", "hierarchy", "tree", "path", "consolidat", "metadata", "attrs", ".zattrs", ".zgroup"],
    "io/serialization": ["open", "save", "load", "read", "write", "zarr.open", "from_array", "to_zarr", "from_zarr"],
    "concurrency/async": ["async", "concurrent", "parallel", "thread", "lock", "dask", "multiprocess"],
    "compatibility/migration": ["v2", "v3", "migrate", "breaking", "backward", "compat", "deprecat", "legacy"],
    "performance": ["slow", "performance", "memory", "speed", "benchmark", "optimize", "fast", "latency"],
    "errors/bugs": ["bug", "error", "crash", "traceback", "exception", "fail", "broken", "corrupt", "fix"],
    "xarray-integration": ["xarray", "dataset", "dataarray", "encoding", "engine", "backend"],
    "dask-integration": ["dask", "distributed", "delayed", "compute", "scheduler"],
    "visualization": ["plot", "display", "repr", "html", "notebook", "widget"],
    "indexing/selection": ["sel", "isel", "loc", "iloc", "where", "query", "mask", "groupby", "resample"],
    "time-series": ["datetime", "timedelta", "cftime", "calendar", "freq", "period", "date_range"],
    "netcdf/io": ["netcdf", "nc4", "hdf5", "h5py", "opendap", "kerchunk", "engine"],
}

def main():
    research_dir = Path(sys.argv[1])
    ranked_file = research_dir / "ranked_issues.jsonl"

    if not ranked_file.exists():
        print(f"Error: {ranked_file} not found", file=sys.stderr)
        sys.exit(1)

    items = [json.loads(line) for line in ranked_file.read_text().splitlines() if line.strip()]

    # Classify items into topics
    topics = defaultdict(list)
    for item in items[:200]:  # Only top 200
        title_lower = item["title"].lower()
        labels_lower = " ".join(item.get("labels", [])).lower()
        text = f"{title_lower} {labels_lower}"

        matched = False
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                topics[topic].append({
                    "number": item["number"],
                    "title": item["title"],
                    "score": item["score"],
                    "comments": item["comments"],
                    "is_pr": item["is_pr"],
                    "state": item["state"],
                })
                matched = True

        if not matched:
            topics["other"].append({
                "number": item["number"],
                "title": item["title"],
                "score": item["score"],
                "comments": item["comments"],
                "is_pr": item["is_pr"],
                "state": item["state"],
            })

    # Sort topics by total engagement
    output = {}
    for topic in sorted(topics, key=lambda t: sum(i["score"] for i in topics[t]), reverse=True):
        items_list = topics[topic]
        output[topic] = {
            "count": len(items_list),
            "total_score": sum(i["score"] for i in items_list),
            "items": items_list[:15],  # Top 15 per topic
        }

    outpath = research_dir / "topic_summary.json"
    outpath.write_text(json.dumps(output, indent=2))
    print(f"Wrote {outpath}")
    print(f"\nTopic distribution:")
    for topic, data in output.items():
        print(f"  {topic}: {data['count']} items, total_score={data['total_score']}")

if __name__ == "__main__":
    main()
