"""Prints what a frame cache folder covers: python runners/baseten/cache_summary.py <cache folder>"""

import json
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
metas = sorted((json.loads(path.read_text()) for path in cache_dir.glob("meta_*.json")), key=lambda meta: meta["lo"])
gaps = sum(first["hi"] != second["lo"] for first, second in zip(metas, metas[1:]))
print(
    f"{cache_dir.name}: {len(metas)} shards, steps {metas[0]['lo']} to {metas[-1]['hi']}, {gaps} gaps, "
    f"{sum(meta.get('missing', 0) for meta in metas)} screenshots missing"
)
