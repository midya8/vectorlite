"""Vector math and dataset loading (pure Python, no numpy)."""

import json
import math


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a, b):
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def load_vectors(path):
    """Load vectors from JSONL: {"id": ..., "vector": [...]} per line."""
    items = []
    dims = None
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d invalid JSON: %s" % (path, lineno, exc)) from exc
            if not isinstance(record, dict) or not isinstance(record.get("vector"), list):
                raise ValueError("%s:%d record needs a list 'vector' field" % (path, lineno))
            vector = [float(x) for x in record["vector"]]
            if dims is None:
                dims = len(vector)
                if dims == 0:
                    raise ValueError("%s:%d vector must be non-empty" % (path, lineno))
            elif len(vector) != dims:
                raise ValueError(
                    "%s:%d vector dim %d != expected %d" % (path, lineno, len(vector), dims)
                )
            vid = record.get("id", record.get("doc_id", "v%d" % (len(items) + 1)))
            if not isinstance(vid, str):
                raise ValueError("%s:%d id must be a string" % (path, lineno))
            items.append({"id": vid, "vector": vector})
    if not items:
        raise ValueError("%s: no vectors found" % path)
    return items, dims
