"""Finite vector math with scale-safe cosine normalization."""
import json
import math


def checked_vector(vector, dims=None, *, empty=False):
    if not isinstance(vector, (list, tuple)) or (not vector and not empty):
        raise ValueError("vector must be a nonempty list or tuple")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
        raise ValueError("vector coordinates must be real numbers, not booleans")
    try:
        out = [float(x) for x in vector]
    except (OverflowError, ValueError) as exc:
        raise ValueError("vector coordinates must be finite") from exc
    if any(not math.isfinite(x) for x in out):
        raise ValueError("vector coordinates must be finite")
    if dims is not None and len(out) != dims:
        raise ValueError("vector dim %d != expected %d" % (len(out), dims))
    return out


def checked_items(items):
    out, seen, dims = [], set(), None
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError("item needs a nonempty string id")
        if item["id"] in seen:
            raise ValueError("duplicate vector id: " + item["id"])
        vector = checked_vector(item.get("vector"), dims)
        dims = len(vector)
        seen.add(item["id"])
        out.append({"id": item["id"], "vector": vector})
    return out


def positive_integer(x, name, zero=False):
    if isinstance(x, bool) or not isinstance(x, int) or x < (0 if zero else 1):
        raise ValueError(name + " must be a " + ("nonnegative" if zero else "positive") + " integer")
    return x


def unit(a):
    a = checked_vector(a)
    scale = max(map(abs, a))
    if scale == 0:
        return [0.0] * len(a)
    scaled = [x / scale for x in a]
    length = math.hypot(*scaled)
    return [x / length for x in scaled]


def dot(a, b):
    a = checked_vector(a)
    b = checked_vector(b, len(a))
    try:
        result = math.fsum(x * y for x, y in zip(a, b))
    except (OverflowError, ValueError) as exc:
        raise ValueError("dot product is outside finite range") from exc
    if not math.isfinite(result):
        raise ValueError("dot product is outside finite range")
    return result


def norm(a):
    return math.hypot(*checked_vector(a))


def cosine_similarity(a, b):
    a = checked_vector(a)
    b = checked_vector(b, len(a))
    return max(-1.0, min(1.0, math.fsum(x*y for x,y in zip(unit(a), unit(b)))))


def l2_distance(a, b):
    a = checked_vector(a)
    b = checked_vector(b, len(a))
    distance = math.dist(a, b)
    if not math.isfinite(distance):
        raise ValueError("L2 distance is outside finite range")
    return distance


def load_vectors(path):
    items, seen, dims = [], set(), None
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                item = checked_items([{"id": record.get("id", record.get("doc_id", "v%d" % (len(items)+1))),
                              "vector": record.get("vector")}])[0]
                if item["id"] in seen:
                    raise ValueError("duplicate vector id: " + item["id"])
                if dims is not None and len(item["vector"]) != dims:
                    raise ValueError("inconsistent dimensions")
                dims = len(item["vector"])
                seen.add(item["id"])
                items.append(item)
            except (ValueError, TypeError) as exc:
                raise ValueError("%s:%d %s" % (path, lineno, exc)) from exc
    if not items:
        raise ValueError("%s: no vectors found" % path)
    items = checked_items(items)
    return items, len(items[0]["vector"])
