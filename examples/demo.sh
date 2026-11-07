#!/usr/bin/env bash
# Fully offline VectorLite demo: exact vs LSH search over a clustered
# dataset, recall measurement against the exact baseline, and both sides
# of the recall gate.
set -u
cd "$(dirname "$0")/.."

OUT="$(mktemp -d /tmp/vectorlite-demo-XXXX)"
trap 'echo "demo artifacts: $OUT"' EXIT

echo "== exact query =="
python3 -m vectorlite query \
  --vectors examples/data/vectors.jsonl \
  --query "0.9,0.1,0,0,0.1,0,0,0.1" -k 3 | tee "$OUT/exact.txt"
grep -qE 'bpe-merges|byte-fallback|vocab-size|pretoken-split' "$OUT/exact.txt" || { echo "FAIL exact nearest missing"; exit 1; }

echo "== benchmark (expect pass) =="
python3 -m vectorlite benchmark \
  --vectors examples/data/vectors.jsonl -k 3 \
  --json "$OUT/bench.json" --min-recall 0.9 || { echo "FAIL benchmark gate that should pass"; exit 1; }

python3 - "$OUT/bench.json" <<'PY'
import json, sys
bench = json.load(open(sys.argv[1]))
assert bench["vectors"] == 12
assert bench["mean_recall"] >= 0.9, bench
print("assertions ok: mean recall %.4f over %d queries" % (bench["mean_recall"], bench["queries"]))
PY

echo "== gate (expect fail) =="
if python3 -m vectorlite benchmark --vectors examples/data/vectors.jsonl -k 3 \
  --min-recall 1.01 >/dev/null 2>&1; then
  echo "FAIL gate that should fail"; exit 1
fi
echo "gate failed as expected"

echo "demo ok"
