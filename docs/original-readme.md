# VectorLite

Exact and approximate kNN vector search from scratch, with no dependencies.

VectorLite implements brute-force exact kNN (cosine and L2) and a random-hyperplane LSH index for approximate cosine search in pure Python, then measures the approximation honestly: recall@k of the LSH index against the exact baseline, gateable in CI. It is dependency-free and runs fully offline.

## Why

Vector search demos usually show an index returning results, not how wrong those results are. VectorLite makes the tradeoff measurable:

- **an exact baseline** - brute-force scan with deterministic tie-breaking, so ground truth is always available at small and medium scale;
- **random-hyperplane LSH** - each vector hashes to a bit signature from seeded Gaussian hyperplanes, candidates share enough signature bits with the query and are rescored exactly; the same seed and data rebuild the identical index bit for bit;
- **recall as a first-class metric** - `benchmark` self-queries every vector through both indexes and reports mean recall@k of LSH against exact, with per-query detail in JSON;
- **a CI gate** - `--min-recall` fails the pipeline when an approximation change (fewer bits, a new seed, a lower sharing threshold) drops recall below policy.

## Usage

Exact search:

```bash
python3 -m vectorlite query \
  --vectors vectors.jsonl --query "0.9,0.1,0,0,0.1,0,0,0.1" -k 3
```

Approximate search and the recall benchmark:

```bash
python3 -m vectorlite query --vectors vectors.jsonl --query "..." -k 3 --approximate --bits 64
python3 -m vectorlite benchmark --vectors vectors.jsonl -k 3 \
  --json bench.json --min-recall 0.9
```

Vectors are JSONL, one `{"id": "...", "vector": [...]}` object per line with uniform dimensions (`doc_id` also accepted; ids default to `v1`, `v2`, ...).

## Verification

- 31 unit tests cover vector math, JSONL loading validation, exact ordering and tie-breaking, LSH build determinism and seed sensitivity, signature-bit similarity, recall metrics, and the CLI end to end including both sides of the recall gate.
- `bash examples/demo.sh` runs fully offline over a clustered 12-vector dataset: an exact query, a benchmark asserting mean recall@3 above 0.9, and both sides of the gate.
- CI runs the same suite across Python 3.10-3.13.

Brute force is O(n) per query by design: it exists to be correct, not fast, and LSH recall is measured against it. Random-hyperplane LSH approximates cosine similarity only; the L2 metric is exact-search only.

## License

MIT
