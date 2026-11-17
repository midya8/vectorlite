# vectorlite

Nearest-neighbor search over vectors in plain Python: brute-force cosine/L2 as the baseline, and random-hyperplane LSH as the approximate version, so you can see how much recall you give up. Small enough to read, no NumPy needed.

```bash
python -m vectorlite query --vectors examples/data/vectors.jsonl \
  --query '0.9,0.1,0,0,0.1,0,0,0.1' -k 3

python -m vectorlite benchmark --vectors examples/data/vectors.jsonl -k 3 --min-recall 0.9
python -m vectorlite benchmark --vectors examples/data/vectors.jsonl -k 3 \
  --exclude-self --bits 64 --min-shared-bits 48
```

Vectors are JSONL, one `{"id": "doc", "vector": [1, 0, ...]}` per line. `benchmark` compares LSH results against exact search and reports recall@k. On the 12-vector example with 64-bit signatures, recall@3 is 1.0 whether or not each query is allowed to find itself. That's a tiny dataset, so take it as "it works", not as a number to compare against anything. `--queries` lets you benchmark with a separate query file instead.

How the LSH part works: each vector gets a bit signature from which side of a set of random hyperplanes it falls on. Vectors whose signatures share at least `--min-shared-bits` bits become candidates, and only those are rescored with real cosine similarity. Lower the threshold and you get more recall but check more candidates.

From Python, `ExactIndex` and `LSHIndex` are the two main classes. There's also `BucketLSHIndex`, which splits signatures into blocks and looks them up in hash tables instead of scanning every signature, falling back to a scan when the buckets get too crowded. BUCKETS.md and METHOD.md have the details.

The browser pages let you edit the vectors and hyperplanes and watch which neighbors get missed: https://midya8.github.io/vectorlite/

## Tests

`python -m unittest discover -s tests` (62 tests), `bash examples/demo.sh`.
