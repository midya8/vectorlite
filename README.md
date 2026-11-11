# VectorLite

**A signature is not a neighbor.** A dependency-free cosine/L2 search library and a working finite random-hyperplane retrieval inspector.

[Live inspector](https://midya8.github.io/vectorlite/) · [Numerical/probability contract](METHOD.md) · [Original README](docs/original-readme.md)

Every projection, admission and missed canonical exact neighbor is inspectable. Edit a query, vector table or the actual planes; recompute all signatures and exact scores in independent scalar JavaScript. Sweep every candidate threshold, distinguish observed fixed-index recall from iid-plane expected recall, and export the complete witness JSON. No model API or training is required.

## What changed

- **Scale-safe math:** independently normalized cosine avoids direct square overflow/underflow. Finite/dimension/id validation, stable L2 and copied input vectors prevent silent invalid results.
- **Full candidate witnesses:** raw planes, normalized dots, every signature bit, shared-bit count, exclusions, exact/approximate ranking and every threshold 0..bits. This is a linear signature scan, not a bucketed or sublinear index. Candidate rescoring fractions are not latency measurements.
- **Probability reference:** binomial admission tails from random-hyperplane angular agreement, expected candidate count and expected canonical recall. These are marginal expectations over ideal iid isotropic rebuilds, not conditional probabilities for these fixed planes. Zero vectors retain cosine 0/all-positive bits but have undefined angular theory.
- **Protocol controls:** original self-inclusive benchmark retained; leave-one-id-out and externally supplied queries are explicit alternatives. A two-vector control changes recall from 1 to 0 when only self exclusion changes. Neither protocol automatically establishes held-out relevance quality.
- **Honest gates:** full-precision means, not 4-decimal rounded means, drive recall gates. Approximate L2 and invalid/nonfinite inputs now fail clearly.

## Run

```bash
python -m vectorlite query --vectors examples/data/vectors.jsonl \
  --query '0.9,0.1,0,0,0.1,0,0,0.1' -k 3

# Historical self-inclusive protocol, still the default.
python -m vectorlite benchmark --vectors examples/data/vectors.jsonl \
  -k 3 --json bench.json --min-recall 0.9

# Explicit id exclusion from both exact and approximate ground truth.
python -m vectorlite benchmark --vectors examples/data/vectors.jsonl \
  -k 3 --exclude-self --bits 64 --min-shared-bits 48

# External supplied queries do not themselves prove a held-out split.
python -m vectorlite benchmark --vectors index.jsonl --queries queries.jsonl -k 3

python -m vectorlite audit --spec examples/audit.json --json report.json
python -m http.server 8000 --directory docs
```

Vector JSONL rows are `{"id":"doc","vector":[1,0]}`. `doc_id` and generated `v1`, `v2` ids remain supported. All coordinates must be finite numeric values; ids are unique. An audit spec contains items/query, optional bits/seed/k/threshold/exclude_ids and optional explicit hyperplanes. Audit bounds: 64 items, 16 dimensions, 128 bits, 128KiB strict JSON. Full CLI search permits larger indexes; exhaustive witness output is intentionally small.

## Verification and retained controls

```bash
python -m unittest discover -s tests -v
bash examples/demo.sh
python scripts/build_audit.py
git diff --exit-code docs/audit.json
node --test tests/engine.cjs tests/test_browser.cjs
bash scripts/wheel_smoke.sh
```

50 Python tests include all **31 original tests unchanged**. The original corpus/demo are byte-identical and retain self-inclusive mean recall@3 of **1.0**. The original README is preserved, including its older claims, as a historical document; METHOD.md defines the corrected contract.

Independent checks use 90-digit Decimal cosine references, rational binomial tails, exhaustive sign patterns, full-circle geometry and 1,200 seeded rebuilds. Thirty-five retained audits have 643 threshold rows; independent scalar JavaScript checks 20,784 numerical fields. Actual Chrome checks 332 vector rows, 643 threshold rows and 5,856 projection rows, export/invalidation/bad inputs and 320/390/1280 containment. Python 3.10-3.13 and installed-wheel checks exercise the CLI outside the checkout.

The monotone threshold result relies on fixed planes/query/k and the same deterministic exact score/id order. Near ties and near-zero projections remain floating-point sensitive. These synthetic vectors are not learned embeddings or human relevance judgments. There are no speed, production-scale, confidence-interval, model-quality or new-algorithm claims. See [METHOD.md](METHOD.md) for equations, bounds and six prior-art sources.

## License

MIT.


## Optional block-bucket retrieval (v0.3)

The original `LSHIndex` remains an unchanged signature scan. New `HammingBuckets`
and `BucketLSHIndex` add reusable exact-block posting tables with full Hamming
verification and unchanged cosine/id rescoring. At radius r<bits, r+1 disjoint
blocks guarantee every true radius neighbor shares an exact block. Duplicate
signatures retain separate row ids. Dense cases and large-radius/block-cap cases
fall back explicitly; auto reports its posting-length rule and work counters.

A constructed512-row24-bit/radius2 control checks9 unique codes after3 lookups and
11 posting visits instead of512 scan checks, while a dense duplicate control
would visit1,536 postings and triggers scan fallback. These are structural counts,
not timing, measured RAM, semantic relevance or a general sublinear guarantee.
The34 complete native/independent BigInt browser reports retain every checked
row. Old12-vector corpus,35 audit cases, scan and ranking conventions are intact.

```python
from vectorlite import BucketLSHIndex
index = BucketLSHIndex(items, num_bits=24, min_shared_bits=22)
neighbors = index.query(query, k=3, mode="auto")
```

59 Python tests include all50 prior tests. Eight Node/actual-Chrome tests include
all3 prior tests. Exhaustive cube oracles check5,376 query/radius/mode combinations
per language;480 random excluded-query/mode cases,360 random cosine rankings and
105 old-spec backend rankings check equivalence with independent scans. A clean
installed8-module artifact checks original commands/audits and both new CLI
entrypoints. See [BUCKETS.md](BUCKETS.md) for algorithm credit, caps, costs and
limits. Multi-index hashing is established prior work, not a new algorithm here.
