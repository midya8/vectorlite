# VectorLite

Cosine/L2 search, random-hyperplane candidate filtering and an optional
block-bucket backend, with finite browser inspectors. Dependency-free Python.

[Signature inspector](https://midya8.github.io/vectorlite/) · [Block-bucket inspector](https://midya8.github.io/vectorlite/buckets.html) · [Numerical/probability contract](METHOD.md) · [Original README](docs/original-readme.md)

Every projection, admission and missed canonical exact neighbor is inspectable. Edit a query, vector table or the actual planes; recompute all signatures and exact scores in independent scalar JavaScript. Sweep every candidate threshold, distinguish observed fixed-index recall from iid-plane expected recall, and export the complete witness JSON. No model API or training is required.

## Choose a retrieval path

| Question | API | What is exact or bounded |
|---|---|---|
| Rank supplied vectors | `ExactIndex` | All cosine or negative-L2 scores, then deterministic ID ties; floating arithmetic |
| Filter by shared hyperplane bits | `LSHIndex` | Scan every signature, then rescore admitted candidates by cosine |
| Reuse fixed-radius signature postings | `HammingBuckets`, `BucketLSHIndex` | Exact Hamming-radius candidates, not exact cosine neighbors; explicit scan fallback |
| Inspect every projection and threshold | `vectorlite.audit.audit` / `python -m vectorlite audit` | 64 items, 16 dimensions, 128 bits, 128 KiB JSON |
| Inspect block lookups and checked rows | `python -m vectorlite.bucket_cli` | 1,024 rows, 128 bits, 128 KiB JSON |

A signature is not a neighbor. Canonical exact-neighbor recall is defined by
this implementation's score/ID ordering, not human relevance. Posting visits
and rescoring counts are not latency measurements. The original signature
scan remains the default; block buckets are an optional fixed-code lookup path.

## Numerical and probability contract

- **Scale-safe math:** independently normalized cosine avoids direct square overflow/underflow. Finite/dimension/id validation, stable L2 and copied input vectors prevent silent invalid results.
- **Full candidate witnesses:** raw planes, normalized dots, every signature bit, shared-bit count, exclusions, exact/approximate ranking and every threshold 0..bits. This audit uses the original linear signature scan, not the optional block backend. Candidate rescoring fractions are not latency measurements.
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

## Reusable block buckets with explicit fallback

The original `LSHIndex` remains an unchanged signature scan. `HammingBuckets`
and `BucketLSHIndex` add reusable exact-block posting tables with full Hamming
verification and unchanged cosine/id rescoring. At radius `r<bits`, `r+1` disjoint
blocks guarantee every true radius neighbor shares an exact block. Duplicate
signatures retain separate row ids. Dense cases and large-radius/block-cap cases
fall back explicitly; auto reports its posting-length rule and work counters.

A constructed 512-row, 24-bit/radius-2 control checks 9 unique codes after 3 lookups
and 11 posting visits instead of 512 scan checks. A dense duplicate control
would visit 1,536 postings and triggers scan fallback. These are structural counts,
not timing, measured RAM, semantic relevance or a general sublinear guarantee.
The 34 complete native/independent BigInt browser reports retain every checked
row. The original 12-vector corpus, 35 audit cases, scan and ranking conventions
remain intact.

```python
from vectorlite import BucketLSHIndex
index = BucketLSHIndex(items, num_bits=24, min_shared_bits=22)
neighbors = index.query(query, k=3, mode="auto")
```

A shared block is necessary, not sufficient; the backend checks full-code
distance after the posting union. `auto` sums
posting lengths and scans when that total is at least the table size. Radius
covering all bits and block-cap cases explicitly fall back, even in forced
`blocks` mode. The native block API allows up to 4,096 bits; the bounded CLI and
browser have the smaller limits above. No online insert/delete is implemented;
a different radius or threshold requires rebuilding. See [BUCKETS.md](BUCKETS.md)
for modes, exclusions, work accounting and established multi-index hashing credit.

## Verification and retained controls

```sh
python -m unittest discover -s tests -v
bash examples/demo.sh
python scripts/build_audit.py
python scripts/build_buckets.py
git diff --exit-code docs/audit.json examples/buckets.json docs/buckets-data.js
node --test --test-concurrency=1 tests/engine.cjs tests/test_browser.cjs \
  tests/test_buckets.cjs tests/test_buckets_browser.cjs
bash scripts/wheel_smoke.sh
```

The current suite has **59 Python tests and eight Node/Chrome tests**. The
original corpus/demo retain self-inclusive mean recall@3 of **1.0**. The original
README, including its older claims, is preserved as a historical document;
[METHOD.md](METHOD.md) defines the corrected signature-scan contract.

- **Numerical and probability references:** 90-digit Decimal cosine checks,
  rational binomial tails, exhaustive sign patterns, full-circle geometry and
  1,200 seeded rebuilds. These are finite checks, not proof of ideal PRNG behavior.
- **Scan audit:** 35 retained reports, 643 threshold rows and 20,784 numerical
  fields checked in independent scalar JavaScript. Actual Chrome checks 332
  vector rows, 643 threshold rows and 5,856 projection rows, plus export,
  invalidation, bad inputs and 320/390/1280 containment.
- **Buckets:** 5,376 exhaustive query/radius/mode comparisons per language,
  480 random excluded-query/mode cases, 360 random cosine rankings and 105
  original-spec backend rankings. Independent native/BigInt reports agree
  exactly for all 34 retained controls, including sparse and dense cases.
  Chrome checks all 1,908 selected rows, policy changes, invalidation, strict
  errors, two trusted downloads and 24 views at 1280/768/390/320 widths.
- **Installed wheel:** eight named import origins checked outside the checkout,
  all 35 original audits, 105 original-spec backend rankings, 34 bucket reports,
  original query/audit commands and both bucket CLI entry points. CI covers
  Python 3.10-3.13; the installed job uses Python 3.13.

Lowering the shared-bit threshold cannot reduce canonical recall when planes,
query, k and score/ID order stay fixed. It does not follow that more bits or a
new seed improve recall. Near ties and near-zero projections remain
floating-point sensitive. Synthetic vectors are not learned embeddings or
human relevance judgments. An external query file is not proof of a held-out
split. Ideal-plane expectations are not confidence intervals for fixed planes.

No new algorithm, timing, measured-RAM, production-scale, semantic-relevance,
recall-gain or model-quality claim is made. Method credit and detailed limits
remain in [METHOD.md](METHOD.md) and [BUCKETS.md](BUCKETS.md).

## License

MIT.
