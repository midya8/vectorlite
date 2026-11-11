# Real block buckets, unchanged signature filter

VectorLite0.3 adds `HammingBuckets` and `BucketLSHIndex`. The original
`LSHIndex` signature scan is unchanged and remains the default. The new backend
changes how exact Hamming-radius candidates are found, not how cosine scores,
ids or canonical ranking are defined. No new retrieval probability model is used.

## Exact-block specialization of established multi-index hashing

For b bits and radius r<b, split all bit positions into m=r+1 disjoint nonempty
contiguous blocks. Their lengths are floor(b/m) or ceil(b/m), with longer blocks
first. Block0 starts at the least-significant bit. If every block differed, there
would be at least r+1 differing bits. Thus every full code at distance at most r
shares at least one exact block. The converse is false. Query each block's exact
key, union row-index postings, deduplicate and verify full-code XOR/popcount.

This is the zero-substring-radius special case of multi-index hashing's
pigeonhole rule. It is not the full Norouzi/Punjani/Fleet implementation, optimal
partition selection, multiprobe algorithm or a new algorithm. Their reported
speedups and uniform-code complexity results are not claimed for this code.

The posting tables store row positions, not signatures as unique identities.
Distinct rows with equal signatures remain distinct. Exclusions remove supplied
row positions before full Hamming verification. `BucketLSHIndex` maps excluded
ids to positions and preserves existing unknown-id exclusion behavior. Final
scores use the unchanged stable cosine and score-descending/id-ascending order.
This is exact relative to the previous threshold scan, not exact cosine nearest
neighbors and not human relevance evaluation.

## Reusable native index and explicit fallback

```python
from vectorlite import HammingBuckets, BucketLSHIndex

h = HammingBuckets([0, 1, 8, 64, 73, 11, 255], bits=8, radius=2)
report = h.lookup(0, mode="blocks")
assert report["admitted_indices"] == [0, 1, 2, 3]

index = BucketLSHIndex(items, num_bits=24, min_shared_bits=22, seed=17)
neighbors = index.query(query, k=3, mode="auto", exclude_ids=["query-id"])
full_query_report = index.query_audit(query, k=3, mode="auto")
```

Construction copies signatures, vectors and ids; caller mutations do not alter
this index. It supports repeated queries, not online insertion, deletion or
changing the threshold in place. New radius/threshold requires rebuilding.
`HammingBuckets.signatures` is a detached tuple; posting tuples are behind
read-only mappings. Returned reports are detached. Private Python internals are
not a security boundary against deliberate mutation.

Modes:

- `scan`: no bucket probes; check every nonexcluded signature.
- `blocks`: use the posting union whenever a block index was built.
- `auto`: probe keys and sum posting lengths without traversing them. If this
  sum is at least N, use a scan. Otherwise visit the postings and deduplicate.

At r=b every signature qualifies, so construction skips tables and reports
`radius-covers-all`. A declared `max_blocks` cap (default32, maximum64) skips
construction when r+1 exceeds it and reports `block-cap`. Even forced `blocks`
falls back with that explicit reason. This cap prevents an accidental high-radius
index with thousands of posting references per row. The native generic signature
API supports1..4096 bits and arbitrary finite input row counts; the strict CLI
and browser cap input at128 bits,1,024 rows and128KiB. Hex values must be lowercase,
fit the declared bit width, and use1..32 digits. Empty tables, zero radius,
duplicate codes, all-excluded sets and cap fallbacks are valid.

## Work counts are not a speed benchmark

A construction report counts occupied table keys and N*m posting references, not
RAM bytes. Query reports show exact block lookup count, posting-length sum,
posting references actually visited, unique union rows, excluded pool entries,
full Hamming checks and admitted positions. The scan reference makes N-excluded
full Hamming checks. In scan mode there is no posting-union candidate claim.
The cosine wrapper additionally reports actual cosine rescoring count and ranking.

The auto rule is a simple structural heuristic. It does not estimate timing,
CPU/cache costs, allocations, sorting/deduplication costs, integer word lengths,
hash operations, query projection cost or cosine cost. It can choose poorly.
Index building and retained postings also cost work and memory. A low number of
Hamming checks alone does not establish a faster overall query. This Python
implementation returns full audit witnesses, which add overhead.

A fixed seeded512-row24-bit signature control, radius2, uses3 lookups, visits11
posting references and checks9 unique codes versus512 scan checks. Both admit the
same one row. This is a constructed table, not learned embeddings or a latency
measurement. A512-row all-zero control yields1,536 posting references; auto
chooses a scan after3 lookups and visits0 postings, while forced blocks visits
all1,536, deduplicates to512 and checks512 codes. Both admit all512. No favorable
case is substituted for the dense case.

At the default r=bits-threshold often used by the original64-bit signature
filter, many blocks can be very short, so dense buckets and fallback are plausible.
There is no general sublinear bound for arbitrary codes or thresholds. All original
12 vectors, original35 audit cases and original scan evidence are retained.
The new original-corpus control XORs every stored signature with the first query's
signature, making query code0 while preserving all Hamming distances. This is a
signature-coordinate translation, not a new embedding or cosine evaluation.

## Independent browser and exports

The separate page uses BigInt XOR/popcount and native JavaScript Map postings,
not Python reports. Only witness input signatures are shipped. It shows
construction partitions/occupied keys, per-query posting lengths, strategy/work
bars, every checked/admitted/rejected row, selected XOR and matching block ids.
The page rebuilds explicitly; native API reuse is a separate capability.

Edits and policy changes hide stale results and disable both downloads. Strict
JSON rejects duplicate keys, booleans in integer fields, malformed hex, unknown
keys and invalid exclusions. Downloaded input can be replayed through either:

```sh
python -m vectorlite.bucket_cli --input bucket-input.json > report.json
vectorlite-buckets --input bucket-input.json > report.json
```

Reports contain only exact integers, strings, booleans and null. Independent
native/JS bucket reports must match exactly, not within a numeric tolerance.
Cosine ranking equivalence is checked using the unchanged native arithmetic,
with identical original hyperplanes rather than a cross-language PRNG claim.

## Reproduction evidence

-59 Python tests retain all50 previous tests. New tests exhaust all binary codes,
  queries and radii through7 bits across all3 modes:5,376 comparisons against a
  separate string-distance oracle.160 random tables across1..128 bits produce480
  excluded-query/mode comparisons with independent bit-position distances.
-360 random original scan-versus-bucket cosine/id rankings cover zeros, duplicate
  vectors, exclusions and thresholds. All35 old complete audit specs run through
  all3 backend modes for105 further exact ranking comparisons.
-34 new full reports include sparse/dense controls, forced posting work, false
  candidates, cap/all-radius fallbacks, duplicates/exclusions, empty table,
  original12-vector translated signatures and24 seeded tables. There are1,908
  checked rows in those reports, including duplicates across controls.
-Eight Node/actual-Chrome tests retain the prior3. Independent JS checks all34
  exact reports and repeats all5,376 exhaustive cube comparisons without using
  the production popcount for its oracle. Actual Chrome checks all1,908 selected
  rows, policy changes, invalid-state clearing, strict errors,2 trusted JSON
  downloads and24 views at1280/768/390/320 widths.
-An installed8-module wheel is checked outside the checkout: all35 original
  audits, original query/audit commands,105 old-spec backend rankings, all34
  new bucket reports and both new CLI entrypoints. Fixture regeneration retains
  the original35-report artifact byte-for-byte.

No new algorithm, speed, production scalability, semantic relevance, recall gain,
RAM saving or statistical complexity guarantee is claimed. An optional candidate
lookup backend is the engineering change; sparse and dense costs stay separate.

## Attribution and sources

- Norouzi, Punjani and Fleet, [Fast Exact Search in Hamming Space with Multi-Index Hashing](https://arxiv.org/html/1307.2982),2013/2014:
  disjoint-substring pigeonhole proposition, candidate union and full-code pruning.
- [Author primary paper PDF](https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf):
  primary mathematical formulation and uneven contiguous block sizes.
- [Author poster](https://norouzi.github.io/research/posters/mih_poster.pdf):
  explicit15-bit/radius2/3-block zero-substring-radius example.
- [Author project](https://norouzi.github.io/research/mih/) and
  [reference implementation](https://github.com/norouzi/mih/): optimized method
  and experiments, not copied here or treated as this implementation's results.
- [Python integer documentation](https://docs.python.org/3/library/stdtypes.html):
  integer bit_count and exact unbounded integer behavior.
- [Python collections documentation](https://docs.python.org/3/library/collections.html):
  posting-map implementation context. This project uses ordinary dicts and tuples.
- Charikar, [Similarity estimation techniques from rounding algorithms](https://dl.acm.org/doi/10.1145/509907.509965),2002:
  original random-hyperplane similarity background, distinct from exact retrieval
  of fixed codes. Existing VectorLite angular probability assumptions stay unchanged.

Source files for the previous scan remain unchanged. Historical README/case
statements describing an O(N) signature scan still apply to the original default,
not to the optional block backend's sparse-query mechanism.
