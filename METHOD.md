# Geometry, candidate and recall contract

This is a finite audit of VectorLite's signature scan, not a new approximate-neighbor algorithm, learned embedding benchmark or speed result.

## Stable numerical core

Vectors must be nonempty, equal-dimensional finite real coordinates, not booleans or numeric strings. Ids are unique nonempty strings. Index construction copies caller-owned vectors. Cosine independently scales each input by its largest magnitude before `hypot` normalization and a finite dot product. This avoids the original sum-of-squares overflow/underflow: `[1e300,1e300]` and `[1e300,0]` now give approximately `sqrt(.5)`, as do their `1e-300` counterparts. The old direct norms produced nonfinite/zero arithmetic. Dot products reject dimension mismatch rather than silently zip-truncating. L2 uses `math.dist` and rejects an unrepresentable final distance.

Cosine with a zero vector retains the original convention 0. A zero vector's projections are zero and signature bits are all 1. Its angle and random-hyperplane probability are undefined, so probability fields are `null`. They are not silently treated as orthogonal.

The Gaussian planes are generated in the original order from `random.Random(seed).gauss(0,1)`. Signs use normalized plane/vector dots and `>=0` for bit 1. Normalizing positive scales preserves the mathematical halfspace, but finite arithmetic can move an almost-zero dot across a sign boundary. Raw planes, normalized coordinates and every projection are exported. The same runtime/data/seed rebuilds deterministically; no cross-language or cross-version PRNG promise is made. The browser replays actual exported planes, not Python's random generator.

All scores use floating point, not exact real arithmetic. `acos(cosine)` is sensitive near endpoints, so the equivalent normalized-vector half-angle identity computes theta as `2*atan2(||u-v||,||u+v||)` for the theoretical agreement probability. Displayed numbers use 9 significant digits, with magnitudes below 1e-12 displayed as 0. Signs and rankings use unrounded values. Independent-language comparisons have tolerances; near ties can rank differently. Canonical ties within an implementation sort by score descending, then id ascending. The browser audit requires ASCII ids to keep Python/JavaScript string ordering aligned.

## Search, exclusions and monotone threshold curve

Exact search computes all cosine scores and keeps k by score/id. LSH first computes a query signature, scans **every** stored signature and counts matching bits, admits those with shared bits >= threshold, then uses the same exact score/id rule to rescore and keep k. There are no hash buckets, fallback candidates, multi-table banding or HNSW graph here. Counts of candidate rescoring are not operation totals or time measurements. The signature scan remains linear in corpus size.

Excluded ids are removed before ranking in both indexes. The audit exports total signature scan count, eligible count, candidate ids, rescored fraction of eligible vectors and exact/approximate top-k. Exclusion does not remove the stored signature scan. Empty ground truth has recall 1 by original convention; k=0 is supported by search but audit k must be positive. When fewer than k eligible vectors exist, recall divides by the actual canonical exact result count, not requested k.

Let G be the canonical exact top-k. If g in G is admitted, there are at most k-1 corpus ids ahead of g in the total score/id ordering. Therefore g cannot be pruned by exact candidate rescoring. Recall is exactly `|G intersect admitted| / |G|`. Lowering threshold creates a nested candidate set and cannot decrease recall under this fixed index/query/k/order. This does **not** say increasing bits or changing seed improves recall. Original README's suggestion that lowering the sharing threshold can reduce recall was incorrect.

Id-based ties are declared ground truth, not semantic relevance judgments. Different equally scored ids can change canonical recall without changing a retrieval utility measure. This project does not claim tie-invariant recall.

## Probability reference, not a confidence interval

For a fixed pair of nonzero vectors with angle theta, one independent isotropic random plane agrees with probability p=1-theta/pi. With b independent such planes, shared bits S follow Binomial(b,p). Admission at threshold t is

`A(b,t,p) = sum(j=t..b) choose(b,j) p^j (1-p)^(b-j)`.

For bounded b<=128, a direct positive-term sum gives the reference probability. It has no cancellation from subtracting a near-one CDF; tiny terms can still underflow. Python uses integer combinations and `fsum`; independent JavaScript uses multiplicative combinations and scalar summation. Rational binomial tails and full 8-bit sign enumeration check the results.

Expected candidate count is the sum of marginal admission probabilities across eligible items. Expected canonical recall is their average across G, by the admitted-ground-truth property above and linearity of expectation. This does not assume independent candidate events. They share planes and are generally correlated. There is no claim that aggregate candidate count follows a binomial distribution. These are expectations over ideal iid isotropic plane rebuilds for a fixed query/data set. They are not probabilities conditional on the displayed fixed planes, statistical confidence intervals or predictions validated for edited arbitrary plane sets. Any eligible zero vector makes expected candidate count undefined; expected recall can still be defined if no exact top-k member is zero.

An explicit full-circle angular check and 1,200 seeded Gaussian rebuilds are sanity controls, not a broad probabilistic calibration study or proof of a PRNG's ideal distribution.

## Evaluation protocols

The unchanged original 12-vector, 8D corpus benchmark still achieves self-inclusive mean recall@3 of 1.0. Original demo, both gate sides and all 31 original tests remain. The original self-query mode remains the default, labeled `self-inclusive`.

`benchmark --exclude-self` excludes the query id from both exact and approximate rankings, labeled `leave-one-id-out`. It does not exclude other equal vectors and is not a held-out evaluation. `benchmark --queries queries.jsonl` runs externally supplied query vectors, labeled `external-supplied`. It does not prove those queries are independent or held out. These flags cannot be combined. A two-item original synthetic fixture with query east, orthogonal north, 8 bits, seed 0 and threshold 8 scores recall 1 with self included and 0 after excluding it. This isolates that protocol change on this fixture only; no general method-quality claim follows.

Mean recall is no longer rounded before CI gate comparison. The CLI still formats display to 4 decimals; JSON and gates use the full computed mean. Reject duplicate ids to prevent set-recall collapse, nonfinite inputs/thresholds, invalid k/bits/seed and the previously silently ignored `--approximate --metric l2` combination. Gate values above 1 remain accepted to preserve the original demo's explicit failing control.

## Bounds, reproducibility and verification

`audit --spec spec.json [--json report.json]` accepts strict JSON with only `items`, `query`, `bits`, `seed`, `k`, `threshold`, `exclude_ids`, `hyperplanes`. Duplicate keys, nonfinite literals and files over 128KiB are rejected. Audit is bounded to 1..64 items, 1..16 dimensions, 8..128 bits, k in 1..64, ASCII ids up to 64 characters and existing distinct excluded ids. Every threshold 0..bits and every item/bit projection is included. Supplied nonzero planes must match dimension/count. General LSH permits 8..4096 bits.

35 retained cases include the original 8D corpus, synthetic 2D and 5D controls, exact sign boundaries, zero vectors and the self-exclusion pair. `python scripts/build_audit.py` regenerates them byte-for-byte in the same environment. Tests independently check 120 random cosine pairs against 90-digit Decimal norms, rational binomial tails, exhaustive sign patterns, full-circle geometry, all threshold results versus index queries, exclusion/invalid input and CLI protocols. The browser independently recomputes 20,784 numerical fields across 643 threshold rows. Actual Chrome checks every 332 item rows, 643 threshold rows and 5,856 projection rows, invalidation, one JSON export, bad input and 320/390/1280 containment. The installed wheel is tested outside the checkout with six module origins, all 35 audits and both CLI paths. CI uses Python 3.10-3.13.

## Prior work

- Moses S. Charikar, [Similarity Estimation Techniques from Rounding Algorithms](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf?file=CharikarEstim.pdf), STOC 2002. Random-hyperplane angle sketches are prior work.
- Cyrus Rashtchian, [Algorithms for Big Data, lectures 14-15](https://madscience.ucsd.edu/2020/notes/lec14and15.pdf), 2020. Isotropic Gaussian planes and collision geometry.
- [SciPy binomial distribution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html). Probability mass definition, not an implementation dependency.
- [scikit-learn NearestNeighbors documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Explicit self-neighbor query semantics; our API/protocol is independently defined above.
- [Python math documentation](https://docs.python.org/3/library/math.html). `hypot`, `dist`, finite checks and floating-point behavior.
- [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks). Broader evaluated ANN methods; this small synthetic audit is not a comparable performance result. No implementation/data copied.

Implementation, original synthetic fixtures and audit by Midya Rahmani. No private data, paid API, learned encoder or new LSH method is claimed.
