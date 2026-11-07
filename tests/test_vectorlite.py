import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vectorlite import ExactIndex, LSHIndex, cosine_similarity, l2_distance, load_vectors
from vectorlite.recall import evaluate_recall, recall_at_k

ITEMS = [
    {"id": "x", "vector": [1.0, 0.0, 0.0]},
    {"id": "y", "vector": [0.9, 0.1, 0.0]},
    {"id": "z", "vector": [0.0, 1.0, 0.0]},
    {"id": "w", "vector": [0.0, 0.0, 1.0]},
]


class TestVectorMath(unittest.TestCase):
    def test_cosine_identical(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [1, 2]), 1.0)

    def test_cosine_orthogonal(self):
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_l2(self):
        self.assertAlmostEqual(l2_distance([0, 0], [3, 4]), 5.0)


class TestLoadVectors(unittest.TestCase):
    def write(self, tmp, text):
        path = Path(tmp) / "v.jsonl"
        path.write_text(text)
        return str(path)

    def test_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, '{"id": "a", "vector": [1, 0]}\n{"vector": [0, 1]}\n')
            items, dims = load_vectors(path)
        self.assertEqual(dims, 2)
        self.assertEqual([i["id"] for i in items], ["a", "v2"])

    def test_rejects_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, '{"vector": [1, 0]}\n{"vector": [1, 0, 0]}\n')
            with self.assertRaises(ValueError):
                load_vectors(path)

    def test_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "nope\n")
            with self.assertRaises(ValueError):
                load_vectors(path)

    def test_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "\n")
            with self.assertRaises(ValueError):
                load_vectors(path)

    def test_rejects_zero_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, '{"vector": []}\n')
            with self.assertRaises(ValueError):
                load_vectors(path)


class TestExactIndex(unittest.TestCase):
    def setUp(self):
        self.index = ExactIndex(ITEMS)

    def test_nearest_first(self):
        result = self.index.query([1.0, 0.05, 0.0], k=1)
        self.assertEqual(result[0][0], "x")

    def test_k_limits_results(self):
        self.assertEqual(len(self.index.query([1, 0, 0], k=2)), 2)

    def test_ordering(self):
        result = self.index.query([1.0, 0.0, 0.0], k=4)
        ids = [vid for vid, _ in result]
        self.assertEqual(ids[0], "x")
        self.assertEqual(ids[1], "y")
        # z and w tie at cosine 0; the tie breaks on id, so w precedes z
        self.assertEqual(ids[-1], "z")

    def test_l2_metric(self):
        index = ExactIndex(ITEMS, metric="l2")
        result = index.query([0.95, 0.05, 0.0], k=1)
        self.assertEqual(result[0][0], "y")

    def test_bad_metric(self):
        with self.assertRaises(ValueError):
            ExactIndex(ITEMS, metric="hamming")

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            self.index.query([1.0, 0.0], k=1)

    def test_ties_break_on_id(self):
        items = [{"id": "b", "vector": [1.0]}, {"id": "a", "vector": [1.0]}]
        result = ExactIndex(items).query([1.0], k=2)
        self.assertEqual([vid for vid, _ in result], ["a", "b"])


class TestLSHIndex(unittest.TestCase):
    def test_deterministic_build(self):
        a = LSHIndex(ITEMS, num_bits=32, seed=7)
        b = LSHIndex(ITEMS, num_bits=32, seed=7)
        self.assertEqual(a.signatures, b.signatures)

    def test_different_seed_different_signatures(self):
        a = LSHIndex(ITEMS, num_bits=32, seed=7)
        b = LSHIndex(ITEMS, num_bits=32, seed=8)
        self.assertNotEqual(a.signatures, b.signatures)

    def test_num_bits_floor(self):
        with self.assertRaises(ValueError):
            LSHIndex(ITEMS, num_bits=4)

    def test_similar_vectors_share_bits(self):
        a = LSHIndex(ITEMS, num_bits=64, seed=17)
        sx, sy = a.signatures[0], a.signatures[1]
        sw = a.signatures[3]
        shared_close = 64 - bin(sx ^ sy).count("1")
        shared_far = 64 - bin(sx ^ sw).count("1")
        self.assertGreater(shared_close, shared_far)

    def test_query_returns_results(self):
        index = LSHIndex(ITEMS, num_bits=64, seed=17, min_shared_bits=0)
        result = index.query([1.0, 0.05, 0.0], k=2)
        self.assertEqual(result[0][0], "x")

    def test_high_threshold_can_return_fewer(self):
        index = LSHIndex(ITEMS, num_bits=64, seed=17, min_shared_bits=64)
        result = index.query([1.0, 0.05, 0.0], k=4)
        self.assertLessEqual(len(result), 1)


class TestRecall(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(recall_at_k([("a", 1), ("b", 0)], [("a", 1), ("b", 0)]), 1.0)

    def test_partial(self):
        self.assertEqual(recall_at_k([("a", 1), ("c", 0)], [("a", 1), ("b", 0)]), 0.5)

    def test_empty_truth(self):
        self.assertEqual(recall_at_k([], []), 1.0)

    def test_evaluate_recall_keys(self):
        exact = ExactIndex(ITEMS)
        approx = LSHIndex(ITEMS, num_bits=64, seed=17)
        result = evaluate_recall(exact, approx, [[1.0, 0.0, 0.0]], k=2)
        self.assertEqual(result["queries"], 1)
        self.assertEqual(result["k"], 2)
        self.assertIn("mean_recall", result)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vectors = Path(self.tmp.name) / "v.jsonl"
        with self.vectors.open("w") as fh:
            for item in ITEMS:
                fh.write(json.dumps(item) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "vectorlite", *args], capture_output=True, text=True
        )

    def test_query_exact(self):
        r = self.run_cli("query", "--vectors", str(self.vectors), "--query", "1,0.05,0", "-k", "1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("x", r.stdout.splitlines()[0])

    def test_query_dim_mismatch(self):
        r = self.run_cli("query", "--vectors", str(self.vectors), "--query", "1,0")
        self.assertEqual(r.returncode, 2)

    def test_query_l2(self):
        r = self.run_cli(
            "query", "--vectors", str(self.vectors), "--query", "0.95,0.05,0",
            "--metric", "l2", "-k", "1",
        )
        self.assertIn("y", r.stdout.splitlines()[0])

    def test_benchmark_reports_and_gates(self):
        out = Path(self.tmp.name) / "bench.json"
        r = self.run_cli(
            "benchmark", "--vectors", str(self.vectors), "-k", "2",
            "--json", str(out), "--min-recall", "0.5",
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(out.read_text())
        self.assertEqual(data["vectors"], 4)
        self.assertIn("mean_recall", data)
        r = self.run_cli(
            "benchmark", "--vectors", str(self.vectors), "-k", "2", "--min-recall", "1.01"
        )
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL", r.stderr)

    def test_query_approximate(self):
        r = self.run_cli(
            "query", "--vectors", str(self.vectors), "--query", "1,0.05,0",
            "-k", "2", "--approximate",
        )
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
