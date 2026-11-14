"""Hamming isometries preserve neighbors, not block lookup work."""
import itertools
import unittest

from vectorlite.buckets import HammingBuckets


def permute(code, order):
    return sum(((code >> old) & 1) << new for new, old in enumerate(order))


class HammingIsometryTests(unittest.TestCase):
    def test_all_four_bit_permutations_and_xor_translations(self):
        codes = [0, 1, 1, 3, 5, 10, 15]
        count = 0
        for order in itertools.permutations(range(4)):
            for mask in (0, 5, 15):
                transformed = [permute(code, order) ^ mask for code in codes]
                for query in (0, 6, 15):
                    moved_query = permute(query, order) ^ mask
                    for radius in (0, 1, 2, 4):
                        expected = [index for index, code in enumerate(codes)
                                    if index != 1 and (code ^ query).bit_count() <= radius]
                        table = HammingBuckets(transformed, 4, radius)
                        for mode in ("auto", "blocks", "scan"):
                            report = table.lookup(moved_query, exclude_indices=[1], mode=mode)
                            with self.subTest(order=order, mask=mask, query=query,
                                              radius=radius, mode=mode):
                                self.assertEqual(report["admitted_indices"], expected)
                                for row in report["checks"]:
                                    self.assertEqual(row["distance"],
                                                     (codes[row["index"]] ^ query).bit_count())
                                self.assertEqual(report["work"]["full_hamming_checks"],
                                                 len(report["checks"]))
                            count += 1
        self.assertEqual(count, 2592)

    def test_bit_permutation_can_change_block_work(self):
        # Both nonzero codes miss radius 1. Only the first shares a two-bit block.
        before = HammingBuckets([0, 3], 4, 1).lookup(0, mode="blocks")
        after = HammingBuckets([0, 5], 4, 1).lookup(0, mode="blocks")
        self.assertEqual(before["admitted_indices"], after["admitted_indices"])
        self.assertEqual(before["admitted_indices"], [0])
        self.assertEqual(before["work"]["full_hamming_checks"], 2)
        self.assertEqual(after["work"]["full_hamming_checks"], 1)
        self.assertNotEqual(before["work"]["posting_lengths_total"],
                            after["work"]["posting_lengths_total"])

    def test_row_permutation_remaps_exclusions_and_duplicate_ids(self):
        codes = [0, 1, 1, 3, 15]
        for order in itertools.permutations(range(len(codes))):
            old_to_new = {old: new for new, old in enumerate(order)}
            table = HammingBuckets([codes[index] for index in order], 4, 1)
            excluded = [old_to_new[1]]
            for mode in ("auto", "blocks", "scan"):
                report = table.lookup(0, exclude_indices=excluded, mode=mode)
                restored = sorted(order[index] for index in report["admitted_indices"])
                self.assertEqual(restored, [0, 2])
                self.assertEqual(report["work"]["scan_reference_hamming_checks"], 4)
