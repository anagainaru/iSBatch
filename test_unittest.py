import unittest
import numpy as np
import HPCRequest as rqs


# test the sequence extraction

# test the cost model
class TestCostModel(unittest.TestCase):
    def test_with_checkpoint(self):
        sequence = [(4,1), (10, 0)]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)

    def test_without_checkpoint(self):
        sequence = [4, 10]
        handler = rqs.LogDataCost(sequence)
        self.assertEqual(len(handler.sequence), 2)
        self.assertEqual(handler.sequence[0], 4)
