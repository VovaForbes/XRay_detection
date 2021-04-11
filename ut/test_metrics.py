import unittest
from metrics.metrics import *


class TestIOUFunction(unittest.TestCase):
    def test_calculate_iou(self):
        gt = [0, 0, 0, 0]
        pred = [0, 0, 1, 1]
        self.assertEqual(calculate_iou(gt, pred), 0.25)

        gt = [0, 0, 1, 1]
        pred = [0, 0, 1, 1]
        self.assertEqual(calculate_iou(gt, pred), 1.)

        gt = [0, 0, 1, 1]
        pred = [1, 1, 2, 2]
        self.assertEqual(calculate_iou(gt, pred), 1/7)

        gt = [0, 0, 1, 1]
        pred = [2, 2, 3, 3]
        self.assertEqual(calculate_iou(gt, pred), 0)

    def test_calculate_metric(self):
        preds = [[1, 1, 2, 2], [2, 4, 6, 7], [8, 7, 9, 8]]
        gts = [[2, 3, 6, 6], [8, 7, 9, 8], [3, 9, 4, 10]]
        self.assertEqual(calculate_metric(gts.copy(), preds, 0.5), 0.5)
        self.assertEqual(calculate_metric(gts.copy(), preds, 0.95), 0.2)

    def test_find_best_match(self):
        preds = [[1, 1, 2, 2], [2, 4, 6, 7], [8, 7, 9, 8]]
        gts = [[2, 3, 6, 6], [8, 7, 9, 8], [3, 9, 4, 10]]
        self.assertEqual(find_best_match(preds[2], gts, 0.99), True)
        self.assertEqual(gts[1], None)
        self.assertEqual(gts[0], [2, 3, 6, 6])
        self.assertEqual(gts[2], [3, 9, 4, 10])

        self.assertEqual(find_best_match(preds[1], gts, 0.6), True)
        self.assertEqual(gts[1], None)
        self.assertEqual(gts[0], None)
        self.assertEqual(gts[2], [3, 9, 4, 10])

        gts = [[2, 3, 6, 6], [8, 7, 9, 8], [3, 9, 4, 10]]
        self.assertEqual(find_best_match(preds[1], gts, 0.61), False)


if __name__ == '__main__':
    unittest.main()
