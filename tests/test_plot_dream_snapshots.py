import unittest

from scripts.rq1.plot_dream_snapshots import confidence_intervals, select_rows


class PlotDreamSnapshotsTest(unittest.TestCase):
    def test_select_rows_uses_largest_protocol_and_sorts_by_training(self):
        rows = [
            {
                "evaluation_id": "complete",
                "iteration": 20,
                "imagined_steps": 200,
                "success_rate": 0.8,
                "episodes": 10,
            },
            {
                "evaluation_id": "partial",
                "iteration": 10,
                "imagined_steps": 100,
                "success_rate": 0.7,
                "episodes": 10,
            },
            {
                "evaluation_id": "complete",
                "iteration": 0,
                "imagined_steps": 0,
                "success_rate": 0.5,
                "episodes": 10,
            },
        ]

        selected, evaluation_id = select_rows(rows)

        self.assertEqual(evaluation_id, "complete")
        self.assertEqual([row["iteration"] for row in selected], [0, 20])

    def test_select_rows_honors_explicit_protocol(self):
        rows = [
            {
                "evaluation_id": protocol,
                "iteration": iteration,
                "success_rate": 0.5,
                "episodes": 10,
            }
            for protocol, iteration in (("a", 0), ("a", 10), ("b", 0))
        ]

        selected, evaluation_id = select_rows(rows, "b")

        self.assertEqual(evaluation_id, "b")
        self.assertEqual(len(selected), 1)

    def test_confidence_intervals_contain_observed_success(self):
        rows = [
            {"success_rate": 0.5, "episodes": 100},
            {"success_rate": 0.8, "episodes": 150},
        ]

        lows, highs = confidence_intervals(rows)

        for row, low, high in zip(rows, lows, highs):
            self.assertLess(low, row["success_rate"])
            self.assertGreater(high, row["success_rate"])


if __name__ == "__main__":
    unittest.main()
