"""Hermetic regressions for audible fault placement, not the DSP implementation."""

import math
import struct
import unittest

from compare_playout import schedule, speech_activity


class FaultPlacementTests(unittest.TestCase):
    def test_silence_is_rejected(self):
        with self.assertRaises(ValueError):
            speech_activity(bytes(24_000 * 10 * 2))

    def test_fault_is_placed_in_speech_and_preserves_order(self):
        # Leading/trailing silence and an internal pause must not attract fault placement.
        pcm = [0] * (24_000 * 12)
        for i in range(24_000 * 2, 24_000 * 9):
            if not 24_000 * 4 <= i < 24_000 * 5:
                pcm[i] = int(5000 * math.sin(i * math.tau / 120))
        activity = speech_activity(struct.pack(f"<{len(pcm)}h", *pcm))
        anchor = activity["anchor_chunk"]
        self.assertIn(anchor, activity["speech_chunks"])
        self.assertGreaterEqual(activity["active_fraction_next_second"], 0.7)
        for scenario in [
            "late_20ms",
            "repeated_stalls",
            "burst_recovery",
            "long_outage",
            "no_catchup",
        ]:
            rows = schedule(scenario, len(pcm), anchor, set(activity["speech_chunks"]))
            self.assertGreater(rows[anchor][0], anchor * 100_000)
            self.assertEqual(sum(row[2] for row in rows), len(pcm))
            self.assertEqual([row[0] for row in rows], sorted(row[0] for row in rows))
            if scenario == "repeated_stalls":
                for index, (arrival, start, _) in enumerate(rows):
                    if arrival > start * 1_000_000 // 24_000:
                        self.assertIn(index, activity["speech_chunks"])


if __name__ == "__main__":
    unittest.main()
