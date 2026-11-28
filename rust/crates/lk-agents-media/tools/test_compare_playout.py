"""Hermetic regressions for audible fault placement, not the DSP implementation."""

import math
import struct
import unittest

from compare_playout import schedule, speech_activity


class FaultPlacementTests(unittest.TestCase):
    def test_no_reserve_case_delays_exactly_thirty_ms(self):
        rows = schedule("late_30ms_no_reserve", 24_000 * 5, 20)
        self.assertEqual(rows[20][0], 2_030_000)
        self.assertEqual(rows[20][1], 48_000)
        self.assertEqual(rows[21][0], 2_100_000)
        self.assertEqual(sum(row[2] for row in rows), 24_000 * 5)

    def test_requested_onset_uses_the_nearest_verified_speech(self):
        pcm = [
            int(5000 * math.sin(i * math.tau / 120)) if 5.1 <= i / 24_000 < 8 else 0
            for i in range(24_000 * 10)
        ]
        activity = speech_activity(struct.pack(f"<{len(pcm)}h", *pcm), preferred_ms=5000)
        self.assertEqual(activity["anchor_ms"], 5100)
        self.assertEqual(min(activity["window_rms"][510:515]) > 150, True)
        for scenario in ["variable_chunks", "clock_drift"]:
            rows = schedule(scenario, len(pcm), 51)
            for arrival, start, count in rows[:51]:
                self.assertEqual(arrival, start * 1_000_000 // 24_000)
                self.assertEqual(count, 2400)
            if scenario == "variable_chunks":
                self.assertEqual(rows[51][2], 480)
            else:
                self.assertGreater(rows[52][0], rows[52][1] * 1_000_000 // 24_000)

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
