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
        self.assertFalse(activity["startup_active"])
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

    def test_slow_jitter_is_sustained_ordered_and_reproducible(self):
        active = set(range(300))
        first = schedule("slow_jittery", 24_000 * 30, 51, active)
        self.assertEqual(first, schedule("slow_jittery", 24_000 * 30, 51, active))
        self.assertEqual(sum(row[2] for row in first), 24_000 * 30)
        self.assertEqual([row[0] for row in first], sorted(row[0] for row in first))
        self.assertTrue(
            all(arrival == start * 1_000_000 // 24_000 for arrival, start, _ in first[:51])
        )
        intervals = [
            second[0] - previous[0]
            for previous, second in zip(first[51:], first[52:], strict=False)
        ]
        self.assertGreater(sum(interval > 120_000 for interval in intervals), 20)
        self.assertGreater(sum(interval < 50_000 for interval in intervals), 10)
        self.assertGreater(len(set(intervals)), 50)
        # Delay keeps accumulating, rather than causing a single initial underrun and then
        # returning to the same playout position forever.
        self.assertGreater(first[-1][0] - first[-1][1] * 1_000_000 // 24_000, 400_000)

    def test_first_answer_packets_have_real_gaps_without_missing_source(self):
        rows = schedule("startup_jitter", 24_000 * 4, 51)
        self.assertEqual(
            [row[0] for row in rows[:8]],
            [0, 160_000, 300_000, 460_000, 700_000, 700_000, 700_000, 700_000],
        )
        self.assertEqual(sum(row[2] for row in rows), 24_000 * 4)
        self.assertTrue(all(row[1] == i * 2400 for i, row in enumerate(rows)))

    def test_report_inspired_large_stalls_only_target_verified_speech(self):
        speech = {60, 77}
        rows = schedule("reported_jitter", 24_000 * 12, 51, speech)
        stalls = [
            i for i, row in enumerate(rows) if row[0] - row[1] * 1_000_000 // 24_000 == 300_000
        ]
        self.assertEqual(stalls, [60, 77])
        self.assertEqual(rows, schedule("reported_jitter", 24_000 * 12, 51, speech))
        self.assertEqual([row[0] for row in rows], sorted(row[0] for row in rows))

    def test_catchup_is_a_complete_provider_burst_with_one_second_of_lead(self):
        rows = schedule("catchup", 24_000 * 10, 22)
        self.assertEqual(rows[22][0], 2_200_000)
        self.assertEqual(rows[37][0], 2_700_000)
        self.assertEqual(rows[38][0], 2_800_000)
        self.assertEqual([row[0] for row in rows], sorted(row[0] for row in rows))
        self.assertEqual(sum(row[2] for row in rows), 24_000 * 10)
        self.assertTrue(all(start == index * 2400 for index, (_, start, _) in enumerate(rows)))

    def test_sustained_jitter_continues_twenty_seconds_then_stabilizes(self):
        active = set(range(300))
        rows = schedule("sustained_jitter", 24_000 * 27, 22, active)
        self.assertEqual(rows, schedule("sustained_jitter", 24_000 * 27, 22, active))
        self.assertEqual(sum(row[2] for row in rows), 24_000 * 27)
        self.assertEqual([row[0] for row in rows], sorted(row[0] for row in rows))
        delays = [arrival - start * 1_000_000 // 24_000 for arrival, start, _ in rows]
        self.assertEqual(delays[:22], [0] * 22)
        for begin in [22, 72, 122, 172]:
            variation = [delays[i] - (i - 22) * 1000 for i in range(begin, begin + 50)]
            self.assertTrue(all(0 <= value <= 30_000 for value in variation))
            self.assertGreater(len(set(variation)), 40)
        self.assertEqual(delays[222:], [200_000] * len(delays[222:]))
        available = consumed = next_chunk = 0
        gaps = []
        in_gap = False
        for at in range(0, 27_000, 10):
            while next_chunk < len(rows) and rows[next_chunk][0] <= at * 1000:
                available += rows[next_chunk][2]
                next_chunk += 1
            count = min(240, available - consumed)
            if count < 240 and consumed < 24_000 * 27:
                if not in_gap:
                    gaps.append(at)
                in_gap = True
            else:
                in_gap = False
            consumed += count
        self.assertGreater(len(gaps), 15)
        self.assertGreater(gaps[-1], 20_000)
        for begin in [2200, 7200, 12200, 17200]:
            self.assertTrue(any(begin <= at < begin + 5000 for at in gaps))
        silence = schedule("sustained_jitter", 24_000 * 27, 22, set())
        self.assertEqual(silence[172][0] - silence[172][1] * 1_000_000 // 24_000, 150_000)

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
        self.assertEqual(anchor, 20)  # Default preference is two seconds, in active speech.
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
