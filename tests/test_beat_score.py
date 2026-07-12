import unittest
from unittest.mock import patch

import numpy as np
import torch

from audioldm_train.metrics import beat_score


class BeatScoreTests(unittest.TestCase):
    def test_beat_scores_handles_empty_and_mismatched_sequences(self):
        self.assertEqual(beat_score.beat_scores([], []), (0, 0))
        self.assertEqual(beat_score.beat_scores([1, 0, 1], [1]), (1.0, 1.0))
        self.assertEqual(beat_score.f1_score(0, 0), 0)

    def test_alignment_score_handles_empty_and_aligned_beats(self):
        self.assertEqual(beat_score.alignment_score([0, 0], [0, 0]), 0)
        self.assertAlmostEqual(
            beat_score.alignment_score([0, 1, 0], [0, 1, 0]),
            1,
        )

    def test_motion_peak_accepts_tensor_input(self):
        motion = torch.zeros(30, 17, 3)
        peaks = beat_score.motion_peak_onehot(motion)
        self.assertEqual(peaks.shape, (30,))
        self.assertEqual(peaks.dtype, np.bool_)

    def test_calculate_beat_score_includes_bas(self):
        with (
            patch.object(
                beat_score,
                "calculate_audio_beat_score",
                return_value=(0.5, 0.25, 1 / 3, 2.0),
            ),
            patch.object(
                beat_score,
                "music_peak_onehot",
                return_value=np.array([0, 1, 0]),
            ),
            patch.object(
                beat_score,
                "motion_peak_onehot",
                return_value=np.array([0, 1, 0]),
            ),
        ):
            result = beat_score.calculate_beat_score(
                torch.zeros(3, 17, 3),
                torch.zeros(16),
                torch.zeros(16),
                16_000,
            )
        np.testing.assert_allclose(result, (0.5, 0.25, 1 / 3, 2.0, 1.0))


if __name__ == "__main__":
    unittest.main()
