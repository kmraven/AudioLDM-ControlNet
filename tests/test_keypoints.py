import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from audioldm_train.utilities.data.keypoints import (
    coco2h36m,
    crop_scale,
    extract_motion_beat,
    interp_nan_keypoints,
    load_keypoints,
    make_cam,
    resample_keypoints_1d,
    resample_keypoints_2d,
)


def make_keypoints(frames=5):
    keypoints = np.zeros((frames, 17, 3), dtype=np.float32)
    keypoints[..., 0] = np.arange(frames)[:, None]
    keypoints[..., 1] = np.arange(17)[None, :]
    keypoints[..., 2] = 1
    return keypoints


class KeypointTests(unittest.TestCase):
    def test_load_keypoints_validates_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keypoints.pkl"
            with path.open("wb") as file:
                pickle.dump(make_keypoints(), file)
            self.assertEqual(load_keypoints(path).shape, (5, 17, 3))

            with path.open("wb") as file:
                pickle.dump([1, 2, 3], file)
            with self.assertRaises(TypeError):
                load_keypoints(path)

    def test_interp_nan_keypoints_interpolates_each_channel(self):
        keypoints = make_keypoints()
        keypoints[2, 3, 0] = np.nan
        keypoints[:, 4, 1] = np.nan
        result = interp_nan_keypoints(keypoints)
        self.assertAlmostEqual(result[2, 3, 0], 2.0)
        self.assertTrue(np.isnan(result[:, 4, 1]).all())
        self.assertTrue(np.isnan(keypoints[2, 3, 0]))

    def test_resampling_preserves_endpoints_and_shape(self):
        result_1d = resample_keypoints_1d(np.array([0.0, 2.0]), 5)
        np.testing.assert_allclose(result_1d, [0.0, 0.5, 1.0, 1.5, 2.0])

        source = make_keypoints(3)
        result_2d = resample_keypoints_2d(source, 7)
        self.assertEqual(result_2d.shape, (7, 17, 3))
        np.testing.assert_allclose(result_2d[[0, -1]], source[[0, -1]])

    def test_coordinate_conversions_and_scaling(self):
        source = make_keypoints(2)
        source[:, 11, :2] = [2, 4]
        source[:, 12, :2] = [4, 8]
        converted = coco2h36m(source)
        np.testing.assert_allclose(converted[:, 0, :2], [3, 6])

        camera = make_cam(np.array([[[960.0, 540.0]]]), (1080, 1920))
        np.testing.assert_allclose(camera, [[[0.0, -0.4375]]])

        scaled = crop_scale(source, (1, 1))
        self.assertLessEqual(np.max(np.abs(scaled[..., :2])), 1)
        no_confidence = source.copy()
        no_confidence[..., 2] = 0
        self.assertEqual(np.count_nonzero(crop_scale(no_confidence)), 0)

    def test_extract_motion_beat_shape_and_validation(self):
        beats = extract_motion_beat(
            make_keypoints(30),
            beat_dim=2,
            target_length=16,
            fps_keypoints=60,
            duration=0.5,
        )
        self.assertEqual(beats.shape, (16, 2))
        self.assertEqual(beats.dtype, torch.int64)

        with self.assertRaises(ValueError):
            extract_motion_beat(make_keypoints(), 0, 16, 60, 1)


if __name__ == "__main__":
    unittest.main()
