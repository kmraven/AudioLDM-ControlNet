import json
import tempfile
import unittest
from pathlib import Path

import torch

from BeatDance.model.clip_transformer import (
    DanceOnlyBeatDanceEncoder,
    DanceOnlyBeatDanceWrapper,
)


def make_encoder(directory):
    config_path = Path(directory) / "beatdance.json"
    config_path.write_text(
        json.dumps({"num_mha_heads": 2, "num_layers": 1, "dropout1": 0.0})
    )
    return DanceOnlyBeatDanceEncoder(
        beatdance_config_path=config_path,
        kpt_num_joints=17,
        kpt_feat_dim=4,
        beat_dim=2,
        beatdance_output_dim=8,
        num_frames=6,
    )


class DanceOnlyEncoderTests(unittest.TestCase):
    def test_shape_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(directory).eval()
        motion = torch.randn(2, 6, 17, 4)
        motion_beats = torch.randn(2, 6, 2)
        self.assertEqual(encoder(motion, motion_beats).shape, (2, 6, 8))

        parameter_names = set(dict(encoder.named_parameters()))
        self.assertFalse(any(name.startswith("music_") for name in parameter_names))
        self.assertFalse(
            any(name.startswith("video_transformer_fuse") for name in parameter_names)
        )

    def test_legacy_alias_and_checkpoint_loading(self):
        self.assertIs(DanceOnlyBeatDanceWrapper, DanceOnlyBeatDanceEncoder)
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(directory)
        legacy_state = dict(encoder.state_dict())
        legacy_state["music_linear.weight"] = torch.empty(1)
        incompatible = encoder.load_state_dict(legacy_state, strict=False)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, ["music_linear.weight"])


if __name__ == "__main__":
    unittest.main()
