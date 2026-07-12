import importlib
import unittest
from pathlib import Path

import yaml


CONFIG_DIR = Path("audioldm_train/config/2025_11_23_dance_controlnet_beatdance")
EXPECTED_TARGET = "BeatDance.model.clip_transformer.DanceOnlyBeatDanceEncoder"


def find_targets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "target":
                yield item
            yield from find_targets(item)
    elif isinstance(value, list):
        for item in value:
            yield from find_targets(item)


def import_target(target):
    module_name, attribute = target.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


class ConfigTargetTests(unittest.TestCase):
    def test_all_configs_use_canonical_encoder_target(self):
        configs = sorted(CONFIG_DIR.glob("*.yaml"))
        self.assertEqual(len(configs), 7)
        for config_path in configs:
            config = yaml.safe_load(config_path.read_text())
            targets = set(find_targets(config))
            self.assertIn(EXPECTED_TARGET, targets, str(config_path))
        self.assertEqual(
            import_target(EXPECTED_TARGET).__name__,
            "DanceOnlyBeatDanceEncoder",
        )

    def test_paper_ablation_mapping(self):
        def encoder_params(filename):
            config = yaml.safe_load((CONFIG_DIR / filename).read_text())
            return config["model"]["params"]["controlnet_stage_config"][
                "dance_controlnet"
            ]["params"]["feature_encoder_config"]["params"]

        full = encoder_params("audioldm_original_medium_stretch_pretrained_frozen.yaml")
        full_mapper = full["feature_mapper_config"]["params"]
        self.assertEqual(
            full["motion_bert_pretrained_weights_path"],
            "data/checkpoints/latest_epoch.bin",
        )
        self.assertEqual(
            full_mapper["beatdance_pretrained_weights_path"],
            "data/checkpoints/model_best.pth",
        )
        self.assertTrue(full_mapper["freeze_beatdance"])

        without_contrastive = encoder_params("audioldm_original_medium.yaml")
        self.assertNotIn(
            "beatdance_pretrained_weights_path",
            without_contrastive["feature_mapper_config"]["params"],
        )

        without_motionbert = encoder_params(
            "audioldm_original_medium_stretch_wo_mb.yaml"
        )
        self.assertIsNone(without_motionbert["motion_bert_pretrained_weights_path"])


if __name__ == "__main__":
    unittest.main()
