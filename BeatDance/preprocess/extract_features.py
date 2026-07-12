"""Extract MERT, beat, and MotionBERT features from AIST++ clips."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import librosa
import numpy as np
import torch
import torch.nn.functional as torch_functional
import torchaudio
import torchaudio.transforms as audio_transforms
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

from audioldm_train.modules.motion_encoder.MotionBERT.DSTformer import (
    DSTformerWrapper,
)
from audioldm_train.utilities.data.keypoints import (
    coco2h36m,
    crop_scale,
    extract_motion_beat,
    interp_nan_keypoints,
    load_keypoints,
    make_cam,
    resample_keypoints_2d,
)


DEFAULT_TARGET_LENGTH = 128
MOTION_FPS = 60
MOTION_BEAT_DIM = 2
MUSIC_BEAT_DIM = 3


def _resample_sequence(features, target_length):
    features = features.transpose(0, 1).unsqueeze(0)
    features = torch_functional.interpolate(
        features,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
    return features.squeeze(0).transpose(0, 1)


class Extractor_fm:
    """Extract fixed-length framewise MERT music features."""

    def __init__(self, device=None, use_amp=True, target_length=DEFAULT_TARGET_LENGTH):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        config = AutoConfig.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
        )
        config.conv_pos_batch_norm = False
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            config=config,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
        )
        self.use_amp = use_amp and self.device.type == "cuda"
        self.target_length = target_length

    def extract(self, waveform, sample_rate):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        model_sample_rate = getattr(
            self.processor,
            "sampling_rate",
            sample_rate,
        )
        if sample_rate != model_sample_rate:
            waveform = audio_transforms.Resample(
                orig_freq=sample_rate,
                new_freq=model_sample_rate,
            )(waveform)

        inputs = self.processor(
            waveform.squeeze().float(),
            sampling_rate=model_sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
        selected_layer = hidden_states[2]
        return _resample_sequence(selected_layer, self.target_length)


class Extractor_fbm:
    """Extract the three-channel music beat representation used by BeatDance."""

    def __init__(self, target_length=DEFAULT_TARGET_LENGTH):
        self.target_length = target_length

    def extract(self, audio, sample_rate):
        _, beat_times = librosa.beat.beat_track(
            y=audio,
            sr=sample_rate,
            units="time",
            hop_length=512,
        )
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        beat_presence_length = int(44100 / 512 * duration)
        beat_indices = np.round(beat_times * (44100 / 512)).astype(int)
        beat_indices = beat_indices[beat_indices < beat_presence_length]

        beat_presence = np.zeros(beat_presence_length, dtype=np.int64)
        beat_presence[beat_indices] = 1
        target_size = MUSIC_BEAT_DIM * self.target_length
        beat_presence = np.pad(
            beat_presence[:target_size],
            (0, max(0, target_size - beat_presence_length)),
        )
        return torch.from_numpy(beat_presence).view(
            MUSIC_BEAT_DIM,
            self.target_length,
        )


class Extractor_fd:
    """Extract fixed-length MotionBERT features from COCO keypoints."""

    def __init__(
        self,
        motionbert_ckpt="data/checkpoints/latest_epoch.bin",
        device=None,
        target_length=DEFAULT_TARGET_LENGTH,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        motionbert = DSTformerWrapper(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=2,
            num_joints=17,
            maxlen=243,
        )
        self.model = nn.DataParallel(motionbert).to(self.device)
        checkpoint = torch.load(
            motionbert_ckpt,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_pos"], strict=True)
        self.model.eval()
        self.max_length = 243
        self.target_length = target_length
        self.camera_shape = (1080, 1920)

    def extract(self, keypoints_path):
        keypoints = interp_nan_keypoints(load_keypoints(keypoints_path))
        confidence = keypoints[:, :, 2:3]
        coordinates = make_cam(
            keypoints[np.newaxis, ..., :2],
            self.camera_shape,
        )[0]
        keypoints = np.concatenate([coordinates, confidence], axis=2)
        keypoints = coco2h36m(keypoints[np.newaxis])[0]
        keypoints = crop_scale(keypoints, (1, 1))

        if keypoints.shape[0] > self.max_length:
            keypoints = resample_keypoints_2d(keypoints, self.max_length)

        keypoints = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.module.get_representation(keypoints)
        features = features.squeeze(0).cpu()

        features = features.permute(1, 2, 0).reshape(1, 17 * 512, -1)
        features = torch_functional.interpolate(
            features,
            size=self.target_length,
            mode="linear",
            align_corners=False,
        )
        return (
            features.reshape(17, 512, self.target_length)
            .permute(2, 0, 1)
            .reshape(
                self.target_length,
                -1,
            )
        )


class Extractor_fbd:
    """Extract the two-channel motion beat representation used by BeatDance."""

    def __init__(self, target_length=DEFAULT_TARGET_LENGTH):
        self.target_length = target_length

    def extract(self, keypoints_path):
        keypoints = interp_nan_keypoints(load_keypoints(keypoints_path))
        return extract_motion_beat(
            keypoints,
            beat_dim=MOTION_BEAT_DIM,
            target_length=self.target_length,
            fps_keypoints=MOTION_FPS,
            duration=keypoints.shape[0] / MOTION_FPS,
        )


def _find_files(root, suffix):
    return sorted(path for path in Path(root).rglob(f"*{suffix}") if path.is_file())


def _save_feature(feature, source_path, source_root, output_root):
    relative = source_path.relative_to(source_root).with_suffix(".pt")
    output_path = output_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature, output_path)


def extract_dataset_features(args):
    audio_dir = Path(args.audio_dir)
    keypoints_dir = Path(args.keypoints_dir)
    output_dir = Path(args.output_dir)

    music_feature_dir = output_dir / "music_feature"
    music_beat_dir = output_dir / "music_beat"
    motion_feature_dir = output_dir / "video_feature"
    motion_beat_dir = output_dir / "video_beat"

    music_extractor = Extractor_fm(
        device=args.device,
        use_amp=not args.disable_amp,
        target_length=args.target_length,
    )
    music_beat_extractor = Extractor_fbm(args.target_length)
    motion_extractor = Extractor_fd(
        motionbert_ckpt=args.motionbert_checkpoint,
        device=args.device,
        target_length=args.target_length,
    )
    motion_beat_extractor = Extractor_fbd(args.target_length)

    audio_files = _find_files(audio_dir, ".wav")
    for audio_path in tqdm(audio_files, desc="Extracting audio features"):
        waveform, sample_rate = torchaudio.load(audio_path)
        _save_feature(
            music_extractor.extract(waveform, sample_rate),
            audio_path,
            audio_dir,
            music_feature_dir,
        )
        audio, librosa_sample_rate = librosa.load(audio_path, sr=None)
        _save_feature(
            music_beat_extractor.extract(audio, librosa_sample_rate),
            audio_path,
            audio_dir,
            music_beat_dir,
        )

    keypoint_files = _find_files(keypoints_dir, ".pkl")
    for keypoints_path in tqdm(keypoint_files, desc="Extracting motion features"):
        _save_feature(
            motion_extractor.extract(keypoints_path),
            keypoints_path,
            keypoints_dir,
            motion_feature_dir,
        )
        _save_feature(
            motion_beat_extractor.extract(keypoints_path),
            keypoints_path,
            keypoints_dir,
            motion_beat_dir,
        )

    print(
        f"Extracted {len(audio_files)} audio clips and "
        f"{len(keypoint_files)} motion clips into {output_dir}"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        default="data/dataset/aist/audio_clips",
    )
    parser.add_argument(
        "--keypoints-dir",
        default="data/dataset/aist/keypoints_clips",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dataset/aist/beatdance_features",
    )
    parser.add_argument(
        "--motionbert-checkpoint",
        default="data/checkpoints/latest_epoch.bin",
    )
    parser.add_argument("--target-length", type=int, default=DEFAULT_TARGET_LENGTH)
    parser.add_argument("--device")
    parser.add_argument("--disable-amp", action="store_true")
    return parser


def main():
    extract_dataset_features(build_parser().parse_args())


if __name__ == "__main__":
    main()
