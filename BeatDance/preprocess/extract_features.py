"""
Extract features from AIST++ at 21.5 FPS
"""

import os
import sys

# Set environment variables BEFORE any imports
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Prevent transformers from checking for TensorFlow
os.environ["USE_TF"] = "0"  # Disable TensorFlow usage

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add AudioLDM-ControlNet to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor, AutoConfig
from tqdm import tqdm
import librosa
import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter as G
import pickle
import math
import pandas as pd

# Import MotionBERT modules at module level
from audioldm_train.modules.motion_encoder.MotionBERT.DSTformer import DSTformerWrapper
from audioldm_train.utilities.data.keypoints import (
    interp_nan_keypoints,
    coco2h36m,
    make_cam,
    crop_scale,
    resample_keypoints_2d,
)

output_dir = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST/beatdance_features"
os.makedirs(output_dir, exist_ok=True)

fm_dirpath  = os.path.join(output_dir, r"music_feature")
fbm_dirpath = os.path.join(output_dir, r"music_beat")
fd_dirpath  = os.path.join(output_dir, r"video_feature")
fbd_dirpath = os.path.join(output_dir, r"video_beat")
os.makedirs(fm_dirpath, exist_ok=True)
os.makedirs(fbm_dirpath, exist_ok=True)
os.makedirs(fd_dirpath, exist_ok=True)
os.makedirs(fbd_dirpath, exist_ok=True)

class Extractor_fm:
    def __init__(self, device=None, use_amp: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        config.conv_pos_batch_norm = False

        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", config=config, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.use_amp = use_amp and (self.device.type == "cuda")

    def extract(self, waveform, sample_rate) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        resample_rate = getattr(self.processor, "sampling_rate", sample_rate)
        if sample_rate != resample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=resample_rate)
            waveform = resampler(waveform)

        input_audio = waveform.squeeze().float()
        inputs = self.processor(
            input_audio, sampling_rate=resample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad # Run Model Inference

        # Gets output from all 13 layers
        with ctx():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

        stacked = torch.stack(outputs.hidden_states) # ([13, 1, 384, 768])
        squeezed = stacked.squeeze() # ([13, 384, 768])
        sliced = squeezed[1:, :, :] # ([12, 384, 768]) Removing the first layer because it is just a embedding
        all_layer_hidden_states = sliced[1, :, :]
        #all_layer_hidden_states = sliced.mean(dim=0) # ([384, 768])

        reshaped = all_layer_hidden_states.transpose(0, 1).unsqueeze(0)

        interpolated = torch.nn.functional.interpolate(
            reshaped,
            size=128,
            mode='linear',
            align_corners=False
        )

        result = interpolated.squeeze(0).transpose(0, 1)

        #print(f"fm shape {result.shape}") [128, 768]
        return result

'''
class Extractor_fm:
    class WeightedAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)

        def forward(self, hidden_states):
            return self.aggregator(hidden_states).squeeze(1)  # Output: [10, 768]

    def __init__(self, device=None, use_amp: bool = True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        config = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        config.conv_pos_batch_norm = False

        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", config=config, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.aggregator = self.WeightedAggregator().to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")

    def extract(self, waveform, sample_rate) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        resample_rate = getattr(self.processor, "sampling_rate", sample_rate)
        if sample_rate != resample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=resample_rate)
            waveform = resampler(waveform)

        input_audio = waveform.squeeze().float()
        inputs = self.processor(
            input_audio, sampling_rate=resample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with ctx():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze(1)

        return all_layer_hidden_states

    def aggregate(self, all_layer_hidden_states: torch.Tensor, L: int = 110) -> torch.Tensor:
        all_layer_hidden_states = all_layer_hidden_states.to(self.device)

        total_time_steps = all_layer_hidden_states.shape[1]
        interval_frames = max(1, total_time_steps // L)

        interval_features = []
        for i in range(L):
            start_idx = i * interval_frames
            end_idx = min((i + 1) * interval_frames, total_time_steps)
            interval_avg = all_layer_hidden_states[:, start_idx:end_idx, :].mean(dim=1)  # [13, 768]
            interval_features.append(interval_avg)

        interval_features = torch.stack(interval_features)            # [10, 13, 768]
        interval_features = interval_features.permute(2, 1, 0)       # [768, 13, 10]

        weighted = self.aggregator(interval_features)                 # [768, 10]
        weighted = weighted.permute(1, 0).contiguous()                # [10, 768]

        return weighted
'''
class Extractor_fbm:
    def extract(self, y, sr, target_fps: float=21.5):
        _, beats_time = librosa.beat.beat_track(y=y, sr=sr, units='time', hop_length=512)
        duration = librosa.get_duration(y=y, sr=sr)
        L = 128

        # Beat Presence Calculation
        sr_original = 44100
        hop_size = 512
        beat_presence_length = int(sr_original / hop_size * duration)  # 441
        detailed_beats_idx = np.round(beats_time * (sr_original / hop_size)).astype(int)
        detailed_beats_idx = detailed_beats_idx[detailed_beats_idx < beat_presence_length]
        beat_presence = np.zeros(beat_presence_length, dtype=int)
        beat_presence[detailed_beats_idx] = 1

        # 3. Resample
        target_total = 3 * 128  # 384
        if beat_presence_length >= target_total:
            resampled_beats = beat_presence[:target_total]
        else:
            resampled_beats = np.pad(beat_presence, (0, target_total - beat_presence_length), 'constant')

        # 4. Reshape to [4, 110]
        resampled_torch_beats = torch.tensor(resampled_beats).view(3 ,128) # [4, 110]
        # print(f"Shape of fbm: {resampled_torch_beats.shape}")  [3, 128]
        return resampled_torch_beats

def interpolate_nan_in_keypoints(keypoints):
    T, J, C = keypoints.shape
    keypoints_flatten = keypoints.reshape(T, -1)
    keypoints_interpolated = pd.DataFrame(keypoints_flatten).interpolate(method='linear', limit_direction='both', axis=0)
    return keypoints_interpolated.values.reshape(T, J, C)

class Extractor_fd:
    """
    MotionBERT-based feature extractor for motion features.
    Extracts motion representation features from 2D keypoints using MotionBERT model.
    """
    def __init__(self, motionbert_ckpt="/home/sangheon/Desktop/AudioLDM-ControlNet/data/checkpoints/latest_epoch.bin", device=None):
        """
        Initialize MotionBERT extractor.

        Args:
            motionbert_ckpt: Path to MotionBERT checkpoint file
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize MotionBERT model
        self.model = DSTformerWrapper(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=2,
            num_joints=17,
            maxlen=243
        )

        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # Load checkpoint
        # checkpoint = torch.load(motionbert_ckpt, map_location='cpu')
        # self.model.load_state_dict(checkpoint['model_pos'], strict=True)
        self.model.eval()

        self.maxlen = 243
        self.camera_shape = (1080, 1920)

    def extract(self, keypoints_path, target_fps: float = 21.5, original_fps: float = 60):
        """
        Extract MotionBERT features from 2D keypoints.

        Args:
            keypoints_path: Path to pickle file containing COCO format keypoints [T, 17, 3]
            target_fps: Target frame rate for output features (default: 21.5) - NOT USED
            original_fps: Original frame rate of keypoints (default: 60) - NOT USED

        Returns:
            Flattened motion features tensor of shape [128, 8704] (17 joints × 512 dims)
        """
        # Load raw keypoints
        with open(keypoints_path, 'rb') as f:
            raw_keypoints = pickle.load(f)

        assert isinstance(raw_keypoints, np.ndarray), "keypoints should be a numpy array"
        assert raw_keypoints.shape[1] == 17 and raw_keypoints.shape[2] == 3, \
            "keypoints should have shape [T, 17, 3]"

        # Preprocess keypoints
        k2d = interp_nan_keypoints(raw_keypoints)
        k2d_score = k2d[:, :, 2]
        k2d = k2d[:, :, :2]

        k2d = make_cam(k2d[np.newaxis, ...], self.camera_shape)[0]
        k2d = np.concatenate([k2d, k2d_score[:, :, np.newaxis]], axis=2)
        k2d = coco2h36m(k2d[np.newaxis, ...])[0]
        k2d = crop_scale(k2d, [1, 1])

        # Handle sequences longer than MotionBERT's maxlen (243)
        if k2d.shape[0] > self.maxlen:
            k2d = resample_keypoints_2d(k2d, self.maxlen)

        k2d_tensor = torch.from_numpy(k2d).float().unsqueeze(0).to(self.device)

        # Extract MotionBERT features
        with torch.no_grad():
            motion_features = self.model.module.get_representation(k2d_tensor)

        motion_features = motion_features.squeeze(0).cpu()

        # Fixed target length of 128
        target_length = 128

        # Resample from MotionBERT output length to fixed target length (128)
        current_length = motion_features.shape[0]
        indices = torch.linspace(0, current_length - 1, target_length)
        indices_floor = torch.floor(indices).long()
        indices_ceil = torch.clamp(torch.ceil(indices).long(), max=current_length - 1)

        weight = (indices - indices_floor.float()).unsqueeze(-1).unsqueeze(-1)
        motion_features_resampled = motion_features[indices_floor] * (1 - weight) + motion_features[indices_ceil] * weight
        motion_features_flattened = motion_features_resampled.reshape(target_length, -1)

        return motion_features_flattened

class Extractor_fbd:
    """
    Extract motion beat features using the same approach as AISTBeatDanceDataset.
    Produces binary beat features based on local minima in kinetic velocity.
    """
    def __init__(self):
        pass

    def extract(self, keypoints_path, target_fps: float = 21.5):
        # keypoints: [T, 17, 3] == COCO format
        with open(keypoints_path, 'rb') as f:
            keypoints = pickle.load(f)
        assert isinstance(keypoints, np.ndarray), "keypoints should be a numpy array"
        assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should have shape [T, 17, 3]"

        # Interpolate NaN values
        keypoints = interp_nan_keypoints(keypoints)

        # Calculate motion beat feature using extract_motion_beat logic
        fps_keypoints = 60
        duration = keypoints.shape[0] / fps_keypoints
        target_length = 128
        beat_dim = 2

        beat_feature = self._extract_motion_beat(
            keypoints,
            beat_dim=beat_dim,
            target_length=target_length,
            fps_keypoints=fps_keypoints,
            duration=duration,
        )  # shape: (target_length, beat_dim)

        return beat_feature

    def _extract_motion_beat(self, keypoints, beat_dim, target_length, fps_keypoints, duration):
        """
        Extract motion beat features from keypoints.
        Same logic as extract_motion_beat from audioldm_train/utilities/data/keypoints.py

        Args:
            keypoints: numpy array of shape [T, 17, 3]
            beat_dim: dimension of beat feature (typically 2)
            target_length: target sequence length (typically 128)
            fps_keypoints: FPS of keypoints (typically 60)
            duration: duration in seconds

        Returns:
            Binary beat feature tensor of shape [target_length, beat_dim]
        """
        assert isinstance(keypoints, np.ndarray), "keypoints should be a numpy array"
        assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should have shape [T, 17, 3]"

        # Calculate velocity (mean across all joints)
        velocity = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)

        # Create kinetic velocity array
        kinetic_vel = np.zeros(keypoints.shape[0])
        valid_len = min(len(velocity), len(kinetic_vel) - 1)
        kinetic_vel[1:valid_len + 1] = np.nan_to_num(velocity[:valid_len], nan=0.0)

        # Normalize
        max_val = np.max(kinetic_vel)
        if max_val > 0:
            kinetic_vel /= max_val

        # Apply Gaussian filter
        kinetic_vel = np.nan_to_num(G(kinetic_vel, sigma=2))

        # Find local minima as motion beats
        motion_beats = argrelextrema(kinetic_vel, np.less)[0]
        motion_beats = motion_beats[motion_beats < keypoints.shape[0]]

        # Convert to flattened target space
        flatten_target_length = target_length * beat_dim
        beats_time = motion_beats / fps_keypoints
        fps = flatten_target_length / duration
        beats_idx = np.round(beats_time * fps).astype(int)
        beats_idx = np.unique(beats_idx)
        beats_idx = beats_idx[beats_idx < flatten_target_length]

        # Create binary beat vector
        beats_vector = np.zeros(flatten_target_length, dtype=int)
        beats_vector[beats_idx] = 1

        # Reshape to [target_length, beat_dim]
        beats_feature = torch.tensor(beats_vector).view(target_length, beat_dim)
        print(f"Beats_feature shape: {beats_feature.shape}")  # Expected: [128, 2]
        return beats_feature

def main():
    fm_extractor = Extractor_fm()
    fbm_extractor = Extractor_fbm()
    fd_extractor = Extractor_fd()
    fbd_extractor = Extractor_fbd()

    audio_dir = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST/audio_clips"
    audio_files = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    total_audio_files = len(audio_files)

    for audio_path in tqdm(audio_files, total=total_audio_files, desc="Extracting features"):
        rel_path = os.path.relpath(os.path.dirname(audio_path), audio_dir)
        pt_name = os.path.splitext(os.path.basename(audio_path))[0] + '.pt'

        # Load audio to get duration
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        # Extract music features
        output_subdir = os.path.join(fm_dirpath, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        pt_path = os.path.join(output_subdir, pt_name)

        fm = fm_extractor.extract(waveform, sample_rate)
        #fm = fm_extractor.aggregate(fm_extractor.extract(waveform, sample_rate))
        #print(f"Extracted fm shape: {fm.shape}")
        torch.save(fm, pt_path)

        # Extract music beat features
        output_subdir = os.path.join(fbm_dirpath, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        pt_path = os.path.join(output_subdir, pt_name)
        y, sr = librosa.load(audio_path, sr=None)
        fbm = fbm_extractor.extract(y, sr)
        torch.save(fbm, pt_path)

    keypoints_dir = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST/keypoints_clips"
    keypoints_files = []
    for root, _, files in os.walk(keypoints_dir):
        for file in files:
            if file.lower().endswith('.pkl'):
                keypoints_files.append(os.path.join(root, file))
    total_keypoints = len(keypoints_files)

    for keypoints_path in tqdm(keypoints_files, total=total_keypoints, desc="Extracting features"):
        rel_path = os.path.relpath(os.path.dirname(keypoints_path), keypoints_dir)
        pt_name = os.path.splitext(os.path.basename(keypoints_path))[0] + '.pt'

        # Extract video features
        output_subdir = os.path.join(fd_dirpath, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        pt_path = os.path.join(output_subdir, pt_name)
        fd = fd_extractor.extract(keypoints_path)
        torch.save(fd, pt_path)

        # Extract video beat features
        output_subdir = os.path.join(fbd_dirpath, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        pt_path = os.path.join(output_subdir, pt_name)
        fbd = fbd_extractor.extract(keypoints_path)
        torch.save(fbd, pt_path)

    print("Feature extraction complete!")

if __name__ == "__main__":
    main()