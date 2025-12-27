"""
Extract features from AIST++ at 21.5 FPS
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        L = int(duration * target_fps)  # 110

        # Beat Presence Calculation
        sr_original = 44100
        hop_size = 512
        beat_presence_length = int(sr_original / hop_size * duration)  # 441
        detailed_beats_idx = np.round(beats_time * (sr_original / hop_size)).astype(int)
        detailed_beats_idx = detailed_beats_idx[detailed_beats_idx < beat_presence_length]
        beat_presence = np.zeros(beat_presence_length, dtype=int)
        beat_presence[detailed_beats_idx] = 1

        # 3. Resample
        target_total = 4 * L  # 440
        if beat_presence_length >= target_total:
            resampled_beats = beat_presence[:target_total]
        else:
            resampled_beats = np.pad(beat_presence, (0, target_total - beat_presence_length), 'constant')

        # 4. Reshape to [4, 110]
        resampled_torch_beats = torch.tensor(resampled_beats).view(4,L) # [4, 110]
        #print(f"Shape of resampled torch beats: {resampled_torch_beats.shape}")
        return resampled_torch_beats

def interpolate_nan_in_keypoints(keypoints):
    T, J, C = keypoints.shape
    keypoints_flatten = keypoints.reshape(T, -1)
    keypoints_interpolated = pd.DataFrame(keypoints_flatten).interpolate(method='linear', limit_direction='both', axis=0)
    return keypoints_interpolated.values.reshape(T, J, C)

class Extractor_fd:

    COCO_BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4),   # face
        (0, 5),  (5, 7), (7, 9), (5, 11),     # left upper body
        (0, 6), (6, 8), (8, 10), (6, 12),     # right upper body
        (11, 13), (13, 15),                  # Left lowerbody
        (12, 14), (14, 16),                   # Right lowerbody
    ]

    def __init__(self):
        pass

    def __angle_difference(self, a1, a2):
        """Compute wrapped angle difference within [-pi, pi]"""
        delta = a1 - a2
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta

    def __compute_bone_angles(self, positions, bones):
        """
        Compute 2D joint orientation angles for each bone. (in radian on the image)
        positions: (T, J, 2)
        Returns: (T, num_bones)
        """
        T, J, _ = positions.shape
        angles = []
        for p1_idx, p2_idx in bones:
            vec = positions[:, p2_idx, :] - positions[:, p1_idx, :]  # (T, 2)
            bone_angle = torch.atan2(vec[:, 1], vec[:, 0])     # (T,)
            angles.append(bone_angle)
        return torch.stack(angles, dim=1)  # (T, num_bones)

    def resample_to_target_fps(self, features, original_fps=60, target_fps=21.5):
        """Resample features from original fps to target fps using linear interpolation from 60 fps to 21.5 fps"""
        original_length = features.shape[0]
        target_length = int(original_length * target_fps / original_fps)

        if target_length == original_length:
            return features

        indices = torch.linspace(0, original_length - 1, target_length)
        indices_floor = torch.floor(indices).long()
        indices_ceil = torch.clamp(torch.ceil(indices).long(), max=original_length - 1)

        weight = (indices - indices_floor.float()).unsqueeze(-1)
        resampled = features[indices_floor] * (1 - weight) + features[indices_ceil] * weight

        return resampled

    def extract(self, keypoints_path, target_fps: float = 21.5):
        """
        Input: 2d keypoints in COCO format (T, 17, 2)
        Output: [root position,
                root velocity,
                joint position (rel to root),
                joint velocity (rel to root),
                joint acceleration (rel to root),
                joint angles (each bone)
                joint angular velocity (each bone)
            ]
        """
        with open(keypoints_path, 'rb') as f:
            keypoints = pickle.load(f)

        assert isinstance(keypoints, np.ndarray), "keypoints should be a list of numpy arrays"
        assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should be a list of numpy arrays with shape [T, 17, 3]"

        pose_feature = torch.tensor(interpolate_nan_in_keypoints(keypoints)[:, :, :2])  # reshape to (T, 17, 2)
        pose_feature = torch.nan_to_num(pose_feature, nan=0.0)
        T, J, _ = pose_feature.shape # (T,J,2)

        # using coco format, extract joint position, velocity, acceleration in root space
        root = (pose_feature[:, 11, :] + pose_feature[:, 12, :]) / 2 # (T, 2) midpoint between left and right hip
        root_vel = torch.cat((torch.zeros(1, 2), root[1:] - root[:-1]), 0) #(T, 2)
        joint_pos = pose_feature - root.unsqueeze(1) #(T, J, 2) joint position in root space
        joint_vel = torch.cat((torch.zeros(1, J, 2), joint_pos[1:] - joint_pos[:-1]), 0) #(T, J, 2) joint linear velocity in root space
        joint_acc = torch.cat((torch.zeros(1, J, 2), joint_vel[1:] - joint_vel[:-1]), 0) #(T, J, 2) joint linear acceleration in root space

        # 2d joint orientations, from each parent joint
        # Bone angles on the 2D image
        joint_angles = self.__compute_bone_angles(joint_pos, self.COCO_BONES)  # (T, num_bones)
        # Bone angular velocity (wrapped between -pi and pi) and pad zeros
        joint_angle_vel = torch.cat((torch.zeros(1, len(self.COCO_BONES)), self.__angle_difference(joint_angles[1:], joint_angles[:-1])), 0)  # (T, num_bones)
        all_feat = torch.cat((root, # (T, 4+6J+2(J-1)) = (T, 138)
                                root_vel,
                                joint_pos.flatten(start_dim=1),
                                joint_vel.flatten(start_dim=1),
                                joint_acc.flatten(start_dim=1),
                                joint_angles,
                                joint_angle_vel), 1)

        # Resample from 60fps to target_fps
        all_feat_resampled = self.resample_to_target_fps(all_feat, 60, target_fps)
        return all_feat_resampled  # shape: (target_length, 138)

class Extractor_fbd:
    def __init__(self):
        pass

    def extract(self, keypoints_path, target_fps: float = 21.5):
        # keypoints: [T, 17, 3] == COCO format
        with open(keypoints_path, 'rb') as f:
            keypoints = pickle.load(f)
        assert isinstance(keypoints, np.ndarray), "keypoints should be a list of numpy arrays"
        assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should be a list of numpy arrays with shape [T, 17, 3]"

        keypoints = interpolate_nan_in_keypoints(keypoints)
        velocity = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
        kinetic_vel = np.zeros(keypoints.shape[0])
        valid_len = min(len(velocity), len(kinetic_vel) - 1)
        kinetic_vel[1:valid_len + 1] = np.nan_to_num(velocity[:valid_len], nan=0.0)
        max_val = np.max(kinetic_vel)
        if max_val > 0:
            kinetic_vel /= max_val
        kinetic_vel = np.nan_to_num(G(kinetic_vel, sigma=2))
        motion_beats = argrelextrema(kinetic_vel, np.less)[0][argrelextrema(kinetic_vel, np.less)[0] < keypoints.shape[0]]

        fps_keypoints = 60
        beats_time = motion_beats / fps_keypoints

        # Calculate duration from keypoints length
        duration = keypoints.shape[0] / fps_keypoints

        L = 110
        target_total = 3 * L  # 330

        motion_presence_length = len(kinetic_vel)  # ~307

        # Create motion beat presence vector
        motion_presence = np.zeros(motion_presence_length, dtype=float)
        motion_presence[:] = kinetic_vel  # Use velocity as continuous feature
        motion_presence[motion_beats] = 1.0  # Mark beat points explicitly

        if motion_presence_length == target_total:
            resampled_motion = motion_presence
        elif motion_presence_length > target_total:
            indices = np.linspace(0, motion_presence_length - 1, target_total)
            indices_floor = np.floor(indices).astype(int)
            indices_ceil = np.clip(np.ceil(indices).astype(int), 0, motion_presence_length - 1)
            weight = indices - indices_floor
            resampled_motion = motion_presence[indices_floor] * (1 - weight) + motion_presence[indices_ceil] * weight
        else:
            # Upsample using linear interpolation
            indices = np.linspace(0, motion_presence_length - 1, target_total)
            indices_floor = np.floor(indices).astype(int)
            indices_ceil = np.clip(np.ceil(indices).astype(int), 0, motion_presence_length - 1)
            weight = indices - indices_floor
            resampled_motion = motion_presence[indices_floor] * (1 - weight) + motion_presence[indices_ceil] * weight

        # Reshape to [3, 110]
        resampled_motion_torch = torch.tensor(resampled_motion).view(3, L) # Shape: [3, 110]
        print(f"Resampled motion torch {resampled_motion_torch.shape}")
        return resampled_motion_torch

def main():
    fm_extractor = Extractor_fm()
    fbm_extractor = Extractor_fbm()
    fd_extractor = Extractor_fd()
    fbd_extractor = Extractor_fbd()

    audio_dir = "/home/sangheon/Desktop/SonyCSL/dance2music/AIST/audio_clips"
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

    keypoints_dir = "/home/sangheon/Desktop/SonyCSL/dance2music/AIST/keypoints_clips"
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



