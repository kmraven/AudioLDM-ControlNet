import pickle
import math
import os

import numpy as np
import torch
import pandas as pd
import torchaudio.transforms as T

from audioldm_train.utilities.data.dataset import AudioDataset, spectral_normalize_torch
from librosa.filters import mel as librosa_mel_fn


class AISTDataset(AudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_preprocessor = MotionPreprocessor()

    def __getitem__(self, index):
        data = super().__getitem__(index)
        datum = self.data[index]
        kpt_path = datum.get("motion", None)
        # k2d = self.motion_preprocessor._load_keypoints(kpt_path)  # shape: (T, 17, 3)
        # k2d = self.motion_preprocessor._interpolate_nan_in_keypoints(k2d)[:, :, :2]  # shape: (T, 17, 2)
        k2d = self.motion_preprocessor.extract(kpt_path)  # shape: (T, 138)
        k2d = self._time_resample_sequence(k2d, self.target_length)
        data.update({"motion": k2d})
        return data

    def feature_extraction(self, index):
        """
        Omits below parts from original 'feature_extraction' methos:
         - skipping index when it's out of range
        """
        label_indices = np.zeros(self.label_num, dtype=np.float32)
        datum = self.data[index]
        (
            log_mel_spec,
            stft,
            waveform,
            random_start,
        ) = self.read_audio_file(datum["wav"])
        mix_datum = None
        if self.label_num > 0 and "labels" in datum.keys():
            for label_str in datum["labels"].split(","):
                label_indices[int(self.index_dict[label_str])] = 1.0

        # If the key "label" is not in the metadata, return all zero vector
        label_indices = torch.FloatTensor(label_indices)

        # The filename of the wav file
        fname = datum["wav"]
        waveform = torch.FloatTensor(waveform)

        return (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_indices,
            (datum, mix_datum),
            random_start,
        )

    def _time_resample_sequence(self, seq, target_T):
        seq = np.asarray(seq)
        if seq.ndim == 2:
            T, D = seq.shape
            if T == target_T:
                return seq
            # 線形補間
            x_old = np.linspace(0, 1, T, dtype=np.float32)
            x_new = np.linspace(0, 1, target_T, dtype=np.float32)
            out = np.empty((target_T, D), dtype=np.float32)
            for d in range(D):
                out[:, d] = np.interp(x_new, x_old, seq[:, d])
            return out
        elif seq.ndim == 3:
            T, J, C = seq.shape
            assert C == 2, f"Expected last dim=2, got {C}"
            if T == target_T:
                return seq
            x_old = np.linspace(0, 1, T, dtype=np.float32)
            x_new = np.linspace(0, 1, target_T, dtype=np.float32)
            out = np.empty((target_T, J, C), dtype=np.float32)
            for j in range(J):
                out[:, j, 0] = np.interp(x_new, x_old, seq[:, j, 0])
                out[:, j, 1] = np.interp(x_new, x_old, seq[:, j, 1])
            return out
        else:
            raise ValueError(f"Unsupported keypoints shape: {seq.shape}")

    def _relative_path_to_absolute_path(self, metadata, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i in range(len(metadata["data"])):
            metadata["data"][i]["wav"] = os.path.join(
                root_path, metadata["data"][i]["wav"]
            )
            metadata["data"][i]["motion"] = os.path.join(
                root_path, metadata["data"][i]["motion"]
            )
        return metadata

    def mel_spectrogram_train(
            self,
            y,
            max_expand_rate=0.1,
            stretch_prob=0.5,
    ):
        if torch.min(y) < -1.0:
            print("train min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("train max value is ", torch.max(y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel,
                self.mel_fmin,
                self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # shape: [channels, freq_bins, time_frames], complex dtype
        
        if self.split == "train" and torch.rand(1).item() < stretch_prob:
            original_length = stft_spec.size(2)
            rate = float(torch.empty(1).uniform_(1.0, 1.0 + max_expand_rate))
            ts = T.TimeStretch(hop_length=self.hop_length, n_freq=stft_spec.size(1), fixed_rate=rate).to(y.device)
            stft_spec = ts(stft_spec)
            stft_spec = stft_spec[:, :, :original_length]

        stft_spec = torch.abs(stft_spec)
        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]


class MotionPreprocessor:

    COCO_BONES = [
        (0, 1), (0, 2), (1, 3), (2, 4),   # face
        (0, 5),  (5, 7), (7, 9), (5, 11),     # left upper body
        (0, 6), (6, 8), (8, 10), (6, 12),     # right upper body
        (11, 13),  (13, 15),                  # Left lowerbody
        (12, 14), (14, 16),                   # Right lowerbody
    ]

    def __init__(self):
        pass

    def _load_keypoints(self, keypoints_path):
        with open(keypoints_path, 'rb') as f:
            keypoints = pickle.load(f)
        assert isinstance(keypoints, np.ndarray), "keypoints should be a list of numpy arrays"
        assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should be a list of numpy arrays with shape [T, 17, 3]"
        return keypoints

    def _interpolate_nan_in_keypoints(self, keypoints):
        T, J, C = keypoints.shape
        keypoints_flatten = keypoints.reshape(T, -1)
        keypoints_interpolated = pd.DataFrame(keypoints_flatten).interpolate(method='linear', limit_direction='both', axis=0)
        return keypoints_interpolated.values.reshape(T, J, C)

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

    def extract(self, keypoints_path):
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
        keypoints = self._load_keypoints(keypoints_path)  # shape: (T, 17, 3)
        pose_feature = torch.tensor(self._interpolate_nan_in_keypoints(keypoints)[:, :, :2])  # reshape to (T, 17, 2)
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
        return all_feat  # shape: (T, 138)
