import pickle

import numpy as np
import torch
import pandas as pd

from audioldm_train.utilities.data.dataset import AudioDataset


class AISTDataset(AudioDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)

        datum = self.data[index]
        kpt_path = datum.get("motion", None)
        k2d = self._load_keypoints_2d(kpt_path)

        tgt_T = self.target_length
        k2d = self._time_resample_sequence(k2d, tgt_T)  # (tgt_T, J, 2) or (tgt_T, 2*J)

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

    def _load_keypoints_2d(self, path):
        with open(path, "rb") as f:
            keypoints = pickle.load(f)
        keypoints = self._interpolate_nan_in_keypoints(keypoints)
        return torch.tensor(keypoints)

    def _interpolate_nan_in_keypoints(keypoints):
        T, J, C = keypoints.shape
        keypoints_flatten = keypoints.reshape(T, -1)
        keypoints_interpolated = pd.DataFrame(keypoints_flatten).interpolate(method='linear', limit_direction='both', axis=0)
        return keypoints_interpolated.values.reshape(T, J, C)

    def _time_resample_sequence(self, seq, target_T):
        """
        seq: (T, J, 2) または (T, 2J)
        target_T: int
        return: (target_T, J, 2) または (target_T, 2J)
        """
        seq = np.asarray(seq)
        if seq.ndim == 2:
            # (T, 2J) のときは (T, D) とみなす
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
            # (T, J, 2)
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
