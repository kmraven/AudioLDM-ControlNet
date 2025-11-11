import json
import os

import numpy as np
import torch

from audioldm_train.utilities.data.dataset import AudioDataset


class AISTDataset(AudioDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)

        datum = self.data[index]
        kpt_path = datum.get("motion", None)
        k2d = self._load_keypoints_2d(kpt_path)  # (T_raw, J, 2) or (T_raw, 2*J)

        tgt_T = self.target_length
        k2d = self._time_resample_sequence(k2d, tgt_T)  # (tgt_T, J, 2) or (tgt_T, 2*J)

        if isinstance(k2d, np.ndarray):
            k2d = torch.from_numpy(k2d).float()
        else:
            k2d = k2d.float()

        data.update({"motion": k2d})
        return data

    def feature_extraction(self, index):
        """
        Omits below parts from parent class' feature_extraction:
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
        """
        期待する形式:
          - .npy: (T, J, 2) または (T, 2J)
          - .json: {"keypoints": [[x1,y1, x2,y2, ...], ...]} 等に緩く対応
        """
        if path is None or (not os.path.exists(path)):
            # 無い場合はゼロ配列を返す（tgt_T で後から埋め、shape は推論）
            # ただし shape が決まらないため、後段のモデル都合に合わせて J を固定したい場合は config で指定してください
            return np.zeros((1, 34, 2), dtype=np.float32)  # 仮の(1, J=34, 2)

        ext = os.path.splitext(path)[-1].lower()
        if ext == ".npy":
            arr = np.load(path)
            # (T, 2J) → (T, J, 2) に直しておくと扱いやすい
            if arr.ndim == 2 and arr.shape[1] % 2 == 0:
                J = arr.shape[1] // 2
                arr = arr.reshape(arr.shape[0], J, 2)
            return arr.astype(np.float32)
        elif ext == ".json":
            with open(path, "r") as f:
                obj = json.load(f)
            # よくある構造: {"keypoints": [[x1,y1,x2,y2,...], ...]}
            if "keypoints" in obj:
                kp = np.array(obj["keypoints"], dtype=np.float32)
                if kp.ndim == 2 and kp.shape[1] % 2 == 0:
                    J = kp.shape[1] // 2
                    kp = kp.reshape(kp.shape[0], J, 2)
                return kp
            raise ValueError(f"Unsupported json structure in {path}")
        else:
            raise ValueError(f"Unsupported keypoint file: {path}")

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
