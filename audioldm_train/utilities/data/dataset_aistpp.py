import os

import numpy as np
import torch
import torchaudio.transforms as T

from audioldm_train.utilities.data.dataset import AudioDataset, spectral_normalize_torch
from librosa.filters import mel as librosa_mel_fn
from audioldm_train.utilities.data.keypoints import (
    load_keypoints,
    interp_nan_keypoints,
    resample_keypoints_2d,
    coco2h36m,
    make_cam,
    crop_scale,
    extract_motion_beat,
)


class AISTDataset(AudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strech_rate = None

    def build_setting_parameters(self):
        super().build_setting_parameters()
        self.additional_strech_rate = self.config["preprocessing"].get("strech_augmentation", {}).get("additional_strech_rate", 0.)  # type: ignore
        self.stretch_prob = self.config["preprocessing"].get("strech_augmentation", {}).get("stretch_prob", 0.)  # type: ignore
        self.camera_shape = (
            self.config["preprocessing"]["motion"]["camera_height"],  # type: ignore
            self.config["preprocessing"]["motion"]["camera_width"],   # type: ignore
        )
        self.motion_target_length = self.config["preprocessing"]["motion"]["target_length"]  # type: ignore
        if "train" in self.split:
            self.scale_range = self.config["preprocessing"]["motion"].get("scale_range_train", [1, 1])  # type: ignore
        else:
            self.scale_range = self.config["preprocessing"]["motion"].get("scale_range_test", [1, 1])  # type: ignore

    def __getitem__(self, index):
        data = super().__getitem__(index)
        datum = self.data[index]
        kpt_path = datum.get("motion", None)
        k2d = self.process_keypoints(kpt_path)
        data.update({"motion": k2d})
        if self.split=="train" and "text" in data and np.random.rand() < 0.5:
            data["text"] = ""
        return data
    
    def process_keypoints(self, kpt_path):
        k2d = load_keypoints(kpt_path)  # shape: (T, 17, 3)
        k2d = interp_nan_keypoints(k2d)  # shape: (T, 17, 3)
        if self.strech_rate is not None:
            k2d = resample_keypoints_2d(k2d, int(self.motion_target_length / self.strech_rate))
            k2d = k2d[:self.motion_target_length]
            self.strech_rate = None
        else:
            k2d = resample_keypoints_2d(k2d, self.motion_target_length)
        k2d_score = k2d[:, :, 2]  # shape: (T, 17)
        k2d = k2d[:, :, :2]  # shape: (T, 17, 2)
        k2d = make_cam(k2d[np.newaxis, ...], self.camera_shape)[0]  # shape: (T, 17, 2)
        k2d = np.concatenate([k2d, k2d_score[:, :, np.newaxis]], axis=2)  # shape: (T, 17, 3)
        k2d = coco2h36m(k2d[np.newaxis, ...])[0]  # shape: (T, 17, 3)
        k2d = crop_scale(k2d, self.scale_range)  # shape: (T, 17, 3)
        return k2d

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

    def mel_spectrogram_train(self, y):
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
        
        if self.split == "train" and torch.rand(1).item() < self.stretch_prob:
            original_length = stft_spec.size(2)
            self.strech_rate = float(torch.empty(1).uniform_(1.0 - self.additional_strech_rate, 1.0))
            ts = T.TimeStretch(n_freq=stft_spec.size(1), fixed_rate=self.strech_rate).to(y.device)
            stft_spec = ts(stft_spec)
            stft_spec = stft_spec[:, :, :original_length]

        stft_spec = torch.abs(stft_spec)
        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]


class AISTBeatDanceDataset(AISTDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_setting_parameters(self):
        super().build_setting_parameters()
        self.fps_keypoints = self.config["preprocessing"]["motion"]["fps_keypoints"]  # type: ignore
        self.beat_dim = self.config["preprocessing"]["motion"]["beat_dim"]  # type: ignore

    def __getitem__(self, index):
        data = super(AISTDataset).__getitem__(index)
        datum = self.data[index]
        kpt_path = datum.get("motion", None)
        beat_feature = self.process_beat_feature(kpt_path)  # call first not to lose strech_rate
        k2d = self.process_keypoints(kpt_path)
        data.update({"motion": {
            'motion_feature': k2d,
            'motion_beat_feature': beat_feature
        }})
        if self.split=="train" and "text" in data and np.random.rand() < 0.5:
            data["text"] = ""
        return data
    
    # TODO pre-calculate and save beat feature to speed up data loading
    def process_beat_feature(self, kpt_path):
        k2d = load_keypoints(kpt_path)  # shape: (T, 17, 3)
        k2d = interp_nan_keypoints(k2d)  # shape: (T, 17, 3)
        if self.strech_rate is not None:
            original_length = len(k2d)
            k2d = resample_keypoints_2d(k2d, int(original_length / self.strech_rate))
            k2d = k2d[:original_length]
        beat_feature = extract_motion_beat(
            k2d,
            beat_dim=self.beat_dim,
            target_length=self.motion_target_length,
            fps_keypoints=self.fps_keypoints,
            duration=self.duration,
        )  # shape: (beat_dim, target_length)
        return beat_feature
