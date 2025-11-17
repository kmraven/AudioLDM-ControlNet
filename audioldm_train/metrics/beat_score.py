"""
from LORIS: https://github.com/OpenGVLab/LORIS/blob/443e4b992a1dc32ab594373012e9fef9928ce404/beats_scores.py
- time resolution of beat detection is modified to 5 Hz (following the setting of textual inversion)
- tempo_difference is originally implemented since BeatDance authors doesn't 
"""

import librosa
import numpy as np


class BeatScoreCalculator:
    def __beat_detect(self, x, sr=22050):
        if x.ndim > 1:
            x = librosa.to_mono(x)
        onsets = librosa.onset.onset_detect(
            y=x,
            sr=sr,
            wait=1,
            delta=0.2,
            pre_avg=3,
            post_avg=3,
            pre_max=3,
            post_max=3,
            units="time",
        )
        time_resolution = 5  # 5 Hz
        n = np.ceil(len(x) / sr * time_resolution)
        beats = [0] * int(n)
        for time in onsets:
            beats[int(np.trunc(time * time_resolution))] = 1
        return beats

    def __beat_scores(self, gt, syn):
        gt = gt[: len(syn)]
        assert len(gt) == len(syn)
        total_beats = sum(gt)
        cover_beats = sum(syn)
        hit_beats = 0
        for i in range(len(gt)):
            if gt[i] == 1 and gt[i] == syn[i]:
                hit_beats += 1
        hit_rate = hit_beats / total_beats if total_beats else 0
        cover_rate = hit_beats / cover_beats if cover_beats else 0
        return cover_rate, hit_rate

    def __f1_score(self, cover_rate, hit_rate):
        if (cover_rate + hit_rate) == 0:
            return 0
        return 2 * (cover_rate * hit_rate) / (cover_rate + hit_rate)

    def __estimate_bpm(self, x, sr=22050, aggregate=np.mean):
        if x.ndim > 1:
            x = librosa.to_mono(x)
        onset_env = librosa.onset.onset_strength(y=x, sr=sr)
        if onset_env is None or np.allclose(onset_env.sum(), 0.0):
            return 0.0
        # Global tempo (BPM): aggregate with median for robustness
        bpm_arr = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=aggregate
        )
        # librosa returns a 1-element array for global tempo
        bpm = float(np.squeeze(bpm_arr))
        # Guard against NaN/Inf
        if not np.isfinite(bpm):
            return 0.0
        return bpm

    def __tempo_difference(self, gt_audio, syn_audio, sr=22050):
        bpm_gt = self.__estimate_bpm(gt_audio, sr=sr)
        bpm_syn = self.__estimate_bpm(syn_audio, sr=sr)
        return abs(bpm_syn - bpm_gt)

    def calculate(self, gt_audio, syn_audio, sr=22050):
        gt_beats = self.__beat_detect(gt_audio, sr)
        syn_beats = self.__beat_detect(syn_audio, sr)
        cover_rate, hit_rate = self.__beat_scores(gt_beats, syn_beats)
        f1 = self.__f1_score(cover_rate, hit_rate)
        td = self.__tempo_difference(gt_audio, syn_audio, sr=sr)
        return cover_rate, hit_rate, f1, td
