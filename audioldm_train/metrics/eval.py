import os
from audioldm_eval.datasets.load_mel import WaveDataset

import torch
from torch.utils.data import DataLoader
from audioldm_eval.audio.tools import write_json
from audioldm_eval import EvaluationHelper
from audioldm_eval.metrics.fad import FrechetAudioDistance
from tqdm import tqdm
from audioldm_train.metrics.beat_score import calculate_beat_score
from audioldm_train.metrics.clap import CLAPScoreCalculator
from audioldm_train.utilities.data.keypoints import (
    load_keypoints,
    interp_nan_keypoints,
    resample_keypoints_1d,
    resample_keypoints_2d
)


class CorrespondedWaveDataset(WaveDataset):
    def __init__(self, gen_music_path, gt_music_path, sampling_rate, gt_motion_path, limit_num=None):
        self.sr = sampling_rate

        gen_music_datalist = [
            os.path.join(gen_music_path, x)
            for x in os.listdir(gen_music_path)
        ]
        gt_music_datalist = [
            os.path.join(gt_music_path, x)
            for x in os.listdir(gt_music_path)
        ]
        gt_motion_datalist = [
            os.path.join(gt_motion_path, x)
            for x in os.listdir(gt_motion_path)
        ]

        gen_music_datalist = sorted(gen_music_datalist)
        gt_music_datalist = sorted(gt_music_datalist)
        gt_motion_datalist = sorted(gt_motion_datalist)

        def to_key(path):
            return os.path.splitext(os.path.basename(path))[0]

        gen_map = {to_key(p): p for p in gen_music_datalist}
        gt_map  = {to_key(p): p for p in gt_music_datalist}
        gt_motion_map  = {to_key(p): p for p in gt_motion_datalist}
        common_keys = sorted(set(gen_map.keys()) & set(gt_map.keys()) & set(gt_motion_map.keys()))

        if limit_num is not None:
            common_keys = common_keys[:limit_num]

        self.datalist = [
            {
                "gen_music": gen_map[k],
                "gt_music": gt_map[k],
                "gt_motion": gt_motion_map[k],
            }
            for k in common_keys
        ]

    def __getitem__(self, index):
        while True:
            try:
                file_dict = self.datalist[index]
                waveform_dict = {}
                for key, filename in file_dict.items():
                    if "music" in key:
                        w = self.read_from_file(filename)
                        if w.size(-1) < 1:
                            raise ValueError("empty file %s" % filename)
                    elif "motion" in key:
                        w = load_keypoints(filename)
                        w = interp_nan_keypoints(w)
                        w = resample_keypoints_2d(w, T_target=int(5.12 * 60))  # TODO fix hardcoding
                        w = w[:, :, :2]  # remove confidence channel
                        if w.shape[0] < 1:
                            raise ValueError("empty file %s" % filename)
                    waveform_dict[key] = w
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.datalist)

        return waveform_dict, os.path.basename(filename)


class ControlNetEvaluationHelper(EvaluationHelper):
    def __init__(self, sampling_rate):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_rate = sampling_rate
        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        self.frechet.model = self.frechet.model.to(self.device)
        self.clap_score_calculator = CLAPScoreCalculator()

    def calculate_metrics(self, gen_music_path, gt_music_path, gt_motion_path, same_name, limit_num=None, calculate_psnr_ssim=False, calculate_lsd=False, recalculate=False):
        torch.manual_seed(0)
        num_workers = 6

        dataloader = DataLoader(
            CorrespondedWaveDataset(
                gen_music_path,
                gt_music_path,
                self.sampling_rate,
                gt_motion_path,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=num_workers,
        )
        out = {}

        # FAD
        ######################################################################################################################
        if(recalculate): 
            print("Calculate FAD score from scratch")
        fad_score = self.frechet.score(gen_music_path, gt_music_path, limit_num=limit_num, recalculate=recalculate)
        out.update(fad_score)

        # Beat Score & CLAP Score
        ######################################################################################################################
        avg_scores = self.__calculate_beat_clap_score(dataloader)
        out.update(avg_scores)

        print("Evaluation Result:")
        print("\n".join((f"{k}: {v:.7f}" for k, v in out.items())))
        return out
    
    def __calculate_beat_clap_score(self, dataloader):
        scores = {
            "beat_coverage_score": [],
            "beat_hit_score": [],
            "f1_score": [],
            "tempo_difference": [],
            "clap_score": [],
        }
        for waveform_dict, _ in tqdm(dataloader):
            gen_music = waveform_dict["gen_music"].squeeze(0)
            gt_music = waveform_dict["gt_music"].squeeze(0)
            gt_motion = waveform_dict["gt_motion"]
            bcs, bhs, f1, td, bas = calculate_beat_score(gt_motion, gt_music, gen_music, self.sampling_rate)
            clap_score = self.clap_score_calculator.calculate(gen_music, gt_music)
            scores["beat_coverage_score"].append(bcs)
            scores["beat_hit_score"].append(bhs)
            scores["f1_score"].append(f1)
            scores["tempo_difference"].append(td)
            scores["beat_alignment_score"].append(bas)
            scores["clap_score"].append(clap_score)
        avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
        return avg_scores
