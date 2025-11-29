import os
from audioldm_eval.datasets.load_mel import WaveDataset

import torch
from torch.utils.data import DataLoader
from audioldm_eval.audio.tools import write_json
from audioldm_eval import EvaluationHelper
from audioldm_eval.metrics.fad import FrechetAudioDistance
from tqdm import tqdm
from audioldm_train.metrics.beat_score import BeatScoreCalculator
from audioldm_train.metrics.clap import CLAPScoreCalculator


class CorrespondedWaveDataset(WaveDataset):
    def __init__(self, generated_path, groundtruth_path, sampling_rate, limit_num=None):
        self.sr = sampling_rate

        generated_datalist = [
            os.path.join(generated_path, x)
            for x in os.listdir(generated_path)
        ]
        groundtruth_datalist = [
            os.path.join(groundtruth_path, x)
            for x in os.listdir(groundtruth_path)
        ]

        generated_datalist = sorted(generated_datalist)
        groundtruth_datalist = sorted(groundtruth_datalist)

        def to_key(path):
            return os.path.splitext(os.path.basename(path))[0]

        gen_map = {to_key(p): p for p in generated_datalist}
        gt_map  = {to_key(p): p for p in groundtruth_datalist}
        common_keys = sorted(set(gen_map.keys()) & set(gt_map.keys()))

        if limit_num is not None:
            common_keys = common_keys[:limit_num]

        self.datalist = [
            {
                "generated": gen_map[k],
                "groundtruth": gt_map[k],
            }
            for k in common_keys
        ]

    def __getitem__(self, index):
        while True:
            try:
                file_dict = self.datalist[index]
                waveform_dict = {}
                for key, filename in file_dict.items():
                    w = self.read_from_file(filename)
                    if w.size(-1) < 1:
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
        self.beat_score_calculator = BeatScoreCalculator()
        self.clap_score_calculator = CLAPScoreCalculator()

    def calculate_metrics(self, generate_files_path, groundtruth_path, same_name, limit_num=None, calculate_psnr_ssim=False, calculate_lsd=False, recalculate=False):
        torch.manual_seed(0)
        num_workers = 6

        dataloader = DataLoader(
            CorrespondedWaveDataset(
                generate_files_path,
                groundtruth_path,
                self.sampling_rate,
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
        fad_score = self.frechet.score(generate_files_path, groundtruth_path, limit_num=limit_num)
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
            generated = waveform_dict["generated"].squeeze(0)
            groundtruth = waveform_dict["groundtruth"].squeeze(0)
            cover_rate, hit_rate, f1, td = self.beat_score_calculator.calculate(groundtruth, generated, self.sampling_rate)
            clap_score = self.clap_score_calculator.calculate(generated, groundtruth)
            scores["beat_coverage_score"].append(cover_rate)
            scores["beat_hit_score"].append(hit_rate)
            scores["f1_score"].append(f1)
            scores["tempo_difference"].append(td)
            scores["clap_score"].append(clap_score)
        avg_scores = {k: sum(v) / len(v) for k, v in scores.items()}
        return avg_scores
