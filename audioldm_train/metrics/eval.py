import inspect
import os
import tempfile

import torch
from audioldm_eval import EvaluationHelper
from audioldm_eval.datasets.load_mel import WaveDataset
from audioldm_eval.metrics.fad import FrechetAudioDistance
from torch.utils.data import DataLoader
from tqdm import tqdm

from audioldm_train.metrics.beat_score import (
    calculate_audio_beat_score,
    calculate_beat_score,
)
from audioldm_train.metrics.clap import CLAPScoreCalculator
from audioldm_train.utilities.data.keypoints import (
    interp_nan_keypoints,
    load_keypoints,
    resample_keypoints_2d,
)


def _collect_files(base_path, suffix=None):
    files = []
    for root, _, filenames in os.walk(base_path):
        for filename in filenames:
            if suffix is None or filename.lower().endswith(suffix):
                files.append(os.path.join(root, filename))
    return sorted(files)


def _map_files_by_stem(paths, kind):
    result = {}
    for path in paths:
        key = os.path.splitext(os.path.basename(path))[0]
        if key in result:
            raise ValueError(f"Duplicate {kind} filename stem: {key}")
        result[key] = path
    return result


class CorrespondedWaveDataset(WaveDataset):
    def __init__(
        self,
        gen_music_path,
        gt_music_path,
        sampling_rate,
        gt_motion_path=None,
        limit_num=None,
    ):
        self.sr = sampling_rate

        gen_map = _map_files_by_stem(
            _collect_files(gen_music_path, suffix=".wav"), "generated audio"
        )
        gt_map = _map_files_by_stem(
            _collect_files(gt_music_path, suffix=".wav"), "ground-truth audio"
        )
        common_keys = set(gen_map) & set(gt_map)

        gt_motion_map = None
        if gt_motion_path is not None:
            gt_motion_map = _map_files_by_stem(
                _collect_files(gt_motion_path), "ground-truth motion"
            )
            common_keys &= set(gt_motion_map)

        common_keys = sorted(common_keys)
        if limit_num is not None:
            common_keys = common_keys[:limit_num]
        if not common_keys:
            raise ValueError("No matching generated, ground-truth, and motion files found")

        self.datalist = []
        for key in common_keys:
            item = {
                "gen_music": gen_map[key],
                "gt_music": gt_map[key],
            }
            if gt_motion_map is not None:
                item["gt_motion"] = gt_motion_map[key]
            self.datalist.append(item)

    def __getitem__(self, index):
        file_dict = self.datalist[index]
        waveform_dict = {}
        for key, filename in file_dict.items():
            if "music" in key:
                value = self.read_from_file(filename)
                if value.size(-1) < 1:
                    raise ValueError(f"Empty audio file: {filename}")
            else:
                value = load_keypoints(filename)
                value = interp_nan_keypoints(value)
                value = resample_keypoints_2d(value, T_target=round(5.12 * 60))
                value = value[:, :, :2]
                if value.shape[0] < 1:
                    raise ValueError(f"Empty motion file: {filename}")
            waveform_dict[key] = value

        return waveform_dict, os.path.basename(file_dict["gen_music"])


class ControlNetEvaluationHelper(EvaluationHelper):
    def __init__(self, sampling_rate):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sampling_rate = sampling_rate
        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )
        self.frechet.model = self.frechet.model.to(self.device)
        self.clap_score_calculator = CLAPScoreCalculator()

    def _calculate_fad(
        self, generate_files_path, groundtruth_path, limit_num, recalculate
    ):
        gen_files = _collect_files(generate_files_path, suffix=".wav")
        gt_files = _collect_files(groundtruth_path, suffix=".wav")
        if not gen_files or not gt_files:
            raise ValueError("FAD requires generated and ground-truth WAV files")

        gen_is_nested = any(
            os.path.dirname(path) != os.path.abspath(generate_files_path)
            for path in map(os.path.abspath, gen_files)
        )
        gt_is_nested = any(
            os.path.dirname(path) != os.path.abspath(groundtruth_path)
            for path in map(os.path.abspath, gt_files)
        )

        if gen_is_nested or gt_is_nested:
            with tempfile.TemporaryDirectory(prefix="audioldm-fad-") as temp_dir:
                flat_gen = os.path.join(temp_dir, "generated")
                flat_gt = os.path.join(temp_dir, "groundtruth")
                os.makedirs(flat_gen)
                os.makedirs(flat_gt)
                for source in gen_files:
                    os.symlink(
                        os.path.abspath(source),
                        os.path.join(flat_gen, os.path.basename(source)),
                    )
                for source in gt_files:
                    os.symlink(
                        os.path.abspath(source),
                        os.path.join(flat_gt, os.path.basename(source)),
                    )
                return self._score_fad(flat_gen, flat_gt, limit_num, recalculate)

        return self._score_fad(
            generate_files_path, groundtruth_path, limit_num, recalculate
        )

    def _score_fad(self, generate_files_path, groundtruth_path, limit_num, recalculate):
        kwargs = {"limit_num": limit_num}
        if "recalculate" in inspect.signature(self.frechet.score).parameters:
            kwargs["recalculate"] = recalculate
        score = self.frechet.score(
            generate_files_path,
            groundtruth_path,
            **kwargs,
        )
        if not isinstance(score, dict):
            raise RuntimeError(f"FAD calculation failed: {score}")
        return score

    def calculate_metrics(
        self,
        generate_files_path,
        groundtruth_path,
        same_name,
        limit_num=None,
        calculate_psnr_ssim=False,
        calculate_lsd=False,
        recalculate=False,
        gt_motion_path=None,
    ):
        del same_name, calculate_psnr_ssim, calculate_lsd
        torch.manual_seed(0)

        dataloader = DataLoader(
            CorrespondedWaveDataset(
                generate_files_path,
                groundtruth_path,
                self.sampling_rate,
                gt_motion_path=gt_motion_path,
                limit_num=limit_num,
            ),
            batch_size=1,
            sampler=None,
            num_workers=6,
        )

        if recalculate:
            print("Calculate FAD score from scratch")
        out = self._calculate_fad(
            generate_files_path, groundtruth_path, limit_num, recalculate
        )
        out.update(self.__calculate_beat_clap_score(dataloader))

        print("Evaluation Result:")
        print("\n".join(f"{key}: {value:.7f}" for key, value in out.items()))
        return out

    def __calculate_beat_clap_score(self, dataloader):
        scores = {
            "beat_coverage_score": [],
            "beat_hit_score": [],
            "f1_score": [],
            "tempo_difference": [],
            "clap_score": [],
        }
        include_alignment = "gt_motion" in dataloader.dataset.datalist[0]
        if include_alignment:
            scores["beat_alignment_score"] = []

        for waveform_dict, _ in tqdm(dataloader):
            gen_music = waveform_dict["gen_music"].squeeze(0)
            gt_music = waveform_dict["gt_music"].squeeze(0)
            if include_alignment:
                gt_motion = waveform_dict["gt_motion"].squeeze(0)
                bcs, bhs, f1, td, bas = calculate_beat_score(
                    gt_motion, gt_music, gen_music, self.sampling_rate
                )
                scores["beat_alignment_score"].append(bas)
            else:
                bcs, bhs, f1, td = calculate_audio_beat_score(
                    gt_music, gen_music, self.sampling_rate
                )

            clap_score = self.clap_score_calculator.calculate(gen_music, gt_music)
            scores["beat_coverage_score"].append(bcs)
            scores["beat_hit_score"].append(bhs)
            scores["f1_score"].append(f1)
            scores["tempo_difference"].append(td)
            scores["clap_score"].append(clap_score)

        return {key: sum(values) / len(values) for key, values in scores.items()}
