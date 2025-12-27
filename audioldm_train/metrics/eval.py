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

        # Helper function to collect all audio files, handling both flat and nested structures
        def collect_audio_files(base_path):
            audio_files = []
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    # If it's a directory, collect all .wav files inside
                    for audio_file in os.listdir(item_path):
                        if audio_file.endswith('.wav'):
                            audio_files.append(os.path.join(item_path, audio_file))
                elif item_path.endswith('.wav'):
                    # If it's a .wav file directly
                    audio_files.append(item_path)
            return sorted(audio_files)

        generated_datalist = collect_audio_files(generated_path)
        groundtruth_datalist = collect_audio_files(groundtruth_path)

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

    def _collect_audio_files(self, base_path):
        """Collect all .wav files from a directory, handling nested structures."""
        audio_files = []
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # If it's a directory, collect all .wav files inside
                for audio_file in os.listdir(item_path):
                    if audio_file.endswith('.wav'):
                        audio_files.append(os.path.join(item_path, audio_file))
            elif item_path.endswith('.wav'):
                # If it's a .wav file directly
                audio_files.append(item_path)
        return sorted(audio_files)

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
        try:
            import soundfile as sf
            import resampy

            # Collect audio files from both directories (handles nested structures)
            gen_files = self._collect_audio_files(generate_files_path)
            gt_files = self._collect_audio_files(groundtruth_path)

            if limit_num is not None:
                gen_files = gen_files[:limit_num]
                gt_files = gt_files[:limit_num]

            # Load audio files and prepare them as (audio, sr) tuples
            def load_audio_files(file_list):
                audio_list = []
                for fpath in tqdm(file_list, desc="Loading audio files"):
                    try:
                        wav_data, sr = sf.read(fpath, dtype="int16")
                        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

                        # Convert to mono
                        if len(wav_data.shape) > 1:
                            wav_data = wav_data.mean(axis=1)

                        # Resample to 16kHz for VGGish
                        if sr != 16000:
                            wav_data = resampy.resample(wav_data, sr, 16000)

                        audio_list.append((wav_data, 16000))
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")
                        continue
                return audio_list

            print(f"Calculating FAD: {len(gen_files)} generated files, {len(gt_files)} ground truth files")
            gen_audio_list = load_audio_files(gen_files)
            gt_audio_list = load_audio_files(gt_files)

            # Get embeddings
            gen_embd = self.frechet.get_embeddings(gen_audio_list, sr=16000)
            gt_embd = self.frechet.get_embeddings(gt_audio_list, sr=16000)

            # Calculate statistics and FAD score
            mu_gen, sigma_gen = self.frechet.calculate_embd_statistics(gen_embd)
            mu_gt, sigma_gt = self.frechet.calculate_embd_statistics(gt_embd)
            fad_score_value = self.frechet.calculate_frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)

            out['frechet_audio_distance'] = float(fad_score_value)
        except Exception as e:
            print(f"Error calculating FAD score: {e}")
            import traceback
            traceback.print_exc()
            out['frechet_audio_distance'] = float('nan')

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
