# Dance-to-Music Generation with Contrastive Pretraining

[![arXiv](https://img.shields.io/badge/arXiv-ARXIV_ID-b31b1b.svg)](https://arxiv.org/abs/ARXIV_ID)

Official research code for **“Dance to Music Generation leveraging Pre-training
with Unpaired data and Contrastive Alignment.”**

The system is trained in two stages:

1. MotionBERT and MERT features are aligned with beat-guided contrastive
   pretraining based on BeatDance.
2. The pretrained dance encoder conditions a frozen AudioLDM backbone through a
   ControlNet adapter.

## Environment

The reference environment uses Python 3.10 and CUDA. Create it with:

```bash
conda env create -f environment.yaml
conda activate audioldm_train
```

Feature extraction and model training require a CUDA-capable GPU. Install
`ffmpeg` separately if audio must be extracted from video files.

## Data and checkpoints

Prepare the AIST++ split used by the [Textual Inversion dance-to-music work](https://github.com/lsfhuihuiff/Dance-to-music_Siggraph_Asia_2024) ([goole drive link wrote at the repo](https://drive.google.com/drive/folders/1h02WhJtA4hzsVMeWinTqXYeoHN16waIz?usp=sharing)) and
place it under `data/dataset/aist/`. The expected inputs for preprocessing are:

```text
data/dataset/aist/
├── audio_clips/{train,val,test}/.../*.wav
└── keypoints_clips/{train,val,test}/.../*.pkl
```

Each keypoint pickle must contain a NumPy array with shape `[frames, 17, 3]` in
COCO joint order. Dataset metadata is stored under
`data/dataset/metadata/aist/`; update those JSON files if local filenames differ.

Download the following pretrained weights into `data/checkpoints/`:

```text
data/checkpoints/
├── audioldm-m-full.ckpt
├── vae_mel_16k_64bins.ckpt
├── clap_music_speech_audioset_epoch_15_esc_89.98.pt
└── latest_epoch.bin                         # MotionBERT
```

AudioLDM weights follow the upstream
[AudioLDM training repository](https://github.com/haoheliu/AudioLDM-training-finetuning),
and `latest_epoch.bin` is the pretrained
[MotionBERT](https://github.com/Walter0807/MotionBERT) checkpoint.

Paper checkpoints will be published on
[Google Drive (link placeholder)](https://drive.google.com/drive/folders/PLACEHOLDER).
The same folder will include test-set dance videos paired with generated music.
They should load normally for inference, evaluation, and weight-only fine-tuning.
Exact training resume, including optimizer state and global step, may be
incompatible with the refactored modules. If exact resume fails, use
`git checkout 0c1f4e3197a7c339586ce5769c6a4510dfbc6f30`.

## Stage 1: contrastive dance-encoder pretraining

Extract the 128-frame MERT, beat, and MotionBERT features:

```bash
python -m BeatDance.preprocess.extract_features
```

Input and output locations can be changed with `--audio-dir`,
`--keypoints-dir`, `--output-dir`, and `--motionbert-checkpoint`. See
`python -m BeatDance.preprocess.extract_features --help` for all options.

Train the BeatDance encoder used by the paper:

```bash
python BeatDance/aist_05_train_L110.py
```

The script uses 128 frames despite its historical filename. Copy the best
checkpoint to the location consumed by Stage 2:

```bash
cp log/beatdance/MotionBERT_pos/model_best.pth data/checkpoints/model_best.pth
```

## Stage 2: AudioLDM ControlNet training

The default wrapper trains the full model from the paper:

```bash
bash bash_train_controlnet_beatdance.sh
```

The wrapper accepts environment overrides without editing source files:

```bash
CUDA_VISIBLE_DEVICES=1 \
CONFIG=path/to/config.yaml \
CHECKPOINT=path/to/initial.ckpt \
bash bash_train_controlnet_beatdance.sh
```

Paper configurations are mapped as follows:

| Paper variant | Configuration |
| --- | --- |
| Full model | `audioldm_original_medium_stretch_pretrained_frozen.yaml` |
| w/o Contrastive Pretraining | `audioldm_original_medium.yaml` |
| w/o MotionBERT | `audioldm_original_medium_stretch_wo_mb.yaml` |
| AudioLDM default | `2025_11_08_dance_controlnet/audioldm_original_medium_stretch.yaml` |

The first three files are under
`audioldm_train/config/2025_11_23_dance_controlnet_beatdance/`.

## Evaluation

Generate the test-set outputs for the full model:

```bash
bash bash_eval_controlnet_beatdance.sh
```

Override `CONFIG`, `CHECKPOINT`, and `CUDA_VISIBLE_DEVICES` in the same way as
the training wrapper. To calculate audio-reference metrics separately:

```bash
python calculate_all_metrics.py \
  --generated path/to/generated_wavs \
  --groundtruth path/to/reference_wavs
```

Add `--motion path/to/keypoint_pickles` to include Beat Alignment Score (BAS).
Generated audio, reference audio, and motion files are matched by filename stem.

## Additional documentation

- [MotionBERT feature extraction](TEST_MOTIONBERT.md)
- [MotionBERT sequence-length handling](MOTIONBERT_MAXLEN_HANDLING.md)
- [Quick feature-extraction guide](QUICK_START.md)
- [BeatDance upstream documentation](BeatDance/README.md)

## Acknowledgements

This repository builds on AudioLDM, ControlNet, MotionBERT, MERT, and BeatDance.
Please cite the corresponding projects as well as this work when using the code.
