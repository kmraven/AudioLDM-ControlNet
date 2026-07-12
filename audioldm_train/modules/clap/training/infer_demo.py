"""Minimal CLAP text and audio inference example."""

import argparse

import librosa
import torch
from transformers import RobertaTokenizer

from audioldm_train.modules.clap.open_clip import create_model
from audioldm_train.modules.clap.training.data import (
    float32_to_int16,
    get_audio_features,
    int16_to_float32,
)


def tokenizer(text, tokenizer_model):
    result = tokenizer_model(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {key: value.squeeze(0) for key, value in result.items()}


def create_clap_model(pretrained_path, device):
    return create_model(
        "HTSAT-tiny",
        "roberta",
        pretrained_path,
        precision="fp32",
        device=device,
        enable_fusion=False,
        fusion_type="aff_2d",
    )


def infer_text(pretrained_path, device):
    model, _ = create_clap_model(pretrained_path, device)
    tokenizer_model = RobertaTokenizer.from_pretrained("roberta-base")
    text_data = tokenizer(
        ["I love contrastive learning", "I love the pretrained model"],
        tokenizer_model,
    )
    print(model.get_text_embedding(text_data).size())


def infer_audio(pretrained_path, audio_path, device):
    model, model_cfg = create_clap_model(pretrained_path, device)
    waveform, _ = librosa.load(audio_path, sr=48_000)
    waveform = torch.from_numpy(int16_to_float32(float32_to_int16(waveform))).float()
    audio_features = get_audio_features(
        {},
        waveform,
        480_000,
        data_truncating="fusion",
        data_filling="repeatpad",
        audio_cfg=model_cfg["audio_cfg"],
    )
    print(model.get_audio_embedding([audio_features]).size())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True, help="CLAP checkpoint path")
    parser.add_argument("--audio", help="Optional audio file for audio inference")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    infer_text(args.pretrained, args.device)
    if args.audio:
        infer_audio(args.pretrained, args.audio, args.device)


if __name__ == "__main__":
    main()
