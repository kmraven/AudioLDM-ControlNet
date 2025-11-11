"""
Annotate tags based on the top-50 tags provided by the MTG-Jamendo dataset.
https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/tags/top50.txt

python tagging.py --root path/to/audio --splits train val test --tag-file ./top50.txt --out ./tags.csv
"""

import csv
import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from pathlib import Path

import laion_clap
import torch
import torchaudio
import tqdm


# -------------------------
# CLAP embedding calculator
# -------------------------
class CLAPEmbeddingCalculator:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.clapmodel = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        self.clapmodel.load_ckpt()
        self.clapmodel.model.audio_branch.requires_grad_(False)
        self.clapmodel.model.audio_branch.eval()
        self.clapmodel.model.text_branch.requires_grad_(False)
        self.clapmodel.model.text_branch.eval()

    @torch.no_grad()
    def audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: (time,) or (channels, time)
        returns: (1, D)
        """
        if audio.ndim == 2:
            audio = torch.mean(audio, dim=0)
        with torch.cuda.amp.autocast(enabled=False):
            embed = self.clapmodel.get_audio_embedding_from_data(
                audio.float()[None, :].to(self.device), use_tensor=True
            )
        # normalize for cosine similarity stability
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed

    @torch.no_grad()
    def text(self, text: str) -> torch.Tensor:
        """
        returns: (1, D)
        """
        with torch.cuda.amp.autocast(enabled=False):
            embed = self.clapmodel.get_text_embedding([text], use_tensor=True)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed


# -------------
# Tag annotator
# -------------
class TagAnnotator:
    """
    tag types in MTG top50: 'genre', 'instrument', 'mood/theme'
    """

    def __init__(self, tag_filepath: str):
        self.clap_calculator = CLAPEmbeddingCalculator()
        raw_tags = self.__load_tags(tag_filepath)
        self.tag_embeddings = self.__precalculate_embeddings_of_tags(raw_tags)

        self.topk_map = {"genre": 1, "mood/theme": 1, "instrument": 3}

    def __load_tags(self, filepath: str) -> dict[str, list[str]]:
        tag_dict = defaultdict(list)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "---" not in line:
                    continue
                tag_type, tag = line.split("---", 1)
                tag = tag.strip()
                tag_dict[tag_type].append(tag)
        return tag_dict

    @torch.no_grad()
    def __precalculate_embeddings_of_tags(
        self, tag_dict: dict[str, list[str]]
    ) -> dict[str, dict[str, torch.Tensor]]:
        tag_embeddings = defaultdict(dict)
        for tag_type, tags in tag_dict.items():
            for tag in tags:
                embedding = self.clap_calculator.text(tag)  # (1, D), normalized
                tag_embeddings[tag_type][tag] = embedding
        return tag_embeddings

    @torch.no_grad()
    def annotate_top(self, music: torch.Tensor) -> dict[str, list[str]]:
        """
        music: (C, T) or (T,)
        returns: {
            'genre': [top1],
            'mood/theme': [top1],
            'instrument': [top3]
        }
        """
        music_embedding = self.clap_calculator.audio(music)  # (1, D)

        result: dict[str, list[str]] = {}

        for tag_type, tag_embeds_dict in self.tag_embeddings.items():
            # stack tag embeddings
            tags = list(tag_embeds_dict.keys())
            embeds = torch.cat([tag_embeds_dict[t] for t in tags], dim=0)  # (N, D)
            # cosine similarity with normalized embeddings = dot product
            sims = (music_embedding @ embeds.T).squeeze(0)  # (N,)

            k = self.topk_map.get(tag_type, 1)
            k = min(k, sims.numel())
            top_vals, top_idx = torch.topk(sims, k)
            top_tags = [tags[i] for i in top_idx.tolist()]

            # 重複/順序の安定化（スコア順）
            result[tag_type] = top_tags

        # 必要なキーが無い場合のフォールバック（空配列を入れない）
        for need in ("genre", "mood/theme", "instrument"):
            result.setdefault(need, [])

        result["genre"] = result["genre"][: self.topk_map["genre"]]
        result["mood/theme"] = result["mood/theme"][: self.topk_map["mood/theme"]]
        result["instrument"] = result["instrument"][: self.topk_map["instrument"]]

        return result


def load_audio_tensor(path: Path):
    """
    Returns a float32 torch.Tensor shaped (channels, time).
    Uses torchaudio if available; otherwise falls back to soundfile.
    """
    if torchaudio is not None:
        wav, sr = torchaudio.load(str(path))  # (channels, time), float32/float64
        # Ensure float32
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        return wav
    else:
        # Fallback using soundfile
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True)  # (time, channels)
        wav = (
            torch.from_numpy(data).to(torch.float32).transpose(0, 1)
        )  # (channels, time)
        return wav


def iter_audio_files(audio_root: Path, splits):
    """
    Yield (split, path) for each .wav under audios/<split>/**.
    Deterministic ordering (sorted path strings).
    """
    for split in splits:
        base = audio_root / split
        if not base.exists():
            continue
        paths = sorted(base.rglob("*.wav"), key=lambda p: p.as_posix())
        for p in paths:
            yield split, p


def main():
    parser = argparse.ArgumentParser(
        description="Annotate tags for WAV files by scanning directories and reusing tagging.TagAnnotator."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing audios/ (and optionally keypoints/).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Target splits under audios/ (default: train val test).",
    )
    parser.add_argument(
        "--tag-file",
        type=str,
        default="./top50.txt",
        help="Path to the tag file (default: ./top50.txt).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tags_scanned.csv"),
        help="Output CSV path (default: tags_scanned.csv).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip IDs already present in the output CSV (if the file exists).",
    )

    args = parser.parse_args()

    # Instantiate annotator
    # Reuse TagAnnotator and CLAPEmbeddingCalculator as-is
    tag_annotator = TagAnnotator(args.tag_file)

    audio_root = args.root.resolve()

    # Prepare skip set if resuming
    seen_ids = set()
    if args.skip_existing and args.out.exists():
        with args.out.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = (row.get("id") or "").strip()
                if rid:
                    seen_ids.add(rid)

    # Prepare output CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists() or not args.skip_existing
    f_out = args.out.open(
        "a" if args.skip_existing and args.out.exists() else "w",
        encoding="utf-8",
        newline="",
    )
    writer = csv.writer(f_out)
    if write_header:
        writer.writerow(["id", "genre", "instrument", "mood/theme"])

    # Iterate audio files
    total = 0
    skipped = 0

    for split, wav_path in tqdm.tqdm(
        list(iter_audio_files(audio_root, args.splits)), desc="Annotating"
    ):
        file_stem = wav_path.stem  # ID is the stem, e.g., gBR_sBM_c01_...
        if args.skip_existing and file_stem in seen_ids:
            skipped += 1
            continue

        # Load audio
        audio = load_audio_tensor(wav_path)  # (C, T) float32

        # Annotate
        with torch.no_grad():
            tags = tag_annotator.annotate_top(audio)

        genre = ",".join(tags.get("genre", [])) if tags.get("genre") else ""
        mood = (tags.get("mood/theme", []) or [""])[0]
        instrument = (
            ",".join(tags.get("instrument", [])) if tags.get("instrument") else ""
        )

        writer.writerow([file_stem, genre, instrument, mood])
        total += 1

    f_out.close()
    print(f"Wrote {total} annotations to: {args.out}")
    if skipped:
        print(f"Skipped {skipped} already-present IDs (skip_existing).")


if __name__ == "__main__":
    main()
