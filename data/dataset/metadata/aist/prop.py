"""
 - python prop.py --root ../../aist --splits train --out ./aist_train.json --tags tags.csv
 - python prop.py --root ../../aist --splits val --out ./aist_val.json --tags tags.csv
 - python prop.py --root ../../aist --splits test --out ./aist_test.json --tags tags.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

audio_dir_name = "audio_clips"
keypoint_dir_name = "keypoints_clips"

class TagPrompter:
    def __init__(self, csv_path: Path):
        self.tag_map = self._load_csv(csv_path)

    @staticmethod
    def _split_list_field(text: str):
        """
        CSV field may be:
          - single token like 'electronic'
          - comma-joined list like 'piano,guitar,drums'
          - possibly quoted with spaces after commas
        Return a list of lowercase, trimmed tokens; empty list if missing.
        """
        if text is None:
            return []
        text = text.strip()
        if not text:
            return []
        # Split on commas; tolerate spaces
        parts = [p.strip() for p in text.split(",")]
        return [p for p in (s.lower() for s in parts) if p]

    def _load_csv(self, csv_path: Path):
        tag_map = {}
        if csv_path is None:
            return tag_map
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # Normalize header names to handle variants like 'mood/theme'
            fieldnames = {name.lower(): name for name in reader.fieldnames or []}
            for row in reader:
                rid = (row.get(fieldnames.get("id", "id")) or "").strip()
                if not rid:
                    continue
                genre_raw = row.get(fieldnames.get("genre", "genre"), "")
                inst_raw = row.get(fieldnames.get("instrument", "instrument"), "")
                mood_raw = row.get(fieldnames.get("mood/theme", "mood/theme"), "")

                genre = self._split_list_field(genre_raw)
                instrument = self._split_list_field(inst_raw)
                mood = (mood_raw or "").strip()

                record = {
                    "genre": genre,
                    "instrument": instrument,
                    "mood/theme": mood,
                }

                # Index by several keys to make lookup robust
                keys = [
                    rid,
                    os.path.basename(rid),
                    os.path.basename(rid).removesuffix(".pt").removesuffix(".wav"),
                ]
                for k in set(keys):
                    tag_map[k] = record
        return tag_map

    def _lookup_tags(self, id_like: str):
        cand = [
            str(id_like),
            os.path.basename(str(id_like)),
            os.path.basename(str(id_like)).removesuffix(".pt").removesuffix(".wav"),
        ]
        for k in cand:
            if k in self.tag_map:
                return self.tag_map[k]
        return {"genre": [], "mood/theme": "", "instrument": []}

    @staticmethod
    def _join_instrument(items):
        """
        ['piano'] -> 'piano'
        ['piano','guitar'] -> 'piano and guitar'
        ['piano','guitar','drums'] -> 'piano, guitar and drums'
        """
        n = len(items)
        if n == 0:
            return ""
        if n == 1:
            return items[0]
        if n == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])} and {items[-1]}"

    def make_prompt(self, id_like: str) -> str:
        tags = self._lookup_tags(id_like)
        mood = (tags.get("mood/theme", "") or "").strip().lower()
        genre = [s.lower() for s in (tags.get("genre") or []) if s]
        instrument = [s.lower() for s in (tags.get("instrument") or []) if s]

        genre = genre[:2]
        instrument = instrument[:3]

        inst_phrase = self._join_instrument(instrument)

        pieces = []
        if mood:
            pieces.append(mood)
        if genre:
            pieces += genre
        mg = " ".join(pieces).strip()

        if mg and inst_phrase:
            prompt = f"A {mg} track featuring {inst_phrase}."
        elif mg:
            prompt = f"A {mg} track."
        elif inst_phrase:
            prompt = f"A track featuring {inst_phrase}."
        else:
            prompt = "A music track."

        return prompt


def index_keypoints(key_root: Path, split: str) -> dict:
    """Index keypoints/<split> .pkl files by stem -> Path."""
    base = key_root / split
    idx = {}
    if not base.exists():
        return idx
    for p in base.rglob("*.pkl"):
        idx[p.stem] = p
    return idx


def collect_pairs_with_caption(audio_root: Path, key_root: Path, split: str, relative_to: Path, prompter: TagPrompter):
    """Match audio_clips/<split> .wav with keypoints_clips/<split> .pkl using file stem and build caption."""
    kp_idx = index_keypoints(key_root, split)
    items = []
    missing = []
    audio_base = audio_root / split
    if not audio_base.exists():
        return items, missing

    wavs = sorted(audio_base.rglob("*.wav"), key=lambda p: (p.parent.as_posix(), p.name))
    for wav in wavs:
        stem = wav.stem
        pkl = kp_idx.get(stem)
        if pkl is None:
            missing.append(wav)
            continue
        caption = prompter.make_prompt(stem)
        items.append({
            "wav": str(wav.relative_to(relative_to)),
            "motion": str(pkl.relative_to(relative_to)),
            "caption": caption,
        })
    return items, missing


def main():
    parser = argparse.ArgumentParser(
        description=f"Build dataset JSON pairing {audio_dir_name}/ and {keypoint_dir_name}/ and add a caption from tags CSV."
    )
    parser.add_argument("--root", type=Path, default=Path("."),
                        help=f"Project root containing {audio_dir_name}/ and {keypoint_dir_name}/")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"],
                        help="Target splits (default: train val test)")
    parser.add_argument("--tags", type=Path, required=True,
                        help="CSV file path for tags with columns: id, genre, instrument, mood/theme")
    parser.add_argument("--out", type=Path, default=Path("data.json"),
                        help="Output JSON path (default: data.json)")
    args = parser.parse_args()

    root = args.root.resolve()
    audio_root = root / audio_dir_name
    key_root = root / keypoint_dir_name

    prompter = TagPrompter(args.tags)

    all_data = []
    total_missing = defaultdict(int)

    for split in args.splits:
        items, missing = collect_pairs_with_caption(audio_root, key_root, split, relative_to=root, prompter=prompter)
        all_data.extend(items)
        total_missing[split] += len(missing)

    payload = {"data": all_data}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_data)} records to {args.out}")
    for split in args.splits:
        miss = total_missing.get(split, 0)
        if miss:
            print(f"[WARN] split={split}: {miss} wav(s) had no matching .pkl and were skipped.")
    if not all_data:
        print("[INFO] No pairs found. Check your directory structure, split names, and tags CSV.")


if __name__ == "__main__":
    main()
