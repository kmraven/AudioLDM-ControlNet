#!/usr/bin/env python3
"""Combine generated WAV files with their matching AIST++ videos."""

import argparse
import subprocess
from pathlib import Path

from tqdm import tqdm


def find_video_file(video_base_dir, filename_stem):
    for path in Path(video_base_dir).rglob("*.mp4"):
        if path.stem == filename_stem:
            return path
    return None


def mix_audio_video(audio_path, video_path, output_path):
    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-y",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"Error mixing {audio_path} and {video_path}: {error}")
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument(
        "--video-dir",
        default="data/dataset/aist/video_clips/test",
    )
    parser.add_argument("--output-dir", default="output_videos")
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    successful = failed = not_found = 0
    for audio_path in tqdm(audio_files, desc="Processing videos"):
        video_path = find_video_file(video_dir, audio_path.stem)
        if video_path is None:
            print(f"Video not found for: {audio_path.name}")
            not_found += 1
            continue
        output_path = output_dir / f"{audio_path.stem}_mixed.mp4"
        if mix_audio_video(audio_path, video_path, output_path):
            successful += 1
        else:
            failed += 1

    print(
        f"Processed {len(audio_files)} files: {successful} succeeded, "
        f"{failed} failed, {not_found} videos not found. Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
