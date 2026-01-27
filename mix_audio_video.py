#!/usr/bin/env python3
"""
Script to mix generated audio files with corresponding video files.
Matches audio and video by filename and combines them using ffmpeg.
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# Directory paths
AUDIO_DIR = "/home/sangheon/Desktop/AudioLDM-ControlNet/log/latent_diffusion_controlnet_beatdance/2025_11_23_dance_controlnet_beatdance/audioldm_original_medium_stretch/val_0_01-01-15:44_cfg_scale_3.5_ddim_200_n_cand_3"
VIDEO_BASE_DIR = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST/video_clips/test"
OUTPUT_DIR = "/home/sangheon/Desktop/AudioLDM-ControlNet/output_videos"

def find_video_file(video_base_dir, filename_stem):
    """
    Find the corresponding video file for a given audio filename stem.

    Args:
        video_base_dir: Base directory containing video files
        filename_stem: Stem of the audio filename (without extension)

    Returns:
        Path to the video file if found, None otherwise
    """
    # Search for the video file recursively
    for root, dirs, files in os.walk(video_base_dir):
        for file in files:
            if file.endswith('.mp4') and Path(file).stem == filename_stem:
                return os.path.join(root, file)
    return None


def mix_audio_video(audio_path, video_path, output_path):
    """
    Mix audio and video files using ffmpeg.

    Args:
        audio_path: Path to the audio file
        video_path: Path to the video file
        output_path: Path for the output video file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove audio from video and replace with new audio
        # -i: input files
        # -map 0:v: take video stream from first input
        # -map 1:a: take audio stream from second input
        # -c:v copy: copy video codec (no re-encoding)
        # -c:a aac: encode audio as AAC
        # -shortest: finish when shortest input ends
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-map', '0:v',
            '-map', '1:a',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-y',  # Overwrite output file if exists
            output_path
        ]

        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error mixing {audio_path} and {video_path}: {e}")
        return False


def main():
    """Main function to process all audio files and mix with videos."""

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all audio files
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])

    print(f"Found {len(audio_files)} audio files to process")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Video directory: {VIDEO_BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    successful = 0
    failed = 0
    not_found = 0

    # Process each audio file
    for audio_file in tqdm(audio_files, desc="Processing videos"):
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        filename_stem = Path(audio_file).stem

        # Find corresponding video file
        video_path = find_video_file(VIDEO_BASE_DIR, filename_stem)

        if video_path is None:
            print(f"Video not found for: {audio_file}")
            not_found += 1
            continue

        # Create output path
        output_filename = f"{filename_stem}_mixed.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Mix audio and video
        if mix_audio_video(audio_path, video_path, output_path):
            successful += 1
        else:
            failed += 1

    # Print summary
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total audio files: {len(audio_files)}")
    print(f"Successfully mixed: {successful}")
    print(f"Failed to mix: {failed}")
    print(f"Video not found: {not_found}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
