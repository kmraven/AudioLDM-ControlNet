"""
Extract audio from AIST++ video files and organize in audio_clips directory.
This script processes all videos in video_clips/{train,val,test} and extracts
audio to audio_clips/{train,val,test} maintaining the same folder structure.
"""

import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def extract_audio(video_path, output_path):
    """
    Extract audio from a video file using ffmpeg.

    Args:
        video_path: Path to input video file
        output_path: Path to output audio file (WAV format)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use ffmpeg to extract audio
    # -i: input file
    # -vn: disable video
    # -acodec pcm_s16le: use PCM 16-bit little-endian audio codec
    # -ar 48000: set audio sampling rate to 48kHz (common for AIST++)
    # -ac 2: set audio channels to stereo
    # -y: overwrite output file if exists
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '48000',
        '-ac', '2',
        '-y',
        str(output_path)
    ]

    try:
        # Run ffmpeg with suppressed output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True, str(video_path)
    except subprocess.CalledProcessError as e:
        return False, f"Error processing {video_path}: {e.stderr.decode()}"


def process_single_video(args):
    """Wrapper for multiprocessing."""
    video_file, video_dir, audio_dir = args

    # Get relative path from video_clips directory
    rel_path = os.path.relpath(video_file, video_dir)

    # Create corresponding audio path
    audio_file = os.path.join(audio_dir, rel_path)
    audio_file = os.path.splitext(audio_file)[0] + '.wav'

    # Skip if already exists
    if os.path.exists(audio_file):
        return True, f"Skipped (exists): {audio_file}"

    return extract_audio(video_file, audio_file)


def main():
    # Base directories
    base_dir = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST"
    video_dir = os.path.join(base_dir, "video_clips")
    audio_dir = os.path.join(base_dir, "audio_clips")

    # Create audio_clips directory
    os.makedirs(audio_dir, exist_ok=True)

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg is not installed or not in PATH")
        print("Install ffmpeg: sudo apt-get install ffmpeg")
        return

    # Collect all video files
    video_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(video_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue

        # Find all video files (mp4, avi, etc.)
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))

    if not video_files:
        print("No video files found!")
        return

    print(f"Found {len(video_files)} video files")
    print(f"Video directory: {video_dir}")
    print(f"Audio directory: {audio_dir}")
    print("\nStarting audio extraction...")

    # Prepare arguments for multiprocessing
    args_list = [(vf, video_dir, audio_dir) for vf in video_files]

    # Process videos in parallel
    num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers
    print(f"Using {num_workers} parallel workers\n")

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, args_list),
            total=len(args_list),
            desc="Extracting audio"
        ))

    # Print summary
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"{'='*60}")

    # Print failures if any
    if failed > 0:
        print("\nFailed extractions:")
        for success, msg in results:
            if not success:
                print(f"  - {msg}")

    # Print directory structure
    print(f"\nAudio files saved to: {audio_dir}")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(audio_dir, split)
        if os.path.exists(split_path):
            count = sum(1 for _ in Path(split_path).rglob('*.wav'))
            print(f"  {split}: {count} files")


if __name__ == "__main__":
    main()
