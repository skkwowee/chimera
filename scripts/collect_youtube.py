#!/usr/bin/env python3
"""
Collect CS2 screenshots from YouTube gameplay videos.

Downloads videos at 1080p (preferred) or 720p and extracts frames at regular
intervals for training data. Higher resolution improves HUD element readability.

Usage:
    python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID"
    python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID" --interval 10
    python scripts/collect_youtube.py "https://youtube.com/watch?v=VIDEO_ID" --max-size 500
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.manifest import append_to_manifest

# Get ffmpeg from imageio_ffmpeg
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


def get_video_info(url: str) -> dict:
    """Get video metadata using yt-dlp."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True,
        text=True,
        check=True
    )
    import json
    return json.loads(result.stdout)


def download_video(url: str, output_path: Path, max_size_mb: int = 100) -> Path:
    """Download video with size limit."""
    print(f"Downloading video (max {max_size_mb}MB)...")

    # Prefer 1080p for high-quality training data
    # Fall back to 720p if 1080p unavailable
    # Avoid m3u8/HLS streams that often fail
    format_selector = (
        f"bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/"  # 1080p with audio merge
        f"best[height<=1080][ext=mp4]/"  # 1080p combined if available
        f"best[height<=720][ext=mp4]/"   # 720p fallback
        f"best[height<=1080]/"           # Any 1080p format
        f"best"                          # Last resort
    )

    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--ffmpeg-location", FFMPEG,
        "-o", str(output_path),
        "--no-playlist",
        "--no-warnings",
        url
    ]

    subprocess.run(cmd, check=True)
    return output_path


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval: float = 5.0,
    prefix: str = "frame"
) -> list[Path]:
    """Extract frames from video at regular intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration first
    probe_cmd = [
        FFMPEG, "-i", str(video_path),
        "-f", "null", "-"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)

    # Extract frames using ffmpeg
    # fps=1/interval means one frame every `interval` seconds
    output_pattern = str(output_dir / f"{prefix}_%04d.png")

    cmd = [
        FFMPEG,
        "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",  # High quality
        output_pattern,
        "-y"  # Overwrite existing
    ]

    print(f"Extracting frames every {interval} seconds...")
    subprocess.run(cmd, check=True, capture_output=True)

    # Find all extracted frames
    frames = sorted(output_dir.glob(f"{prefix}_*.png"))
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Collect CS2 screenshots from YouTube videos"
    )
    parser.add_argument(
        "url",
        help="YouTube video URL"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Output directory for frames (default: data/raw)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=5.0,
        help="Seconds between frame captures (default: 5.0)"
    )
    parser.add_argument(
        "--max-size", "-s",
        type=int,
        default=500,
        help="Maximum video download size in MB (default: 500, suitable for 1080p)"
    )
    parser.add_argument(
        "--prefix", "-p",
        default=None,
        help="Prefix for output files (default: video title)"
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep the downloaded video file"
    )

    args = parser.parse_args()

    # Get video info for naming
    print(f"Fetching video info...")
    try:
        info = get_video_info(args.url)
        video_title = info.get("title", "cs2_video")
        video_id = info.get("id", "unknown")
        duration = info.get("duration", 0)
        print(f"Video: {video_title}")
        print(f"Duration: {duration}s ({duration/60:.1f} min)")
    except Exception as e:
        print(f"Warning: Could not fetch video info: {e}")
        video_title = "cs2_video"
        video_id = "unknown"

    # Clean prefix for filenames
    prefix = args.prefix or video_id
    prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)

    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download to temp location or keep
    if args.keep_video:
        video_path = output_dir / f"{prefix}.mp4"
    else:
        temp_dir = tempfile.mkdtemp()
        video_path = Path(temp_dir) / f"{prefix}.mp4"

    try:
        # Download video
        download_video(args.url, video_path, args.max_size)

        if not video_path.exists():
            # yt-dlp might add extension
            possible = list(video_path.parent.glob(f"{prefix}.*"))
            if possible:
                video_path = possible[0]

        print(f"Downloaded: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f}MB)")

        # Extract frames
        frames = extract_frames(video_path, output_dir, args.interval, prefix)

        # Append manifest entries for each frame
        manifest_path = output_dir.parent / "manifest.jsonl"
        video_url = args.url
        for i, frame_path in enumerate(frames):
            entry = {
                "id": frame_path.stem,
                "source": "youtube",
                "video_id": video_id,
                "video_title": video_title,
                "video_url": video_url,
                "timestamp": i * args.interval,
            }
            append_to_manifest(manifest_path, entry)

        print(f"\nExtracted {len(frames)} frames to {output_dir}/")
        print(f"Manifest: {manifest_path}")
        if frames:
            print(f"First: {frames[0].name}")
            print(f"Last:  {frames[-1].name}")

    finally:
        # Cleanup temp video if not keeping
        if not args.keep_video and video_path.exists():
            video_path.unlink()
            print(f"Cleaned up temporary video file")


if __name__ == "__main__":
    main()
