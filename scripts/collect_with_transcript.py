#!/usr/bin/env python3
"""
Collect CS2 screenshots aligned to YouTube transcript timestamps.

Uses caster commentary from tournament broadcasts to identify meaningful moments
and extract frames with natural language descriptions for training data.

Usage:
    python scripts/collect_with_transcript.py "https://youtube.com/watch?v=VIDEO_ID"
    python scripts/collect_with_transcript.py URL --min-relevance 2
    python scripts/collect_with_transcript.py URL --keywords weapons,economy

TODO: Update claude_labeler.py to accept transcript context from the manifest.
      When labeling frames, pass the caster commentary as a hint to improve accuracy.
      See: src/labeling/claude_labeler.py
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


# Keywords indicating game-relevant commentary
KEYWORD_CATEGORIES = {
    "weapons": [
        "awp", "ak", "ak-47", "m4", "m4a1", "m4a4", "rifle", "pistol", "deagle",
        "desert eagle", "usp", "glock", "famas", "galil", "aug", "sg", "scout",
        "ssg", "smg", "mac-10", "mac10", "mp9", "mp7", "ump", "p90", "negev",
        "knife", "zeus", "taser"
    ],
    "utility": [
        "smoke", "flash", "flashbang", "molly", "molotov", "incendiary", "nade",
        "grenade", "he grenade", "frag", "utility", "blind", "flashed"
    ],
    "economy": [
        "eco", "save", "force", "force buy", "full buy", "half buy", "bonus",
        "loss bonus", "broke", "reset", "money", "economy", "invested", "saved"
    ],
    "situation": [
        "retake", "post-plant", "post plant", "clutch", "trade", "entry",
        "lurk", "rotate", "flank", "push", "execute", "default", "aggression",
        "peek", "hold", "anchor", "stack"
    ],
    "rounds": [
        "1v1", "1v2", "1v3", "1v4", "1v5",
        "2v1", "2v2", "2v3", "2v4", "2v5",
        "3v1", "3v2", "3v3", "3v4", "3v5",
        "4v1", "4v2", "4v3", "4v4", "4v5",
        "5v1", "5v2", "5v3", "5v4",
        "man advantage", "numbers", "even fight"
    ],
    "locations": [
        "a site", "b site", "a-site", "b-site", "mid", "middle", "connector",
        "palace", "apartments", "ramp", "squeaky", "main", "lobby", "heaven",
        "hell", "pit", "jungle", "stairs", "window", "door", "tunnel", "upper",
        "lower", "cat", "catwalk", "short", "long", "dust", "mirage", "inferno",
        "nuke", "overpass", "vertigo", "anubis", "ancient"
    ],
    "status": [
        "low hp", "low health", "tagged", "hit", "damage", "hurt", "one shot",
        "one-shot", "lit", "dinked", "headshot", "whiff", "miss", "alive",
        "dead", "down", "eliminated", "killed", "fragged"
    ],
    "bomb": [
        "bomb", "plant", "planted", "defuse", "defusing", "stick", "fake",
        "spike", "c4", "explosion", "time", "ticking"
    ],
}


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing information."""
    start: float  # seconds
    end: float    # seconds
    text: str
    relevance_score: int = 0
    matched_categories: list = None

    def __post_init__(self):
        if self.matched_categories is None:
            self.matched_categories = []

    @property
    def midpoint(self) -> float:
        """Get the midpoint timestamp for frame extraction."""
        return (self.start + self.end) / 2

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_vtt_timestamp(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


def parse_vtt(vtt_path: Path) -> list[TranscriptSegment]:
    """Parse VTT subtitle file into segments."""
    content = vtt_path.read_text(encoding="utf-8")
    segments = []

    # VTT pattern: timestamp --> timestamp\ntext
    pattern = r"(\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3})\s*\n((?:(?!\d{1,2}:\d{2}:\d{2}\.\d{3}).+\n?)+)"

    for match in re.finditer(pattern, content):
        start = parse_vtt_timestamp(match.group(1))
        end = parse_vtt_timestamp(match.group(2))
        text = match.group(3).strip()

        # Clean up VTT formatting tags
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            segments.append(TranscriptSegment(start=start, end=end, text=text))

    return segments


def score_relevance(segment: TranscriptSegment, categories: list[str] = None) -> TranscriptSegment:
    """Score how game-relevant a transcript segment is."""
    text_lower = segment.text.lower()
    score = 0
    matched = []

    categories_to_check = categories or list(KEYWORD_CATEGORIES.keys())

    for category in categories_to_check:
        if category not in KEYWORD_CATEGORIES:
            continue
        for keyword in KEYWORD_CATEGORIES[category]:
            if keyword in text_lower:
                score += 1
                if category not in matched:
                    matched.append(category)
                break  # Only count each category once

    segment.relevance_score = score
    segment.matched_categories = matched
    return segment


def merge_segments(
    segments: list[TranscriptSegment],
    max_gap: float = 2.0,
    max_duration: float = 15.0
) -> list[TranscriptSegment]:
    """Merge consecutive segments that are close together."""
    if not segments:
        return []

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        gap = next_seg.start - current.end
        combined_duration = next_seg.end - current.start

        if gap <= max_gap and combined_duration <= max_duration:
            # Merge segments
            current = TranscriptSegment(
                start=current.start,
                end=next_seg.end,
                text=current.text + " " + next_seg.text,
                relevance_score=max(current.relevance_score, next_seg.relevance_score),
                matched_categories=list(set(current.matched_categories + next_seg.matched_categories))
            )
        else:
            merged.append(current)
            current = next_seg

    merged.append(current)
    return merged


def deduplicate_segments(
    segments: list[TranscriptSegment],
    min_time_gap: float = 3.0
) -> list[TranscriptSegment]:
    """Remove segments that are too close together, keeping highest relevance."""
    if not segments:
        return []

    # Sort by timestamp
    sorted_segs = sorted(segments, key=lambda s: s.start)
    result = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        if seg.start - result[-1].midpoint >= min_time_gap:
            result.append(seg)
        elif seg.relevance_score > result[-1].relevance_score:
            # Replace with higher relevance segment
            result[-1] = seg

    return result


def download_video_and_transcript(
    url: str,
    output_dir: Path,
    video_id: str
) -> tuple[Path, Path]:
    """Download video and auto-generated transcript."""
    video_path = output_dir / f"{video_id}.mp4"

    # Download video with transcript
    format_selector = (
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
        "best[height<=1080][ext=mp4]/"
        "best[height<=720][ext=mp4]/"
        "best"
    )

    cmd = [
        "yt-dlp",
        "-f", format_selector,
        "--ffmpeg-location", FFMPEG,
        "--write-auto-subs",
        "--sub-lang", "en",
        "-o", str(output_dir / f"{video_id}.%(ext)s"),
        "--no-playlist",
        url
    ]

    print("Downloading video and transcript...")
    subprocess.run(cmd, check=True)

    # Find the transcript file (could be .en.vtt or .en.vtt3)
    vtt_candidates = list(output_dir.glob(f"{video_id}*.vtt"))
    if not vtt_candidates:
        raise FileNotFoundError(f"No transcript found for {video_id}")

    vtt_path = vtt_candidates[0]

    # Find video file
    video_candidates = list(output_dir.glob(f"{video_id}.mp4"))
    if not video_candidates:
        video_candidates = list(output_dir.glob(f"{video_id}.*"))
        video_candidates = [v for v in video_candidates if v.suffix in (".mp4", ".mkv", ".webm")]

    if not video_candidates:
        raise FileNotFoundError(f"No video found for {video_id}")

    video_path = video_candidates[0]

    return video_path, vtt_path


def extract_frame_at_timestamp(
    video_path: Path,
    timestamp: float,
    output_path: Path
) -> bool:
    """Extract a single frame at a specific timestamp."""
    cmd = [
        FFMPEG,
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
        "-y"
    ]

    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and output_path.exists()


def get_video_info(url: str) -> dict:
    """Get video metadata using yt-dlp."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Collect CS2 screenshots aligned to transcript timestamps"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--min-relevance", "-r",
        type=int,
        default=1,
        help="Minimum relevance score to extract frame (default: 1)"
    )
    parser.add_argument(
        "--min-gap", "-g",
        type=float,
        default=5.0,
        help="Minimum seconds between extracted frames (default: 5.0)"
    )
    parser.add_argument(
        "--categories", "-c",
        default=None,
        help="Comma-separated keyword categories to use (default: all)"
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep the downloaded video file"
    )
    parser.add_argument(
        "--keep-transcript",
        action="store_true",
        help="Keep the transcript VTT file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse transcript and show matches without downloading video"
    )

    args = parser.parse_args()

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Get video info
    print("Fetching video info...")
    try:
        info = get_video_info(args.url)
        video_id = info.get("id", "unknown")
        video_title = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        print(f"Video: {video_title}")
        print(f"Duration: {duration}s ({duration/60:.1f} min)")
    except Exception as e:
        print(f"Error fetching video info: {e}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For dry run, just download transcript
    if args.dry_run:
        print("\nDry run: downloading transcript only...")
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--skip-download",
            "-o", str(output_dir / f"{video_id}.%(ext)s"),
            args.url
        ]
        subprocess.run(cmd, check=True)
        vtt_candidates = list(output_dir.glob(f"{video_id}*.vtt"))
        if not vtt_candidates:
            print("No English transcript available for this video")
            sys.exit(1)
        vtt_path = vtt_candidates[0]
        video_path = None
    else:
        video_path, vtt_path = download_video_and_transcript(args.url, output_dir, video_id)
        print(f"Video: {video_path}")
        print(f"Transcript: {vtt_path}")

    # Parse and process transcript
    print("\nParsing transcript...")
    segments = parse_vtt(vtt_path)
    print(f"Found {len(segments)} raw segments")

    # Score relevance
    segments = [score_relevance(s, categories) for s in segments]

    # Filter by relevance
    relevant = [s for s in segments if s.relevance_score >= args.min_relevance]
    print(f"Found {len(relevant)} relevant segments (score >= {args.min_relevance})")

    # Merge close segments
    merged = merge_segments(relevant)
    print(f"Merged into {len(merged)} segments")

    # Deduplicate
    final = deduplicate_segments(merged, min_time_gap=args.min_gap)
    print(f"Final: {len(final)} segments after deduplication (min gap: {args.min_gap}s)")

    if args.dry_run:
        print("\n=== Matched Segments ===")
        for i, seg in enumerate(final[:20]):  # Show first 20
            ts = f"{int(seg.midpoint//60)}:{seg.midpoint%60:05.2f}"
            cats = ", ".join(seg.matched_categories)
            print(f"\n[{ts}] (score: {seg.relevance_score}, {cats})")
            print(f"  {seg.text[:100]}...")
        if len(final) > 20:
            print(f"\n... and {len(final) - 20} more segments")

        # Cleanup transcript if not keeping
        if not args.keep_transcript:
            vtt_path.unlink()
        sys.exit(0)

    # Extract frames
    print("\nExtracting frames...")
    manifest = []

    for i, seg in enumerate(final):
        frame_name = f"{video_id}_{i:04d}.png"
        frame_path = output_dir / frame_name

        success = extract_frame_at_timestamp(video_path, seg.midpoint, frame_path)

        if success:
            manifest.append({
                "frame": frame_name,
                "timestamp": seg.midpoint,
                "start": seg.start,
                "end": seg.end,
                "transcript": seg.text,
                "relevance_score": seg.relevance_score,
                "categories": seg.matched_categories
            })
            print(f"  [{i+1}/{len(final)}] {frame_name} @ {seg.midpoint:.1f}s")
        else:
            print(f"  [{i+1}/{len(final)}] FAILED: {frame_name}")

    # Save manifest
    manifest_path = output_dir / f"{video_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "video_id": video_id,
            "video_title": video_title,
            "video_url": args.url,
            "duration": duration,
            "min_relevance": args.min_relevance,
            "min_gap": args.min_gap,
            "categories": categories,
            "frames": manifest
        }, f, indent=2)

    print(f"\nExtracted {len(manifest)} frames")
    print(f"Manifest: {manifest_path}")

    # Cleanup
    if not args.keep_video and video_path:
        video_path.unlink()
        print("Cleaned up video file")
    if not args.keep_transcript:
        vtt_path.unlink()
        print("Cleaned up transcript file")


if __name__ == "__main__":
    main()
