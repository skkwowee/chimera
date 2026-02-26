#!/usr/bin/env python3
"""
Capture CS2 demo screenshots via netcon automation.

Connects to a running CS2 instance (launched with -netconport), loads a demo,
and iterates through a capture plan: seeking to each tick, switching POV, and
taking a JPEG screenshot.

Prerequisites:
    1. CS2 launched with: -netconport 2121 -console -novid
    2. Demo files copied to a Windows-accessible path (e.g. C:\\cs2demos\\)
    3. Run in CS2 console before starting:
         spec_mode 4; spec_autodirector 0; jpeg_quality 95

Usage:
    python scripts/capture_screenshots.py data/captures/furia-vs-vitality-m1-mirage/capture_plan.json
    python scripts/capture_screenshots.py data/captures/furia-vs-vitality-m1-mirage/capture_plan.json --resume
    python scripts/capture_screenshots.py data/captures/furia-vs-vitality-m1-mirage/capture_plan.json --limit 10
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.netcon import CS2Netcon

# Default timing (seconds). Conservative — decrease after testing.
SEEK_SETTLE = 1.0   # wait after demo_gototick for world to render
POV_SETTLE = 0.5    # wait after spec_player for camera transition
JPEG_SETTLE = 0.2   # wait after jpeg for file write

# CS2 setup commands to run at the start of each session
SETUP_COMMANDS = [
    "spec_mode 4",           # first-person spectator
    "spec_autodirector 0",   # disable auto-director
    "cl_drawhud 1",          # ensure HUD visible
    "jpeg_quality 95",       # high quality screenshots
    "hud_scaling 1",         # full size HUD
]


def find_screenshots(cs2_dir: Path, screenshot_id: str) -> list[Path]:
    """Find screenshot files matching a screenshot_id in the CS2 directory.

    CS2's jpeg command may produce files named:
    - {name}.jpg
    - {name}0000.jpg (with counter suffix)
    """
    matches = list(cs2_dir.glob(f"{screenshot_id}*.jpg"))
    return matches


def capture_plan(
    plan_path: Path,
    demo_dir: str,
    netcon_port: int,
    cs2_screenshot_dir: Path,
    output_dir: Path,
    resume: bool = False,
    limit: int = 0,
    seek_settle: float = SEEK_SETTLE,
    pov_settle: float = POV_SETTLE,
    jpeg_settle: float = JPEG_SETTLE,
):
    """Execute a capture plan against a running CS2 instance."""
    plan = json.loads(plan_path.read_text())
    demo_file = plan["demo_file"]
    demo_stem = plan["demo_stem"]
    captures = plan["captures"]

    # Sort by tick for efficient sequential seeking
    captures = sorted(captures, key=lambda c: (c["tick"], c["player_name"]))

    # Resume: skip already-captured screenshots
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        existing = {p.stem for p in raw_dir.glob("*.jpg")}
        before = len(captures)
        captures = [c for c in captures if c["screenshot_id"] not in existing]
        skipped = before - len(captures)
        if skipped:
            print(f"Resuming: skipping {skipped} already-captured screenshots")

    if limit > 0:
        captures = captures[:limit]

    if not captures:
        print("No captures to make.")
        return

    print(f"Capture plan: {len(captures)} screenshots from {demo_stem}")
    print(f"Demo file: {demo_dir}/{demo_file}")
    print(f"CS2 screenshots: {cs2_screenshot_dir}")
    print(f"Output: {raw_dir}")
    print(f"Timing: seek={seek_settle}s, pov={pov_settle}s, jpeg={jpeg_settle}s")
    per_capture = seek_settle + pov_settle + jpeg_settle
    est_minutes = len(captures) * per_capture / 60
    print(f"Estimated time: {est_minutes:.0f} minutes")
    print()

    con = CS2Netcon(port=netcon_port)
    con.connect()

    # Setup CS2 for spectating
    print("Configuring CS2...")
    con.exec_cfg(SETUP_COMMANDS)
    time.sleep(1)

    # Load the demo
    demo_path = f"{demo_dir}/{demo_file}" if demo_dir else demo_file
    print(f"Loading demo: {demo_path}")
    con.playdemo(demo_path, load_wait=15.0)
    con.pause()
    time.sleep(2)

    current_tick = -1
    captured = 0
    failed = 0

    try:
        for i, cap in enumerate(captures):
            tick = cap["tick"]
            player = cap["player_name"]
            ss_id = cap["screenshot_id"]

            # Seek to tick (skip if same as current)
            if tick != current_tick:
                con.goto_tick(tick, settle=seek_settle)
                current_tick = tick

            # Switch POV
            con.spec_player(player, settle=pov_settle)

            # Capture
            con.screenshot(ss_id, settle=jpeg_settle)

            # Move screenshot from CS2 dir to output
            matches = find_screenshots(cs2_screenshot_dir, ss_id)
            if matches:
                src = matches[0]
                dst = raw_dir / f"{ss_id}.jpg"
                shutil.move(str(src), str(dst))
                captured += 1
            else:
                failed += 1

            # Progress
            if (i + 1) % 50 == 0 or i == len(captures) - 1:
                pct = (i + 1) / len(captures) * 100
                print(f"  [{i+1}/{len(captures)}] {pct:.0f}% — "
                      f"tick={tick} {player} — "
                      f"{captured} ok, {failed} failed")

    except KeyboardInterrupt:
        print(f"\nInterrupted at capture {i+1}/{len(captures)}")
    finally:
        con.disconnect()

    print(f"\nDone: {captured} captured, {failed} failed, "
          f"{len(captures) - captured - failed} remaining")


def main():
    parser = argparse.ArgumentParser(
        description="Capture CS2 demo screenshots via netcon automation"
    )
    parser.add_argument(
        "plan",
        help="Path to capture_plan.json",
    )
    parser.add_argument(
        "--demo-dir",
        default="",
        help="Directory containing .dem files (Windows path, e.g. C:\\cs2demos). "
             "If empty, uses just the filename (CS2 searches its default paths).",
    )
    parser.add_argument(
        "--port", type=int, default=2121,
        help="CS2 netcon port (default: 2121)",
    )
    parser.add_argument(
        "--cs2-screenshot-dir",
        default=r"/mnt/c/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/screenshots",
        help="Path to CS2's screenshot output directory (WSL path)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output directory (default: same as capture plan directory)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-captured screenshots",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit to first N captures (for testing)",
    )
    parser.add_argument(
        "--seek-settle", type=float, default=SEEK_SETTLE,
        help=f"Seconds to wait after seeking (default: {SEEK_SETTLE})",
    )
    parser.add_argument(
        "--pov-settle", type=float, default=POV_SETTLE,
        help=f"Seconds to wait after POV switch (default: {POV_SETTLE})",
    )
    parser.add_argument(
        "--jpeg-settle", type=float, default=JPEG_SETTLE,
        help=f"Seconds to wait after screenshot (default: {JPEG_SETTLE})",
    )

    args = parser.parse_args()
    plan_path = Path(args.plan)

    if not plan_path.exists():
        print(f"Capture plan not found: {plan_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else plan_path.parent
    cs2_ss_dir = Path(args.cs2_screenshot_dir)

    capture_plan(
        plan_path=plan_path,
        demo_dir=args.demo_dir,
        netcon_port=args.port,
        cs2_screenshot_dir=cs2_ss_dir,
        output_dir=output_dir,
        resume=args.resume,
        limit=args.limit,
        seek_settle=args.seek_settle,
        pov_settle=args.pov_settle,
        jpeg_settle=args.jpeg_settle,
    )


if __name__ == "__main__":
    main()
