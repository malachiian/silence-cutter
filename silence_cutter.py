#!/usr/bin/env python3
"""
Silence Cutter — Remove pauses from video using FFmpeg + NVIDIA GPU encoding.

Usage:
    python silence_cutter.py input.mp4
    python silence_cutter.py input.mp4 -o output.mp4
    python silence_cutter.py input.mp4 --threshold -30dB --min-silence 0.5 --padding 0.1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

# OpenClaw gateway for phone-home reports
OPENCLAW_GATEWAY = "http://127.0.0.1:18789"


def detect_silences(input_file: str, threshold: str = "-30dB", min_duration: float = 0.5) -> list[dict]:
    """Use ffmpeg silencedetect to find silent segments."""
    cmd = [
        "ffmpeg", "-i", input_file,
        "-af", f"silencedetect=noise={threshold}:d={min_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences = []
    current = {}
    for line in stderr.split("\n"):
        if "silence_start:" in line:
            try:
                current["start"] = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "silence_end:" in line:
            try:
                parts = line.split("silence_end:")[1].strip().split("|")
                current["end"] = float(parts[0].strip().split()[0])
                if "start" in current:
                    silences.append(current)
                current = {}
            except (ValueError, IndexError):
                current = {}

    return silences


def get_duration(input_file: str) -> float:
    """Get video duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def has_nvenc() -> bool:
    """Check if NVIDIA NVENC is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def build_segments(duration: float, silences: list[dict], padding: float = 0.1) -> list[tuple[float, float]]:
    """Convert silence list into keep-segments (the non-silent parts)."""
    segments = []
    pos = 0.0

    for s in silences:
        seg_start = pos
        seg_end = s["start"] + padding  # keep a tiny bit into the silence for natural feel
        if seg_end > seg_start + 0.05:  # skip tiny segments
            segments.append((seg_start, min(seg_end, duration)))
        pos = max(s["end"] - padding, 0)  # start next segment slightly before speech resumes

    # Final segment
    if pos < duration:
        segments.append((pos, duration))

    return segments


def cut_and_concat(input_file: str, output_file: str, segments: list[tuple[float, float]], use_gpu: bool = True):
    """Cut segments and concatenate using ffmpeg. GPU-accelerated encoding if available."""
    if not segments:
        print("No segments to keep — entire video is silent?")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = []
        total = len(segments)

        print(f"\n✂️  Cutting {total} segments...")
        for i, (start, end) in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.ts")
            duration = end - start

            # Use stream copy for cutting (fast, no re-encode per segment)
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", input_file,
                "-t", str(duration),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                seg_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            segment_files.append(seg_path)

            pct = (i + 1) / total * 50  # first 50% is cutting
            print(f"\r  Progress: [{'█' * int(pct/2)}{'░' * (25 - int(pct/2))}] {pct:.0f}%  ({i+1}/{total} segments)", end="", flush=True)

        # Build concat file
        concat_path = os.path.join(tmpdir, "concat.txt")
        with open(concat_path, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        print(f"\n\n🔗 Concatenating and encoding...")

        # Concat + re-encode for clean output
        if use_gpu:
            encode_args = [
                "-c:v", "h264_nvenc",
                "-preset", "p4",        # balanced speed/quality
                "-cq", "20",            # constant quality (lower = better, 18-23 is great)
                "-b:v", "0",
                "-c:a", "aac", "-b:a", "192k"
            ]
            print("  🚀 Using NVIDIA GPU encoding (h264_nvenc)")
        else:
            encode_args = [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-c:a", "aac", "-b:a", "192k"
            ]
            print("  🖥️  Using CPU encoding (libx264)")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            *encode_args,
            "-movflags", "+faststart",
            output_file
        ]
        subprocess.run(cmd, capture_output=True, check=True)

    print(f"  ✅ Done!")


def send_report(gateway_url: str, gateway_token: str, report: dict):
    """Send a run report to OpenClaw gateway as a wake event."""
    status = "✅ SUCCESS" if report.get("success") else "❌ FAILED"
    lines = [
        f"🔇 **Silence Cutter Report** — {status}",
        f"**File:** `{report.get('input', '?')}`",
    ]
    if report.get("success"):
        lines.extend([
            f"**Original:** {report.get('original_duration', '?')} ({report.get('input_size', '?')} MB)",
            f"**Output:** {report.get('output_duration', '?')} ({report.get('output_size', '?')} MB)",
            f"**Removed:** {report.get('silence_removed', '?')} ({report.get('silence_pct', '?')}%)",
            f"**Silences found:** {report.get('silence_count', 0)}",
            f"**Segments kept:** {report.get('segment_count', 0)}",
            f"**Encoding:** {report.get('encoder', 'unknown')}",
            f"**Processing time:** {report.get('elapsed', '?')}",
            f"**Settings:** threshold={report.get('threshold', '?')}, min_silence={report.get('min_silence', '?')}s, padding={report.get('padding', '?')}s",
            f"**Output file:** `{report.get('output', '?')}`",
        ])
    else:
        lines.append(f"**Error:** {report.get('error', 'Unknown error')}")

    text = "\n".join(lines)

    try:
        payload = json.dumps({"text": text, "mode": "now"}).encode()
        req = urllib.request.Request(
            f"{gateway_url}/api/cron/wake",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {gateway_token}"
            },
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
        print("📡 Report sent to Daemon")
    except Exception as e:
        print(f"⚠️  Could not send report: {e}")


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def main():
    parser = argparse.ArgumentParser(
        description="🔇 Silence Cutter — Remove pauses from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                          # Auto-detect settings
  %(prog)s video.mp4 -o clean.mp4             # Custom output name
  %(prog)s video.mp4 --threshold -25dB        # More aggressive (cuts quieter speech)
  %(prog)s video.mp4 --threshold -35dB        # Less aggressive (only deep silence)
  %(prog)s video.mp4 --min-silence 0.3        # Cut shorter pauses too
  %(prog)s video.mp4 --padding 0.2            # More breathing room at cuts
  %(prog)s video.mp4 --no-gpu                 # Force CPU encoding
        """
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output file (default: input_cut.mp4)")
    parser.add_argument("--threshold", default="-30dB",
                        help="Silence threshold (default: -30dB). Lower = only deep silence, higher = more aggressive")
    parser.add_argument("--min-silence", type=float, default=0.5,
                        help="Minimum silence duration to cut, in seconds (default: 0.5)")
    parser.add_argument("--padding", type=float, default=0.12,
                        help="Seconds to keep at edges of speech (default: 0.12)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU encoding")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cut without processing")
    parser.add_argument("--report", action="store_true", help="Send run report to Daemon via OpenClaw")
    parser.add_argument("--gateway-token", default=os.environ.get("OPENCLAW_TOKEN", ""),
                        help="OpenClaw gateway token (or set OPENCLAW_TOKEN env var)")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"❌ File not found: {args.input}")
        sys.exit(1)

    if not args.output:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + "_cut"))

    use_gpu = not args.no_gpu and has_nvenc()

    print(f"🎬 Silence Cutter")
    print(f"   Input:     {args.input}")
    print(f"   Output:    {args.output}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Min pause: {args.min_silence}s")
    print(f"   Padding:   {args.padding}s")
    print(f"   GPU:       {'✅ NVENC' if use_gpu else '❌ CPU'}")

    start_time = time.time()

    # Step 1: Get duration
    duration = get_duration(args.input)
    print(f"\n📏 Video duration: {format_time(duration)}")

    # Step 2: Detect silences
    print(f"\n🔍 Detecting silences...")
    silences = detect_silences(args.input, args.threshold, args.min_silence)
    total_silence = sum(s["end"] - s["start"] for s in silences)
    print(f"   Found {len(silences)} silent segments ({format_time(total_silence)} total)")
    print(f"   Will remove {total_silence/duration*100:.1f}% of video")

    if not silences:
        print("\n✨ No silence detected — nothing to cut!")
        sys.exit(0)

    # Step 3: Build segments
    segments = build_segments(duration, silences, args.padding)
    kept_duration = sum(end - start for start, end in segments)
    print(f"   Keeping {len(segments)} segments ({format_time(kept_duration)})")

    if args.dry_run:
        print("\n📋 Segments to keep:")
        for i, (start, end) in enumerate(segments):
            print(f"   {i+1:3d}. {start:.2f}s → {end:.2f}s ({end-start:.2f}s)")
        print(f"\n⏱️  Would save {format_time(total_silence)} ({total_silence/duration*100:.1f}%)")
        return

    # Step 4: Cut and concat
    report = {"input": args.input, "output": args.output, "threshold": args.threshold,
              "min_silence": args.min_silence, "padding": args.padding}
    try:
        cut_and_concat(args.input, args.output, segments, use_gpu)

        elapsed = time.time() - start_time
        output_size = os.path.getsize(args.output) / (1024 * 1024)
        input_size = os.path.getsize(args.input) / (1024 * 1024)

        print(f"\n📊 Summary:")
        print(f"   Original:  {format_time(duration)} ({input_size:.1f} MB)")
        print(f"   Output:    {format_time(kept_duration)} ({output_size:.1f} MB)")
        print(f"   Removed:   {format_time(total_silence)} ({total_silence/duration*100:.1f}%)")
        print(f"   Processed in {elapsed:.1f}s")
        print(f"\n🎉 Saved to: {args.output}")

        report.update({
            "success": True,
            "original_duration": format_time(duration),
            "output_duration": format_time(kept_duration),
            "input_size": f"{input_size:.1f}",
            "output_size": f"{output_size:.1f}",
            "silence_removed": format_time(total_silence),
            "silence_pct": f"{total_silence/duration*100:.1f}",
            "silence_count": len(silences),
            "segment_count": len(segments),
            "encoder": "h264_nvenc (GPU)" if use_gpu else "libx264 (CPU)",
            "elapsed": f"{elapsed:.1f}s",
        })
    except Exception as e:
        report.update({"success": False, "error": f"{e}\n{traceback.format_exc()}"})
        print(f"\n❌ Error: {e}")

    if args.report:
        report_path = Path(args.output).with_suffix(".report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📋 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
