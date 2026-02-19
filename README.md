# 🔇 Silence Cutter

Remove pauses and dead air from videos. GPU-accelerated with NVIDIA NVENC.

Built for content creators who talk fast but still end up with gaps.

## Requirements

- **Python 3.10+**
- **FFmpeg** (with your GPU drivers if using NVENC)
- **NVIDIA GPU** (optional but recommended — falls back to CPU)

```bash
# Check you have ffmpeg
ffmpeg -version

# That's it. No pip install, no dependencies. Just Python + FFmpeg.
```

## Usage

```bash
# Basic — auto-detects GPU, uses sensible defaults
python silence_cutter.py video.mp4

# Custom output name
python silence_cutter.py video.mp4 -o clean_video.mp4

# More aggressive — cuts quieter parts too
python silence_cutter.py video.mp4 --threshold -25dB

# Less aggressive — only cuts dead silence
python silence_cutter.py video.mp4 --threshold -35dB

# Cut shorter pauses (default is 0.5s minimum)
python silence_cutter.py video.mp4 --min-silence 0.3

# Preview what would be cut (no processing)
python silence_cutter.py video.mp4 --dry-run

# Force CPU encoding
python silence_cutter.py video.mp4 --no-gpu
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `input_cut.mp4` | Output filename |
| `--threshold` | `-30dB` | Silence threshold. Lower = only deep silence, higher = more aggressive |
| `--min-silence` | `0.5` | Minimum pause length to cut (seconds) |
| `--padding` | `0.12` | Breathing room at speech edges (seconds) |
| `--no-gpu` | off | Force CPU encoding |
| `--dry-run` | off | Preview cuts without processing |

## How It Works

1. **Detect** — FFmpeg's `silencedetect` finds all silent segments
2. **Segment** — Calculates which parts to keep (with configurable padding)
3. **Cut** — Stream-copies segments (fast, no quality loss)
4. **Concat** — Joins segments and re-encodes with GPU (NVENC) or CPU (libx264)

## Tips

- Start with `--dry-run` to preview before committing
- For talking-head videos: `--threshold -30dB --min-silence 0.4` works great
- For podcasts with background noise: try `--threshold -25dB`
- If cuts feel abrupt: increase `--padding` to 0.2 or 0.25
