# 🎤 Live Transcriber

Real-time speech-to-text on Mac Apple Silicon, powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Runs entirely on-device.

Supports two audio sources:
- **Microphone** — your voice
- **System Audio** — Zoom meetings, YouTube, etc. (via BlackHole)

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [Homebrew](https://brew.sh)

## Setup

The quickest way to get started is using the automated installer script. It will automatically install Homebrew, Python, PortAudio, BlackHole (for system audio), and all required Python packages.

```bash
# Make the installer executable
chmod +x install.sh

# Run the installer
./install.sh
```

### System Audio Capture (Zoom, YouTube, etc.)

The installer automatically downloads **BlackHole 2ch**. To use it:

2. **Create Multi-Output Device** (so you can still hear audio):
   - Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
   - Click **+** → **Create Multi-Output Device**
   - Check both your **speakers/headphones** and **BlackHole 2ch**
   - Make sure your speakers are listed **first** (drag to reorder)

3. **Set the Multi-Output as your system output** before joining a Zoom call:
   - System Settings → Sound → Output → select your Multi-Output Device

## Usage

```bash
python3 transcriber.py
```

Select your microphone and (optionally) BlackHole for system audio. Transcriptions appear with `[MIC]` or `[SYS]` labels. Press **Ctrl+C** to stop.

A timestamped log file is saved automatically to the `transcripts/` folder.

## Configuration

Edit the top of `transcriber.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL` | `mlx-community/whisper-small-mlx` | Whisper model size |
| `CHUNK_DURATION` | `3` | Seconds per transcription chunk |
| `SILENCE_THRESHOLD` | `5` | RMS threshold for silence detection |
