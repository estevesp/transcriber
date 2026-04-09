"""
Simple Live Microphone + System Audio Transcriber — Mac Apple Silicon
Captures mic and system audio (via BlackHole) and transcribes in real time.
"""

import argparse
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pyaudio

import lameenc
import mlx_whisper

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL = "mlx-community/whisper-small-mlx"  # HuggingFace repo — downloaded on first run
SAMPLE_RATE = 16_000                       # Whisper expects 16 kHz
CHUNK_DURATION = 3                         # Seconds per chunk
SILENCE_THRESHOLD = 20                     # RMS below this = silence (skip)

# ── ANSI colors ────────────────────────────────────────────────────────────────

CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ── Helpers ────────────────────────────────────────────────────────────────────

def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio chunk."""
    # Convert to float32 to prevent int16 overflow when squaring
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))


def is_silent(audio: np.ndarray) -> bool:
    """Check if an audio chunk is below the silence threshold."""
    return rms(audio) < SILENCE_THRESHOLD


def get_meter_string(rms_value: float, threshold: float, width: int = 40) -> str:
    """Generate a visual meter string for the live terminal UI."""
    max_log = 4.0 # up to 10000 RMS
    
    val_log = np.log10(max(1.0, rms_value))
    thresh_log = np.log10(max(1.0, threshold))
    
    val_pos = int((val_log / max_log) * width)
    val_pos = min(width, max(0, val_pos))
    
    thresh_pos = int((thresh_log / max_log) * width)
    thresh_pos = min(width - 1, max(0, thresh_pos))
    
    out = []
    for i in range(width):
        if i == thresh_pos:
            out.append(f"{RED}│{RESET}")
        elif i < val_pos:
            out.append(f"{GREEN}█{RESET}")
        else:
            out.append(f"{DIM}─{RESET}")
            
    return "".join(out)


# Known Whisper hallucination phrases
HALLUCINATION_PHRASES = [
    "thank you", "thanks for watching", "subscribe", "like and subscribe",
    "see you next time", "bye bye", "goodbye", "you", "ʕ ʔ ʔ",
    "well that was good", "well that was good.", "well that was good context", 
    "well that is what we are going to do with that", "well that is what we are going to do with that.",
    "well that is what we are living with right now", "well that is what we are living with right now."
]

def is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucinations: repetitive or known junk phrases."""
    t = text.lower().strip()
    t_clean = t.rstrip(".!?")
    
    # Check known phrases
    if t_clean in HALLUCINATION_PHRASES:
        return True
        
    # Check for excessive consecutive substring repetition (e.g., "noinoinoi...", "w w w w ")
    # Matches any 1 to 15 character sequence that repeats at least 4 times in a row.
    if re.search(r'(.{1,15}?)\1{4,}', t):
        return True
        
    # Check for excessive word-level repetition
    t_words = re.sub(r'[^\w\s]', '', t)
    words = t_words.split()
    if len(words) >= 3:
        max_n = min(len(words) // 2, 10)
        # Check from the start of the string
        for n in range(1, max_n + 1):
            chunk = " ".join(words[:n])
            if not chunk: continue
            count = t_words.count(chunk)
            if count >= 3 and len(chunk) * count >= len(t_words) * 0.5:
                return True
            # For 2 repetitions, it must dominate 80%+ of the text
            if count >= 2 and len(chunk) * count >= len(t_words) * 0.8:
                return True
                
        # Check from the end of the string (in case hallucination loop starts at the end)
        words_rev = words[::-1]
        for n in range(1, max_n + 1):
            chunk = " ".join(words_rev[:n][::-1])
            if not chunk: continue
            count = t_words.count(chunk)
            if count >= 3 and len(chunk) * count >= len(t_words) * 0.5:
                return True
                
    return False


def save_audio_mp3(audio_buffers: Dict[str, list], filepath: str) -> None:
    """Mix all recorded audio sources and encode to MP3."""
    mixed = None
    for chunks in audio_buffers.values():
        if not chunks:
            continue
        source_audio = np.concatenate(chunks).astype(np.float32)
        if mixed is None:
            mixed = source_audio
        else:
            # Pad shorter array to match longer
            if len(mixed) < len(source_audio):
                mixed = np.pad(mixed, (0, len(source_audio) - len(mixed)))
            elif len(source_audio) < len(mixed):
                source_audio = np.pad(source_audio, (0, len(mixed) - len(source_audio)))
            mixed = mixed + source_audio

    if mixed is None or len(mixed) == 0:
        return

    # Normalize to int16 range instead of hard-clipping
    peak = np.max(np.abs(mixed))
    if peak > 32767:
        mixed = mixed * (32767.0 / peak)
    mixed = mixed.astype(np.int16)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)

    mp3_data = encoder.encode(mixed.tobytes())
    mp3_data += encoder.flush()

    with open(filepath, "wb") as f:
        f.write(mp3_data)


def timestamp() -> str:
    """Current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def list_input_devices(pa: pyaudio.PyAudio) -> List[Dict]:
    """Return a list of available input devices."""
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({"index": i, "name": info["name"], "channels": info["maxInputChannels"]})
    return devices


def pick_device(pa: pyaudio.PyAudio, devices: List[Dict], prompt: str) -> Optional[int]:
    """Prompt user to select a device. Returns device index or None to skip."""
    while True:
        try:
            choice = input(prompt).strip()
            if choice.lower() == "s":
                return None
            idx = int(choice)
            if 0 <= idx < len(devices):
                selected = devices[idx]
                print(f"  {DIM}→ {selected['name']}{RESET}")
                return selected["index"]
            else:
                print(f"  Please enter 0-{len(devices) - 1}, or 's' to skip.")
        except ValueError:
            print("  Please enter a valid number, or 's' to skip.")
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Cancelled.{RESET}")
            pa.terminate()
            sys.exit(0)


def find_blackhole(devices: List[Dict]) -> Optional[int]:
    """Find BlackHole in the device list, return its list index or None."""
    for i, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower():
            return i
    return None


# ── Zoom Meeting Detection ────────────────────────────────────────────────────

def is_zoom_meeting_active() -> bool:
    """Detect if a Zoom meeting is currently active.

    Uses two complementary signals:
    1. UDP connections to port 8801 (Zoom media servers) — primary signal.
    2. CptHost process — Zoom's conferencing host, only runs during meetings.

    Either signal being present means the meeting is active. This avoids false
    stops when UDP briefly drops (e.g. when starting Zoom recording).
    """
    # Check 1: UDP media connections to port 8801
    try:
        result = subprocess.run(
            ["lsof", "-i", "UDP", "-n", "-P"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("zoom") and ":8801" in line:
                return True
    except (subprocess.TimeoutExpired, OSError):
        pass

    # Check 2: CptHost process (stays alive even when UDP briefly drops)
    try:
        result = subprocess.run(
            ["pgrep", "-f", "CptHost"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, OSError):
        pass

    return False


# Lock to serialize GPU access (mlx-whisper is not thread-safe)
transcribe_lock = threading.Lock()

# Lock to serialize file writes and guarantee real-time disk flushing
file_lock = threading.Lock()


def transcription_loop(
    pa: pyaudio.PyAudio,
    device_index: int,
    label: str,
    color: str,
    log,
    running: threading.Event,
    shared_state: Dict[str, Dict],
    audio_buffers: Dict[str, list],
):
    """Capture audio from a device and transcribe in a loop.

    Uses a separate reader thread so the stream is read continuously
    even while transcription is running, preventing buffer overflow and
    audio glitches in the recording.
    """
    frames_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)
    frames_per_read = 1024
    chunk_queue: queue.Queue[np.ndarray] = queue.Queue()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=frames_per_read,
    )

    def reader():
        """Continuously read from the audio stream into a queue and recording buffer."""
        while running.is_set():
            try:
                data = stream.read(frames_per_read, exception_on_overflow=False)
            except Exception:
                break
            chunk_audio = np.frombuffer(data, dtype=np.int16)
            chunk_queue.put(chunk_audio)
            if label in audio_buffers:
                audio_buffers[label].append(chunk_audio.copy())
            shared_state[label]["rms"] = rms(chunk_audio)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    try:
        while running.is_set():
            # Accumulate one chunk worth of samples from the queue
            frames = []
            collected = 0
            while collected < frames_per_chunk and running.is_set():
                try:
                    chunk = chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                frames.append(chunk)
                collected += len(chunk)

                # Update live meter
                meter_strs = []
                meter_width = 20 if len(shared_state) > 1 else 40
                for lbl, data_dict in shared_state.items():
                    c = data_dict["color"]
                    r = data_dict["rms"]
                    meter = get_meter_string(r, SILENCE_THRESHOLD, meter_width)
                    meter_strs.append(f"{c}{lbl}{RESET} {meter} {DIM}{r:4.0f}{RESET}")
                sys.stdout.write(f"\r\033[K  " + "   ".join(meter_strs))
                sys.stdout.flush()

            if not frames or not running.is_set():
                break

            audio_i16 = np.concatenate(frames)

            # Skip silence
            if is_silent(audio_i16):
                continue

            # Normalize
            audio_f32 = audio_i16.astype(np.float32) / 32768.0

            # Transcribe (serialize GPU access)
            with transcribe_lock:
                result = mlx_whisper.transcribe(
                    audio_f32,
                    path_or_hf_repo=MODEL,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    compression_ratio_threshold=2.4,
                )

            text = result.get("text", "").strip()
            lang = result.get("language", "?")
            if text and not is_hallucination(text):
                ts = timestamp()
                line = f"[{ts}] {label} [{lang}] {text}\n"
                print(f"\r\033[K  {CYAN}{ts}{RESET}  {color}{label}{RESET} {DIM}[{lang}]{RESET} {text}")

                # Write to file and force sync to disk immediately
                with file_lock:
                    log.write(line)
                    log.flush()
                    os.fsync(log.fileno())

    finally:
        reader_thread.join(timeout=2)
        stream.stop_stream()
        stream.close()


# ── Transcription Session ─────────────────────────────────────────────────────

def run_transcription_session(
    pa: pyaudio.PyAudio,
    mic_index: Optional[int],
    sys_index: Optional[int],
    meeting_name: str,
    record_audio: bool,
    running: threading.Event,
) -> str:
    """Run a single transcription session. Returns the log file path."""
    os.makedirs("transcripts", exist_ok=True)

    ts_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join("transcripts", f"transcript-{ts_file} - [{meeting_name}].txt")

    print(f"{DIM}Log: {log_file}{RESET}")
    print(f"{GREEN}▶ Listening...{RESET}\n")

    log = open(log_file, "a", encoding="utf-8")
    with file_lock:
        log.write(f"# {meeting_name}\n")
        log.write(f"# Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log.flush()
        os.fsync(log.fileno())

    running.set()
    threads = []

    shared_state = {}
    audio_buffers = {}
    if mic_index is not None:
        shared_state["[MIC]"] = {"rms": 0.0, "color": CYAN}
        if record_audio:
            audio_buffers["[MIC]"] = []
    if sys_index is not None:
        shared_state["[SYS]"] = {"rms": 0.0, "color": YELLOW}
        if record_audio:
            audio_buffers["[SYS]"] = []

    if mic_index is not None:
        t = threading.Thread(
            target=transcription_loop,
            args=(pa, mic_index, "[MIC]", CYAN, log, running, shared_state, audio_buffers),
            daemon=True,
        )
        threads.append(t)
        t.start()

    if sys_index is not None:
        t = threading.Thread(
            target=transcription_loop,
            args=(pa, sys_index, "[SYS]", YELLOW, log, running, shared_state, audio_buffers),
            daemon=True,
        )
        threads.append(t)
        t.start()

    try:
        while running.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    for t in threads:
        t.join(timeout=5)

    with file_lock:
        log.write(f"\n# Stopped at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()
        os.fsync(log.fileno())
        log.close()
    print(f"\r\033[K\n{DIM}⏹ Stopped. Log saved to {log_file}{RESET}")

    if record_audio:
        audio_file = log_file.rsplit(".", 1)[0] + ".mp3"
        print(f"{DIM}Encoding audio...{RESET}", end="", flush=True)
        save_audio_mp3(audio_buffers, audio_file)
        print(f"\r\033[K{DIM}⏹ Audio saved to {audio_file}{RESET}")

    return log_file


# ── Setup Helpers ─────────────────────────────────────────────────────────────

def setup_devices(pa: pyaudio.PyAudio):
    """Interactive device selection. Returns (mic_index, sys_index)."""
    devices = list_input_devices(pa)

    if not devices:
        print("No input devices found.")
        pa.terminate()
        sys.exit(1)

    print(f"{BOLD}Available input devices:{RESET}\n")
    for i, dev in enumerate(devices):
        tag = ""
        if "blackhole" in dev["name"].lower():
            tag = f"  {DIM}<- system audio{RESET}"
        print(f"  {CYAN}[{i}]{RESET}  {dev['name']}{tag}")
    print()

    print(f"{BOLD}1. Microphone{RESET} (your voice)")
    mic_index = pick_device(pa, devices, f"   Select mic device (0-{len(devices)-1}): ")

    print(f"\n{BOLD}2. System Audio{RESET} (Zoom/other apps via BlackHole)")
    bh_hint = find_blackhole(devices)
    if bh_hint is not None:
        print(f"   {DIM}BlackHole detected at [{bh_hint}]{RESET}")
    else:
        print(f"   {DIM}BlackHole not detected. Install it for system audio capture.{RESET}")
        print(f"   {DIM}See: https://existential.audio/blackhole/{RESET}")
    sys_index = pick_device(pa, devices, f"   Select system audio device (0-{len(devices)-1}, or 's' to skip): ")

    if mic_index is None and sys_index is None:
        print("No devices selected. Exiting.")
        pa.terminate()
        sys.exit(1)

    return mic_index, sys_index


def warmup_model():
    """Load and warm up the Whisper model."""
    print(f"\n{DIM}Loading model...{RESET}")
    warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(warmup_audio, path_or_hf_repo=MODEL)
    print(f"{GREEN}✓ Model loaded.{RESET}\n")


# ── Watch Mode (Auto-detect Zoom Meetings) ───────────────────────────────────

WATCH_POLL_INTERVAL = 3  # seconds between Zoom detection checks

def watch_loop(pa, mic_index, sys_index, record_audio):
    """Watch for Zoom meetings and auto-start/stop transcription."""
    running = threading.Event()
    stop_watch = threading.Event()

    def on_sigint(sig, frame):
        if running.is_set():
            running.clear()
        else:
            stop_watch.set()

    signal.signal(signal.SIGINT, on_sigint)

    print(f"{BOLD}👀 Watching for Zoom meetings...{RESET}")
    print(f"{DIM}Will auto-start transcription when a meeting is detected.{RESET}")
    print(f"{DIM}Press Ctrl+C to exit.{RESET}\n")

    while not stop_watch.is_set():
        sys.stdout.write(f"\r\033[K  {DIM}Waiting for Zoom meeting...{RESET}")
        sys.stdout.flush()

        # Wait for a meeting to start
        while not stop_watch.is_set():
            if is_zoom_meeting_active():
                break
            time.sleep(WATCH_POLL_INTERVAL)

        if stop_watch.is_set():
            break

        # Meeting detected — run transcription in a background thread so we
        # can keep polling for meeting end in this thread.
        meeting_name = f"zoom-{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        print(f"\r\033[K\n{GREEN}✓ Zoom meeting detected!{RESET} Starting transcription...\n")

        # Set running BEFORE starting the session thread to avoid a race
        # where the main loop checks running.is_set() before the thread has
        # had a chance to call running.set().
        running.set()

        session_thread = threading.Thread(
            target=run_transcription_session,
            args=(pa, mic_index, sys_index, meeting_name, record_audio, running),
            daemon=True,
        )
        session_thread.start()

        # Poll until the meeting ends or user hits Ctrl+C.
        consecutive_inactive = 0
        while running.is_set() and not stop_watch.is_set():
            time.sleep(WATCH_POLL_INTERVAL)
            if not is_zoom_meeting_active():
                consecutive_inactive += 1
                if consecutive_inactive >= 5:
                    print(f"\r\033[K\n{YELLOW}Meeting ended.{RESET} Stopping transcription...")
                    running.clear()
            else:
                consecutive_inactive = 0

        session_thread.join(timeout=10)

        if stop_watch.is_set():
            break

        print(f"\n{BOLD}👀 Watching for next Zoom meeting...{RESET}\n")

        # Wait until meeting is confirmed inactive before watching again,
        # to avoid immediately re-triggering on the same meeting.
        while not stop_watch.is_set():
            if not is_zoom_meeting_active():
                break
            time.sleep(WATCH_POLL_INTERVAL)

    print(f"\n{DIM}Exiting watcher.{RESET}")


# ── Manual Mode ───────────────────────────────────────────────────────────────

def manual_loop(pa, mic_index, sys_index, record_audio):
    """Original interactive mode: prompt for meeting names, Ctrl+C to stop each."""
    running = threading.Event()

    def on_stop(sig, frame):
        running.clear()

    signal.signal(signal.SIGINT, on_stop)

    while True:
        try:
            print("-" * 50)
            meeting_name = input(f"\n{BOLD}Meeting name{RESET} (or Ctrl+D to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Exiting transcriber.{RESET}")
            break

        if not meeting_name:
            meeting_name = "untitled"

        print(f"{DIM}Press Ctrl+C to stop this meeting{RESET}")
        run_transcription_session(
            pa, mic_index, sys_index, meeting_name, record_audio, running,
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live audio transcriber")
    parser.add_argument(
        "--watch", action="store_true",
        help="Auto-detect Zoom meetings and start/stop transcription",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record audio as MP3 alongside the transcript",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}🎤 Live Transcriber{RESET}")
    mode_label = "watch mode" if args.watch else "manual mode"
    print(f"{DIM}Model: {MODEL} · Chunk: {CHUNK_DURATION}s · {mode_label}{RESET}\n")

    pa = pyaudio.PyAudio()
    mic_index, sys_index = setup_devices(pa)

    # In manual mode, ask about recording interactively (unless --record is set)
    if args.watch:
        record_audio = args.record
        if record_audio:
            print(f"  {DIM}→ Audio recording enabled{RESET}")
    elif args.record:
        record_audio = True
        print(f"  {DIM}→ Audio recording enabled{RESET}")
    else:
        print(f"\n{BOLD}3. Record Audio{RESET} (save MP3 alongside transcript)")
        while True:
            try:
                rec_choice = input(f"   Record audio? (y/n) [n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}Cancelled.{RESET}")
                pa.terminate()
                sys.exit(0)
            if rec_choice in ("y", "yes"):
                record_audio = True
                print(f"  {DIM}→ Audio recording enabled{RESET}")
                break
            elif rec_choice in ("n", "no", ""):
                record_audio = False
                print(f"  {DIM}→ Audio recording disabled{RESET}")
                break
            else:
                print("  Please enter 'y' or 'n'.")

    warmup_model()

    if args.watch:
        watch_loop(pa, mic_index, sys_index, record_audio)
    else:
        manual_loop(pa, mic_index, sys_index, record_audio)

    pa.terminate()


if __name__ == "__main__":
    main()
