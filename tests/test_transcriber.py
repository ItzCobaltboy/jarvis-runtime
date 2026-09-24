"""Throwaway script to verify Transcriber wraps WhisperONNX correctly. Run from repo root."""
import sys
from pathlib import Path

import yaml
import torchaudio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from transcriber import Transcriber, SAMPLE_RATE  # noqa: E402

ROOT = Path(__file__).parent.parent

with open(ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

print("[*] Loading transcriber (this downloads the ONNX model on first run)...")
t = Transcriber(config["transcriber"])

print("[*] Warming up...")
t.warmup()

wav_path = ROOT / "Whisper" / "audio_files" / "1089-134686-0000.wav"
waveform, sr = torchaudio.load(str(wav_path))
if sr != SAMPLE_RATE:
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
audio = waveform.squeeze(0).numpy().astype(np.float32)

print(f"[*] Transcribing {wav_path} ({len(audio)/SAMPLE_RATE:.1f}s)...")
text = t.transcribe(audio)
print(f"[result] {text!r}")
