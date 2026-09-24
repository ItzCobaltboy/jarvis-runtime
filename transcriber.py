"""
Whisper transcription via AMD's RyzenAI-SW WhisperONNX (Whisper/run_whisper.py),
run in-process using onnxruntime's VitisAI execution provider for NPU offload.
"""

import os
import sys
from pathlib import Path

import numpy as np

_WHISPER_DIR = Path(__file__).parent / "Whisper"
sys.path.insert(0, str(_WHISPER_DIR))

from run_whisper import WhisperONNX, load_provider_options, download_whisper_onnx  # noqa: E402

SAMPLE_RATE = 16000

# In a compiled (PyInstaller/Nuitka onefile) build, __file__ resolves inside a
# temp extraction directory that's wiped after the process exits — caching the
# NPU-compiled model there would force a ~15min recompile on every single
# launch. Redirect the cache to a stable per-user location instead.
if getattr(sys, "frozen", False) or "__compiled__" in globals():
    _CACHE_DIR = Path(os.environ["LOCALAPPDATA"]) / "jarvis-runtime" / "Whisper" / "cache"
else:
    _CACHE_DIR = _WHISPER_DIR / "cache"


def _resolve_config_paths(model_config: dict, base_dir: Path):
    """model_config.json's config_file/cache_dir entries are relative to Whisper/;
    onnxruntime's VitisAI EP resolves them against the process cwd, not this file,
    so rewrite them to absolute paths regardless of where the caller runs from.
    cache_dir is redirected to _CACHE_DIR so compiled builds persist the NPU
    compile cache across runs instead of losing it with the temp extraction dir."""
    for model_variants in model_config.get("whisper", {}).values():
        for device_opts in model_variants.values():
            for stage in ("encoder", "decoder"):
                opts = device_opts.get(stage, {})
                if opts.get("config_file"):
                    opts["config_file"] = str((base_dir / opts["config_file"]).resolve())
                if opts.get("cache_dir"):
                    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    opts["cache_dir"] = str(_CACHE_DIR)


class Transcriber:
    def __init__(self, config: dict):
        self.model_type = config["model_type"]  # e.g. "whisper-base"
        self.device = config.get("device", "cpu")  # "cpu" or "npu"
        self.language = config.get("language")
        config_file = _WHISPER_DIR / config.get("config_file", "config/model_config.json")

        import json
        with open(config_file) as f:
            model_config = json.load(f)
        _resolve_config_paths(model_config, _WHISPER_DIR)

        encoder_providers, decoder_providers = load_provider_options(
            model_config, self.model_type, self.device
        )

        encoder_path = config.get("encoder_path")
        decoder_path = config.get("decoder_path")
        if not encoder_path or not decoder_path:
            encoder_path, decoder_path = download_whisper_onnx(self.model_type)

        self.model = WhisperONNX(
            encoder_path,
            decoder_path,
            self.model_type,
            encoder_providers=encoder_providers,
            decoder_providers=decoder_providers,
            language=self.language,
        )

    def warmup(self):
        """Run a dummy transcription to force model load / NPU compile+cache."""
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        try:
            self.model.transcribe(dummy, is_mic=True)
        except Exception:
            pass  # warmup failure is non-fatal

        print("[transcriber] Warmed up")

    def transcribe(self, audio: np.ndarray) -> str:
        """audio is float32 at 16kHz, shape (N,), range [-1, 1]."""
        text, _ = self.model.transcribe(audio, is_mic=True)
        return text.strip()
