# jarvis-runtime

A local, offline-first voice assistant for Windows, built for ASUS ROG Zephyrus laptops with a Ryzen AI (XDNA) NPU. Say a wake word, speak a command, and it runs — transcription happens on the NPU, nothing is sent to the cloud.

Pipeline: **wake word → NPU speech-to-text (Whisper) → intent matching → action**, with a translucent GPU-rendered overlay reflecting what's happening on screen.

---

## Stack

| Layer | Tech |
|---|---|
| Wake word | [openWakeWord](https://github.com/dscripka/openWakeWord) (ONNX, 80ms chunks) |
| Transcription | Whisper (small), NPU-accelerated via [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW)'s `WhisperONNX` + onnxruntime's `VitisAIExecutionProvider` |
| Intent matching | `all-MiniLM-L6-v2` sentence embeddings, cosine similarity |
| Actions | subprocess / [WinRT](https://github.com/pywinrt/pywinrt) / [pycaw](https://github.com/AndreMiras/pycaw) |
| Overlay UI | PyQt6 + a hand-written OpenGL fragment shader |

---

## Requirements

- Windows 11
- An AMD Ryzen AI laptop with an NPU (developed on a ROG Zephyrus G14 GA403, Ryzen AI 300 series)
- [AMD's RyzenAI SDK](https://ryzenai.docs.amd.com/) installed, which provisions a **conda environment** (`ryzen-ai-1.8.0` by default) with an NPU-patched build of `onnxruntime` (`onnxruntime-vitisai`) that exposes `VitisAIExecutionProvider`
- [AMD's RyzenAI-SW repo](https://github.com/amd/RyzenAI-SW), specifically `Demos/ASR/Whisper/`, copied into this project as `Whisper/`

### Why a conda env, not a venv

The NPU execution provider (`VitisAIExecutionProvider`) only exists inside the conda environment the RyzenAI SDK installer sets up — it is **not** a normal `pip install onnxruntime` package. A plain venv cannot see it. Every command below assumes you're using that conda environment's `python.exe` directly.

**Do not `pip install` anything without `--no-deps`** in this environment. Several packages (`onnxruntime`, `torch`, `torchvision`, `transformers`) pull in generic versions as transitive dependencies that will silently overwrite the RyzenAI-patched builds and kill NPU support. If `onnxruntime.get_available_providers()` ever stops listing `VitisAIExecutionProvider`, something clobbered it — reinstall `onnxruntime-vitisai` and `onnxruntime-providers-ryzenai` from the RyzenAI SDK's install directory (`C:\Program Files\RyzenAI\<version>\*.whl`).

---

## Setup

1. Install the RyzenAI SDK (creates the conda env).
2. Activate it:
   ```powershell
   conda activate ryzen-ai-1.8.0
   ```
3. Install dependencies, **with `--no-deps`**:
   ```powershell
   pip install --no-deps -r requirements.txt
   ```
4. Copy AMD's Whisper ASR demo into this repo as `Whisper/` (needs `run_whisper.py`, `config/`, and its own `requirements.txt` for reference — model weights are downloaded automatically on first run from Hugging Face).
5. Run `python overlay_gl.py` standalone first — it cycles through all UI states without needing the mic or NPU, and is the fastest way to confirm PyQt6/OpenGL/transparency work on your machine before touching the rest of the pipeline.

---

## Run

```powershell
conda activate ryzen-ai-1.8.0
python main.py
```

First run downloads the Whisper ONNX model from Hugging Face and compiles it for the NPU — this can take several minutes. Subsequent runs use the cached, pre-compiled model and start in seconds.

---

## Commands

Wake word (`config.yaml` → `wake_word.name`) is currently `"alexa"` — any [openWakeWord](https://github.com/dscripka/openWakeWord) built-in model name works (`hey_jarvis`, `hey_mycroft`, etc). Say the wake word, then one of:

| Say | Does |
|---|---|
| "open chrome" | Launches Chrome |
| "open spotify" | Launches Spotify |
| "open edge" | Launches Edge |
| "open vs code" | Launches VS Code |
| "open steam" | Launches Steam |
| "open claude" / "open cloud" | Launches the Claude desktop app (falls back to Edge → claude.ai if not installed) |
| "start dev mode" / "work mode" / "open everything" / "daddy's home" | Opens Claude + Edge + VS Code together |
| "bravo six going dark" / "lock out" | Suspends the system |
| "what time is it" | Prints the time |
| "turn on/off hotspot" | Toggles mobile hotspot (WinRT, no admin needed) |
| "toggle playback" / "next song" / "previous song" | Media key controls (works with Spotify, YouTube, anything) |
| "volume up" / "volume down" / "mute" | System volume via `pycaw` |
| "lock screen" | Locks the workstation |
| "cancel" / "nevermind" / "forget it" | No-op — matches so a misfire doesn't trigger an unrelated command |

New commands: add the phrase → action key mapping in `config.yaml` under `commands:`, implement the handler in `actions/apps.py` or `actions/system.py`, register it in `actions/registry.py`.

Intent matching is pure cosine similarity against `config.yaml`'s command list, not an LLM — phrasing that's semantically close to a registered phrase will match; anything else falls through as unrecognised.

---

## The overlay

`overlay_gl.py` is a full-screen, click-through, always-on-top window with no taskbar entry, showing an edge-glow vignette + a small top-right pill with status text. All color/gradient math runs in a GLSL fragment shader (not CPU-drawn `QPainter` gradients), specifically to avoid burning CPU/battery redrawing dozens of gradients at 30fps on a laptop that's also running NPU inference.

- **Hidden** while waiting for the wake word.
- **Listening** (angular rainbow) — shown the instant the wake word fires, for the duration of the recording.
- **Processing** (pulsing amber) — while Whisper transcribes.
- **Success** (green) — command matched and dispatched; the pill shows the transcript.
- **Unrecognised** (red, pill shakes) — no intent matched, or transcript was empty.

States cross-fade into each other in the shader (not a hard color cut) and auto-dismiss back to hidden after a short hold.

Test it standalone, independent of the mic/NPU pipeline:
```powershell
python overlay_gl.py
```

---

## Project structure

```
jarvis-runtime/
├── main.py             # entry point: wires listener -> transcriber -> intent -> action -> overlay
├── listener.py         # mic capture (48kHz/4ch native -> downmixed to 16kHz mono) + wake word detection
├── transcriber.py      # wraps Whisper/run_whisper.py's WhisperONNX for NPU-accelerated transcription
├── intent.py           # sentence-embedding cosine similarity intent matching
├── overlay_gl.py        # GPU-shader status overlay (PyQt6 + OpenGL)
├── actions/
│   ├── registry.py     # action key -> handler function map
│   ├── apps.py         # app launchers
│   └── system.py       # sleep, hotspot, media/volume, lock screen, cancel
├── Whisper/             # AMD's RyzenAI-SW Whisper ASR demo (run_whisper.py, config/) — not authored here
├── tests/               # standalone smoke tests (mic, wake word, transcriber, intent)
├── config.yaml
└── requirements.txt
```

---

## Known quirks

- **Mic must run in shared WASAPI mode.** Some laptop mic arrays (e.g. Realtek "Microphone Array") only expose themselves for exclusive-mode capture at their native format (commonly 48kHz/4-channel), not the 16kHz mono PyAudio would normally request directly. `listener.py` opens the stream at the device's native format and downmixes/decimates to 16kHz mono in software.
- **`pip install` without `--no-deps` will break the NPU.** See "Why a conda env" above. This has happened more than once during development — `sentence-transformers`, `torchaudio`, and plain `pip install onnxruntime` are the usual culprits.
- **The Whisper ONNX model's `config_file` paths must be absolute.** AMD's `run_whisper.py` reads `model_config.json`'s `config_file`/`cache_dir` entries as literal strings passed straight to onnxruntime's native `VitisAIExecutionProvider`, which resolves them relative to the process's *current working directory*, not the script's location. `transcriber.py` rewrites these to absolute paths on load so it works regardless of where you run `main.py` from.
- **ASUS ATK WMI (performance/GPU mode switching) is not implemented.** It was attempted and removed — the ASUS ATK WMI provider (`root\wmi`, `AsusAtkWmi_WMNB`) requires the calling process to be elevated (`WBEM_E_ACCESS_DENIED` otherwise) and was unreliable even then. Not present in this codebase.
- **"Sleep" hibernates instead of sleeping**, if your system's power plan has "Hibernate after" set to 0 on AC power combined with Modern Standby (S0) — this is a Windows power policy interaction, not a bug in `sleep_system()`. Considered acceptable since hibernate also has a security benefit (full power-down).
- **Silence-cutoff timing matters.** `listener.py`'s `SILENCE_CHUNKS` controls how long a pause is tolerated before recording auto-stops. Too short, and it cuts you off mid-sentence right after the wake word; too long, and commands take longer to process. Currently tuned to ~4 seconds.
- **First NPU run is slow.** The Whisper model is JIT-compiled for the NPU on first load and cached (`Whisper/cache/`); this can take up to ~15 minutes once, then seconds on every subsequent run.

---

## Credits

- [AMD RyzenAI-SW](https://github.com/amd/RyzenAI-SW) — `Whisper/run_whisper.py`'s `WhisperONNX` class (NPU-accelerated Whisper inference via onnxruntime's VitisAI EP) is used directly, not reimplemented.
- [openWakeWord](https://github.com/dscripka/openWakeWord) — wake word detection models and inference.
- [sentence-transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` embeddings for intent matching.
- [pycaw](https://github.com/AndreMiras/pycaw) — Windows Core Audio volume/mute control.
