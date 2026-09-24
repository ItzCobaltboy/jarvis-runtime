"""
Main entry point. Wires listener -> transcriber -> intent -> action, with a
GPU-rendered overlay (overlay_gl.py) reflecting pipeline state on screen.
Run this. Everything else is a module.
"""

import os
import sys
import threading
import time

import yaml
from PyQt6.QtWidgets import QApplication

UNRECOGNISED_COOLDOWN_SECONDS = 1.0


def _fixup_npu_dll_path():
    """The VitisAI execution provider's vaiml.dll (from the flexml package)
    is loaded by onnxruntime's native code via LoadLibrary at runtime, not a
    Python import — Nuitka's dependency scanner can't see that reference, so
    it won't get bundled/pathed automatically like a normal import would be.

    In dev, this DLL lives inside the conda env's site-packages; in a
    compiled build, build.bat copies that same folder next to the exe.
    Either way, add it to the DLL search path *before* importing anything
    that transitively imports onnxruntime.
    """
    candidates = []
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        # Compiled build: bundled next to the exe by build.bat / build_pyinstaller.bat.
        candidates.append(os.path.join(os.path.dirname(sys.executable), "flexml_lib"))
        # PyInstaller onefile extracts data files under sys._MEIPASS instead.
        if hasattr(sys, "_MEIPASS"):
            candidates.append(os.path.join(sys._MEIPASS, "flexml_lib"))
    else:
        # Dev: locate via the currently-running conda env's onnxruntime install.
        try:
            import onnxruntime  # noqa: F401 -- just to resolve site-packages path
            site_packages = os.path.dirname(os.path.dirname(onnxruntime.__file__))
            candidates.append(os.path.join(site_packages, "flexml", "flexml_extras", "lib"))
        except ImportError:
            pass

    for path in candidates:
        if os.path.isdir(path):
            os.add_dll_directory(path)
            return

    print(f"[main] Warning: flexml DLL directory not found (tried {candidates}). "
          f"NPU acceleration may fail to load.")


_fixup_npu_dll_path()

from listener import Listener
from transcriber import Transcriber
from intent import IntentMatcher
from actions.registry import dispatch
from overlay_gl import JarvisOverlayGL


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def audio_pipeline(overlay: JarvisOverlayGL):
    config = load_config()

    print("[*] Warming up transcriber...")
    transcriber = Transcriber(config["transcriber"])
    transcriber.warmup()

    print("[*] Loading intent matcher...")
    intent = IntentMatcher(config["intent"], config["commands"])

    print("[*] Starting listener...")
    listener = Listener(config["wake_word"])

    print("[*] Ready. Listening for wake word...")

    while True:
        # Overlay stays hidden while waiting for the wake word — it only
        # appears once the wake word actually fires and recording begins.
        audio = listener.wait_for_command(
            on_wake_word=lambda: overlay.show_state("listening")
        )
        if audio is None:
            continue

        overlay.show_state("processing")
        transcript = transcriber.transcribe(audio)
        if not transcript:
            overlay.show_state("unrecognised")
            time.sleep(UNRECOGNISED_COOLDOWN_SECONDS)
            continue

        print(f"[transcript] {transcript}")

        action_key, confidence = intent.match(transcript)
        if action_key:
            print(f"[intent] {action_key} ({confidence:.2f})")
            overlay.show_state("success", transcript=transcript)
            dispatch(action_key)
        else:
            print(f"[intent] no match (best confidence: {confidence:.2f})")
            overlay.show_state("unrecognised")
            time.sleep(UNRECOGNISED_COOLDOWN_SECONDS)


def main():
    qt_app = QApplication(sys.argv)
    overlay = JarvisOverlayGL()

    thread = threading.Thread(target=audio_pipeline, args=(overlay,), daemon=True)
    thread.start()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
