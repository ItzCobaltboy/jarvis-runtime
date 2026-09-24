"""
Main entry point. Wires listener -> transcriber -> intent -> action, with a
GPU-rendered overlay (overlay_gl.py) reflecting pipeline state on screen.
Run this. Everything else is a module.
"""

import sys
import threading
import time

import yaml
from PyQt6.QtWidgets import QApplication

from listener import Listener
from transcriber import Transcriber
from intent import IntentMatcher
from actions.registry import dispatch
from overlay_gl import JarvisOverlayGL

UNRECOGNISED_COOLDOWN_SECONDS = 1.0


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
