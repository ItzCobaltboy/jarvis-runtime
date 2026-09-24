"""
Handles mic input and wake word detection.
Streams 80ms chunks to openWakeWord, records after trigger until silence.
"""

import numpy as np
import pyaudio
from openwakeword.model import Model


CHUNK_MS = 80
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1280 samples

# The Realtek mic only supports shared WASAPI capture at its native format:
# 48000 Hz, 4-channel. We downmix to mono and decimate 3:1 to get 16kHz.
NATIVE_SAMPLE_RATE = 48000
NATIVE_CHANNELS = 4
DECIMATION_FACTOR = NATIVE_SAMPLE_RATE // SAMPLE_RATE  # 3
NATIVE_CHUNK_SIZE = CHUNK_SIZE * DECIMATION_FACTOR  # 3840 samples per channel

MAX_RECORD_SECONDS = 5
SILENCE_THRESHOLD = 500  # RMS below this = silence
SILENCE_CHUNKS = 50      # consecutive silent chunks before stop (~4s at 80ms/chunk)


def _to_mono_16k(data: bytes) -> np.ndarray:
    """Downmix native 48kHz/4ch int16 audio to mono, then decimate to 16kHz."""
    raw = np.frombuffer(data, dtype=np.int16).reshape(-1, NATIVE_CHANNELS)
    mono = raw.mean(axis=1)
    return mono[::DECIMATION_FACTOR].astype(np.int16)


class Listener:
    def __init__(self, config: dict):
        self.wake_word = config["name"]
        self.threshold = config["confidence_threshold"]

        self.oww = Model(wakeword_models=[self.wake_word], inference_framework="onnx")
        self.pa = pyaudio.PyAudio()

    def _open_stream(self):
        return self.pa.open(
            rate=NATIVE_SAMPLE_RATE,
            channels=NATIVE_CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=NATIVE_CHUNK_SIZE,
        )

    def wait_for_command(self, on_wake_word=None) -> np.ndarray | None:
        """Block until wake word, then record and return audio array.

        on_wake_word, if given, is called the instant the wake word fires —
        before the recording phase starts — so a caller (e.g. main.py driving
        an overlay) can react to "listening has actually begun" rather than
        only finding out once the whole recording is done.
        """
        stream = self._open_stream()

        try:
            # --- wake word loop ---
            while True:
                chunk = stream.read(NATIVE_CHUNK_SIZE, exception_on_overflow=False)
                audio_np = _to_mono_16k(chunk)
                self.oww.predict(audio_np)

                scores = self.oww.prediction_buffer.get(self.wake_word, [0])
                if scores and scores[-1] >= self.threshold:
                    print("[listener] Wake word detected")
                    # Without this, the model's rolling prediction buffer can
                    # still read above threshold on the very next cycle's
                    # first chunks (residual from this same utterance), which
                    # re-triggers detection instantly with no real new speech.
                    self.oww.reset()
                    if on_wake_word:
                        on_wake_word()
                    break

            # --- record until silence ---
            frames = []
            silent_chunks = 0
            max_chunks = int(SAMPLE_RATE / CHUNK_SIZE * MAX_RECORD_SECONDS)

            for _ in range(max_chunks):
                chunk = stream.read(NATIVE_CHUNK_SIZE, exception_on_overflow=False)
                mono_16k = _to_mono_16k(chunk)
                frames.append(mono_16k.tobytes())

                rms = np.sqrt(np.mean(mono_16k.astype(np.float32) ** 2))
                if rms < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks >= SILENCE_CHUNKS:
                    break

            audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
            return audio

        finally:
            stream.stop_stream()
            stream.close()
