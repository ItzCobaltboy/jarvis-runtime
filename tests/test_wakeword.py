"""Throwaway script to verify openWakeWord reacts to real mic audio."""
import numpy as np
import pyaudio
from openwakeword.model import Model

NATIVE_SAMPLE_RATE = 48000
NATIVE_CHANNELS = 4
TARGET_SAMPLE_RATE = 16000
DECIMATION_FACTOR = NATIVE_SAMPLE_RATE // TARGET_SAMPLE_RATE  # 3
TARGET_CHUNK = 1280  # 80ms at 16kHz
NATIVE_CHUNK = TARGET_CHUNK * DECIMATION_FACTOR  # 3840 samples per channel

WAKE_WORD = "hey_jarvis"
THRESHOLD = 0.5


def to_mono_16k(data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.int16).reshape(-1, NATIVE_CHANNELS)
    mono = raw.mean(axis=1)
    return mono[::DECIMATION_FACTOR].astype(np.int16)


def main():
    print(f"Loading openWakeWord model: {WAKE_WORD}")
    oww = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")

    pa = pyaudio.PyAudio()
    wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    device_info = pa.get_device_info_by_index(wasapi_info["defaultInputDevice"])
    print(f"Using input device: {device_info['name']}")

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=NATIVE_CHANNELS,
        rate=NATIVE_SAMPLE_RATE,
        input=True,
        frames_per_buffer=NATIVE_CHUNK,
        input_device_index=device_info["index"],
    )

    print(f"Listening for wake word '{WAKE_WORD}'... (Ctrl+C to stop)")

    import time
    start = time.time()
    RUN_SECONDS = 20

    try:
        while time.time() - start < RUN_SECONDS:
            chunk = stream.read(NATIVE_CHUNK, exception_on_overflow=False)
            audio_np = to_mono_16k(chunk)

            prediction = oww.predict(audio_np)
            score = prediction.get(WAKE_WORD, 0.0)

            print(f"score = {score:.4f}")

            if score >= THRESHOLD:
                print("!!!!!!!!!! WAKE WORD DETECTED !!!!!!!!!!")

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
