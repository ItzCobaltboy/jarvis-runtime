"""Throwaway script to verify mic input via PyAudio (WASAPI shared mode)."""
import wave
import numpy as np
import pyaudio

TARGET_RATE = 16000
CHANNELS = 1
TARGET_CHUNK = 1280  # 80ms at 16kHz
DURATION = 5
FORMAT = pyaudio.paInt16
OUTPUT_FILE = "test_output.wav"

pa = pyaudio.PyAudio()

wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
device_info = pa.get_device_info_by_index(wasapi_info["defaultInputDevice"])

# Stock PyAudio has no WASAPI auto-convert flag, so shared-mode streams must
# be opened at the device's native mix rate; we resample down to 16kHz after.
NATIVE_RATE = int(device_info["defaultSampleRate"])
NATIVE_CHANNELS = int(device_info["maxInputChannels"])
NATIVE_CHUNK = int(NATIVE_RATE * TARGET_CHUNK / TARGET_RATE)

print(
    f"Using input device: {device_info['name']} "
    f"(native rate: {NATIVE_RATE} Hz, native channels: {NATIVE_CHANNELS})"
)

stream = pa.open(
    format=FORMAT,
    channels=NATIVE_CHANNELS,
    rate=NATIVE_RATE,
    input=True,
    frames_per_buffer=NATIVE_CHUNK,
    input_device_index=device_info["index"],
)

resampled_frames = []
num_chunks = int(NATIVE_RATE / NATIVE_CHUNK * DURATION)

print(f"Recording for {DURATION} seconds...")
for i in range(num_chunks):
    data = stream.read(NATIVE_CHUNK, exception_on_overflow=False)
    raw = np.frombuffer(data, dtype=np.int16).astype(np.float64)
    raw = raw.reshape(-1, NATIVE_CHANNELS)
    samples = raw.mean(axis=1)  # downmix to mono

    # Linear-interpolation resample from NATIVE_RATE to TARGET_RATE.
    src_idx = np.arange(len(samples))
    dst_idx = np.linspace(0, len(samples) - 1, TARGET_CHUNK)
    resampled = np.interp(dst_idx, src_idx, samples).astype(np.int16)
    resampled_frames.append(resampled.tobytes())

    rms = np.sqrt(np.mean(samples ** 2))
    print(f"Chunk {i:3d}: RMS = {rms:8.2f}")

stream.stop_stream()
stream.close()
pa.terminate()

with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(TARGET_RATE)
    wf.writeframes(b"".join(resampled_frames))

print(f"Saved recording to {OUTPUT_FILE}")
