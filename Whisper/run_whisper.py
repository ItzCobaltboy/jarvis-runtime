import numpy as np
import onnxruntime as ort
import time
import os
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from huggingface_hub import snapshot_download

SAMPLE_RATE = 16000


class WhisperONNX:
    def __init__(self, encoder_path, decoder_path,
                 model_type, encoder_providers=None, decoder_providers=None, language=None):

        self.encoder = ort.InferenceSession(encoder_path, providers=encoder_providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=decoder_providers)

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_type}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/{model_type}")
        self.decoder_start_token = self.sot_token = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
        if not isinstance(self.max_length, int):
            raise ValueError("Invalid/Dynamic input shapes")

        self.language = language
        if self.language:
            self.tokenizer.set_prefix_tokens(language=self.language, task="transcribe")
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        else:
            self.initial_tokens = [self.decoder_start_token]

    def preprocess(self, audio):
        """
        Convert raw audio to Whisper log-mel spectrogram
        """
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        return inputs["input_features"]

    def encode(self, input_features):
        """
        Run encoder ONNX model
        """
        input_name = self.encoder.get_inputs()[0].name
        return self.encoder.run(None, {input_name: input_features})[0]

    def decode(self, encoder_out):
        """
        Greedy decode with fixed-length input_ids
        """
        tokens = list(self.initial_tokens)
        first_token_delay = None
        decode_start = time.time()

        # Get decoder input names
        decoder_inputs = self.decoder.get_inputs()
        input_ids_name = decoder_inputs[0].name
        encoder_out_name = decoder_inputs[1].name

        # Distinguish inputs by data type if the order is not guaranteed
        if decoder_inputs[0].type != 'tensor(int64)':
            input_ids_name, encoder_out_name = encoder_out_name, input_ids_name

        for _ in range(len(tokens), self.max_length):
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens

            outputs = self.decoder.run(None, {
                input_ids_name: decoder_input,
                encoder_out_name: encoder_out
            })
            logits = outputs[0]
            next_token = int(np.argmax(logits[0, len(tokens) - 1]))

            if next_token == self.eos_token:
                break
            tokens.append(next_token)
            if first_token_delay is None:
                first_token_delay = time.time() - decode_start
        return tokens, first_token_delay

    def transcribe(self, audio, chunk_length_s=30, is_mic=False):
        """
        Full encode-decode pipeline with support for long-form transcription using chunking.
        """
        chunk_size = SAMPLE_RATE * chunk_length_s
        total_samples = len(audio)
        transcription = []
        chunk_idx = 0
        total_start_time = time.time()

        overlap = SAMPLE_RATE * 1  # Tune this
        for start in range(0, total_samples, chunk_size - overlap):
            end = min(start + chunk_size, total_samples)
            audio_chunk = audio[start:end]

            input_features = self.preprocess(audio_chunk)
            encoder_out = self.encode(input_features)
            tokens, first_token_delay = self.decode(encoder_out)
            decoded_text = self.tokenizer.decode(
                tokens[len(self.initial_tokens):],
                skip_special_tokens=True
            ).strip()
            transcription.append(decoded_text)
            chunk_idx += 1
            if not is_mic:
                if first_token_delay is not None:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(f" Time to First Token for this chunk: {first_token_delay:.2f} seconds")
                else:
                    print(f"\nPerformance Metric (Chunk {chunk_idx}):")
                    print(" Time to First Token for this chunk: n/a (no token before EOS)")

        total_end_time = time.time()
        input_audio_duration = total_samples / SAMPLE_RATE
        rtf = (total_end_time - total_start_time) / input_audio_duration
        if not is_mic:
            print(f" RTF: {rtf:.2f}")

        return " ".join(transcription), rtf


def load_provider_options(config, model_name, device):
    model_key = model_name.replace("whisper-", "")
    if model_key not in config["whisper"]:
        raise ValueError(f"Model type '{model_key}' not found in config")

    if device not in config["whisper"][model_key]:
        raise ValueError(f"Device '{device}' not found in config for model type '{model_key}'")

    model_config = config["whisper"][model_key][device]
    encoder_opts = model_config["encoder"]
    decoder_opts = model_config["decoder"]

    def build_provider_opts(opts):
        if opts.get("config_file"):
            return [
                (
                    "VitisAIExecutionProvider",
                    {
                        "config_file": opts["config_file"],
                        "cache_dir": opts.get("cache_dir", ""),
                        "cache_key": opts.get("cache_key", "")
                    }
                )
            ]
        else:
            return ["CPUExecutionProvider"]

    print("Selected Provider Options: ")
    print("Decoder: ", build_provider_opts(decoder_opts))
    print("Encoder: ", build_provider_opts(encoder_opts))
    return build_provider_opts(encoder_opts), build_provider_opts(decoder_opts)


def download_whisper_onnx(model_type: str):
    """
    Download Whisper ONNX encoder/decoder from Hugging Face if not already present.
    Returns paths to encoder and decoder model files.
    """
    hf_model_map = {
        "whisper-small": "amd/whisper-small-onnx-npu",
        "whisper-medium": "amd/whisper-medium-onnx-npu",
        "whisper-large-v3-turbo": "amd/whisper-large-turbo-onnx-npu"
    }

    repo_id = hf_model_map.get(model_type)
    if repo_id is None:
        raise ValueError(f"Unsupported model_type '{model_type}' for ONNX auto-download.")

    local_dir = snapshot_download(
        repo_id=repo_id,
    )

    # Construct paths to encoder/decoder ONNX files
    encoder_path = os.path.join(local_dir, "encoder_model.onnx")
    decoder_path = os.path.join(local_dir, "decoder_model.onnx")

    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        raise FileNotFoundError(f"Could not find encoder/decoder in {local_dir}")

    return encoder_path, decoder_path
