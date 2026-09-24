"""
Intent matching via cosine similarity on sentence embeddings.
Compares transcript against known command strings, returns best match above threshold.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class IntentMatcher:
    def __init__(self, config: dict, commands: dict):
        self.threshold = config["confidence_threshold"]
        self.model = SentenceTransformer(config["model_name"])

        self.command_texts = list(commands.keys())
        self.command_keys = list(commands.values())
        self.embeddings = self.model.encode(self.command_texts, normalize_embeddings=True)

    def match(self, transcript: str) -> tuple[str | None, float]:
        """Returns (action_key, confidence). action_key is None if below threshold."""
        query = self.model.encode([transcript.lower()], normalize_embeddings=True)
        scores = (self.embeddings @ query.T).flatten()

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            return self.command_keys[best_idx], best_score

        return None, best_score
