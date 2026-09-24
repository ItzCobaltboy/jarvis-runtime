"""Throwaway script to sanity-check IntentMatcher thresholds and matching quality."""
import yaml
from intent import IntentMatcher

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

matcher = IntentMatcher(config["intent"], config["commands"])

test_phrases = [
    "open chrome",
    "launch chrome browser",
    "hey open spotify for me",
    "bravo six going dark",
    "go to sleep",
    "what time is it",
    "turn on hotspot",
    "enable my hotspot",
    "random gibberish that should not match anything",
]

for phrase in test_phrases:
    action, score = matcher.match(phrase)
    print(f"{phrase!r:55} -> action={action!r:20} confidence={score:.4f}")
