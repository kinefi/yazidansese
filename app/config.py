import os

import torch

MODEL_ID = "facebook/mms-tts-tur"
# Shared cache directory for models
CACHE_DIR = "cache/models"

# Directory for caching synthesized audio files
CACHE_AUDIO_DIR = "cache/audio"
MAX_CACHE_SIZE_MB = 500  # Maximum size of the audio cache in MB

os.makedirs(CACHE_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Set in HF Space secrets.
os.makedirs(CACHE_AUDIO_DIR, exist_ok=True) # Ensure audio cache directory exists
MAX_CHARS = 25_000                                 # Hard cap per request.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
