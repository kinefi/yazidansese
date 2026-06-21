import os

import torch

SUPPORTED_MODELS = {
    "MMS-TTS (facebook)": "facebook/mms-tts-tur",
    "Multi-Language VITS (OpenBible)": "multilingual-tts/VITS-OpenBible-Turkish",
    "EveryVoice VITS (OpenBible)": "multilingual-tts/EveryVoice-OpenBible-Turkish",
    "F5-TTS (OpenBible)": "multilingual-tts/F5-TTS-OpenBible-Turkish",
}

FALLBACK_MODEL_NAMES = [
    "MMS-TTS (facebook)",
]

DEFAULT_MODEL_NAME = "MMS-TTS (facebook)"
# Shared cache directory for models
CACHE_DIR = "cache/models"

# Directory for caching synthesized audio files
CACHE_AUDIO_DIR = "cache/audio"
MAX_CACHE_SIZE_MB = 500  # Maximum size of the audio cache in MB

os.makedirs(CACHE_DIR, exist_ok=True)

VOICE_SAMPLES = {
    "Erken (Varsayılan)": 0, # Bu değerler, kullanılan çok konuşmacılı modelin speaker_id'lerine göre ayarlanmalıdır.
    "Kadın (Deneysel)": 1,   # Mevcut 'facebook/mms-tts-tur' modeli tek konuşmacılıdır ve bu seçimi desteklemez.
} # Bu sözlük, seçilen modele göre dinamik olarak güncellenebilir.
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # Set in HF Space secrets.
os.makedirs(CACHE_AUDIO_DIR, exist_ok=True) # Ensure audio cache directory exists
MAX_CHARS = 25_000                                 # Hard cap per request.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
