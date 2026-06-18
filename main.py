"""
Yazıdan Sese
Model: facebook/mms-tts-tur (VITS)
"""

import logging
import os
import re
import tempfile
import time
from functools import lru_cache

import gradio as gr
import nltk
import numpy as np
import torch
from pydub import AudioSegment
from transformers import AutoTokenizer, VitsModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# ── Config ────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "MMS-TTS (facebook)": "facebook/mms-tts-tur",
    "Multi-Speaker VITS (yuvuz)": "yuvuz/vits-turkish-multi-speaker",
    "Multi-Speaker VITS (ahmetvural)": "ahmetvural/vits-turkish-multi-speaker",
    "VITS (kan-bayrak)": "kan-bayrak/vits-turkish",
    "MMS-TTS (ylacombe)": "ylacombe/mms-tur-vits",
}
DEFAULT_MODEL_NAME = "MMS-TTS (facebook)"
# Shared cache directory for models
CACHE_DIR = "cache/models"
os.makedirs(CACHE_DIR, exist_ok=True)

VOICE_SAMPLES = {
    "Kadın (Varsayılan)": 0, # Bu değerler, kullanılan çok konuşmacılı modelin speaker_id'lerine göre ayarlanmalıdır.
    "Erkek (Deneysel)": 1,   # Mevcut 'facebook/mms-tts-tur' modeli tek konuşmacılıdır ve bu seçimi desteklemez.
} # Bu sözlük, seçilen modele göre dinamik olarak güncellenebilir.
API_KEY = os.environ.get("HF_TOKEN", "")   # Set in HF Space secrets.
MAX_CHARS = 25_000                                 # Hard cap per request.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading (singleton) ─────────────────────────────────────────────────


@lru_cache(maxsize=5)
def load_model(model_id: str) -> tuple[VitsModel, AutoTokenizer]:
    logger.info("Loading %s on %s using cache %s …", model_id, DEVICE, CACHE_DIR)
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    model = VitsModel.from_pretrained(model_id, cache_dir=CACHE_DIR).to(DEVICE)
    model.eval()
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)
    return model, tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove characters VITS can't handle."""
    text = re.sub(r'\s+', ' ', text)
    # Strip URLs
    text = re.sub(r'https?://\S+', '', text)
    # Strip anything outside printable Turkish charset + common punctuation
    text = re.sub(r'[^\w\s.,!?;:\'\"\-–—()\u00C0-\u024F]',
                  ' ', text, flags=re.UNICODE)
    return text.strip()


def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """
    Split at sentence boundaries to keep each VITS forward pass short.
    Longer inputs produce degraded prosody; 500-char chunks are safe.
    """
    sentences = nltk.sent_tokenize(text, language="turkish")
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence.strip()
        elif sentence.strip():
            if current:
                chunks.append(current)
            # Sentence itself longer than max — hard split on word boundary.
            if len(sentence) > max_chars:
                words = sentence.split()
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 <= max_chars:
                        current = (current + " " + word).strip() if current else word.strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = word
                if current: # Add any remaining words in the buffer
                    chunks.append(current)
                current = "" # Reset current after hard splitting a long sentence
            else:
                # The sentence itself fits, but didn't fit with previous sentences
                chunks.append(sentence.strip())
                current = "" # Reset current as this sentence is now a chunk

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk] # Filter out any empty chunks


def synthesize_chunks(
    chunks: list[str],
    model: VitsModel,
    tokenizer: AutoTokenizer,
    seed: int,
    speaker_id: int = None
) -> np.ndarray:
    """Run inference per chunk, concatenate waveforms."""
    torch.manual_seed(seed)
    segments: list[np.ndarray] = []

    with torch.no_grad():
        for chunk in chunks:
            if not chunk.strip():
                continue
            inputs = tokenizer(chunk, return_tensors="pt").to(DEVICE)
            
            # speaker_ids is used for multi-speaker models.
            # For single-speaker models like facebook/mms-tts-tur, this parameter is ignored or may cause an error.
            if speaker_id is not None and hasattr(model.config, 'num_speakers') and model.config.num_speakers > 1:
                speaker_ids = torch.LongTensor([speaker_id]).to(DEVICE)
                output = model(**inputs, speaker_ids=speaker_ids)
            else:
                output = model(**inputs) 
            
            # output.waveform shape: (1, 1, samples)
            audio = output.waveform.squeeze().cpu().numpy()
            segments.append(audio)
            segments.append(
                np.zeros(int(model.config.sampling_rate * 0.2), dtype=np.float32))

    return np.concatenate(segments).astype(np.float32)


# ── Gradio Interface Function ─────────────────────────────────────────────────

def synthesize_for_gradio(
    text: str,
    model_name: str,
    voice_type: str,
    seed: int = 42,
    rate: float = 1.0,
    pitch: float = 0.0,
    progress=gr.Progress()
) -> tuple[tuple[int, np.ndarray], str | None]:
    """
    Synthesizes text into audio using the loaded VITS model.
    Returns a tuple of (sample_rate, audio_numpy_array) for Gradio's Audio component.
    Returns None for the file path if synthesis fails.
    """
    if progress:
        progress(0, desc="Initializing synthesis...")

    
    # Eğer model_name bir anahtar değilse (özel ID girilmişse) direkt ID olarak kullan
    model_id = SUPPORTED_MODELS.get(model_name, model_name)
    
    model, tokenizer = load_model(model_id) # Model is loaded based on the fixed Turkish ID

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long! Please keep it under {MAX_CHARS} characters.")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise gr.Error(
            "Text is empty after cleaning. Please provide valid Turkish text.")

    chunks = chunk_text(cleaned_text)
    logger.info("chars=%d  chunks=%d", len(cleaned_text), len(chunks)) # Log after chunking

    try:
        speaker_idx = VOICE_SAMPLES.get(voice_type) # Get speaker_id, can be None if voice_type not found.
        audio = synthesize_chunks(
            progress.tqdm(chunks, desc="Sentezleniyor...") if progress else chunks,
            model,
            tokenizer,
            seed,
            speaker_id=speaker_idx
        )
    except Exception as exc:
        logger.exception("Synthesis failed for text: %s", cleaned_text[:100])
        raise gr.Error(f"Synthesis failed: {str(exc)}") from exc

    sample_rate = model.config.sampling_rate

    # Create an AudioSegment for processing
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="sentez_")
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_int16.dtype.itemsize,
        channels=1
    )

    # Adjust speaking rate (speed)
    if rate != 1.0:
        # For speed > 1.0 pydub has speedup; for < 1.0 we adjust frame rate
        # Note: changing frame rate changes pitch. 
        # For professional speed shifting without pitch change, librosa is usually preferred.
        if rate > 1.0:
            audio_segment = audio_segment.speedup(playback_speed=rate, chunk_size=150, crossfade=25)
        else:
            # Slow down by changing frame rate and then re-sampling to original rate
            # This will change pitch, but is a simple way to adjust speed with pydub
            audio_segment = audio_segment.set_frame_rate(int(audio_segment.frame_rate * rate))
            audio_segment = audio_segment.set_frame_rate(sample_rate) # Resample back to original for consistency

    # Adjust pitch (Frequency shift)
    if pitch != 0:
        # pitch is in octaves. rate_shift = 2.0^pitch.
        # e.g., 1 octave up doubles frequency, shifting the voice higher.
        rate_shift = 2.0 ** pitch
        new_sample_rate = int(audio_segment.frame_rate * rate_shift)
        audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_sample_rate})
        audio_segment = audio_segment.set_frame_rate(sample_rate)

    audio_segment.export(temp_file.name, format="mp3")
    
    # Convert processed segment back to numpy for the Gradio player
    processed_audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32767.0
    return (sample_rate, processed_audio), temp_file.name

def update_char_count(text: str) -> str:
    """Updates the character count display."""
    current_chars = len(text)
    remaining_chars = MAX_CHARS - current_chars
    color = "green" if remaining_chars >= 0 else "red"
    return (
        f"<p style='color: {color}; font-weight: bold;'>"
        f"Karakter Sayısı: {current_chars} / {MAX_CHARS} (Kalan: {remaining_chars})</p>"
    )


# ── Gradio App ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Warm up the model on startup
    load_model(SUPPORTED_MODELS[DEFAULT_MODEL_NAME]) # Load the default Turkish model

    # Define the Gradio Interface
    with gr.Blocks(title="Yazıdan Sese TTS — Türkçe Metin Okuma") as demo:
        gr.Markdown("# 🔊 Yazıdan Sese TTS — Türkçe Metin Okuma")
        gr.Markdown(
            "Bu uygulama, Türkçe metinleri konuşmaya dönüştürmek için `facebook/mms-tts-tur` VITS modelini kullanır. "
            "Metninizi girin, tohum değerini ayarlayın ve sesi oluşturmak için 'Sentezle' düğmesine tıklayın."
        )

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                value=DEFAULT_MODEL_NAME,
                label="Model Seçimi",
                allow_custom_value=True,
                info="Listeden seçin veya HuggingFace model ID girin (örn: facebook/mms-tts-tur)."
            )
            voice_dropdown = gr.Dropdown(
                choices=list(VOICE_SAMPLES.keys()),
                value="Kadın (Varsayılan)",
                label="Ses Seçimi",
                info="Çok konuşmacılı modeller için ses tonu seçin. Tek konuşmacılı modellerde bu ayar etkisizdir."
            )
        
        text_input = gr.Textbox(
                label="Metin (Türkçe)",
                lines=10,
                placeholder=(
                    "Merhaba dünya! Bu bir test cümlesidir. "
                    "Lütfen buraya sentezlemek istediğiniz Türkçe metni girin."
                ),
                info=f"Maksimum {MAX_CHARS} karakter."
            )
        char_count_display = gr.Markdown(update_char_count(""))

        with gr.Row():
            seed_slider = gr.Slider(
                minimum=0,
                maximum=10000,  # Increased max seed for more variety if needed
                step=1,
                value=42,
                label="Tohum (Seed)",
                info="Sabit bir tohum değeri, aynı metin için her zaman aynı sesin üretilmesini sağlar."
            )
            rate_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                step=0.1,
                value=1.0,
                label="Konuşma Hızı",
                info="1.0 normal hızdır. Daha yüksek değerler konuşmayı hızlandırır, daha düşük değerler yavaşlatır."
            )
            pitch_slider = gr.Slider(
                minimum=-1.0,
                maximum=1.0,
                step=0.1,
                value=0.0,
                label="Ses Tonu (Pitch)",
                info="Sesin kalınlığını veya inceliğini ayarlar. Pozitif değerler sesi inceltir."
            )
        
        with gr.Row():
            synthesize_button = gr.Button("Sentezle")
            clear_button = gr.ClearButton(value="Temizle")

        audio_output = gr.Audio(
                label="Sesi Dinle", type="numpy"
            )
        file_output = gr.File(
                label="MP3 Olarak İndir"
            )

        # Event Handlers
        text_input.change(update_char_count, inputs=[text_input], outputs=[char_count_display])
        
        synthesize_button.click(
            synthesize_for_gradio,
            inputs=[text_input, model_dropdown, voice_dropdown, seed_slider, rate_slider, pitch_slider],
            outputs=[audio_output, file_output]
        )
        
        clear_button.add(
            [
                text_input, char_count_display, audio_output, file_output,
                model_dropdown, voice_dropdown, seed_slider, rate_slider, pitch_slider
            ]
        )

    demo.launch()
