import hashlib
import os
import shutil  # For copying files
import tempfile

import gradio as gr
import numpy as np
from pydub import AudioSegment

from app.audio_effects import create_audio_segment_from_numpy
from app.audio_synthesis import synthesize_chunks
from app.cache_management import clear_oldest_cache_files
from app.config import (
    CACHE_AUDIO_DIR,
    DEFAULT_MODEL_NAME,
    MAX_CACHE_SIZE_MB,
    MAX_CHARS,
    SUPPORTED_MODELS,
    VOICE_SAMPLES,
)
from app.health_check import run_health_check
from app.model_service import load_model
from app.text_processing import chunk_text, clean_text
from app.logger import logger


def synthesize_for_gradio(
    text: str,
    model_name: str,
    voice_type: str,
    normalize: bool,
    progress=gr.Progress()  # noqa: B008
) -> tuple[tuple[int, np.ndarray], str, str]:
    """
    Synthesizes text into audio using the loaded VITS model.
    Returns a tuple of (sample_rate, audio_numpy_array) for Gradio's Audio component.
    Returns None for the file path if synthesis fails.
    """
    if progress:
        # Initialize progress if it's None (e.g., when called without a Gradio context)
        progress = progress if progress is not None else gr.Progress()
        progress(0, desc="Initializing synthesis...")

    # Eğer model_name bir anahtar değilse (özel ID girilmişse) direkt ID olarak kullan
    model_id = SUPPORTED_MODELS.get(model_name, model_name)

    # --- Caching Logic Start ---
    # Create a unique key for the cache based on relevant inputs
    cache_key_str = f"{text}|{model_id}|{voice_type}|{normalize}"
    cache_filename = hashlib.md5(cache_key_str.encode('utf-8')).hexdigest() + ".mp3"
    cache_path = os.path.join(CACHE_AUDIO_DIR, cache_filename)

    if os.path.exists(cache_path):
        if progress:
            progress(1.0, desc="Önbellekten yükleniyor...")
        
        # Load the cached MP3 using pydub
        cached_audio_segment = AudioSegment.from_mp3(cache_path)
        sample_rate = cached_audio_segment.frame_rate
        processed_audio = np.array(cached_audio_segment.get_array_of_samples()).astype(np.float32) / 32767.0
        
        return (
            (sample_rate, processed_audio),
            gr.update(value=cache_path, interactive=True),
            "<p style='color: green;'>✅ Önbellekten yüklendi.</p>",
        )
    # --- Caching Logic End ---

    try:
        model, tokenizer = load_model(model_id, progress=progress) # Pass progress to model_service
    except Exception as e:
        raise gr.Error(f"Model yüklenemedi: {str(e)}") from e

    sample_rate = model.config.sampling_rate

    if len(text) > MAX_CHARS:
        raise gr.Error(
            f"Text too long! Please keep it under {MAX_CHARS} characters.")

    cleaned_text = clean_text(text, normalize=normalize)
    if not cleaned_text:
        raise gr.Error(
            "Text is empty after cleaning. Please provide valid Turkish text.")

    # Use a reasonable max_chars for chunking
    chunks = chunk_text(cleaned_text, max_chars=500)
    logger.info("chars=%d  chunks=%d", len(cleaned_text), len(chunks))

    speaker_idx = VOICE_SAMPLES.get(voice_type)
    audio_numpy = synthesize_chunks(
        progress.tqdm(chunks, desc="Sentezleniyor...") if progress else chunks,
        model,
        tokenizer,
        42, # Fixed seed for reproducibility
        speaker_id=speaker_idx
    )

    audio_segment = create_audio_segment_from_numpy(audio_numpy, sample_rate)
    
    # Export to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", prefix="sentez_") as temp_output_file:
        audio_segment.export(temp_output_file.name, format="mp3")
        temp_file_path = temp_output_file.name

    # --- Caching Logic: Save to cache ---
    shutil.copy(temp_file_path, cache_path)
    os.remove(temp_file_path)
    logger.info(f"Audio cached to {cache_path}. Checking cache size...")
    clear_oldest_cache_files(CACHE_AUDIO_DIR, MAX_CACHE_SIZE_MB) # Clear cache after adding new file
    # --- Caching Logic End ---

    return (
        (sample_rate, audio_numpy),
        gr.update(value=cache_path, interactive=True),
        "<p style='color: blue;'>✨ Yeni sentezlendi ve önbelleğe alındı.</p>",
    )
def update_char_count(text: str) -> str:
    """Updates the character count display."""
    current_chars = len(text)
    remaining_chars = MAX_CHARS - current_chars
    color = "green" if remaining_chars >= 0 else "red"
    return (
        f"<p style='color: {color}; font-weight: bold;'>"
        f"Karakter Sayısı: {current_chars} / {MAX_CHARS} (Kalan: {remaining_chars})</p>"
    )


def render_interface(initial_loaded_model_id: str = None):
    """Renders the Gradio interface."""
    # Define the Gradio Interface
    with gr.Blocks(title="Yazıdan Sese TTS — Türkçe Metin Okuma") as demo:
        gr.Markdown("# 🔊 Yazıdan Sese TTS — Türkçe Metin Okuma")
        gr.Markdown(
            "Bu uygulama, Türkçe metinleri konuşmaya dönüştürmek için `facebook/mms-tts-tur` VITS modelini kullanır. "
            "Metninizi girin, tohum değerini ayarlayın ve sesi oluşturmak için 'Sentezle' düğmesine tıklayın."
        )

        with gr.Tab("Metin Sentezi"):
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_MODELS.keys()),
                    value=(
                        initial_loaded_model_id
                        if initial_loaded_model_id
                        else DEFAULT_MODEL_NAME
                    ),  # Set initial value based on successful warm-up
                    label="Model Seçimi",
                    allow_custom_value=True,
                    info="Listeden seçin veya HuggingFace model ID girin (örn: facebook/mms-tts-tur)."
                )
                voice_dropdown = gr.Dropdown(
                    choices=list(VOICE_SAMPLES.keys()),
                    value="Kadın (Deneysel)", # Corrected to match an existing key in VOICE_SAMPLES
                    label="Ses Seçimi",
                    info="Çok konuşmacılı modeller için ses tonu seçin. Tek konuşmacılı modellerde bu ayar etkisizdir."
                )
            
            normalize_checkbox = gr.Checkbox(
                label="Metin Normalizasyonu", 
                value=True, 
                info="Sayıları yazıya çevirir ve kısaltmaları açar."
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
                synthesize_button = gr.Button("Sentezle", variant="primary") # Make Sentezle button primary
                clear_button = gr.ClearButton(value="Temizle")
                download_button = gr.DownloadButton("MP3 İndir", interactive=False) # Initially disabled
            
            cache_status_indicator = gr.Markdown("") # Visual indicator for cache status

            audio_output = gr.Audio(label="Sesi Dinle", type="numpy") # Now in its own row implicitly

            synthesize_button.click(
                synthesize_for_gradio,
                inputs=[text_input, model_dropdown, voice_dropdown, normalize_checkbox],
                outputs=[audio_output, download_button, cache_status_indicator]
            )
            clear_button.add([
                text_input,
                char_count_display,
                audio_output,
                download_button,
                model_dropdown,
                voice_dropdown,
                normalize_checkbox,
                cache_status_indicator,
            ])

        with gr.Tab("Sistem Durumu"):
            gr.Markdown(
                "Uygulamanın ve sistem kaynaklarının durumunu kontrol edin.")
            health_check_button = gr.Button("Durum Kontrolü Yap")
            health_check_output = gr.Markdown()
            health_check_button.click(
                run_health_check, outputs=health_check_output)
        return demo
