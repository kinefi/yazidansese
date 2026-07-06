from app.cache_management import clear_oldest_cache_files
from app.config import CACHE_AUDIO_DIR, DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAMES, MAX_CACHE_SIZE_MB, SUPPORTED_MODELS
from app.model_service import load_model  # Assuming load_model now accepts progress
from app.text_processing import setup_nltk
from app.ui import render_interface
from app.logger import logger

# Download NLTK data for sentence tokenization
setup_nltk()


# ── Gradio Application ────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Warm up the default model on startup
    # This initial load doesn't have a Gradio progress bar, so it will log to console.
    
    # Clear cache on startup
    clear_oldest_cache_files(CACHE_AUDIO_DIR, MAX_CACHE_SIZE_MB)

    # Variable to store the model ID that was successfully loaded at startup
    initial_loaded_model_id = None

    try:
        load_model(SUPPORTED_MODELS[DEFAULT_MODEL_NAME])
        initial_loaded_model_id = DEFAULT_MODEL_NAME
        logger.info(
            f"Default model '{DEFAULT_MODEL_NAME}' warmed up successfully.")
    except Exception as e:
        logger.error(
            f"Failed to warm up default model '{DEFAULT_MODEL_NAME}': {str(e)}")
        logger.info("Attempting to load fallback models...")
        for fallback_name in FALLBACK_MODEL_NAMES:
            try:
                fallback_model_id = SUPPORTED_MODELS.get(fallback_name, fallback_name)
                load_model(fallback_model_id)
                initial_loaded_model_id = fallback_name
                logger.info(f"Fallback model '{fallback_name}' warmed up successfully.")
                break # Stop at the first successful fallback
            except Exception as fe:
                logger.warning(f"Failed to load fallback model '{fallback_name}': {str(fe)}")
    
    # Pass the initially loaded model ID to the UI to set the default dropdown value
    # Assign the Blocks object to a 'demo' variable for Gradio's hot-reloader
    demo = render_interface(initial_loaded_model_id=initial_loaded_model_id)
    demo.launch()
