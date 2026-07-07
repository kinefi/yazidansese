from app.cache_management import clear_oldest_cache_files
from app.config import CACHE_AUDIO_DIR, MAX_CACHE_SIZE_MB, MODEL_ID
from app.model_service import load_model
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
        load_model(MODEL_ID)
        initial_loaded_model_id = MODEL_ID
        logger.info(f"Default model '{MODEL_ID}' warmed up successfully.")
    except Exception as e:
        logger.error(f"Failed to warm up default model '{MODEL_ID}': {str(e)}")
        initial_loaded_model_id = None
    
    # Pass the initially loaded model ID to the UI to set the default dropdown value
    # Assign the Blocks object to a 'demo' variable for Gradio's hot-reloader
    demo = render_interface(initial_loaded_model_id=initial_loaded_model_id)
    demo.launch()
