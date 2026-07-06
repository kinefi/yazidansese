import nltk
import torch

from .config import DEFAULT_MODEL_NAME, DEVICE, HF_TOKEN, SUPPORTED_MODELS
from .model_service import load_model


def run_health_check():
    """Performs various checks to verify system health."""
    status_messages = []

    # 1. Check NLTK data
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
        status_messages.append(("✅ NLTK 'punkt' and 'punkt_tab' data found.", "green"))
    except LookupError:
        status_messages.append(("❌ NLTK tokenizer data not found. "
        "Run `setup_nltk()` or `nltk.download('punkt')` and `nltk.download('punkt_tab')`.", "red"))

    # 2. Check GPU availability
    if DEVICE == "cuda":
        if torch.cuda.is_available():
            status_messages.append((f"""✅ GPU available: 
                                    {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} devices)""", "green"))
        else:
            status_messages.append(("❌ GPU not available, but CUDA device was requested. Running on CPU.", "red"))
    else:
        status_messages.append(("ℹ️ Running on CPU (GPU not requested or not available).", "yellow"))

    # 3. Check HuggingFace Token
    if HF_TOKEN:
        status_messages.append(("✅ HuggingFace Token (HF_TOKEN) found in environment.", "green"))
    else:
        status_messages.append(("⚠️ HuggingFace Token (HF_TOKEN) not found in environment. "
        "Some models might be inaccessible.", "yellow"))

    # 4. Attempt to load default model (if not already loaded)
    try:
        load_model(SUPPORTED_MODELS[DEFAULT_MODEL_NAME])
        status_messages.append((f"✅ Default model '{DEFAULT_MODEL_NAME}' loaded successfully.", "green"))
    except Exception as e:
        status_messages.append((f"❌ Failed to load default model '{DEFAULT_MODEL_NAME}': {str(e)}", "red"))

    # Format messages for Gradio Markdown
    formatted_messages = ""
    for msg, color in status_messages:
        formatted_messages += f"<p style='color: {color};'>{msg}</p>"
    
    return formatted_messages