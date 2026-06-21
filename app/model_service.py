import logging
import time
from functools import lru_cache

import requests  # To catch connection errors
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer, VitsModel

from .config import CACHE_DIR, DEVICE, HF_TOKEN

logger = logging.getLogger(__name__)

# ── Model loading (singleton) ─────────────────────────────────────────────────

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10), # Wait 4s, 8s, 10s (max)
    stop=stop_after_attempt(3), # Try 3 times
    retry=(retry_if_exception_type(requests.exceptions.ConnectionError) |
           retry_if_exception_type(HfHubHTTPError)),
    reraise=True # Re-raise the last exception if all retries fail
)
@lru_cache(maxsize=5) # Cache models to avoid re-loading
def load_model(model_id: str, progress=None) -> tuple[VitsModel, AutoTokenizer]:
    try:
        logger.info("Attempting to load %s on %s using cache %s …", model_id, DEVICE, CACHE_DIR)
        t0 = time.monotonic()
        
        # Conditionally pass the token only if it's not an empty string
        token_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        
        if progress:
            progress(0, desc=f"Downloading tokenizer for {model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            use_fast=False,  # MMS/VITS models often require the slow tokenizer (SentencePiece)
            trust_remote_code=True,
            **token_kwargs
        )
        model = VitsModel.from_pretrained(
            model_id, cache_dir=CACHE_DIR, trust_remote_code=True, **token_kwargs
        ).to(DEVICE)


        if progress:
            progress(0.5, desc=f"Loading model for {model_id} into memory...")

        model.eval()
        
        logger.info("Model loaded in %.1fs", time.monotonic() - t0)
        return model, tokenizer
    except RepositoryNotFoundError:
        logger.error(f"Model ID '{model_id}' not found on HuggingFace Hub.")
        raise ValueError(f"Model '{model_id}' not found. Please check the ID.") from None
    except GatedRepoError:
        logger.error(f"Access to model '{model_id}' is restricted.")
        raise PermissionError(f"Access denied to '{model_id}'. Check your HF_TOKEN.") from None
    except requests.exceptions.ConnectionError as e:
        logger.warning(f"Connection error while loading model {model_id}: {e}. Retrying...")
        raise # Re-raise to trigger tenacity retry
    except HfHubHTTPError as e:
        logger.warning(f"HuggingFace Hub HTTP error while loading model {model_id}: {e}. Retrying...")
        raise # Re-raise to trigger tenacity retry
    except Exception as e:
        logger.exception(f"Unexpected error loading model {model_id}")
        raise RuntimeError(f"Could not load TTS model: {str(e)}") from e