import logging

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from .config import DEVICE

logger = logging.getLogger(__name__)

def synthesize_chunks(
    chunks: list[str],
    model: VitsModel,
    tokenizer: AutoTokenizer,
    seed: int,
    speaker_id: int = None
) -> np.ndarray:
    """
    Run inference per chunk, concatenate waveforms.
    Handles potential errors during synthesis.
    """
    torch.manual_seed(seed)
    segments: list[np.ndarray] = []

    try:
        with torch.no_grad():
            for chunk in chunks:
                if not chunk.strip():
                    continue
                inputs = tokenizer(chunk, return_tensors="pt").to(DEVICE)
                
                if speaker_id is not None and hasattr(model.config, 'num_speakers') and model.config.num_speakers > 1:
                    speaker_ids = torch.LongTensor([speaker_id]).to(DEVICE)
                    output = model(**inputs, speaker_ids=speaker_ids)
                else:
                    output = model(**inputs) 
                
                audio = output.waveform.squeeze().cpu().numpy()
                segments.append(audio)
                segments.append(np.zeros(int(model.config.sampling_rate * 0.2), dtype=np.float32)) # Add a short pause
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU Out of Memory during synthesis.")
        raise RuntimeError("The GPU ran out of memory. Try shorter text or a smaller model.") from None
    except Exception as exc:
        logger.exception("Synthesis failed.")
        raise RuntimeError(f"An error occurred during audio generation: {str(exc)}") from exc

    return np.concatenate(segments).astype(np.float32)