import numpy as np
from pydub import AudioSegment


def create_audio_segment_from_numpy(audio_numpy: np.ndarray, sample_rate: int) -> AudioSegment:
    """Converts a numpy array audio waveform to a pydub AudioSegment."""
    audio_int16 = (audio_numpy * 32767).astype(np.int16)
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_int16.dtype.itemsize,
        channels=1
    )
