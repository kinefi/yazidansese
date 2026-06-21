import glob
import logging
import os
import tempfile
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_temp_audio_files(age_threshold_seconds: int = 3600):
    """
    Sistemin geçici dizinindeki belirli önekli ve uzantılı ses dosyalarını temizler.
    Belirtilen yaş eşiğinden daha eski olan dosyaları siler.
    """
    temp_dir = tempfile.gettempdir()
    current_time = time.time()
    
    search_pattern = os.path.join(temp_dir, "sentez_*.mp3")
    
    logger.info(f"Geçici dizin taranıyor: {temp_dir} (Desen: {search_pattern})")
    
    for filepath in glob.glob(search_pattern):
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > age_threshold_seconds:
                try:
                    os.remove(filepath)
                    logger.info(f"Silindi: {filepath} (Yaş: {file_age:.1f} saniye)")
                except OSError as e:
                    logger.error(f"Dosya silinirken hata oluştu {filepath}: {e}")

if __name__ == "__main__":
    # 1 saatten (3600 saniye) eski dosyaları temizle
    cleanup_temp_audio_files(age_threshold_seconds=3600)
    logger.info("Geçici ses dosyası temizleme işlemi tamamlandı.")