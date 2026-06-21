import logging
import os

logger = logging.getLogger(__name__)

def get_directory_size(path: str) -> float:
    """Calculates the total size of files in a given directory in MB."""
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024) # Convert bytes to MB

def clear_oldest_cache_files(directory: str, max_size_mb: float):
    """
    Clears the oldest files from the cache directory if its size exceeds max_size_mb.
    Files are deleted until the cache size is below the threshold.
    """
    current_size = get_directory_size(directory)
    logger.info(f"Current audio cache size: {current_size:.2f} MB (Max: {max_size_mb} MB)")

    if current_size <= max_size_mb:
        return

    logger.warning(
        f"Audio cache size ({current_size:.2f} MB) exceeds limit ({max_size_mb} MB). Clearing oldest files..."
    )

    files: list[tuple[float, str]] = [] # List of (modification_time, filepath)
    for dirpath, _dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                files.append((os.path.getmtime(fp), fp))
    
    files.sort() # Sort by modification time (oldest first)

    for _mtime, filepath in files:
        if current_size <= max_size_mb:
            break
        try:
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            os.remove(filepath)
            current_size -= file_size_mb
            logger.info(
                f"Deleted old cache file: {filepath} (Size: {file_size_mb:.2f} MB). "
                f"New cache size: {current_size:.2f} MB"
            )
        except OSError as e:
            logger.error(f"Error deleting cache file {filepath}: {e}")
    
    logger.info(f"Audio cache clearing complete. Final size: {current_size:.2f} MB")