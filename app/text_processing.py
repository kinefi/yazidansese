import json
import logging
import os
import re
from functools import lru_cache

import nltk
import requests
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from num2words import num2words  # New import for number-to-word conversion

from .constants import ACRONYMS_URL

# Set seed for reproducible language detection
DetectorFactory.seed = 42
logger = logging.getLogger(__name__)

def _load_acronyms_from_json(file_path: str) -> dict:
    """Loads acronyms from a JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"Acronym file not found at {file_path}. Using an empty dictionary.")
        return {}
    try:
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading acronyms from {file_path}: {e}. Using an empty dictionary.")
        return {}

@lru_cache(maxsize=1)
def _load_acronyms_from_url(url: str) -> dict:
    """Loads acronyms from a remote JSON file with caching."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Error fetching or decoding acronyms from {url}: {e}. Using an empty dictionary.")
        return {}

def expand_acronyms(text: str, acronym_dict: dict) -> str:
    """Expands common acronyms based on a predefined dictionary."""
    # Sort acronyms by length in descending order to handle cases like "ABD" before "AB"
    sorted_acronyms = sorted(acronym_dict.keys(), key=len, reverse=True)
    for acronym in sorted_acronyms:
        if not acronym:
            continue
        
        escaped = re.escape(acronym)
        start_boundary = r'\b' if acronym[0].isalnum() else r'(?<!\w)'
        end_boundary = r'\b' if acronym[-1].isalnum() else r'(?!\w)'
        pattern = start_boundary + escaped + end_boundary
        
        # Determine case-sensitivity:
        # - Acronyms starting with an uppercase letter (TBMM, Dr., Bul.) → case-sensitive.
        #   This prevents lowercase words like 'bul' matching 'Bul.' → 'Bulvarı'.
        # - Purely lowercase abbreviations (vb., age., bkz.) → case-insensitive so
        #   they match regardless of how the user typed them.
        starts_upper = acronym[0].isupper() if acronym[0].isalpha() else False
        flags = 0 if starts_upper else re.IGNORECASE
        text = re.sub(pattern, acronym_dict[acronym], text, flags=flags)
    return text

# Download NLTK data for sentence tokenization
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

def detect_language(text: str) -> str:
    """Detects the language of the text, falling back to 'tr'."""
    if not text or not text.strip():
        return "tr"
    try:
        return detect(text)
    except LangDetectException:
        return "tr"

def convert_numbers_to_words(text: str, lang: str = 'tr') -> str:
    """Converts numbers in text to their word representation."""
    # Find all numbers (integers and floats) in the text
    # This regex matches integers, decimals, and numbers with commas (Turkish style)
    def replace_number(match):
        raw_num = match.group(0)
        
        # Handle language specific number formatting
        if lang == 'en':
            # In English, commas are usually thousands separators (1,000 -> 1000)
            num_str = raw_num.replace(',', '')
        else:
            # In Turkish, commas are decimal separators (1,5 -> 1.5)
            num_str = raw_num.replace(',', '.')

        try:
            if '.' in num_str:
                # Handle floats
                integer_part, decimal_part = num_str.split('.')
                integer_words = num2words(int(integer_part), lang=lang)
                decimal_words = num2words(int(decimal_part), lang=lang) if decimal_part else ""
                
                # Language-specific decimal separator pronunciation
                sep = " virgül " if lang == "tr" else " point "
                return f"{integer_words}{sep}{decimal_words}" if decimal_words else integer_words
            else:
                # Handle integers
                return num2words(int(num_str), lang=lang)
        except (ValueError, NotImplementedError):
            return match.group(0) # Return original if conversion fails
    
    # Regex to find numbers, including those with commas as decimal separators
    # It's important to handle numbers that might be part of other words or punctuation carefully.
    # This regex tries to capture standalone numbers.
    text = re.sub(r'\b\d+([.,]\d+)?\b', replace_number, text)
    return text

def clean_text(text: str, normalize: bool = True) -> str:
    """Normalize whitespace and remove characters VITS can't handle."""
    if normalize:
        # 0. Detect language for context-aware normalization
        lang = detect_language(text)
        
        # Load acronyms for normalization (prefer local acronyms.json)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_acronyms_path = os.path.join(project_root, "acronyms.json")
        acronym_dict = _load_acronyms_from_json(local_acronyms_path)
        if not acronym_dict:
            acronym_dict = _load_acronyms_from_url(ACRONYMS_URL)

        # 1. Expand acronyms (always apply — langdetect can misclassify short Turkish texts)
        if acronym_dict:
            text = expand_acronyms(text, acronym_dict)
        
        # 2. Convert numbers to words
        text = convert_numbers_to_words(text, lang=lang)

    # 3. General cleaning (existing logic)
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