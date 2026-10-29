# utils.py
"""
Utility functions for Arabic-English text processing
"""

import re
from typing import Dict


def detect_language(text: str) -> str:
    """
    Detect if text is primarily Arabic or English.

    Args:
        text: Input text

    Returns:
        'ar' for Arabic, 'en' for English
    """
    if not text:
        return "en"

    # Count Arabic characters
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    # Count English characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))

    # If more than 30% are Arabic characters, consider it Arabic
    total_chars = len(text)
    if total_chars > 0 and (arabic_chars / total_chars) > 0.3:
        return "ar"

    return "en"


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for better processing.

    Args:
        text: Arabic text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove tatweel (ـ)
    text = re.sub(r'ـ', '', text)

    # Normalize Arabic letters
    text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef
    text = re.sub(r'ى', 'ي', text)  # Normalize Ya
    text = re.sub(r'ة', 'ه', text)  # Normalize Ta Marbuta

    # Remove diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean and normalize text (works for both Arabic and English).

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Detect language
    lang = detect_language(text)

    # Apply Arabic normalization if needed
    if lang == "ar":
        text = normalize_arabic_text(text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)

    return text.strip()


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length in characters
        add_ellipsis: Add ... at the end

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncated = text[:max_length]

    if add_ellipsis:
        truncated += "..."

    return truncated


def format_response(response: Dict, language: str = None) -> Dict:
    """
    Format response based on detected or specified language.

    Args:
        response: Response dictionary
        language: 'ar' or 'en' (auto-detect if None)

    Returns:
        Formatted response
    """
    if language is None:
        # Auto-detect from answer
        language = detect_language(response.get("answer", ""))

    # Add language field
    response["language"] = language

    # Add appropriate labels
    if language == "ar":
        response["labels"] = {
            "answer": "الإجابة",
            "sources": "المصادر",
            "confidence": "مستوى الثقة",
            "query": "السؤال"
        }
    else:
        response["labels"] = {
            "answer": "Answer",
            "sources": "Sources",
            "confidence": "Confidence",
            "query": "Query"
        }

    return response