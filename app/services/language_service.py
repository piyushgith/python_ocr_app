from collections import Counter

from lingua import Language, LanguageDetectorBuilder

MIN_LETTERS = 20
MIN_SCRIPT_SHARE = 0.1
MAX_SAMPLE_CHARS = 20_000

_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))

_latin_detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()


def detect_languages(text: str) -> list[str]:
    """Languages (en, fr, ar) found in `text`, most letters first. Empty when the text has too few letters to tell."""
    return [code for code, _ in language_letter_counts(text).most_common()]


def language_letter_counts(text: str) -> Counter[str]:
    """Letters per detected language (en, fr, ar). Empty when the text has too few letters to tell.

    Arabic is found by script. English vs French is decided by lingua on the Latin-script part.
    A script with under MIN_SCRIPT_SHARE of the letters is ignored as noise.
    """
    sample = text[:MAX_SAMPLE_CHARS]
    arabic_letters = sum(1 for char in sample if char.isalpha() and _is_arabic(char))
    latin_letters = sum(1 for char in sample if char.isalpha() and ord(char) < 0x250)
    total = arabic_letters + latin_letters
    counts: Counter[str] = Counter()
    if total < MIN_LETTERS:
        return counts

    if arabic_letters / total >= MIN_SCRIPT_SHARE:
        counts["ar"] = arabic_letters
    if latin_letters / total >= MIN_SCRIPT_SHARE:
        latin_text = "".join(" " if _is_arabic(char) else char for char in sample)
        language = _latin_detector.detect_language_of(latin_text)
        if language is not None:
            counts[language.iso_code_639_1.name.lower()] = latin_letters
    return counts


def _is_arabic(char: str) -> bool:
    code_point = ord(char)
    return any(start <= code_point <= end for start, end in _ARABIC_RANGES)
