from headline_ocr.text import alphabetic_word_count, lexical_quality, normalize_text, words


def test_normalization_preserves_names_numbers_and_punctuation() -> None:
    assert normalize_text("  Québec's rate: 4.5%\n") == "QUÉBEC'S RATE: 4.5%"


def test_words_support_hyphenated_tokens() -> None:
    assert words("REAL-TIME OCR IS READY") == ["REAL-TIME", "OCR", "IS", "READY"]


def test_quality_does_not_require_dictionary_membership() -> None:
    assert lexical_quality("SUKHRAJ JOINS NOKIA TEAM", 0.9) > 0.8


def test_time_labels_are_not_counted_as_three_headline_words() -> None:
    assert alphabetic_word_count("8:30 PM") == 1
