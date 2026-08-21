"""Compatibility entry point; prefer the installed ``news-ocr`` command."""

from headline_ocr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
