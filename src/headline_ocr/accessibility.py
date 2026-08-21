"""Accessible, dependency-free HTML report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .models import ExtractionResult


def write_accessible_html(result: ExtractionResult, output_path: str | Path) -> Path:
    """Create a high-contrast page with browser-native text-to-speech controls."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    headline = escape(result.headline or "No headline was detected.")
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Extracted news headline</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 52rem; margin: 0 auto; padding: 2rem; line-height: 1.6; }}
    main {{ border: 3px solid currentColor; border-radius: 1rem; padding: 2rem; }}
    #headline {{ font-size: clamp(1.5rem, 4vw, 3rem); font-weight: 750; }}
    button {{ min-height: 3rem; padding: .7rem 1.2rem; font: inherit; font-weight: 700; }}
    button:focus-visible {{ outline: 4px solid #ffbf47; outline-offset: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>Extracted news headline</h1>
    <p id="headline" tabindex="0">{headline}</p>
    <button id="speak" type="button">Read headline aloud</button>
    <button id="stop" type="button">Stop</button>
    <p>
      Source duration: {result.duration_seconds:.1f} seconds.
      Extraction mode: {escape(result.mode_used)}.
    </p>
  </main>
  <script>
    const text = document.getElementById('headline').textContent;
    document.getElementById('speak').addEventListener('click', () => {{
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(text));
    }});
    document.getElementById('stop').addEventListener('click', () => speechSynthesis.cancel());
  </script>
</body>
</html>
"""
    destination.write_text(document, encoding="utf-8")
    return destination
