# Temporal-Consensus OCR for News Headlines

A reproducible, CPU-oriented system for extracting a known news lower-third and reconstructing a
scrolling ticker across video frames. It replaces the original framewise Tesseract prototype with
PP-OCRv6 line recognition, resolution-independent regions of interest, confidence-aware filtering,
static consensus, and chronological fuzzy-overlap assembly.

The accompanying four-page manuscript follows the IEEE conference layout:
**[read the paper](paper/main.pdf)** or inspect the [LaTeX source](paper/main.tex).

> Research boundary: the committed results come from a deterministic 50-clip synthetic benchmark.
> They validate the temporal assembly mechanism; they do not establish accuracy on unconstrained
> television broadcasts. The paper is IEEE-formatted but has not been peer reviewed or accepted by
> IEEE.

## Measured results

All three conditions use identical sampled videos. Success means normalized character similarity
of at least 0.80; precision and recall include ten negative controls.

| Method | CER ↓ | WER ↓ | Exact ↑ | Success ↑ | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Center frame | 12.34% | 26.33% | 37.50% | 50.00% | 100.00% | 50.00% | 66.67% |
| Center + preprocessing | 12.34% | 26.83% | 37.50% | 50.00% | 100.00% | 50.00% | 66.67% |
| **Temporal consensus** | **2.10%** | **9.67%** | **70.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

The paired success-rate improvement is 50 percentage points (5,000-sample paired-bootstrap 95%
CI: 35–65 points; exact two-sided McNemar p = 1.91 × 10⁻⁶). Full predictions and stratified tables
are committed in [`results/`](results/).

## How it works

```text
video → normalized headline ROI → raw / CLAHE / Otsu hypotheses
      → PP-OCRv6 recognition → confidence + lexical filtering
      → static medoid or ticker assembly → JSON + accessible HTML
```

- Static overlays use a quality-weighted medoid across repeated observations.
- Scrolling tickers use timestamp-ordered fuzzy suffix/prefix overlap to recover words that no
  single frame contains.
- Dictionary-free lexical checks preserve names and acronyms while suppressing short channel
  labels and timestamps.
- The accessible report keeps text visible for inspection and uses local browser speech synthesis;
  extracted text is not sent to a cloud or language-model service.

## Install

Python 3.10 or newer is required.

```powershell
git clone https://github.com/Mahal-Raj/NewsHeadlinesExtractor_OCR.git
Set-Location NewsHeadlinesExtractor_OCR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Extract a headline

The ROI is `x,y,width,height` in normalized frame coordinates. Auto mode distinguishes repeated
static text from a moving ticker.

```powershell
news-ocr extract .\video.mp4 `
  --roi "0.02,0.64,0.96,0.32" `
  --sample-fps 2 `
  --mode auto `
  --output outputs\headline.json `
  --accessible-html outputs\headline.html
```

The command prints the recovered headline, writes a machine-readable record of all retained OCR
candidates, and optionally creates a standalone read-aloud page. Use `--mode static` or
`--mode ticker` when the overlay type is known.

## Reproduce the experiment

```powershell
news-ocr benchmark --regenerate --output results
python scripts\make_paper_figures.py
python -m pytest
ruff check .
```

The fixed seed and every rendering parameter are in
[`benchmark_artifacts/manifest.json`](benchmark_artifacts/manifest.json). Generated videos are
ignored because the benchmark recreates them exactly. The benchmark writes per-clip predictions,
aggregate and stratified CSV files, runtime metadata, paired statistics, and the LaTeX result
macros consumed by the paper.

## Repository map

- `src/headline_ocr/` — extraction, preprocessing, OCR, temporal aggregation, metrics, and CLI
- `tests/` — unit tests for ROI contracts, normalization, metrics, and temporal reconstruction
- `results/` — complete benchmark measurements and predictions
- `paper/` — IEEEtran manuscript, bibliography, generated figure, and compiled PDF
- `scripts/make_paper_figures.py` — regenerates the ticker-sequence figure from benchmark clips

## Current limitations

The system expects a supplied, mostly single-line headline band. It has not yet been evaluated on
a licensed real-broadcast corpus, multilingual tickers, vertical text, automatic ROI detection, or
blind and low-vision participants. OCR mistakes in names, numbers, or emergency messages can alter
meaning, so outputs should be shown with source context and treated as machine transcription.

## Author

Sukhraj Singh — COSC 4086, Faculty of Computer Science and Technology, Algoma University.



