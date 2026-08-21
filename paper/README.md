# Paper build

`main.tex` uses the official `IEEEtran` conference class and reads measured values from
`generated/results_macros.tex`. The committed PDF is generated with Tectonic 0.17.0.

From the repository root (PowerShell):

```powershell
news-ocr benchmark --regenerate --output results
python scripts/make_paper_figures.py
Push-Location paper
tectonic -X compile main.tex
Pop-Location
```

The benchmark regenerates its videos, predictions, aggregate tables, paired statistics, and the
LaTeX result macros. The generated videos are intentionally ignored because they are deterministic
and can be recreated from source.

The benchmark is a controlled mechanism test, not a substitute for independent evaluation on
licensed broadcast footage. The committed PDF is four pages and uses embedded Times-style text
and NewTX math fonts.
