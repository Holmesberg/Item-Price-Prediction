"""Render report/report.md to report/report.pdf.

Substitutes {{MODEL_COMPARISON_TABLE}} with the live results table, then
renders via pandoc + xelatex if available, or falls back to weasyprint.

Usage: python -m src.render_report
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .config import REPORT_DIR, RESULTS_DIR


def _table_md(comp: pd.DataFrame) -> str:
    cols = ["model", "cv_rmse_mean", "cv_rmse_std", "oof_rmse", "fit_seconds"]
    show = comp[cols].copy()
    header = "| " + " | ".join(show.columns) + " |"
    sep = "|" + "|".join(["---"] * len(show.columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in show.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def _substitute_placeholder() -> Path:
    src = REPORT_DIR / "report.md"
    raw = src.read_text(encoding="utf-8")
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path).sort_values("cv_rmse_mean")
        block = _table_md(comp)
        raw = raw.replace("```\n{{MODEL_COMPARISON_TABLE}}\n```", block)
    else:
        raw = raw.replace(
            "{{MODEL_COMPARISON_TABLE}}",
            "(model_comparison.csv not found — run `python -m src.train` first.)",
        )
    rendered = REPORT_DIR / "_report_rendered.md"
    rendered.write_text(raw, encoding="utf-8")
    return rendered


def _try_pandoc(md: Path, pdf: Path) -> bool:
    if shutil.which("pandoc") is None:
        return False
    cmd = [
        "pandoc",
        str(md),
        "-o",
        str(pdf),
        "--from=markdown",
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=2.5cm",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  pandoc render failed: {getattr(e, 'stderr', e)}")
        return False


def _try_weasyprint(md: Path, pdf: Path) -> bool:
    try:
        import markdown as md_lib
        from weasyprint import HTML
    except ImportError as e:
        print(f"  weasyprint stack missing: {e}")
        return False
    text = md.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :].lstrip()
    body = md_lib.markdown(
        text, extensions=["tables", "fenced_code", "attr_list"]
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<style>
  @page {{ size: A4; margin: 2.5cm; }}
  body {{ font-family: 'DejaVu Sans', sans-serif; font-size: 11pt;
          line-height: 1.4; color: #222; }}
  h1 {{ font-size: 17pt; margin-top: 1.5em; }}
  h2 {{ font-size: 14pt; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #aaa; padding: 4px 6px; font-size: 10pt; }}
  th {{ background: #eaeef5; }}
  img {{ max-width: 90%; display: block; margin: 1em auto; }}
  blockquote {{ background: #fff7e6; border-left: 4px solid #f0a500;
                padding: 0.5em 1em; margin: 1em 0; }}
  code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 10pt; }}
</style></head><body>{body}</body></html>"""
    html_path = pdf.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(REPORT_DIR)).write_pdf(str(pdf))
    return True


def main() -> None:
    md = _substitute_placeholder()
    pdf = REPORT_DIR / "report.pdf"
    if _try_pandoc(md, pdf):
        print(f"rendered {pdf} via pandoc+xelatex")
        return
    if _try_weasyprint(md, pdf):
        print(f"rendered {pdf} via weasyprint")
        return
    raise RuntimeError(
        "Both pandoc and weasyprint failed. Install one of:\n"
        "  - sudo apt install pandoc texlive-xetex texlive-fonts-recommended\n"
        "  - pip install --user --break-system-packages weasyprint markdown"
    )


if __name__ == "__main__":
    main()
