"""Write reports/final_report.html and print reports/final_report.pdf.

Pandoc is not required. The script uses the installed Markdown package and
headless Chrome. It does not change the Markdown source.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "final_report.md"
HTML = ROOT / "reports" / "final_report.html"
PDF = ROOT / "reports" / "final_report.pdf"

STYLE = """
body { font-family: Georgia, serif; max-width: 900px; margin: 2em auto; line-height: 1.45; color: #222; }
img { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
td, th { border: 1px solid #ccc; padding: 4px 6px; vertical-align: top; }
code, pre { font-family: ui-monospace, monospace; font-size: 0.9em; }
pre { white-space: pre-wrap; }
"""


def main() -> None:
    import markdown

    if not SOURCE.is_file():
        raise SystemExit(f"Missing report source: {SOURCE}")
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome is None:
        raise SystemExit("Headless Chrome is not installed, so the PDF was not written.")
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    HTML.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Final report</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    if PDF.exists():
        PDF.unlink()
    process = subprocess.Popen(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            "--allow-file-access-from-files",
            "--user-data-dir=/tmp/chrome-report-pdf",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            HTML.as_uri(),
        ]
    )
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    if not PDF.is_file() or PDF.stat().st_size < 1000 or not PDF.read_bytes().startswith(b"%PDF"):
        raise SystemExit(f"Chrome did not write a readable PDF at {PDF}")
    if not PDF.read_bytes().rstrip().endswith(b"%%EOF"):
        raise SystemExit(f"The PDF at {PDF} is incomplete.")
    print(f"Wrote {HTML}")
    print(f"Wrote {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError:
        sys.exit("The markdown package is required for this export and is not installed.")
