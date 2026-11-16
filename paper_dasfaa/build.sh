#!/usr/bin/env bash
# Local build of the DASFAA version (Overleaf needs only main.tex, references.bib, figures/).
set -uo pipefail
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1
W="$(pwd -W 2>/dev/null || pwd)"
docker run --rm -v "$W:/work" -w /work echo-tex:1 sh -c "pdflatex -interaction=nonstopmode main >/dev/null 2>&1; bibtex main >/dev/null 2>&1; pdflatex -interaction=nonstopmode main >/dev/null 2>&1; pdflatex -interaction=nonstopmode main >/dev/null 2>&1; grep -nE '^! ' -A3 main.log | head -20; grep -c 'undefined' main.log; grep -E 'Overfull .hbox' main.log | head -8; pdfinfo main.pdf | grep Pages; rm -rf pages; mkdir pages; pdftoppm -r 70 -png main.pdf pages/p; pdftotext main.pdf - | grep -c . "
