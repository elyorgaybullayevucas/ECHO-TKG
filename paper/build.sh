#!/usr/bin/env bash
# Compile the paper in the echo-tex Docker image (TeX Live + pgfplots,
# multirow, cleveref) and render page PNGs for inspection.
#   ./build.sh            # pdf + pages/page-N.png
#   ./build.sh fig_overview   # render one figure standalone (figures/<name>.tex)
set -uo pipefail
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1
IMG=echo-tex:1
W="$(pwd -W 2>/dev/null || pwd)"
if [[ -n "${1:-}" ]]; then
  # standalone figure build: wrap the figure in a minimal document
  mkdir -p pages
  cat > _fig.tex <<EOF
\documentclass[varwidth=18cm,border=4pt]{standalone}
\input{_preamble}
\begin{document}\footnotesize\input{figures/$1}\end{document}
EOF
  docker run --rm -v "$W:/work" -w /work $IMG sh -c \
    "pdflatex -interaction=nonstopmode _fig.tex >/dev/null 2>&1; grep -nE '^! ' _fig.log | head; pdftoppm -r 170 -png _fig.pdf pages/$1 >/dev/null 2>&1 && echo rendered pages/$1-1.png"
  exit 0
fi
docker run --rm -v "$W:/work" -w /work $IMG sh -c "
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
  bibtex main 2>&1 | grep -iE 'warning|error' | head -5
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1
  echo '--- errors / undefined'
  grep -nE '^! |LaTeX Error|Citation .* undefined|Reference .* undefined|Label .* multiply' main.log | head -20
  echo '--- overfull > 20pt'
  grep -E 'Overfull .hbox .([2-9][0-9]|[1-9][0-9]{2,})' main.log | head -10
  echo '--- pages'
  pdfinfo main.pdf 2>/dev/null | grep Pages
  mkdir -p pages && rm -f pages/page-*.png
  pdftoppm -r 110 -png main.pdf pages/page >/dev/null 2>&1 && ls pages | head -20
"
