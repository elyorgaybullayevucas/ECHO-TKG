#!/usr/bin/env bash
# Pre-render every figures/*.tex to figures/*.pdf (standalone, cropped).
# main.tex includes the PDF when it exists and falls back to the TikZ
# source otherwise, so Overleaf compiles in seconds on the free plan while
# the sources stay editable.
set -uo pipefail
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1
W="$(pwd -W 2>/dev/null || pwd)"
for f in figures/*.tex; do
  n=$(basename "$f" .tex)
  case "$n" in
    # wide TikZ drawings define their own size; \includegraphics scales
    # them to \textwidth in the paper
    fig_overview|fig_intro|fig_framework) open=""; close="";;
    # pgfplots figures use \linewidth, so give them one column's width
    *) open='\begin{minipage}{3.42in}\centering\footnotesize'; close='\end{minipage}';;
  esac
  cat > _fig.tex <<EOF
\documentclass[border=1pt]{standalone}
\input{_preamble}
\begin{document}
$open
\input{figures/$n}
$close
\end{document}
EOF
  docker run --rm -v "$W:/work" -w /work echo-tex:1 sh -c \
    "pdflatex -interaction=nonstopmode _fig.tex >/dev/null 2>&1 && mv _fig.pdf figures/$n.pdf && echo \"ok   $n  \$(pdfinfo figures/$n.pdf | grep 'Page size' | cut -c20-)\" || (echo 'FAIL $n'; grep -nE '^! ' _fig.log | head -3)"
done
rm -f _fig.*
