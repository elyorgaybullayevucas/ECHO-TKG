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
  # text width of the main document: 7in minus nothing (full width) for
  # starred floats, \linewidth of one column (3.4in) for the others
  case "$n" in fig_overview) width=7in;; *) width=3.42in;; esac
  cat > _fig.tex <<EOF
\documentclass[border=1pt]{standalone}
\input{_preamble}
\begin{document}
\begin{minipage}{$width}\centering\footnotesize
\input{figures/$n}
\end{minipage}
\end{document}
EOF
  docker run --rm -v "$W:/work" -w /work echo-tex:1 sh -c \
    "pdflatex -interaction=nonstopmode _fig.tex >/dev/null 2>&1 && mv _fig.pdf figures/$n.pdf && echo 'ok   $n' || (echo 'FAIL $n'; grep -nE '^! ' _fig.log | head -3)"
done
rm -f _fig.*
