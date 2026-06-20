#!/bin/bash
echo "=== 清理缓存 ==="
rm -f main.aux main.bbl main.blg main.log main.out main.synctex.gz
echo "=== 第1次 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex
echo "=== bibtex ==="
bibtex main
echo "=== 第2次 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex
echo "=== 第3次 pdflatex ==="
pdflatex -interaction=nonstopmode main.tex
echo "=== 完成！检查 main.pdf ==="
