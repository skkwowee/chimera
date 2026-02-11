# Makefile for NeurIPS LaTeX paper

PAPER = main
LATEX = pdflatex
BIBTEX = bibtex
LATEXMK = latexmk

.PHONY: all paper clean watch

# Default target
all: paper

# Build the paper
paper: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib neurips_2026.sty
	$(LATEX) -interaction=nonstopmode $(PAPER)
	-$(BIBTEX) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	rm -f $(PAPER).pdf

# Continuous compilation with latexmk (if available)
watch:
	@which $(LATEXMK) > /dev/null || (echo "latexmk not found. Install it with: brew install latexmk (macOS) or apt-get install latexmk (Linux)" && exit 1)
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(PAPER).tex
