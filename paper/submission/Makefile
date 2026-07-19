# make iclr | make neurips  (pdflatex+bibtex loop)
TEX=pdflatex -interaction=nonstopmode
iclr:
	$(TEX) main_iclr2027 && bibtex main_iclr2027 || true && $(TEX) main_iclr2027 && $(TEX) main_iclr2027
neurips:
	$(TEX) main_neurips2027 && bibtex main_neurips2027 || true && $(TEX) main_neurips2027 && $(TEX) main_neurips2027
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf
