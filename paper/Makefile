PAPER=paper
TEX=pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean tectonic

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(TEX) $(PAPER).tex
	bibtex $(PAPER)
	$(TEX) $(PAPER).tex
	$(TEX) $(PAPER).tex

tectonic: $(PAPER).tex references.bib
	tectonic $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc
