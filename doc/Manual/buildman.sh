
# Generate PDF and copy to gui folder for inclusion in embedded executable
# Note: convert Mac screenshots to better dpi using:
# convert -density 150 -units pixelsperinch oldfile.png newfile.png

pandoc manual.md -N --toc --include-in-header header.tex --filter pandoc-fignos --variable subparagraph --variable geometry="margin=1in" --variable fontfamily=sans --filter pandoc-citeproc --bibliography=biblio.bib --csl=biblio.csl -o ../SUNCALmanual.pdf

# For an html version:
#pandoc manual.md -N --toc --css manual.css --standalone --self-contained --webtex https://latex.codecogs.com/svg.latex? --filter pandoc-citeproc --bibliography=biblio.bib -o ../index.html

cp ../SUNCALmanual.pdf ../../suncal/gui/SUNCALmanual.pdf


