# Full setup:
#
# conda env update --file environment_osx.yml
# source activate uncertosx
# python setup.py install
# ./buildapp.sh

# Get version number
ver=`python -c "import suncal.version; print(suncal.version.__version__)"`
echo Version $ver

echo Generating License Information...
python ./suncal/gui/gen_licenses.py

# Bundle app
echo Building APP for Sandia PSL Uncert Calc $ver...
pyinstaller --onefile --windowed uncertmac.spec

# Build zip file
echo Zipping package...
cd dist
zip -r SandiaUncertCalc-OSX-$ver.zip "Sandia PSL Uncertainty Calculator.app"
cd ../doc
zip -r ../dist/SandiaUncertCalc-OSX-$ver.zip Examples/*.yaml
zip -r ../dist/SandiaUncertCalc-OSX-$ver.zip Examples/Data/*.dat
zip -r ../dist/SandiaUncertCalc-OSX-$ver.zip Examples/Data/*.txt
zip -r ../dist/SandiaUncertCalc-OSX-$ver.zip SUNCALmanual.pdf
cd ..
