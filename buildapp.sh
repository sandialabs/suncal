# Full setup:
#
# conda env update --file environment_osx.yml
# source activate uncertosx
# python setup.py install
# ./buildapp.sh

# Get version number
ver=`python -c "import psluncert.version; print(psluncert.version.__version__)"`
echo Version $ver

echo Generating License Information...
python ./psluncert/gui/gen_licenses.py

# Bundle app
echo Building APP for PSL Uncert Calc $ver...
pyinstaller --onefile --windowed uncertmac.spec

# Build zip file
echo Zipping package...
cd dist
zip -r PSLUncertCalc-OSX-$ver.zip "PSL Uncertainty Calculator.app"
cd ../doc
zip -r ../dist/PSLUncertCalc-OSX-$ver.zip Examples/*.yaml
zip -r ../dist/PSLUncertCalc-OSX-$ver.zip Examples/Data/*.dat
zip -r ../dist/PSLUncertCalc-OSX-$ver.zip Examples/Data/*.txt
zip -r ../dist/PSLUncertCalc-OSX-$ver.zip PSLUCmanual.pdf
cd ..
