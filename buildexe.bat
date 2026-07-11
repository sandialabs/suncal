echo Generating License Information...
python suncal/gui/gen_licenses.py

echo Building Standalone Exe
python -m nuitka suncal\gui
