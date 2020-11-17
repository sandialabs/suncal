REM Make standalone exe and installer for Windows

for /f %%i in ('python -c "import suncal; print(suncal.version.__version__)"') do set VER=%%i
echo Building Sandia PSL Uncert Calc Executable %VER%

echo Generating License Information...
python suncal/gui/gen_licenses.py

echo Building Standalone Exe
pyinstaller --windowed --onefile --version-file=winexe_version_info.txt uncertwinonefile.spec

echo Building Installer
pyinstaller --windowed  --version-file=winexe_version_info.txt uncertwin.spec
"C:\Program Files (x86)\Inno Setup 6\ISCC" /dMyAppVersion="%VER%" installer.iss