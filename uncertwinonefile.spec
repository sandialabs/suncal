# -*- mode: python -*-

block_cipher = None


a = Analysis(['psluncert\\startui.py'],
             pathex=[],
             binaries=[],
             datas=[('psluncert/gui/PSLUCmanual.pdf', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'matplotlib.backends.backend_tkagg', '_ssl'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='PSLUncertCalc',
          debug=False,
          strip=False,
          icon='psluncert/gui/PSLcal_logo.ico',
          upx=True,
          console=False)

