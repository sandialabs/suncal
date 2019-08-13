# -*- mode: python -*-

block_cipher = None


a = Analysis(['suncal\\startui.py'],
             pathex=[],
             binaries=[],
             datas=[('suncal/gui/SUNCALmanual.pdf', '.')],
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
          exclude_binaries=True,
          name='Sandia Uncertainty Calculator',
          debug=False,
          strip=False,
          icon='suncal/gui/PSLcal_logo.ico',
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Suncal')
