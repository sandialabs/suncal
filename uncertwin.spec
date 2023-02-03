# -*- mode: python -*-

block_cipher = None


a = Analysis(['suncal\\startui.py'],
             pathex=[],
             binaries=[],
             datas=[('suncal/gui/SUNCALmanual.pdf', '.')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={
                'matplotlib': {'backends': ['Qt5Agg', 'SVG', 'AGG', 'PDF']},
             },
             excludes=[
                      '_tkinter', 'tk85.dll', 'tcl85.dll',
                      '_sqlite3', 'zmq', 'tornado', 'IPython'
                      ],
             runtime_hooks=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='Suncal',
          debug=False,
          strip=False,
          icon='suncal/gui/PSLcal_logo.ico',
          upx=True,
          console=False,
          version='winexe_version_info.txt')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Suncal')
