# -*- mode: python -*-

import sysconfig

block_cipher = None

a = Analysis(['suncal/startui.py'],
             pathex=['suncal'],
             binaries=None,
             datas=[('suncal/gui/SUNCALmanual.pdf', '.')],
             hiddenimports=[sysconfig._get_sysconfigdata_name(True), '_sysconfigdata_m_darwin_darwin'],  # Needed to patch yet another pyinstaller bug (https://github.com/pyinstaller/pyinstaller/issues/3105)
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
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Suncal',
          debug=False,
          strip=False,
          upx=True,
          exclude_binaries=False,
          console=False )
app = BUNDLE(exe,
             name='Suncal.app',
             icon='suncal/gui/PSLcal_logo.icns',
             info_plist={'NSHighResolutionCapable': 'True'},
             bundle_identifier=None)
