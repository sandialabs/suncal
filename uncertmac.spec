# -*- mode: python -*-

import sysconfig

block_cipher = None

a = Analysis(['suncal/startui.py'],
             pathex=['suncal'],
             binaries=None,
             datas=[('suncal/gui/SUNCALmanual.pdf', '.')],
             hiddenimports=[sysconfig._get_sysconfigdata_name(True), '_sysconfigdata_m_darwin_darwin'],  # Needed to patch yet another pyinstaller bug (https://github.com/pyinstaller/pyinstaller/issues/3105)
             hookspath=[],
             runtime_hooks=[],
             excludes=[
                      'PyQt4',
                      '_tkinter', 'tk85.dll', 'tcl85.dll',
                      'matplotlib.backends.backend_tkagg',
                      'matplotlib.backends.backend_webagg',
                      '_sqlite3', 'zmq', 'tornado', 'IPython'
                      ],
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
          name='Sandia PSL Uncertainty Calculator',
          debug=False,
          strip=False,
          upx=True,
          console=False )
app = BUNDLE(exe,
             name='Sandia PSL Uncertainty Calculator.app',
             icon='suncal/gui/PSLcal_logo.icns',
             info_plist={'NSHighResolutionCapable': 'True'},
             bundle_identifier=None)
