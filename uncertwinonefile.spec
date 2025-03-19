# -*- mode: python -*-

block_cipher = None


a = Analysis(['suncal\\gui\\__main__.py'],
             pathex=[],
             binaries=[],
             datas=[('suncal/gui/SUNCALmanual.pdf', '.'),
                    ('suncal/common/style/suncal_light.mplstyle', '.'),
                    ('suncal/common/style/suncal_dark.mplstyle', '.')],
             hiddenimports=['scipy._lib.array_api_compat.numpy.fft'],
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
          icon='suncal/gui/icons/PSLcal_logo.ico',
          upx=True,
          console=False,
          exclude_binaries=False,
          version='winexe_version_info.txt')

