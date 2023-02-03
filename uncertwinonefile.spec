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
          icon='suncal/gui/PSLcal_logo.ico',
          upx=True,
          console=False,
          exclude_binaries=False,
          version='winexe_version_info.txt')

