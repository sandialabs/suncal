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
excludes = ['mkl_avx512_mic.dll', 'mkl_avx512.dll', 'mkl_avx2.dll', 'mkl_avx.dll',
            'mkl_mc3.dll', 'mkl_mc.dll', 'mkl_pgi_thread.dll', 'mkl_tbb_thread.dll',
            'mkl_vml_avx512_mic.dll', 'mkl_vml_avx2.dll', 'mkl_vml_avx.dll', 'mkl_sequential.dll',
            'mkl_vml_avx512.dll', 'mkl_vml_mc3.dll', 'mkl_vml_mc.dll', 'mkl_vml_mc2.dll',
            'mkl_scalapack_ilp64.dll', 'mkl_scalapack_lp64.dll', 'mkl_vml_cmpt.dll',
            'mkl_blacs_ip64.dll', 'mkl_blacs_lp64.dll', 'mkl_cdft_core.dll',
            'mkl_blacs_intelmpi_ilp64.dll', 'mkl_blacs_intelmpi_lp64.dll',
            'mkl_blacs_mpich2_ilp64.dll', 'mkl_blacs_mpich2_lp64.dll',
            'mkl_blacs_msmpi_ilp64.dll', 'mkl_blacs_msmpi_lp64.dll'
]
a.binaries = [x for x in a.binaries if x[0] not in excludes]


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

