# -*- mode: python -*-
a = Analysis(['gui.py'],
             pathex=['/Users/ziyuanliu/LyricClassifier'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='gui',
          debug=False,
          strip=None,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=None,
               upx=True,
               name='gui')
app = BUNDLE(coll,
             name='gui.app',
             icon=None)
