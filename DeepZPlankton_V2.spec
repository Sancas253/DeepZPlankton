# -*- mode: python ; coding: utf-8 -*-
# El código de la app vive en el paquete deepzplankton/ (deepzplankton/main.py
# es el entry point); PyInstaller sigue los imports normales del paquete solo,
# así que no hace falta listar sus módulos acá a mano como antes.
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

# Rutas relativas a este .spec (raíz del repo), para que el build funcione
# en cualquier máquina y no solo en la carpeta original.
ROOT = os.path.abspath(SPECPATH)

datas = []
binaries = []
hiddenimports = []

# cv2: dejar en su lugar normal
tmp = collect_all('cv2')
datas    += tmp[0]
binaries += tmp[1]
hiddenimports += tmp[2]

# onnxruntime: mantener DLLs en su lugar original (onnxruntime/capi/)
# el startup code de la app los precarga via ctypes
tmp = collect_all('onnxruntime')
datas    += tmp[0]
binaries += tmp[1]
hiddenimports += tmp[2]

hiddenimports += [
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi.onnxruntime_inference_collection',
]

a = Analysis(
    [os.path.join(ROOT, 'deepzplankton', 'main.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # La app sólo necesita numpy, pandas, cv2, PIL, PyQt5 y onnxruntime.
    # Los frameworks de entrenamiento entran únicamente por los scripts de
    # train_*/export, que no se bundlean: sin estos excludes el .exe pesa ~1.7 GB
    # (tensorflow 1.1 GB + torch 0.5 GB) en lugar de ~0.2 GB.
    excludes=[
        # frameworks de entrenamiento / export
        'torch', 'timm', 'torchvision', 'tensorflow', 'tensorboard', 'keras',
        'transformers', 'tokenizers', 'h5py', 'tf2onnx', 'onnx',
        # nunca se usan en runtime, los arrastra el entorno de Anaconda
        'matplotlib', 'scipy', 'IPython', 'jedi', 'astroid', 'sphinx',
        'black', 'yapf', 'ruff', 'nbformat', 'notebook', 'zmq',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeepZPlanktonV3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[os.path.join(ROOT, 'deepzplankton', 'app_icon.ico')],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='DeepZPlanktonV3',
)
