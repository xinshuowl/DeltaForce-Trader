# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — 三角洲行动交易行助手 v1.0
打包命令:  pyinstaller DeltaForceTrader.spec --noconfirm
"""
import os
import sys
from pathlib import Path

# RapidOCR 模型和配置文件
_site = Path(sys.prefix) / "Lib" / "site-packages"
_rapid = _site / "rapidocr"

rapidocr_datas = []
if _rapid.exists():
    for f in (_rapid / "models").glob("*"):
        if f.is_file():
            rapidocr_datas.append((str(f), "rapidocr/models"))
    for name in ("config.yaml", "default_models.yaml"):
        p = _rapid / name
        if p.exists():
            rapidocr_datas.append((str(p), "rapidocr"))
    arch_cfg = _rapid / "networks" / "arch_config.yaml"
    if arch_cfg.exists():
        rapidocr_datas.append((str(arch_cfg), "rapidocr/networks"))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=rapidocr_datas,
    hiddenimports=[
        "sklearn",
        "sklearn.ensemble",
        "sklearn.ensemble._forest",
        "sklearn.tree",
        "sklearn.tree._classes",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._vector_sentinel",
        "sklearn.neighbors._partition_nodes",
        "joblib",
        "rapidocr",
        "onnxruntime",
        "cv2",
        "numpy",
        "PIL",
        "PyQt5",
        "pytesseract",
        "keyboard",
        "pyautogui",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 不需要的巨型库 — 避免 PyInstaller 分析时崩溃
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "tensorboard",
        "onnx",
        "onnx.reference",
        "scipy",
        "pandas",
        "matplotlib",
        "tkinter",
        "IPython",
        "notebook",
        "jupyter",
        "wandb",
        "omegaconf",
        "hydra",
        "test",
        "unittest",
        "xmlrpc",
        "pydoc",
        "doctest",
        "lib2to3",
        "distutils",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeltaForceTrader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    version="version_info.txt",
    uac_admin=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DeltaForceTrader",
)
