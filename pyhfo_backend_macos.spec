# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPECPATH).resolve()
SOURCE = ROOT / "src" / "pyhfo"
datas = [
    (str(SOURCE / "resources"), "pyhfo/resources"),
    (str(SOURCE / "agent_mcp" / "ui"), "pyhfo/agent_mcp/ui"),
    (str(ROOT / "LICENSE.txt"), "."),
]

binaries = []
datas += collect_data_files("lspopt", includes=["data/*.npy"])
datas += collect_data_files("mne", includes=["**/*.pyi"])
hiddenimports = [
    "mne.html_templates._templates",
    "mne.io.array._array",
    "mne.io.fiff",
    "mne.io.fiff.raw",
    "mne.utils.progressbar",
    "mne.utils._testing",
    "mne.utils._typing",
    "mne.utils.dataframe",
    "mne.utils.linalg",
    "mne.utils.mixin",
    "pyhfo.dl_models.severity.modeling",
    "pyhfo.dl_models.modeling_neuralcnn",
    "pyhfo.classifer",
    "pyhfo.model",
    "pyhfo.preprocessing",
]
hiddenimports.extend(collect_submodules("HFODetector"))

excluded_modules = [
    # The packaged backend is headless; Electron owns the desktop UI.
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "pyqtgraph",
    "tkinter",
    "_tkinter",
    # Development, notebook, and documentation stacks are not runtime features.
    "pytest",
    "_pytest",
    "IPython",
    "ipykernel",
    "jedi",
    "parso",
    "jupyter_client",
    "jupyter_core",
    "nbclient",
    "nbformat",
    "sphinx",
    "sphinxcontrib",
    "docutils",
    "alabaster",
    "babel",
    # Optional cloud/database adapters pulled in by generic dependency hooks.
    "boto3",
    "botocore",
    "s3transfer",
    "sqlalchemy",
    "asyncpg",
    "psycopg",
    "psycopg_binary",
    # Export/training accelerators are not needed for ONNX Runtime inference.
    "onnx",
    "onnxscript",
    "hf_xet",
    # The desktop backend ships BASED severity (torch and transformers) and
    # the legacy GPU classifier (torchvision). Both are lazy imports.
]

a = Analysis(
    [str(ROOT / "scripts" / "pyhfo_backend_entry.py")],
    pathex=[str(ROOT / "src")],
    # Numba keys its cache by MNE source filenames. Transformers also opens
    # the BASED model source to discover supported attention implementations.
    module_collection_mode={"mne": "py", "pyhfo.dl_models.severity": "pyz+py"},
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[str(ROOT / "scripts" / "pyinstaller_headless_runtime.py")],
    excludes=excluded_modules,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pyhfo-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pyhfo-backend",
)
