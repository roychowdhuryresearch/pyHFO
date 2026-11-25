# -*- mode: python ; coding: utf-8 -*-
#
# Headless backend build for the Electron desktop app (mirrors pyhfo_backend_macos.spec).
# There is no PyQt UI to bundle any more — Electron owns the desktop UI and talks to this
# frozen backend (scripts/pyhfo_backend_entry.py) over the local agent_api HTTP server.

import ctypes
import ctypes.wintypes
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


# SPECPATH, not __file__: PyInstaller exec()s the spec without __file__ defined, so the old
# line raised NameError before the build started -- this spec had never actually run. The
# macOS spec already did it this way.
ROOT = Path(SPECPATH).resolve()
SOURCE = ROOT / "src" / "pyhfo"
datas = [
    (str(SOURCE / "resources"), "pyhfo/resources"),
    (str(SOURCE / "agent_mcp" / "ui"), "pyhfo/agent_mcp/ui"),
    (str(ROOT / "LICENSE.txt"), "."),
]


datas += collect_data_files("lspopt", includes=["data/*.npy"])
datas += collect_data_files("mne", includes=["**/*.pyi"])

hiddenimports = [
    "mne.html_templates._templates",
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
# mne is built on lazy_loader: `mne.io.RawArray` etc. resolve to submodules only at runtime, so
# PyInstaller's static analysis cannot see them and the frozen app dies mid-analysis with
# ModuleNotFoundError (observed: 'mne.io.array' during severity scoring). Listing them one by one
# is whack-a-mole across every code path, so collect the package wholesale -- it is pure Python
# and cheap next to torch.
hiddenimports.extend(collect_submodules("mne"))
hiddenimports.extend(collect_submodules("HFODetector"))

# The packaged backend is headless; Electron owns the desktop UI. Keep this list in
# sync with pyhfo_backend_macos.spec — same reasoning applies on both platforms.
excluded_modules = [
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "pyqtgraph",
    "tkinter",
    "_tkinter",
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
    "boto3",
    "botocore",
    "s3transfer",
    "sqlalchemy",
    "asyncpg",
    "psycopg",
    "psycopg_binary",
    "onnx",
    "onnxscript",
    "hf_xet",
    # The desktop backend ships BASED severity (torch and transformers) and
    # the legacy GPU classifier (torchvision). Both are lazy imports.
]

# yasa (spindle detection) imports numba, which loads llvmlite's native
# libllvmlite shared library at runtime. PyInstaller does not auto-collect that
# binary (or numba/lightgbm/sleepecg's compiled extensions and data files), so a
# bundled build fails `import yasa` and SILENTLY disables spindle detection
# (yasa is a soft-optional dep guarded by has_yasa()). collect_all() pulls the
# native binaries, data files, and dynamically imported submodules for the whole
# scientific cluster yasa needs.
binaries = []

# The core stack (numba/llvmlite/yasa) is mandatory for spindle detection, so a
# collection failure must FAIL THE BUILD LOUDLY rather than silently ship a
# bundle whose `import yasa` crashes at runtime. The rest are optional yasa
# extras (sleep staging / phase-amplitude coupling) that are skipped cleanly
# when absent from the build env.
# torch/torchvision/transformers back desktop severity and legacy GPU classification. Torch
# must go through collect_all: relying on the stock hook produced a bundle whose torch/lib DLLs
# failed to initialise (WinError 1114 on c10.dll) because the package was collected incompletely.
_REQUIRED = {"numba", "llvmlite", "yasa", "onnxruntime", "torch"}
for _pkg in (
    "torch", "torchvision", "transformers",
    "numba", "llvmlite", "yasa", "onnxruntime",
    "antropy", "lspopt", "lightgbm", "sleepecg", "tensorpac", "pyriemann",
):
    try:
        _pkg_datas, _pkg_binaries, _pkg_hiddenimports = collect_all(_pkg)
    except Exception as exc:
        if _pkg in _REQUIRED:
            raise
        print(f"[pyhfo_windows.spec] skipping optional collect_all({_pkg!r}): {exc}")
        continue
    datas += _pkg_datas
    binaries += _pkg_binaries
    hiddenimports += _pkg_hiddenimports


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

# --- MSVC runtime version reconciliation -------------------------------------------------
# PyInstaller collects whatever msvcp140.dll it finds first among the dependencies; here that
# was pyzmq's, version 14.16.27033.0 (VS2017-era). Windows resolves DLL imports by MODULE
# NAME, so that one copy served the whole process -- including torch's c10.dll, which needs a
# much newer C++ runtime. Against 14.16 its static initialiser fails and the load dies with
# WinError 1114 ("DLL initialization routine failed"), taking the entire backend down at
# `import torch`. Verified: identical c10.dll + System32's 14.51 runtime loads fine.
#
# So replace every bundled copy of the redistributable with the newest one on the build
# machine. Replacing (rather than dropping and relying on System32) keeps the bundle
# self-contained on machines without the VC++ redistributable installed.
_VCREDIST = {
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll", "concrt140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "vccorlib140.dll",
}
_system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"


def _file_version(path):
    """Return a comparable version tuple from a PE's VERSIONINFO, or None."""
    size = ctypes.windll.version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    ctypes.windll.version.GetFileVersionInfoW(str(path), 0, size, buf)
    block = ctypes.c_void_p()
    length = ctypes.wintypes.UINT()
    if not ctypes.windll.version.VerQueryValueW(
        buf, "\\", ctypes.byref(block), ctypes.byref(length)
    ):
        return None
    ms = ctypes.cast(block, ctypes.POINTER(ctypes.wintypes.DWORD * 4)).contents
    return (ms[2] >> 16, ms[2] & 0xFFFF, ms[3] >> 16, ms[3] & 0xFFFF)


_patched = []
for _i, (_dest, _src, _kind) in enumerate(a.binaries):
    _name = Path(_dest).name.lower()
    if _name not in _VCREDIST:
        continue
    _sys_copy = _system32 / _name
    if not _sys_copy.is_file():
        continue
    _have, _sys = _file_version(_src), _file_version(_sys_copy)
    if _have is not None and _sys is not None and _sys <= _have:
        continue
    a.binaries[_i] = (_dest, str(_sys_copy), _kind)
    _patched.append(f"{_dest}: {_have} -> {_sys}")
if _patched:
    print("[pyhfo_windows.spec] upgraded bundled MSVC runtime:")
    for _line in _patched:
        print(f"    {_line}")

pyz = PYZ(a.pure)

# console=True: Electron reads the {"status": "ready", "url": ...} readiness line this
# process prints to stdout on startup (see pyhfo.agent_api.server.main()) — a windowed
# (console=False) build would swallow that and the backend would never be detected as ready.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pyhfo-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pyhfo-backend",
)
