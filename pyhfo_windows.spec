# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(__file__).resolve().parent


def collect_tree(relative_dir, target_root):
    source_root = ROOT / relative_dir
    items = []
    for path in sorted(source_root.rglob("*")):
        if path.is_file():
            destination = Path(target_root) / path.relative_to(source_root).parent
            items.append((str(path), str(destination)))
    return items


datas = []
datas.extend(collect_tree("src/ui", "src/ui"))
datas.extend(collect_tree("ckpt", "ckpt"))
datas.append((str(ROOT / "LICENSE.txt"), "."))

hiddenimports = [
    "PyQt5.sip",
    "src.classifer",
    "src.model",
]
hiddenimports.extend(collect_submodules("src.dl_models"))
hiddenimports.extend(collect_submodules("HFODetector"))


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PyHFO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name="PyHFO",
)
