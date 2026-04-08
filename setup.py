"""py2app packaging entrypoint for PyHFO."""

import os
import sys
from pathlib import Path

from setuptools import setup

sys.setrecursionlimit(1500)

ROOT = Path(__file__).resolve().parent
APP_NAME = "PyHFO"
APP_VERSION = "3.0.2"
APP_IDENTIFIER = "org.roychowdhuryresearch.pyhfo"
APP = ["main.py"]


def read_requirements(path):
    requirements = []
    for raw_line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        requirements.append(line)
    return requirements


def collect_data_files(*folders):
    data_files = []
    for folder in folders:
        source_root = ROOT / folder
        if not source_root.exists():
            continue
        for current_root, _dirs, files in os.walk(source_root):
            if not files:
                continue
            file_paths = [os.path.join(current_root, file_name) for file_name in files]
            data_files.append((current_root, file_paths))
    return data_files


DATA_FILES = collect_data_files("src", "ckpt")

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'src/ui/images/icon1.icns',
    'plist': {
        'CFBundleDisplayName': APP_NAME,
        'CFBundleIdentifier': APP_IDENTIFIER,
        'CFBundleName': APP_NAME,
        'CFBundleShortVersionString': APP_VERSION,
        'CFBundleVersion': APP_VERSION,
    },
    'packages': [
        'matplotlib',
        'mne',
        'numpy',
        'p_tqdm',
        'pandas',
        'openpyxl',
        'PyQt5',
        'PyQt5.sip',
        'pyqtgraph',
        'scipy',
        'scikit-image',
        'torch',
        'torchvision',
        'transformers',
        'tqdm',
        'ctypes',
        'HFODetector',
        'yasa',
        'antropy',
        'joblib',
        'ipywidgets',
        'lightgbm',
        'lspopt',
        'llvmlite',
        'numba',
        'pyriemann',
        'seaborn',
        'sklearn',
        'sleepecg',
        'tensorpac',
    ],
    'includes': ['liblzma'],
}

setup(
    name=APP_NAME,
    version=APP_VERSION,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    install_requires=[] if "py2app" in sys.argv else read_requirements("requirements.txt"),
)
