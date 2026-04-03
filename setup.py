"""py2app packaging entrypoint for PyHFO."""

from setuptools import setup
import sys
import os
sys.setrecursionlimit(1500)
APP = ['main.py']
DATA_FILES = []
for folder in ['src', 'ckpt']:
    for root, dirs, files in os.walk(folder):
        for file in files:
            DATA_FILES.append((root, [os.path.join(root, file)]))
print(DATA_FILES)

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'src/ui/images/icon1.icns',
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
        'chardet',
        'tqdm',
        'ctypes',
        'HFODetector',
        'yasa',
    ],
    'includes': ['liblzma'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    install_requires=[
        'torch',
        'torchvision',
        'transformers[torch]',
        'PyQt5==5.15.9',
        'pyqtgraph==0.13.3',
    ],
)
