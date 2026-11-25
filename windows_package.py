import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "pyhfo_windows.spec"


def main():
    if sys.platform != "win32":
        raise SystemExit("windows_package.py must be run on Windows.")
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(SPEC_PATH)],
        cwd=ROOT,
        check=True,
    )
    print(f"Built Windows release folder: {ROOT / 'dist' / 'PyHFO'}")


if __name__ == "__main__":
    main()
