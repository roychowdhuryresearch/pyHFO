import argparse
import ctypes.util
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
APP_NAME = "PyHFO"
APP_VERSION = "3.0.0"
APP_IDENTIFIER = "org.roychowdhuryresearch.pyhfo"
PYTHON_LIB_DIRNAME = f"python{sys.version_info.major}.{sys.version_info.minor}"
RELEASE_STEM = f"{APP_NAME}-{APP_VERSION}-macos-arm64"
EXTRA_SITE_PACKAGES = (
    "HFODetector",
    "PIL",
    "antropy",
    "charset_normalizer",
    "contourpy",
    "cycler",
    "dateutil",
    "decorator",
    "filelock",
    "fontTools",
    "fsspec",
    "ipywidgets",
    "jinja2",
    "joblib",
    "lazy_loader",
    "lightgbm",
    "llvmlite",
    "lspopt",
    "markupsafe",
    "mpl_toolkits",
    "networkx",
    "openpyxl",
    "p_tqdm",
    "packaging",
    "platformdirs",
    "pooch",
    "pyparsing",
    "pyriemann",
    "regex",
    "requests",
    "safetensors",
    "seaborn",
    "sleepecg",
    "six",
    "tensorpac",
    "threadpoolctl",
    "tokenizers",
    "typing_extensions",
    "urllib3",
    "yaml",
    "yasa",
    "kiwisolver",
    "zipp",
)
STRIPPED_XATTRS = (
    "com.apple.FinderInfo",
    "com.apple.fileprovider.fpfs#P",
    "com.apple.provenance",
)


def _run(command, cwd=ROOT, check=True):
    subprocess.run(command, cwd=cwd, check=check)


def _find_built_app():
    apps = sorted(DIST_DIR.glob("*.app"))
    if len(apps) != 1:
        raise RuntimeError(f"Expected exactly one .app bundle in {DIST_DIR}, found {len(apps)}")
    return apps[0]


def _site_packages_roots():
    candidates = [
        Path(sys.prefix) / "lib" / PYTHON_LIB_DIRNAME / "site-packages",
        Path(sys.base_prefix) / "lib" / PYTHON_LIB_DIRNAME / "site-packages",
        Path(f"/opt/homebrew/lib/{PYTHON_LIB_DIRNAME}/site-packages"),
        Path(f"/usr/local/lib/{PYTHON_LIB_DIRNAME}/site-packages"),
    ]
    roots = []
    for candidate in candidates:
        if candidate.exists() and candidate not in roots:
            roots.append(candidate)
    return roots


def _find_site_package_entries(name):
    entries = []
    variants = {
        name,
        name.replace("-", "_"),
        name.replace("_", "-"),
    }
    for root in _site_packages_roots():
        for variant in variants:
            for candidate in (root / variant, root / f"{variant}.py"):
                if candidate.exists() and candidate not in entries:
                    entries.append(candidate)
            for pattern in (f"{variant}-*.dist-info", f"{variant}-*.data"):
                for candidate in root.glob(pattern):
                    if candidate not in entries:
                        entries.append(candidate)
    return entries


def _find_qt_root():
    candidates = [
        Path(sys.prefix) / "lib" / PYTHON_LIB_DIRNAME / "site-packages" / "PyQt5" / "Qt5",
        Path(sys.base_prefix) / "lib" / PYTHON_LIB_DIRNAME / "site-packages" / "PyQt5" / "Qt5",
        Path(f"/opt/homebrew/lib/{PYTHON_LIB_DIRNAME}/site-packages/PyQt5/Qt5"),
        Path(f"/usr/local/lib/{PYTHON_LIB_DIRNAME}/site-packages/PyQt5/Qt5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_liblzma_candidates():
    search_roots = [
        Path(sys.prefix) / "lib",
        Path(sys.base_prefix) / "lib",
        Path("/opt/homebrew/opt/xz/lib"),
        Path("/usr/local/opt/xz/lib"),
        Path("/usr/lib"),
    ]
    names = ["liblzma.5.dylib", "liblzma.dylib"]
    resolved = []
    library_name = ctypes.util.find_library("lzma")
    if library_name:
        library_path = Path(library_name)
        if library_path.is_absolute() and library_path.exists():
            resolved.append(library_path)
        else:
            for root in search_roots:
                candidate = root / library_name
                if candidate.exists():
                    resolved.append(candidate)
    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.exists() and candidate not in resolved:
                resolved.append(candidate)
    return resolved


def _copy_path(source, target):
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target, copy_function=shutil.copy)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o644)


def _stage_source_app(source_app):
    stage_root = Path(tempfile.mkdtemp(prefix="pyhfo-release-"))
    staged_app = stage_root / f"{APP_NAME}.app"
    _run(["ditto", "--noextattr", "--noqtn", str(source_app), str(staged_app)])
    return stage_root, staged_app


def _overlay_repo_source(app_path):
    resources_dir = app_path / "Contents" / "Resources"
    lib_root = resources_dir / "lib" / PYTHON_LIB_DIRNAME
    lib_root.mkdir(parents=True, exist_ok=True)

    _copy_path(ROOT / "main.py", resources_dir / "main.py")
    _copy_path(ROOT / "src", lib_root / "src")
    _copy_path(ROOT / "ckpt", lib_root / "ckpt")


def _bundle_python_packages(app_path, package_names):
    lib_root = app_path / "Contents" / "Resources" / "lib" / PYTHON_LIB_DIRNAME
    lib_root.mkdir(parents=True, exist_ok=True)
    for package_name in package_names:
        sources = _find_site_package_entries(package_name)
        if not sources:
            print(f"Warning: site-packages entry not found for {package_name}; skipping.")
            continue
        for source in sources:
            _copy_path(source, lib_root / source.name)


def _bundle_qt_plugins(app_path):
    qt_root = _find_qt_root()
    if qt_root is None:
        raise RuntimeError("PyQt5 Qt5 directory was not found.")

    source_platforms = qt_root / "plugins" / "platforms"
    target_platforms = app_path / "Contents" / "PlugIns" / "platforms"
    if target_platforms.exists():
        shutil.rmtree(target_platforms)
    shutil.copytree(source_platforms, target_platforms, copy_function=shutil.copy)

    qt_conf = app_path / "Contents" / "Resources" / "qt.conf"
    qt_conf.write_text("[Paths]\nPlugins = PlugIns\n", encoding="utf-8")

    plugin_rpath = f"@executable_path/../Resources/lib/{PYTHON_LIB_DIRNAME}/PyQt5/Qt5/lib"
    for dylib in sorted(target_platforms.glob("*.dylib")):
        subprocess.run(
            ["install_name_tool", "-add_rpath", plugin_rpath, str(dylib)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _remove_duplicate_qt_frameworks(app_path):
    frameworks_dir = app_path / "Contents" / "Frameworks"
    for framework in frameworks_dir.glob("Qt*.framework"):
        shutil.rmtree(framework, ignore_errors=True)


def _ensure_liblzma(app_path):
    frameworks_dir = app_path / "Contents" / "Frameworks"
    frameworks_dir.mkdir(parents=True, exist_ok=True)
    copied = set()
    for source in _find_liblzma_candidates():
        if source.name in copied:
            continue
        copied.add(source.name)
        _copy_path(source, frameworks_dir / source.name)


def _update_bundle_metadata(app_path):
    plist_path = app_path / "Contents" / "Info.plist"
    with plist_path.open("rb") as handle:
        info = plistlib.load(handle)
    info.update(
        {
            "CFBundleDisplayName": APP_NAME,
            "CFBundleIdentifier": APP_IDENTIFIER,
            "CFBundleName": APP_NAME,
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
        }
    )
    with plist_path.open("wb") as handle:
        plistlib.dump(info, handle)


def _strip_problem_xattrs(app_path):
    _run(["chmod", "-R", "u+w", str(app_path)])
    subprocess.run(["xattr", "-cr", str(app_path)], cwd=ROOT, check=False)
    for attribute in STRIPPED_XATTRS:
        subprocess.run(
            ["xattr", "-d", attribute, str(app_path)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["xattr", "-r", "-d", attribute, str(app_path)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _cleanup_codesign_tempfiles(app_path):
    for temp_file in app_path.rglob("*.cstemp"):
        temp_file.unlink(missing_ok=True)


def _codesign_app(app_path):
    _strip_problem_xattrs(app_path)

    platforms_dir = app_path / "Contents" / "PlugIns" / "platforms"
    for dylib in sorted(platforms_dir.glob("*.dylib")):
        _run(["codesign", "--force", "--sign", "-", str(dylib)])

    executable = app_path / "Contents" / "MacOS" / APP_NAME
    if executable.exists():
        _run(["codesign", "--force", "--sign", "-", str(executable)])

    _run(["codesign", "--force", "--sign", "-", "--deep", str(app_path)])
    _cleanup_codesign_tempfiles(app_path)
    _run(["codesign", "--verify", "--deep", "--verbose=2", str(app_path)])


def _create_zip(staged_app, output_path):
    output_path.unlink(missing_ok=True)
    _run(["ditto", "-c", "-k", "--keepParent", str(staged_app), str(output_path)])


def _create_dmg(staged_app, stage_root, output_path):
    dmg_source = stage_root / "dmg-root"
    app_target = dmg_source / staged_app.name
    if dmg_source.exists():
        shutil.rmtree(dmg_source)
    dmg_source.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged_app, app_target, copy_function=shutil.copy)
    applications_link = dmg_source / "Applications"
    if applications_link.exists() or applications_link.is_symlink():
        applications_link.unlink()
    applications_link.symlink_to("/Applications")

    output_path.unlink(missing_ok=True)
    _run(
        [
            "hdiutil",
            "create",
            "-volname",
            f"{APP_NAME} {APP_VERSION}",
            "-srcfolder",
            str(dmg_source),
            "-fs",
            "HFS+",
            "-format",
            "UDZO",
            str(output_path),
        ]
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Build PyHFO macOS release artifacts.")
    parser.add_argument(
        "--source-app",
        type=Path,
        help="Existing .app bundle to repackage instead of rebuilding with py2app.",
    )
    return parser.parse_args()


def main():
    if sys.platform != "darwin":
        raise SystemExit("macos_package.py must be run on macOS.")

    args = _parse_args()
    if args.source_app is None:
        _run([sys.executable, "setup.py", "py2app", "--packages=PyQt5"])
        source_app = _find_built_app()
    else:
        source_app = args.source_app.expanduser().resolve()
        if not source_app.exists():
            raise FileNotFoundError(f"Source app does not exist: {source_app}")

    stage_root, staged_app = _stage_source_app(source_app)
    _overlay_repo_source(staged_app)
    _bundle_python_packages(staged_app, EXTRA_SITE_PACKAGES)
    _bundle_qt_plugins(staged_app)
    _remove_duplicate_qt_frameworks(staged_app)
    _ensure_liblzma(staged_app)
    _update_bundle_metadata(staged_app)
    _codesign_app(staged_app)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / f"{RELEASE_STEM}.zip"
    dmg_path = DIST_DIR / f"{RELEASE_STEM}.dmg"
    _create_zip(staged_app, zip_path)
    _create_dmg(staged_app, stage_root, dmg_path)

    print(f"Built macOS app bundle: {staged_app}")
    print(f"Built macOS zip archive: {zip_path}")
    print(f"Built macOS DMG archive: {dmg_path}")


if __name__ == "__main__":
    main()
