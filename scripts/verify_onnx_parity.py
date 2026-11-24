"""Compare Torch and ONNX classifier labels on the same saved PyBrain session."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


LABEL_FIELDS = ("channel_names", "starts", "ends", "artifact", "spike", "ehfo")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _classifier_args(kind: str, model_root: Path, biomarker: str) -> list[str]:
    steps = ["artifact", "spike"] + (["ehfo"] if biomarker.lower() == "hfo" else [])
    args = [
        "--classifier-param", "enabled=true",
        "--classifier-param", f"model_type={'onnx_cpu' if kind == 'onnx' else 'default_cpu'}",
        "--classifier-param", "source_preference=local",
        "--classifier-param", "use_spike=true",
        "--classifier-param", f"use_ehfo={'true' if biomarker.lower() == 'hfo' else 'false'}",
    ]
    suffix = "onnx" if kind == "onnx" else "path"
    extension = "onnx" if kind == "onnx" else "tar"
    for name, tag in (("artifact", "a"), ("spike", "s")):
        args += ["--classifier-param", f"{name}_{suffix}={model_root / f'model_{tag}.{extension}'}"]
    if biomarker.lower() == "hfo":
        args += ["--classifier-param", f"ehfo_{suffix}={model_root / f'model_e.{extension}'}"]
    for step in steps:
        args += ["--classifier-param", f"steps={step}"]
    return args


def _run_classification(session: Path, output: Path, biomarker: str, kind: str, model_root: Path) -> Path:
    command = [
        sys.executable,
        "-m",
        "pyhfo.agent_cli",
        "classify",
        "--session",
        str(session),
        "--output",
        str(output),
        "--biomarker",
        biomarker,
        *_classifier_args(kind, model_root, biomarker),
        "--export",
        "csv",
        "--export",
        "summary-json",
    ]
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    subprocess.run(command, check=True, env=env)
    return output / "events.csv"


def compare_csv(torch_csv: Path, onnx_csv: Path) -> dict:
    torch_rows = _read_rows(torch_csv)
    onnx_rows = _read_rows(onnx_csv)
    mismatches = []
    for index in range(max(len(torch_rows), len(onnx_rows))):
        torch_row = torch_rows[index] if index < len(torch_rows) else {}
        onnx_row = onnx_rows[index] if index < len(onnx_rows) else {}
        if any(torch_row.get(field) != onnx_row.get(field) for field in LABEL_FIELDS):
            mismatches.append({
                "index": index,
                "torch": {field: torch_row.get(field) for field in LABEL_FIELDS},
                "onnx": {field: onnx_row.get(field) for field in LABEL_FIELDS},
            })
    return {
        "status": "passed" if not mismatches else "failed",
        "events_torch": len(torch_rows),
        "events_onnx": len(onnx_rows),
        "compared_fields": list(LABEL_FIELDS),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "torch_csv_sha256": _sha256(torch_csv),
        "onnx_csv_sha256": _sha256(onnx_csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--biomarker", default="HFO", choices=["HFO", "Spindle"])
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="directory holding model_{a,e,s}.tar; the legacy torch checkpoints are NOT in this "
             "repository and must be obtained separately from the maintainers",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("ckpt_onnx"),
        help="directory holding the exported model_*.onnx and model_*.spec.json; generate it "
             "first with scripts/export_onnx.py (the export output is not tracked in git)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch_csv = _run_classification(
        args.session.resolve(), args.out / "torch", args.biomarker, "torch", args.checkpoint_dir.resolve()
    )
    onnx_csv = _run_classification(
        args.session.resolve(), args.out / "onnx", args.biomarker, "onnx", args.onnx_dir.resolve()
    )
    report = compare_csv(torch_csv, onnx_csv)
    report.update({
        "session": str(args.session.resolve()),
        "biomarker": args.biomarker,
        "torch_csv": str(torch_csv.resolve()),
        "onnx_csv": str(onnx_csv.resolve()),
    })
    report_path = args.out / "onnx-parity-report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
