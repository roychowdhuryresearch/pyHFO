from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SUPPORTED_EEG_SUFFIXES = {".edf", ".vhdr", ".fif", ".fif.gz"}
TERMINAL_STATUSES = {"complete", "failed", "skipped"}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _is_supported_recording(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".gz" and len(path.suffixes) >= 2:
        suffix = "".join(part.lower() for part in path.suffixes[-2:])
    return suffix in SUPPORTED_EEG_SUFFIXES


def discover_recordings(input_dir, *, recursive=True) -> list[Path]:
    root = Path(input_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and _is_supported_recording(path))


def _case_id_for_path(path: Path, used_ids: set[str]) -> str:
    name = path.name
    if name.endswith(".fif.gz"):
        base = name[:-7]
    else:
        base = path.stem
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in base)
    normalized = "_".join(part for part in normalized.split("_") if part) or "case"
    candidate = normalized
    index = 2
    while candidate in used_ids:
        candidate = f"{normalized}_{index}"
        index += 1
    used_ids.add(candidate)
    return candidate


@dataclass
class BatchCase:
    case_id: str
    input_path: str
    output_dir: str
    status: str = "pending"
    attempts: int = 0
    error: str = ""
    updated_at: str = field(default_factory=_now_iso)
    result: dict[str, Any] = field(default_factory=dict)

    def mark_running(self):
        self.status = "running"
        self.attempts += 1
        self.error = ""
        self.updated_at = _now_iso()

    def mark_complete(self, result: dict[str, Any] | None = None):
        self.status = "complete"
        self.result = dict(result or {})
        self.error = ""
        self.updated_at = _now_iso()

    def mark_failed(self, error: str):
        self.status = "failed"
        self.error = str(error)
        self.updated_at = _now_iso()

    def mark_skipped(self, reason: str = ""):
        self.status = "skipped"
        self.error = str(reason or "")
        self.updated_at = _now_iso()

    def reset(self):
        self.status = "pending"
        self.error = ""
        self.result = {}
        self.updated_at = _now_iso()

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "status": self.status,
            "attempts": self.attempts,
            "error": self.error,
            "updated_at": self.updated_at,
            "result": dict(self.result),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchCase":
        return cls(
            case_id=payload["case_id"],
            input_path=payload["input_path"],
            output_dir=payload["output_dir"],
            status=payload.get("status", "pending"),
            attempts=int(payload.get("attempts", 0) or 0),
            error=payload.get("error", ""),
            updated_at=payload.get("updated_at", _now_iso()),
            result=dict(payload.get("result", {})),
        )


@dataclass
class BatchProject:
    input_dir: str
    output_dir: str
    recursive: bool = True
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    cases: list[BatchCase] = field(default_factory=list)

    def next_case(self, *, retry_failed=False) -> BatchCase | None:
        allowed_statuses = {"pending"}
        if retry_failed:
            allowed_statuses.add("failed")
        for case in self.cases:
            if case.status in allowed_statuses:
                return case
        return None

    def get_case(self, case_id: str) -> BatchCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(f"Unknown batch case: {case_id}")

    def mark_running(self, case_id: str):
        case = self.get_case(case_id)
        case.mark_running()
        self.updated_at = _now_iso()
        return case

    def mark_complete(self, case_id: str, result: dict[str, Any] | None = None):
        case = self.get_case(case_id)
        case.mark_complete(result)
        self.updated_at = _now_iso()
        return case

    def mark_failed(self, case_id: str, error: str):
        case = self.get_case(case_id)
        case.mark_failed(error)
        self.updated_at = _now_iso()
        return case

    def progress_summary(self):
        counts = {status: 0 for status in ["pending", "running", "complete", "failed", "skipped"]}
        for case in self.cases:
            counts[case.status] = counts.get(case.status, 0) + 1
        counts["total"] = len(self.cases)
        counts["remaining"] = counts.get("pending", 0) + counts.get("running", 0)
        return counts

    def to_dict(self):
        return {
            "format": "pybrain-batch-project",
            "version": 1,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "recursive": self.recursive,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchProject":
        return cls(
            input_dir=payload["input_dir"],
            output_dir=payload["output_dir"],
            recursive=bool(payload.get("recursive", True)),
            created_at=payload.get("created_at", _now_iso()),
            updated_at=payload.get("updated_at", _now_iso()),
            cases=[BatchCase.from_dict(case) for case in payload.get("cases", [])],
        )

    def save(self, path=None):
        manifest_path = Path(path) if path is not None else Path(self.output_dir) / "batch_project.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return manifest_path


def create_batch_project(input_dir, output_dir=None, *, recursive=True) -> BatchProject:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve() if output_dir is not None else input_root / "pybrain_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    used_ids = set()
    cases = []
    for recording_path in discover_recordings(input_root, recursive=recursive):
        case_id = _case_id_for_path(recording_path, used_ids)
        case_output = output_root / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        cases.append(
            BatchCase(
                case_id=case_id,
                input_path=str(recording_path),
                output_dir=str(case_output),
            )
        )
    return BatchProject(
        input_dir=str(input_root),
        output_dir=str(output_root),
        recursive=recursive,
        cases=cases,
    )


def load_batch_project(path) -> BatchProject:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return BatchProject.from_dict(payload)
