from __future__ import annotations

from pathlib import Path
from typing import Any


UI_RESOURCE_URI = "ui://pyhfo/results"
UI_DIR = Path(__file__).resolve().parent / "ui"


def ui_resource_descriptor() -> dict[str, Any]:
    return {
        "uri": UI_RESOURCE_URI,
        "name": "pyhfo-results-ui",
        "title": "PyHFO Results",
        "description": "Interactive PyHFO detection and batch result panel.",
        "mimeType": "text/html",
        "_meta": ui_meta(),
    }


def ui_meta() -> dict[str, Any]:
    return {
        "ui": {
            "resourceUri": UI_RESOURCE_URI,
            "permissions": [],
            "csp": {
                "default-src": ["'none'"],
                "script-src": ["'self'", "'unsafe-inline'"],
                "style-src": ["'self'", "'unsafe-inline'"],
                "img-src": ["'self'", "data:"],
                "connect-src": ["'none'"],
            },
        }
    }


def read_ui_resource(uri: str) -> dict[str, Any]:
    if uri != UI_RESOURCE_URI:
        raise FileNotFoundError(f"UI resource not found: {uri}")
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
    js = (UI_DIR / "app.js").read_text(encoding="utf-8")
    bundled = html.replace("/*__PYHFO_CSS__*/", css).replace("/*__PYHFO_JS__*/", js)
    return {
        "uri": UI_RESOURCE_URI,
        "mimeType": "text/html",
        "text": bundled,
        "_meta": ui_meta(),
    }
