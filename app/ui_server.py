"""Lightweight static UI server (port UI_PORT, default 27756).

Serves the files under ``web/`` so the browser can call the TTS API on PORT.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def default_web_root() -> Path:
    """Resolve ``web/`` relative to the repo / image layout."""
    env = os.environ.get("UI_ROOT", "").strip()
    if env:
        return Path(env).resolve()

    # Preferred layout: /app/web (image) or <repo>/web (dev)
    candidates = [
        Path(__file__).resolve().parent.parent / "web",
        Path("/app/web"),
        Path.cwd() / "web",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler with quieter logs and no-cache for local iteration."""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write(
            f'{{"ts":"","level":"INFO","logger":"ui_server","message":"{self.address_string()} {format % args}"}}\n'
        )


def serve(host: str, port: int, root: Path) -> None:
    if not root.is_dir():
        raise SystemExit(f"UI root not found: {root}")

    handler = partial(QuietHandler, directory=str(root))
    server = ThreadingHTTPServer((host, port), handler)
    print(
        f'{{"ts":"","level":"INFO","logger":"ui_server","message":"Serving UI from {root} on http://{host}:{port}"}}',
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the TTS web UI")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Bind address (default: HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("UI_PORT", "27756")),
        help="UI port (default: UI_PORT or 27756)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory of static files (default: web/)",
    )
    args = parser.parse_args(argv)
    root = (args.root or default_web_root()).resolve()
    serve(args.host, args.port, root)


if __name__ == "__main__":
    main()
