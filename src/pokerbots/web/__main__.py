"""Live tournament viewer.

    PYTHONPATH=src .venv/bin/python -m pokerbots.web --games 1 --rotations 20
    open http://localhost:8765/

Runs a tournament (same bots as `python -m pokerbots.tournament`) in a
background thread and streams every event to live.html over server-sent
events: the deal with everyone's hole cards, every decision with the bot's
explain(), round results, rotation stacks and game results. The page buffers
the stream, so you can pause, step back and replay while the run continues.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pokerbots.tournament import Tournament, TournamentConfig
from pokerbots.tournament.__main__ import DEFAULT_CONFIG, ROUNDLOGS_DIR, default_log_path, prepend_leaderboard

REPO_ROOT = ROUNDLOGS_DIR.parent

HTML = Path(__file__).with_name("live.html")
KEEPALIVE_S = 15.0


class Feed:
    """Append-only event list shared by the tournament thread and every client."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.done = False
        self._cond = threading.Condition()

    def add(self, record: dict) -> None:
        line = json.dumps(record)
        with self._cond:
            self.events.append(line)
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def wait_from(self, start: int, timeout: float) -> tuple[list[str], bool]:
        with self._cond:
            if len(self.events) <= start and not self.done:
                self._cond.wait(timeout)
            return self.events[start:], self.done


def display_path(path: Path | None) -> str | None:
    """Log path as the page shows it: relative to the repo root when inside it,
    so the copied jq command runs from the project root."""
    if path is None:
        return None
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def run_tournament(feed: Feed, config: TournamentConfig) -> None:
    feed.add({"type": "meta", "games": config.games, "rotations": config.rotations, "seats": config.seats,
              "starting_stack": config.starting_stack, "k": config.k, "mode": config.mode,
              "base_seed": config.base_seed, "log_file": display_path(config.log_file)})
    try:
        results = Tournament(config, observer=feed.add).run()
        prepend_leaderboard(results, config)
        feed.add({"type": "done"})
    except Exception as exc:
        feed.add({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        feed.finish()


def make_handler(feed: Feed) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            if url.path in ("/", "/live.html"):
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/events":
                self._stream(url.query)
            else:
                self.send_error(404)

        def _stream(self, query: str) -> None:
            # Resume after a reconnect: EventSource resends the last id it saw.
            start = int(self.headers.get("Last-Event-ID") or parse_qs(query).get("from", ["0"])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    batch, done = feed.wait_from(start, KEEPALIVE_S)
                    if batch:
                        start += len(batch)
                        self.wfile.write(f"id: {start}\ndata: [{','.join(batch)}]\n\n".encode())
                    elif done:
                        self.wfile.write(b"event: end\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    else:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args) -> None:  # noqa: A002 (http.server API)
            pass  # keep the terminal for the tournament output

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--rotations", type=int)
    parser.add_argument("--seats", type=int)
    parser.add_argument("--stack", type=float, help="starting stack in bb")
    parser.add_argument("--k", type=float)
    parser.add_argument("--mode", choices=["dev", "eval"])
    parser.add_argument("--seed", type=int, help="base seed")
    parser.add_argument("--log-file", type=Path, help="JSONL match log (default roundlogs/<date>_<time>.jsonl)")
    parser.add_argument("--no-logs", action="store_true", help="do not write a match log")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    overrides = {
        "games": args.games, "rotations": args.rotations, "seats": args.seats,
        "starting_stack": args.stack, "k": args.k, "mode": args.mode, "base_seed": args.seed,
    }
    config = dataclasses.replace(DEFAULT_CONFIG, **{k: v for k, v in overrides.items() if v is not None})
    if not args.no_logs:
        config = dataclasses.replace(config, log_file=default_log_path(args.log_file))

    feed = Feed()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(feed))
    server.daemon_threads = True
    threading.Thread(target=run_tournament, args=(feed, config), daemon=True).start()
    print(f"Live viewer: http://localhost:{args.port}/  (Ctrl+C to stop)")
    print(f"Match log: {display_path(config.log_file) or 'off (--no-logs)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
