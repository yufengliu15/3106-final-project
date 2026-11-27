"""Live tournament viewer.

    PYTHONPATH=src .venv/bin/python -m pokerbots.web --games 1 --rotations 20
    open http://localhost:8765/

Nothing runs until the page picks a mode (POST /start):
    watch -- the tournament (same bots as `python -m pokerbots.tournament`),
             with its match log and LEADERBOARD.md entry;
    play  -- humans against the same bots in a sandbox (sandbox.py): fresh
             cards every hand, no rotations, nothing written to disk.

The run happens in a background thread and every event streams to live.html
over server-sent events: the deal with everyone's hole cards, every decision
with the bot's explain(), round results, stacks and game results. The page
buffers the stream, so you can pause, step back and replay while it runs.
Humans are HumanSeat bots (humans.py) and can join only in play mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pokerbots.tournament import Tournament, TournamentConfig
from pokerbots.tournament.__main__ import DEFAULT_CONFIG, ROUNDLOGS_DIR, default_log_path, prepend_leaderboard
from pokerbots.web.hands import showdown_hands
from pokerbots.web.humans import HumanSeat, Lobby
from pokerbots.web.sandbox import Sandbox

MODES = ("watch", "play")

REPO_ROOT = ROUNDLOGS_DIR.parent

HTML = Path(__file__).with_name("live.html")
KEEPALIVE_S = 15.0
MAX_BODY = 4096


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


def observe(feed: Feed, record: dict) -> None:
    """Tournament observer: every record goes to the page; showdown rounds also
    get each shown hand's name and best five cards for the result screen."""
    if record["type"] == "round" and record["showdown_cards"]:
        record = {**record, "hands": showdown_hands(record["community_cards"], record["showdown_cards"])}
    feed.add(record)


class Session:
    """One server run: waits for the page to pick watch or play, then runs it once."""

    def __init__(self, config: TournamentConfig, play_seed: int) -> None:
        self.config = config
        self.play_seed = play_seed
        self.feed = Feed()
        self.lobby = Lobby(self.feed.add, config.seats)
        self.mode: str | None = None
        self.run_id = secrets.token_hex(4)
        self._lock = threading.Lock()
        self.feed.add({"type": "hello", "run_id": self.run_id})

    def choose(self, mode: str) -> tuple[bool, str | None]:
        """(this call picked the mode, error). Choosing the mode already running is fine."""
        with self._lock:
            if self.mode is None:
                self.mode = mode
                return True, None
        if self.mode == mode:
            return False, None
        return False, f"this run already started in {self.mode} mode; restart the server to switch"

    def launch(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        watch = self.mode == "watch"
        config = self.config if watch else dataclasses.replace(self.config, log_file=None, base_seed=self.play_seed)
        self.feed.add({"type": "meta", "mode": self.mode, "games": config.games, "rotations": config.rotations,
                       "seats": config.seats, "starting_stack": config.starting_stack, "k": config.k,
                       "base_seed": config.base_seed, "log_file": display_path(config.log_file),
                       "run_id": self.run_id})
        observer = lambda record: observe(self.feed, record)  # noqa: E731
        try:
            if watch:
                prepend_leaderboard(Tournament(config, observer=observer).run(), config)
            else:
                Sandbox(config, observer=observer, extra_players=self.lobby.players).run()
            self.feed.add({"type": "done"})
        except Exception as exc:
            self.feed.add({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            self.feed.finish()


def read_json(headers, rfile) -> dict | None:
    """A POST body that is a JSON object of at most MAX_BODY bytes, else None."""
    length = headers.get("Content-Length") or "0"
    if not length.isdigit() or int(length) > MAX_BODY:
        return None
    try:
        body = json.loads(rfile.read(int(length)) or b"{}")
    except ValueError:  # includes JSONDecodeError and bad UTF-8
        return None
    return body if isinstance(body, dict) else None


def join_request(session: Session, body: dict) -> tuple[int, dict]:
    if session.mode != "play":
        return 409, {"error": "humans can only join in play mode; restart the server and pick Join as a player"}
    if session.feed.done:
        return 409, {"error": "the run has finished; restart the server to play"}
    try:
        seat = session.lobby.join(str(body.get("name", "")))
    except ValueError as exc:
        return 409, {"error": str(exc)}
    return 200, {"human_id": seat.human_id, "token": seat.token, "name": seat.name}


def start_request(session: Session, body: dict) -> tuple[int, dict]:
    """Pick the mode. Play also joins the caller as a human before the first deal."""
    mode = body.get("mode")
    if mode not in MODES:
        return 400, {"error": f"mode must be one of {MODES}"}
    picked, error = session.choose(mode)
    if error:
        return 409, {"error": error}
    reply = join_request(session, body) if mode == "play" else (200, {})
    if picked:
        session.launch()
    return reply


def post_request(session: Session, path: str, body: dict | None) -> tuple[int, dict]:
    """POST /start and /human/{join,act,leave}: (HTTP status, JSON reply). body None = unreadable."""
    if body is None:
        return 400, {"error": f"expected a JSON object of at most {MAX_BODY} bytes"}
    if path == "/start":
        return start_request(session, body)
    if path == "/human/join":
        return join_request(session, body)
    return seat_request(session, path, body)


def seat_request(session: Session, path: str, body: dict) -> tuple[int, dict]:
    """/human/act and /human/leave, for the human holding body["token"]."""
    found: HumanSeat | None = session.lobby.get(str(body.get("token", "")))
    if found is None:
        return 404, {"error": "unknown player (was the server restarted?)"}
    if path == "/human/act":
        error = found.choose(body.get("prompt_id"), str(body.get("action", "")))
        return (409, {"error": error}) if error else (200, {})
    if path == "/human/leave":
        found.leave()
        return 200, {}
    return 404, {"error": f"no endpoint {path}"}


def make_handler(session: Session) -> type[BaseHTTPRequestHandler]:
    feed = session.feed

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

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            body = read_json(self.headers, self.rfile)
            self._json(*post_request(session, urlparse(self.path).path, body))

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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

    session = Session(config, play_seed=args.seed if args.seed is not None else secrets.randbelow(2**31))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    server.daemon_threads = True
    print(f"Live viewer: http://localhost:{args.port}/  (Ctrl+C to stop)")
    print("Open it and pick Watch (tournament) or Join as a player (sandbox, no logs).")
    print(f"Match log in watch mode: {display_path(config.log_file) or 'off (--no-logs)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
