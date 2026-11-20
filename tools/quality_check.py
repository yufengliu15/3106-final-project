"""Quality ratchet: fail if any metric is worse than the committed baseline.

    python tools/quality_check.py            # affected tests only (pytest-testmon)
    python tools/quality_check.py --full     # whole test suite
    python tools/quality_check.py --init     # (re)write the baseline from the current state

Metrics (lower is better unless noted):
    lint            ruff findings (rules in pyproject.toml), incl. security + complexity
    dead_code       vulture findings
    duplication     pylint duplicate-code findings
    test_count      number of test functions         (higher is better: tests can't vanish)
    assert_count    number of asserts in tests/      (higher is better: tests can't be hollowed)
    failing_tests   tests failing (known failures are recorded, new ones fail the check)

Improvements tighten the baseline automatically, so quality can only go up.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / ".quality" / "baseline.json"
PY = sys.executable
LOWER_IS_BETTER = ("lint", "dead_code", "duplication")
HIGHER_IS_BETTER = ("test_count", "assert_count")


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def lint() -> tuple[int, list[str]]:
    res = run([PY, "-m", "ruff", "check", "src", "tests", "tools", "--output-format", "json"])
    items = json.loads(res.stdout or "[]")
    lines = [f"{Path(i['filename']).relative_to(ROOT)}:{i['location']['row']} {i['code']} {i['message']}" for i in items]
    return len(items), lines


def dead_code() -> tuple[int, list[str]]:
    res = run([PY, "-m", "vulture", "src", "--min-confidence", "80"])
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    return len(lines), lines


def duplication() -> tuple[int, list[str]]:
    res = run([PY, "-m", "pylint", "--disable=all", "--enable=duplicate-code",
               "--min-similarity-lines=6", "--output-format=parseable", "src"])
    lines = [ln for ln in res.stdout.splitlines() if "R0801" in ln]
    return len(lines), lines


def test_shape() -> tuple[int, int]:
    tests = asserts = 0
    for f in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                tests += 1
            elif isinstance(node, ast.Assert) or (isinstance(node, ast.Attribute) and node.attr == "raises"):
                asserts += 1
    return tests, asserts


def run_tests(full: bool) -> tuple[set[str], set[str], str]:
    """Return (failed, passed) node ids and pytest's tail output."""
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        args = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"]
        if not full:
            args.append("--testmon")
        res = run(args)
        failed, passed = set(), set()
        if xml.exists():
            for case in ET.parse(xml).iter("testcase"):
                nid = f"{case.get('classname')}::{case.get('name')}"
                if case.find("failure") is not None or case.find("error") is not None:
                    failed.add(nid)
                elif case.find("skipped") is None:
                    passed.add(nid)
        tail = "\n".join(res.stdout.splitlines()[-30:])
        if res.returncode not in (0, 1, 5):  # 5 = no tests collected (nothing affected)
            failed.add(f"<pytest exited {res.returncode}>")
        return failed, passed, tail


def bucket(line: str) -> str:
    """Group a finding by file + rule so line-number shifts don't matter."""
    path = line.split(":", 1)[0]
    rule = line.split(" ")[1] if " " in line else ""
    return f"{path} {rule}"


def buckets(lines: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for ln in lines:
        out[bucket(ln)] = out.get(bucket(ln), 0) + 1
    return out


def regressions(metric: str, lines: list[str], base: dict) -> list[str]:
    """Only the findings in (file, rule) groups that grew since the baseline."""
    old = base.get("buckets", {}).get(metric, {})
    grown = {b for b, n in buckets(lines).items() if n > old.get(b, 0)}
    return [ln for ln in lines if bucket(ln) in grown][:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()

    cur: dict = {}
    detail: dict = {}
    for name, fn in (("lint", lint), ("dead_code", dead_code), ("duplication", duplication)):
        cur[name], detail[name] = fn()
    cur["test_count"], cur["assert_count"] = test_shape()
    failed, passed, tail = run_tests(full=a.full or a.init)

    cur_buckets = {m: buckets(detail[m]) for m in LOWER_IS_BETTER}

    if a.init or not BASELINE.exists():
        cur["failing_tests"] = sorted(failed)
        cur["buckets"] = cur_buckets
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(cur, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written to {BASELINE.relative_to(ROOT)}:\n{json.dumps(cur, indent=2)}")
        return 0

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    known = set(base.get("failing_tests", []))
    errors, improved = compare(cur, detail, base, (failed, passed, tail))

    if improved:
        base["buckets"] = cur_buckets

    if errors:
        print("QUALITY CHECK FAILED (worse than .quality/baseline.json):\n")
        print("\n\n".join(errors))
        return 1
    if improved:
        BASELINE.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
        print("Baseline tightened: " + "; ".join(improved))
    still = sorted(failed & known)
    print(f"Quality check passed. ({len(still)} known failing tests still failing: {', '.join(still) or 'none run'})")
    return 0


def compare(cur: dict, detail: dict, base: dict, tests: tuple[set, set, str]) -> tuple[list[str], list[str]]:
    """Compare against the baseline; tighten `base` in place for every improvement."""
    failed, passed, tail = tests
    known = set(base.get("failing_tests", []))
    errors: list[str] = []
    improved: list[str] = []

    for m in LOWER_IS_BETTER:
        if cur[m] > base[m]:
            errors.append(f"{m}: {base[m]} -> {cur[m]} (worse)\n  " + "\n  ".join(regressions(m, detail[m], base)))
        elif cur[m] < base[m]:
            improved.append(f"{m}: {base[m]} -> {cur[m]}")
            base[m] = cur[m]
    for m in HIGHER_IS_BETTER:
        if cur[m] < base[m]:
            errors.append(f"{m}: {base[m]} -> {cur[m]} (tests or assertions were removed)")
        elif cur[m] > base[m]:
            improved.append(f"{m}: {base[m]} -> {cur[m]}")
            base[m] = cur[m]

    new_failures = failed - known
    if new_failures:
        errors.append("new failing tests:\n  " + "\n  ".join(sorted(new_failures)) + "\n" + tail)
    fixed = known & passed
    if fixed:
        improved.append("now passing: " + ", ".join(sorted(fixed)))
        base["failing_tests"] = sorted(known - fixed)
    return errors, improved


if __name__ == "__main__":
    sys.exit(main())
