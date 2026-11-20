# Project notes for Claude

- Python venv: `.venv\Scripts\python.exe`. Dev tools: `pip install -r requirements-dev.txt`.
- Quality gate: `.venv\Scripts\python.exe tools/quality_check.py` (affected tests via pytest-testmon; `--full` for all).
  It is a ratchet against `.quality/baseline.json`: lint/dead-code/duplication can't rise, test and assert
  counts can't fall, no new failing tests. Improvements tighten the baseline automatically. Never edit the baseline by hand.
- Lint rules live in `pyproject.toml` under `[tool.ruff]`. When a new class of mistake shows up, add a rule there.
- Architecture rule: bots import from `core/` and `opponents/`, never from `rlcard` or `sim/` directly.
- Known failing tests at setup (2026-11-20): three in `tests/test_tournament.py`; they are recorded in the baseline.
