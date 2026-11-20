# Project notes for Claude

- Python venv: `.venv\Scripts\python.exe`. Dev tools: `pip install -r requirements-dev.txt`.
- Quality gate: `.venv\Scripts\python.exe tools/quality_check.py` (affected tests via pytest-testmon; `--full` for all).
  It is a ratchet against `.quality/baseline.json`: lint/dead-code/duplication can't rise, test and assert
  counts can't fall, no new failing tests. Improvements tighten the baseline automatically. Never edit the baseline by hand.
- Lint rules live in `pyproject.toml` under `[tool.ruff]`. When a new class of mistake shows up, add a rule there.
- Architecture rule: bots import from `core/` and `opponents/`, never from `rlcard` or `sim/` directly.
- Known failing tests at setup (2026-11-20): three in `tests/test_tournament.py`; they are recorded in the baseline.

## Default bot approaches
- Bot A: prefer approach A4, an opponent-type table. Classify the opponent as tight/loose × passive/aggressive,
  then play by that type's fixed thresholds (open, call, raise, all-in, bluff). Use a default set until enough
  hands are seen, then switch all at once (no blending).
- Bot B: prefer approach B1, a Mamdani fuzzy controller. Fuzzify hand strength, price edge, opponent looseness and
  aggression, position, and stack-to-pot ratio; fire fixed IF–THEN rules; defuzzify by centroid into one 0–1
  aggression score; map score ranges to fold/call/raise/all-in.
- The user knows best: if they ask for something that conflicts with these defaults, don't bother confirming,
  just do what they ask.
