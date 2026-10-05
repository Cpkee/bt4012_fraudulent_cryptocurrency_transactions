"""Repository locations, resolved from this file so scripts run from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"   # every uploaded / candidate CSV (gitignored)
ARTIFACTS = ROOT / "artifacts"       # CV tables, caches, family_*.json (gitignored)
LEDGER = ROOT / "docs" / "results.md"
DATA = Path.home() / ".cache/bt4012/bt-4012-competition-2026"   # see kaggle_data.py
