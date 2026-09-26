"""Download Kaggle competition data using a KGAT_ API token.

The `kaggle` and `kagglehub` packages (latest: 1.7.4.5 / 0.3.13) authenticate with
HTTP Basic using a username plus the 32-char `key` from a kaggle.json. They have no
local support for the newer `KGAT_...` tokens, which the API accepts only as
`Authorization: Bearer <token>`. This module talks to the REST API directly.

Set KAGGLE_KEY in .env to the KGAT_ token (keep the prefix).
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm

API = "https://www.kaggle.com/api/v1"
DEFAULT_CACHE = Path.home() / ".cache" / "bt4012"


def _headers() -> dict[str, str]:
    token = os.getenv("KAGGLE_KEY")
    if not token:
        msg = "KAGGLE_KEY is not set - check .env and that load_dotenv() ran."
        raise RuntimeError(msg)
    if not token.startswith("KGAT_"):
        msg = (
            "KAGGLE_KEY does not look like a KGAT_ token. Keep the 'KGAT_' prefix; "
            "the API rejects the token without it."
        )
        raise RuntimeError(msg)
    return {"Authorization": f"Bearer {token}"}


def list_files(competition: str) -> list[dict]:
    """Return the competition's file listing (name and totalBytes per file)."""
    r = requests.get(f"{API}/competitions/data/list/{competition}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json().get("files", [])


def download_competition(
    competition: str,
    cache_dir: Path | str = DEFAULT_CACHE,
    *,
    force: bool = False,
) -> Path:
    """Download and extract a competition's data. Returns the extracted directory.

    Cached: re-running is a no-op unless `force=True`.
    """
    dest = Path(cache_dir).expanduser() / competition
    marker = dest / ".complete"
    if marker.exists() and not force:
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "_download.zip"

    url = f"{API}/competitions/data/download-all/{competition}"
    with requests.get(url, headers=_headers(), stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with archive.open("wb") as fh, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=f"{competition} download"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()

    marker.touch()
    return dest
