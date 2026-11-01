"""Download the official Demucs six-source checkpoint into this project only."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "demucs"
URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"
FILENAME = "5c90dfd2-34c22ccb.th"
EXPECTED_PREFIX = "34c22ccb"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODEL_DIR / FILENAME
    if destination.exists():
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if not digest.startswith(EXPECTED_PREFIX):
            raise RuntimeError("Existing Demucs checkpoint failed its official checksum.")
        print(f"Already installed: {destination.name}; sha256={digest}")
        return
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            count = 0
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count % (25 * 1024 * 1024) == 0:
                    print(f"Downloaded {count / 1024**2:.0f} / {total / 1024**2:.0f} MiB", flush=True)
        checksum = digest.hexdigest()
        if not checksum.startswith(EXPECTED_PREFIX):
            raise RuntimeError("Downloaded checkpoint failed the official SHA-256 prefix check.")
        temporary.replace(destination)
        (MODEL_DIR / "SHA256.txt").write_text(f"{checksum}  {FILENAME}\n", encoding="utf-8")
        print(f"Installed {count / 1024**2:.1f} MiB; sha256={checksum}", flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
