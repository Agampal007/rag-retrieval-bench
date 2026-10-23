"""Download the SQuAD v1.1 dev set into ./data and verify its checksum."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
DEST = Path(__file__).resolve().parent.parent / "data" / "dev-v1.1.json"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, DEST)
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if digest != SHA256:
        print(f"Checksum mismatch for {DEST}\n  expected {SHA256}\n  got      {digest}", file=sys.stderr)
        return 1
    print(f"OK: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
