"""Fix mojibake (encoding artifacts) in ground-truth annotation JSON files.

This repository contains some product strings like "ZubehÃ¶r" which are
typically UTF-8 text that was mistakenly decoded as Latin-1/Windows-1252.

We fix these artifacts in-place for the ground-truth annotations.

Scope (intentionally minimal):
- data/annotations/*.json
- data/annotations_aldi5/*.json (if present)

Only modifies textual fields inside offers (product/price).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "€")


def _maybe_fix_mojibake(text: str) -> str:
    if not text or not any(m in text for m in _MOJIBAKE_MARKERS):
        return text

    fixed = None
    for encoding in ("cp1252", "latin-1"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
            break
        except (UnicodeEncodeError, UnicodeDecodeError):
            fixed = None

    if fixed is None:
        return text

    if fixed == text:
        return text

    # Heuristic: accept if we removed typical mojibake markers.
    if any(m in fixed for m in _MOJIBAKE_MARKERS):
        return text

    return fixed


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fix_file(path: Path) -> tuple[int, int]:
    """Return (offers_touched, string_fields_changed)."""
    data = _load_json(path)
    offers = []

    if isinstance(data, dict):
        offers = data.get("offers", []) or []
    else:
        # Unexpected for GT, but keep safe.
        return (0, 0)

    offers_touched = 0
    changed = 0

    for offer in offers:
        if not isinstance(offer, dict):
            continue

        before_product = offer.get("product")
        if isinstance(before_product, str):
            after_product = _maybe_fix_mojibake(before_product)
            if after_product != before_product:
                offer["product"] = after_product
                changed += 1
                offers_touched += 1

        before_price = offer.get("price")
        if isinstance(before_price, str):
            after_price = _maybe_fix_mojibake(before_price)
            if after_price != before_price:
                offer["price"] = after_price
                changed += 1
                offers_touched += 1

    if changed:
        _write_json(path, data)

    return offers_touched, changed


def iter_annotation_files(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if d.exists() and d.is_dir():
            files.extend(sorted(d.glob("*.json")))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix mojibake in GT annotation JSONs")
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["data/annotations", "data/annotations_aldi5"],
        help="Annotation directories to process (default: data/annotations data/annotations_aldi5)",
    )
    args = parser.parse_args()

    dirs = [Path(p) for p in args.dirs]
    files = iter_annotation_files(dirs)

    if not files:
        print("No annotation JSON files found in:")
        for d in dirs:
            print(f"  - {d}")
        return 1

    changed_files = 0
    total_offers_touched = 0
    total_strings_changed = 0

    for f in files:
        offers_touched, strings_changed = fix_file(f)
        if strings_changed:
            changed_files += 1
            total_offers_touched += offers_touched
            total_strings_changed += strings_changed

    print("GT encoding fix complete")
    print(f"  Files scanned:   {len(files)}")
    print(f"  Files changed:   {changed_files}")
    print(f"  Offers touched:  {total_offers_touched}")
    print(f"  Fields changed:  {total_strings_changed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
