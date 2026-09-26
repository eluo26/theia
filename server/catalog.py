"""The sample closet's catalog: one entry per sample box lot, stored as JSON.

The seed data uses made-up drug names (Cardiolex and so on). Replace them with
the names printed on the real mock boxes once those exist.
"""
import json
import os
import re
import threading
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

_write_lock = threading.Lock()


def load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(items):
    tmp_path = CATALOG_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, CATALOG_PATH)


def make_id(drug_name, strength, lot):
    """Build an entry id like 'cardiolex-10-mg-cx26a118' from the fields that identify a lot."""
    slug = "-".join(part for part in (drug_name, strength, lot) if part)
    return re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")


def _same_lot(item, drug_name, strength, lot):
    return (
        item["drug_name"].lower() == drug_name.lower()
        and item["strength"].lower() == strength.lower()
        and (item.get("lot") or "").lower() == (lot or "").lower()
    )


def find_location(catalog, drug_name, strength):
    """Return (shelf, bin) of an existing entry for this drug and strength, or (None, None)."""
    for item in catalog:
        if item["drug_name"].lower() == drug_name.lower() and item["strength"].lower() == strength.lower():
            return item["shelf"], item["bin"]
    return None, None


def add_stock(entry, count=1):
    """Add `count` boxes described by `entry` to the catalog.

    If an entry with the same drug, strength and lot exists, its count grows.
    Otherwise a new entry is created, so each lot keeps its own expiration date.
    Returns the saved entry.
    """
    with _write_lock:
        catalog = load_catalog()
        for item in catalog:
            if _same_lot(item, entry["drug_name"], entry["strength"], entry.get("lot")):
                item["count"] += count
                save_catalog(catalog)
                return item

        new_item = {
            "id": make_id(entry["drug_name"], entry["strength"], entry.get("lot")),
            "drug_name": entry["drug_name"],
            "strength": entry["strength"],
            "box_color": entry.get("box_color"),
            "ndc": entry.get("ndc"),
            "lot": entry.get("lot"),
            "expiration_date": entry.get("expiration_date"),
            "count": count,
            "shelf": entry.get("shelf"),
            "bin": entry.get("bin"),
        }
        catalog.append(new_item)
        save_catalog(catalog)
        return new_item
