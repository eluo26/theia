"""Read a sample box label into structured fields with a vision model.

The model is chosen with environment variables (or a .env file in the project
root; see .env.example) so we can switch from OpenAI to xAI (Grok) without code
changes:

    VISION_PROVIDER   "openai" (default) or "xai"
    OPENAI_API_KEY    key for the openai provider
    XAI_API_KEY       key for the xai provider
    VISION_MODEL      optional model override

Try it on a photo without the UI:

    python server/label_reader.py photo.jpg
"""
import argparse
import base64
import calendar
import difflib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-6-astra"},
    "xai": {"base_url": "https://api.x.ai/v1", "key_env": "XAI_API_KEY", "model": "grok-4.6"},
}

FIELD_LABELS = {
    "drug_name": "drug name",
    "strength": "strength",
    "ndc": "NDC",
    "lot": "lot number",
    "expiration_text": "expiration date",
    "box_color": "box color",
}

_nullable = lambda description: {"type": ["string", "null"], "description": description}  # noqa: E731

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "drug_name": _nullable("Brand name of the drug as printed, without the strength."),
        "strength": _nullable("Dose per unit with its unit, e.g. '10 mg' or '90 mcg'."),
        "ndc": _nullable("The NDC number exactly as printed, keeping its hyphens."),
        "lot": _nullable("Lot or batch number exactly as printed."),
        "expiration_text": _nullable("Expiration date exactly as printed, e.g. 'EXP 03/2027'."),
        "box_color": _nullable("Dominant color of the box in one or two plain words, e.g. 'red' or 'dark blue'."),
        "notes": _nullable("Anything that made the label hard to read (glare, blur, cut off), else null."),
    },
    "required": ["drug_name", "strength", "ndc", "lot", "expiration_text", "box_color", "notes"],
    "additionalProperties": False,
}

PROMPT = (
    "You extract inventory fields from a photo of a drug sample label. "
    "Handwritten notes, printed mock-ups, notebook paper, and real boxes are all valid labels. "
    "Never refuse a photo because it is handwritten or on paper. Read the visible text anyway. "
    "Report only text you can actually see. If a field is not visible or not clearly legible, return null for it. "
    "Never guess or invent a value. "
    "For box_color, report the color of the box or paper. "
    "Do not confuse the lot number with the NDC: the NDC is a hyphenated 10- or 11-digit number, "
    "usually labelled 'NDC'; the lot is labelled 'LOT' or 'Lot No'. "
    "Copy the expiration date exactly as written, including any 'EXP' prefix. "
    "Put observations about handwriting, paper, or glare in notes, not by leaving every field null."
)


class LabelReaderError(RuntimeError):
    pass


def _client_and_model():
    provider = os.environ.get("VISION_PROVIDER", "openai").lower()
    if provider not in PROVIDERS:
        raise LabelReaderError(f"Unknown VISION_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    config = PROVIDERS[provider]
    api_key = os.environ.get(config["key_env"])
    if not api_key:
        raise LabelReaderError(f"Set the {config['key_env']} environment variable to use the {provider} provider.")
    client = OpenAI(api_key=api_key, base_url=config["base_url"])
    return client, os.environ.get("VISION_MODEL", config["model"])


def ask_model(image_bytes, mime_type="image/jpeg"):
    """Send the image to the vision model and return the raw label fields it reports."""
    client, model = _client_and_model()
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this drug sample label."},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "box_label", "strict": True, "schema": LABEL_SCHEMA},
        },
    }
    try:
        response = client.chat.completions.create(**request, temperature=0)
    except BadRequestError as error:
        if "temperature" not in str(error):
            raise
        response = client.chat.completions.create(**request)  # some models reject temperature
    return json.loads(response.choices[0].message.content)


MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_abbr) if name}


def parse_expiration(text):
    """Turn printed expiration text into an ISO date. Month-only dates mean the end of that month."""
    if not text:
        return None
    cleaned = re.sub(r"(?i)\b(exp(?:iry|ires|iration)?\.?|use by|best before)\b[:.]?", " ", text).strip()

    def end_of_month(year, month):
        return date(year, month, calendar.monthrange(year, month)[1])

    def full_year(value):
        year = int(value)
        return year + 2000 if year < 100 else year

    patterns = [
        (r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
        (r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", lambda m: date(int(m[3]), int(m[1]), int(m[2]))),
        (r"^(\d{4})[-/.](\d{1,2})$", lambda m: end_of_month(int(m[1]), int(m[2]))),
        (r"^(\d{1,2})[-/.](\d{2}|\d{4})$", lambda m: end_of_month(full_year(m[2]), int(m[1]))),
        (r"^([A-Za-z]{3,9})\.?\s+(\d{4})$", lambda m: end_of_month(int(m[2]), MONTHS[m[1][:3].lower()])),
        (r"^(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{2,4})$",
         lambda m: date(full_year(m[3]), MONTHS[m[2][:3].lower()], int(m[1]))),
        (r"^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})$",
         lambda m: date(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))),
    ]
    for pattern, build in patterns:
        match = re.match(pattern, cleaned)
        if match:
            try:
                return build(match).isoformat()
            except (KeyError, ValueError):
                return None
    return None


NDC_FORMATS = re.compile(r"^(\d{4}-\d{4}-\d{2}|\d{5}-\d{3}-\d{2}|\d{5}-\d{4}-\d{1}|\d{5}-\d{4}-\d{2})$")


def ndc_is_valid(text):
    return bool(text) and bool(NDC_FORMATS.match(text.strip()))


def match_drug_name(name, catalog):
    """Return (canonical_name, warning). Matches the read name to an existing catalog name when close."""
    if not name:
        return name, None
    known = sorted({item["drug_name"] for item in catalog})
    for candidate in known:
        if candidate.lower() == name.strip().lower():
            return candidate, None
    close = difflib.get_close_matches(name.strip().lower(), [k.lower() for k in known], n=1, cutoff=0.8)
    if close:
        canonical = next(k for k in known if k.lower() == close[0])
        return canonical, f"Read the drug name as '{name}'; matched it to '{canonical}' in the catalog."
    return name.strip(), None


def interpret(raw, catalog=()):
    """Clean up the model's raw answer and collect warnings for the clinician."""
    warnings = []
    fields = {key: (raw.get(key) or None) for key in FIELD_LABELS}
    for key, value in fields.items():
        if isinstance(value, str):
            fields[key] = value.strip() or None

    for key, label in FIELD_LABELS.items():
        if fields[key] is None:
            warnings.append(f"Could not read the {label}.")

    fields["drug_name"], warning = match_drug_name(fields["drug_name"], catalog)
    if warning:
        warnings.append(warning)

    fields["expiration_date"] = parse_expiration(fields["expiration_text"])
    if fields["expiration_text"] and not fields["expiration_date"]:
        warnings.append(f"Could not understand the expiration date '{fields['expiration_text']}'. Enter it by hand.")

    if fields["ndc"] and not ndc_is_valid(fields["ndc"]):
        warnings.append(f"'{fields['ndc']}' does not look like a valid NDC. Check it.")

    if raw.get("notes"):
        warnings.append(f"Model note: {raw['notes']}")

    return {"fields": fields, "warnings": warnings, "raw": raw}


def read_label(image_bytes, catalog=(), mime_type="image/jpeg"):
    """Read a label photo (JPEG/PNG bytes) into {"fields", "warnings", "raw"}."""
    return interpret(ask_model(image_bytes, mime_type), catalog)


def main():
    parser = argparse.ArgumentParser(description="Read a sample box label from a photo.")
    parser.add_argument("photo", help="path to a .jpg or .png photo of the box")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from catalog import load_catalog

    with open(args.photo, "rb") as f:
        image_bytes = f.read()
    mime_type = "image/png" if args.photo.lower().endswith(".png") else "image/jpeg"
    try:
        result = read_label(image_bytes, load_catalog(), mime_type)
    except LabelReaderError as error:
        sys.exit(f"Error: {error}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
