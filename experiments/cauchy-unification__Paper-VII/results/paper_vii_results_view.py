"""Export or reconstruct the dated Paper VII results without altering the input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


EDITIONS = {
    "results_50_domain_validation.json": (
        "c51988968d14e624a059deff6ceb2166d059d75a307f6da2413b5ba8fe7e24bc",
        "f8e650db0fe1b10bd79873ccdb67568860de4f81683334bd3bcbb7ceb14dfaec", "results"),
    "empirical_miss_analysis.json": (
        "edba33113699a9b5ad267861f505840bd131e4c449331fcd27910e2048afd7b2",
        "c49f1d137b24eed3360304fe9c9d6a63b627f2c005845669c9d9e5006a716aac", "misses"),
}


def reject_constant(value):
    raise ValueError("Non-standard JSON number: " + value)


def convert(data, restore=False):
    fingerprint = hashlib.sha256(data).hexdigest()
    for archived_hash, view_hash, row_key in EDITIONS.values():
        if fingerprint == (view_hash if restore else archived_hash):
            break
    else:
        raise ValueError("The input does not match the expected edition and direction")
    target = archived_hash if restore else view_hash
    old, new = (b"null", b"Infinity") if restore else (b"Infinity", b"null")
    # The hashes pin the whitespace and rule out replacing a quoted word or an unrelated null.
    pattern = rb'(?m)^(\s*"(?:aicc|bic)"\s*:\s*)' + old + rb'(?=,)'
    result, count = re.subn(pattern, lambda match: match[1] + new, data)
    if count != 4 or hashlib.sha256(result).hexdigest() != target:
        raise ValueError("The archived fields do not match the four recorded replacements")
    strict = data if restore else result
    rows = json.loads(strict, parse_constant=reject_constant)[row_key]
    row = next(item for item in rows if item["domain_id"] == 20)
    missing = [(fit["name"], key) for fit in row["top_fits"]
               for key in ("aicc", "bic") if fit[key] is None]
    if missing != [("hill", "aicc"), ("hill", "bic"), ("logistic", "aicc"), ("logistic", "bic")]:
        raise ValueError("Unavailable score fields differ from the published record")
    return result


def save_new(path, data):
    # Publish only a complete file and preserve an existing destination on reruns or concurrent writes.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("The destination already contains different bytes")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--restore", action="store_true", help="reconstruct the archived bytes in a separate file")
    args = parser.parse_args()
    try:
        if args.source.resolve() == args.output.resolve():
            raise ValueError("Choose a destination different from the source")
        save_new(args.output, convert(args.source.read_bytes(), args.restore))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
