"""Verify each reader copy and reconstruct its archived file in memory."""
from hashlib import sha256
import json
from pathlib import Path

def reject_constant(value):
    raise ValueError("Invalid JSON number: " + value)

def verify(root):
    key = json.loads((root / "field-key.json").read_bytes())
    for entry in key["files"]:
        original = (root / entry["original"]).read_bytes()
        standard = (root / entry["standard"]).read_bytes()
        if sha256(original).hexdigest() != entry["original_sha256"] or sha256(standard).hexdigest() != entry["standard_sha256"]:
            raise ValueError("File fingerprint differs: " + entry["original"])
        parsed = json.loads(standard, parse_constant=reject_constant)
        restored = bytearray(standard)
        for field in sorted(entry["fields"], key=lambda item: item["view_byte_offset"], reverse=True):
            point = parsed
            for token in field["pointer"][1:].split("/"):
                point = point[token.replace("~1", "/").replace("~0", "~")]
            offset = field["view_byte_offset"]
            if point is not None or restored[offset:offset+4] != b"null":
                raise ValueError("Unexpected value at " + field["pointer"])
            restored[offset:offset+4] = b"NaN"
        if restored != original:
            raise ValueError("Reconstruction differs: " + entry["original"])
        print("Verified", entry["standard"])

if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
