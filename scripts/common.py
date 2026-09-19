from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
def read_text(path):
    return Path(path).read_text(encoding="utf-8")
