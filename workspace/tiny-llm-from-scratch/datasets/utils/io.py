import json

from pathlib import Path


def load_json(path):

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )