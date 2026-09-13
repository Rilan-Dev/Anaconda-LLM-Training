import json

from pathlib import Path

from core.paths import CONFIG_DIR

CONFIG_FILE = CONFIG_DIR / "tiny_100k.json"


def load_config():

    return json.loads(
        Path(CONFIG_FILE).read_text(
            encoding="utf-8"
        )
    )