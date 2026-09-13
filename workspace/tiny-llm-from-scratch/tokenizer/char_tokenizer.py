import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = payload["stoi"]
        itos = {int(k): v for k, v in payload["itos"].items()}
        return cls(stoi, itos)

    def encode(self, text):
        unk_id = self.stoi["<UNK>"]
        return [self.stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids):
        return "".join([self.itos.get(int(i), "<UNK>") for i in ids])
