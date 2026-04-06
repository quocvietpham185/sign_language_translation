"""
vocabulary.py — Vocabulary for gloss and text
"""

import json
from pathlib import Path
from typing import List, Optional


class Vocabulary:
    """
    Token vocabulary hỗ trợ BPE-style encoding đơn giản.
    Tương thích với mBART tokenizer output.
    """
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    BLANK = "<blank>"  # CTC blank

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.pad_id = 0
        self.blank_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self._add_special_tokens()

    def _add_special_tokens(self):
        for tok in [self.BLANK, self.UNK, self.BOS, self.EOS]:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        return self.token2id[token]

    def build_from_texts(self, texts: List[str]):
        """Build vocab từ list of strings."""
        for text in texts:
            for token in text.strip().split():
                self._add(token)
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in text.strip().split():
            ids.append(self.token2id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids if ids else [self.unk_id]

    def ids_to_text(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, self.UNK)
            if tok in (self.BOS, self.EOS, self.PAD, self.BLANK):
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def __len__(self):
        return len(self.token2id)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            data = json.load(f)
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        vocab.pad_id = vocab.token2id.get(cls.PAD, 0)
        vocab.blank_id = vocab.token2id.get(cls.BLANK, 0)
        vocab.unk_id = vocab.token2id.get(cls.UNK, 1)
        vocab.bos_id = vocab.token2id.get(cls.BOS, 2)
        vocab.eos_id = vocab.token2id.get(cls.EOS, 3)
        return vocab

    @classmethod
    def build_from_json(cls, data_path: str, field: str) -> "Vocabulary":
        """Build vocab từ JSON dataset."""
        import json as _json
        vocab = cls()
        with open(data_path) as f:
            samples = _json.load(f)
        for s in samples:
            text = s.get(field, "")
            for tok in text.strip().split():
                vocab._add(tok)
        return vocab
