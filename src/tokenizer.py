class SimpleTokenizer:
    def __init__(self):
        self.special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
        ]
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, texts):
        print("\nBuilding vocabulary...")
        tokens = set()

        for text in texts:
            for token in text.split():
                tokens.add(token)

        vocabulary = self.special_tokens + sorted(tokens)

        for idx, token in enumerate(vocabulary):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        print(f"Vocabulary size: {len(self.token_to_id)}")

    def encode(self, text, max_length):
        tokens = ["[CLS]"] + text.split() + ["[SEP]"]

        ids = [
            self.token_to_id.get(token, self.token_to_id["[UNK]"])
            for token in tokens
        ]

        ids = ids[:max_length]

        while len(ids) < max_length:
            ids.append(self.token_to_id["[PAD]"])

        return ids

    def decode(self, ids):
        return " ".join(
            self.id_to_token.get(idx, "[UNK]")
            for idx in ids
        )

    @property
    def vocab_size(self):
        return len(self.token_to_id)
