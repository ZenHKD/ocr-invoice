import torch

class Tokenizer:
    def __init__(self, charset):
        # 0: BOS (Begin of Sentence)
        # 1: EOS (End of Sentence)
        # 2: PAD (Padding)
        self.bos = '[B]'
        self.eos = '[E]'
        self.pad = '[P]'
        self.specials = [self.bos, self.eos, self.pad]

        self.charset = list(sorted(list(set(charset))))
        self.token_to_id = {t: i + len(self.specials) for i, t in enumerate(self.charset)}

        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2

        self.token_to_id[self.bos] = self.bos_id
        self.token_to_id[self.eos] = self.eos_id
        self.token_to_id[self.pad] = self.pad_id

        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.num_classes = len(self.token_to_id)

    def encode(self, texts, device='cpu', max_length=25):
        # texts: list of strings
        batch_ids = []
        for text in texts:
            ids = [self.bos_id] + [self.token_to_id[c] for c in text if c in self.token_to_id] + [self.eos_id]
            batch_ids.append(ids)

        # Pad
        max_len = min(max([len(ids) for ids in batch_ids]), max_length + 2) # +2 for BOS/EOS
        padded_ids = []
        for ids in batch_ids:
            if len(ids) > max_len:
                ids = ids[:max_len-1] + [self.eos_id]
            else:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            padded_ids.append(ids)

        return torch.tensor(padded_ids, dtype=torch.long, device=device)

    def decode(self, token_ids):
        # token_ids: list of ints or tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        texts = []
        for ids in token_ids:
            text = []
            for i in ids:
                if i == self.eos_id:
                    break
                if i == self.bos_id or i == self.pad_id:
                    continue
                text.append(self.id_to_token[i])
            texts.append("".join(text))
        return texts
