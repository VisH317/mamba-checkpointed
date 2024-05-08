import torch
from torch import Tensor

n_to_idx = {
    "a": 0,
    "g": 1,
    "c": 2,
    "t": 3,
    "u": 4,
    "r": 5,
    "y": 6,
    "k": 7,
    "m": 8,
    "s": 9,
    "w": 10,
    "b": 11,
    "d": 12,
    "h": 13,
    "v": 14,
    "n": 15,
    "<MASK>": 16
}

VOCAB_SIZE = len(n_to_idx.keys())

MAX_SEQ_LEN = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(dna: str, max_len: int = MAX_SEQ_LEN) -> Tensor:
    if len(dna) > 512: dna = dna[:512]
    li = torch.empty(tuple([MAX_SEQ_LEN]), dtype=torch.int64)
    for i, nuc in enumerate(dna.lower()):
        li[i] = n_to_idx[nuc]
    if len(dna) < 512:
        for i in range(512 - len(dna), 512): li[i] = n_to_idx["n"]

    return li

def collate(data: list[tuple[str, Tensor]]) -> tuple[Tensor, Tensor]:
    ten_li = []

    for seq in data: ten_li.append(tokenize(seq[0]))
    target = torch.empty(tuple([len(data)]))
    for ix, item in enumerate(data): target[ix] = item[1]

    ten_li_t = torch.stack(ten_li, dim=0)

    print(ten_li_t.size(), target.size())

    return ten_li_t.to(device=device), target.to(device=device)
