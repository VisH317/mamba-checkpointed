from torch import Tensor, empty

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

def tokenize(dna: str) -> Tensor:
    li = empty((len(dna)))
    for i, nuc in enumerate(dna.lower()):
        li[i] = n_to_idx[nuc]
    return li

def collate(data: list[tuple[str], Tensor]) -> tuple[Tensor, Tensor]:
    ten_li = []
    for seq in data[0]: ten_li.append(tokenize(seq))
    return ten_li, data[1]
