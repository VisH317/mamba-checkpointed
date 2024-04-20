from Bio import SeqIO
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pickle
import random
import logging
import torch

# with open("genome.fna") as f:
#     i = 0
#     lens = []
#     seq = SeqIO.parse(f, "fasta")
#     print(seq[0].id)

# data loading mechanism

NUM_SEQ = 705
MAX_LEN = 1000

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

class Genome(Dataset):
    def __init__(self, file_name: str, max_seq: int = 100, num_seq: int = NUM_SEQ, max_len: int = MAX_LEN, existing_data_name: str | None = None) -> None:
        self.seqs = []

        if existing_data_name is not None:
            with open(existing_data_name, "rb") as f:
                self.seqs = pickle.load(f)
                logging.info("Previous Data Loaded Successfully")
        else:
            with open(file_name) as f:
                for record in tqdm(SeqIO.parse(f, "fasta"), desc="creating dataset...", total=num_seq):
                    l = len(record.seq)
                    for i in range(l//max_len):
                        target = torch.as_tensor([n_to_idx[n.lower()] for n in record.seq[i*max_len:(i+1)*max_len]])

                        mask_idx = torch.randint(low=0, high=5, size=target.size())
                        li = torch.where(mask_idx==0, torch.full_like(target, 16), target)

                        s = set(li)
                        if not (len(s) <= 3 and 15 in s): self.seqs.append((li, target))
                        if i >= max_seq: break

            with open("genome_seq.pkl", "wb") as f:
                pickle.dump(self.seqs, f)
                logging.info("Data Constructed Successfully")

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return torch.tensor(self.seqs[idx][0], dtype=torch.long), torch.tensor(self.seqs[idx][1], dtype=torch.long)

