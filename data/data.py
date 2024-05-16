from Bio import SeqIO
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from pyfaidx import Fasta
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
MAX_LEN = 1024
MAX_LEN_2 = 32768
MAX_LEN_3 = 131072

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

# class FastaRetriever:
#     def __init__(self, file_name: str) -> None:
#         self.genome = Fasta(file_name)


class Genome(Dataset):
    def __init__(self, file_name: str, max_seq: int = 50, num_seq: int = NUM_SEQ, max_len: int = MAX_LEN_3, existing_data_name: str | None = None) -> None:
        self.seqs = []

        MAX_PER_CHR = max_seq

        if existing_data_name is not None:
            with open(existing_data_name, "rb") as f:
                self.seqs = pickle.load(f)
                logging.info("Previous Data Loaded Successfully")
        else:
            with open(file_name) as f:
                for record in tqdm(SeqIO.parse(f, "fasta"), desc="creating dataset...", total=num_seq):
                    l = len(record.seq)
                    # print(l//max_len)
                    if l - max_len - 1 < 0: continue
                    r = torch.randint(l - max_len - 1, tuple([l]))
                    for i in trange(min(MAX_PER_CHR, l//max_len)):
                        target = torch.as_tensor([n_to_idx[n.lower()] for n in record.seq[r[i].item():(r[i].item() + max_len)]])

                        mask_idx = torch.randint(low=0, high=5, size=target.size())
                        mask = mask_idx == 0 
                        li = torch.where(mask_idx==0, torch.full_like(target, 16), target)
                        rand_idx = torch.randint_like(li, low=0, high=4)
                        use_rand = torch.randint_like(li, low=0, high=10)
                        li = torch.where(torch.logical_and(li==16, use_rand==0), rand_idx, li)

                        rand_idx = torch.randint_like(li, low=0, high=4)
                        use_rand = torch.randint_like(li, low=0, high=10)
                        li = torch.where(torch.logical_and(li==16, use_rand==0), target, li)

                        target = torch.where(mask_idx==0, target, torch.full_like(target, -100))

                        s = set(li)
                        if not (len(s) <= 3 and 15 in s): self.seqs.append((li, target, mask))

            with open("genome_seq_131k.pkl", "wb") as f:
                pickle.dump(self.seqs, f)
                logging.info("Data Constructed Successfully")

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return torch.tensor(self.seqs[idx][0], dtype=torch.long), torch.tensor(self.seqs[idx][1], dtype=torch.long)

if __name__ == "__main__":
    Genome("genome.fna")