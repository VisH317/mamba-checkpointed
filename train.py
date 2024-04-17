import torch
from data.data import Genome, n_to_idx
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.mamba_checkpointed import MambaBlock
from modules.lm_head import LMHead
from tqdm import tqdm
import pickle
import logging

# config
d_in = 8
d_model = 16
d_ssm = 32
dt_rank = 8
vocab_size = len(n_to_idx.keys())

# train config
n_epochs = 3
batch_size = 16
val_batch_size = 8
val_step = 8
grad_accum_iter = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(x: list[tuple[list[int], int]]):
    return torch.stack([t[0] for t in x]).to(device=device), torch.stack([t[1] for t in x]).to(device=device)

def train():

    #model setup
    embed = nn.Embedding(vocab_size, d_in)

    inner_model = nn.Sequential(
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
    )

    lm_head = LMHead(d_in, vocab_size)

    model = nn.Sequential(embed, inner_model, lm_head).to(device=device)

    # data setup
    dataset = Genome("data/genome.fna", existing_data_name="genome_seq.pkl")
    train_data, val_data = random_split(dataset, [0.7, 0.3])

    loss_func = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.AdamW(model.parameters(), 3e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    # data
    losses = []
    val_losses = [0]

    # main loop
    for epoch in range(n_epochs):
        train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_data, val_batch_size, shuffle=True, collate_fn=collate)
        val_loader_iter = iter(val_loader)

        logging.info("Data Loaded!")

        opt.zero_grad()
        for ix, data in (bar := tqdm(enumerate(train_loader), desc=f"Epoch: {epoch+1}, Loss: N/A, Val: N/A", total=len(train_data))):
            input, mask = data
            out = model(input)
            loss = loss_func(out, mask)
            loss.backward()

            if ix % grad_accum_iter == 0:
                opt.step()
                opt.zero_grad()
            
            losses.append(loss.item())
            bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")

            if ix % val_step == 0:
                with torch.no_grad():
                    try:
                        input, mask = next(val_loader_iter)
                    except:
                        val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
                        val_loader_iter = iter(val_loader)
                        input, mask = next(val_loader_iter)
                    
                    out = model(torch.tensor(input, device=device))
                    loss = loss_func(out, torch.tensor(mask, device=device))
                    val_losses.append(loss.item())
                    bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")
            
            if ix == 1: break
        scheduler.step()
        break
    
    torch.save(model.state_dict(), "model.pt")
    with open("loss_data.pkl", "wb") as f:
        pickle.dump([losses, val_losses], f)
    
    return model


if __name__ == "__main__":
    model = train()
