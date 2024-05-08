import torch
from data.data import Genome, n_to_idx
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.mamba_checkpointed import MambaBlock
from modules.lm_head import LMHead
from tqdm import tqdm
import pickle
import logging

torch.set_warn_always(False)
torch.set_default_dtype(torch.float)

# config
d_in = 64
d_model = 128
d_ssm = 128
dt_rank = 64
vocab_size = len(n_to_idx.keys())
# d_in = 4
# d_model = 8
# d_ssm = 8
# dt_rank = 2
# vocab_size = len(n_to_idx.keys())

# train config
n_epochs = 3
batch_size = 2
val_batch_size = 4
val_step = 8
grad_accum_iter = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate(x: list[tuple[list[int], int]]):
    return torch.stack([t[0] for t in x]).to(device=device), torch.stack([t[1] for t in x]).to(device=device)

def create_mamba(config: dict, has_lmhead: bool = False):
    embed = nn.Embedding(vocab_size, config["d_in"])
    inner_model = nn.Sequential(
        MambaBlock(config["d_in"], config["d_model"], config["d_ssm"], config["dt_rank"]),
        MambaBlock(config["d_in"], config["d_model"], config["d_ssm"], config["dt_rank"]),
        MambaBlock(config["d_in"], config["d_model"], config["d_ssm"], config["dt_rank"]),
    )

    if has_lmhead: 
        print(vocab_size)
        lmhead = LMHead(config["d_in"], vocab_size)
        return nn.Sequential(embed, inner_model, lmhead).to(device=device)    

    return nn.Sequential(embed, inner_model).to(device=device)


def train():

    #model setup
    embed = nn.Embedding(vocab_size, d_in)

    # scaler = torch.cuda.amp.GradScaler()

    inner_model = nn.Sequential(
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
        MambaBlock(d_in, d_model, d_ssm, dt_rank),
    )

    lm_head = LMHead(d_in, vocab_size).to(device=device)

    model = nn.Sequential(embed, inner_model).to(device=device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("param count: ", pytorch_total_params)

    # data setup
    dataset = Genome("data/genome.fna", existing_data_name="genome_seq.pkl")
    train_data, val_data = random_split(dataset, [0.7, 0.3])

    loss_func = nn.CrossEntropyLoss(reduction="mean")
    opt = optim.AdamW(model.parameters(), 8e-3, betas=(0.95, 0.999))
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
        for ix, data in (bar := tqdm(enumerate(train_loader), desc=f"Epoch: {epoch+1}, Loss: N/A, Val: N/A", total=len(train_data)//batch_size)):
            input, target = data
            with torch.cuda.amp.autocast():
                out = lm_head(model(input))
                loss = loss_func(out.transpose(2, 1), target)
            
            # scaler.scale(loss).backward()
            loss.backward()

            if ix % grad_accum_iter == 0:
                # scaler.step(opt)
                opt.step()
                opt.zero_grad()
                # scaler.update()
            
            losses.append(loss.item())
            bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")

            if ix % val_step == 0:
                with torch.no_grad():
                    try:
                        input, target = next(val_loader_iter)
                    except:
                        val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
                        val_loader_iter = iter(val_loader)
                        input, target = next(val_loader_iter)
                    
                    out = lm_head(model(torch.tensor(input, device=device)))
                    loss = loss_func(out.transpose(2, 1), torch.tensor(target, device=device))
                    val_losses.append(loss.item())
                    bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")

        scheduler.step()

    torch.save(model.state_dict(), "model.pt")
    torch.save(lm_head.state_dict(), "lmhead.pt")
    with open("loss_data.pkl", "wb") as f:
        pickle.dump([losses, val_losses], f)
    
    return model


if __name__ == "__main__":
    model = train()
