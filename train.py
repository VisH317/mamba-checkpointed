import torch
from data.data import Genome, n_to_idx
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.mamba_checkpointed import MambaBlock, Mamba
from modules.lm_head import LMHead
from tqdm import tqdm
import torch.nn.functional as F
# from mamba_ssm import Mamba
import pickle
import logging

torch.set_warn_always(False)
torch.set_default_dtype(torch.float)

# config
d_in = 128
d_model = 64
d_ssm = 64
dt_rank = 8
vocab_size = len(n_to_idx.keys())
# d_in = 4
# d_model = 8
# d_ssm = 8
# dt_rank = 2
# vocab_size = len(n_to_idx.keys())

# train config
n_epochs = 4
batch_size = 8
val_batch_size = 4
val_step = 8
grad_accum_iter = 16
max_epoch_len = 15000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def collate(x: list[tuple[list[int], int, Tensor]]):
    target = torch.stack([t[1] for t in x]).to(device=device)
    return torch.stack([t[0] for t in x]).to(device=device), target

def create_mamba(config: dict, has_lmhead: bool = False):
    embed = nn.Embedding(vocab_size, config["d_in"])
    inner_model = nn.Sequential(
        Mamba(d_in),
        Mamba(d_in),
        Mamba(d_in),
        Mamba(d_in)
    )

    if has_lmhead: 
        print(vocab_size)
        lmhead = LMHead(config["d_in"], vocab_size)
        return nn.Sequential(embed, inner_model, lmhead).to(device=device, dtype=dtype)    

    return nn.Sequential(embed, inner_model).to(device=device, dtype=dtype)


# OG MAMBA d_in 64

def train(data_path: str):

    #model setup
    embed = nn.Embedding(vocab_size, d_in, dtype=dtype)

    # scaler = torch.cuda.amp.GradScaler()

    # inner_model = nn.Sequential(
    #     MambaBlock(d_in, d_model, d_ssm, dt_rank),
    #     MambaBlock(d_in, d_model, d_ssm, dt_rank),
    #     MambaBlock(d_in, d_model, d_ssm, dt_rank),
    # )
    inner_model = nn.Sequential(
        Mamba(d_in),
        Mamba(d_in),
        Mamba(d_in),
        Mamba(d_in),
        # Mamba(d_in),
        # Mamba(d_in)
    )

    lm_head = LMHead(d_in, vocab_size).to(device=device, dtype=dtype)

    model = nn.Sequential(embed, inner_model).to(device=device, dtype=dtype)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("param count: ", pytorch_total_params)

    # data setup
    dataset = Genome("data/genome.fna", existing_data_name=data_path)
    train_data, val_data = random_split(dataset, [0.7, 0.3])

    loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
    opt = optim.AdamW(model.parameters(), 1.5e-3, betas=(0.9, 0.95), weight_decay=0.1) # worked before: 1.5e-3
    # opt = optim.SGD(model.parameters(), lr=5e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    total = len(train_data) * n_epochs
    damp = batch_size * grad_accum_iter
    total_iters = total // (damp*8)
    warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.25, total_iters=total_iters)
    cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total//damp, eta_min=0.4)

    # data
    losses = []
    accuracies = []
    val_losses = [0]

    # main loop
    for epoch in range(n_epochs):
        train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_data, val_batch_size, shuffle=True, collate_fn=collate)
        val_loader_iter = iter(val_loader)

        logging.info("Data Loaded!")

        # test_input = torch.ones(2048, dtype=torch.int64).cuda()
        # test_input[320] = 16
        # target = torch.full_like(test_input, -100, dtype=torch.int64).cuda()
        # target[320] = 1
        # reuse_data = test_input.unsqueeze(0), target.unsqueeze(0)

        opt.zero_grad()
        for ix, data in (bar := tqdm(enumerate(train_loader), desc=f"Epoch: {epoch+1}, Loss: N/A, Val: N/A", total=len(train_data)//batch_size)):
            # if ix == 0: reuse_data = data
            input, target = data
            # if ix % 10 == 0: print(target.numel(), (target==-100).count_nonzero())
            # if ix % 10 == 0: print((input==16).count_nonzero(), (target==-100).count_nonzero(), target.numel())
            # with torch.cuda.amp.autocast():
            out = lm_head(model(input))
            # if ix % 50 == 0: print((torch.argmax(out, dim=-1)==target).count_nonzero(), torch.sum(target!=-100))
            acc = ((torch.argmax(out, dim=-1)==target).count_nonzero()/torch.sum(target!=-100)).detach().item()
            accuracies.append(acc)
            # out = lm_head(out)
            # target[~mask] = -100
            loss = loss_func(out.transpose(2, 1), target)

            # scaler.scale(loss).backward()
            loss.backward()

            losses.append(loss.item())
            # if ix%250 == 0:print(inner_model[0].out_proj.weight.grad)

            if ix % grad_accum_iter == 0:
                # scaler.step(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                nn.utils.clip_grad_norm_(lm_head.parameters(), 2)
                opt.step()
                opt.zero_grad()
                warmup.step()
                cosine.step()
                # if ix != 0: print(sum(losses[-16:])/16)
                # scaler.update()
            
            bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Acc: {round(accuracies[-1], 3)}")

            if ix % val_step == 0:
                with torch.no_grad():
                    try:
                        input, target = next(val_loader_iter)
                    except:
                        val_loader = DataLoader(val_data, val_batch_size, shuffle=True)
                        val_loader_iter = iter(val_loader)
                        input, target = next(val_loader_iter)
                    
                    out = lm_head(model(input))
                    # target[~mask] = -100
                    loss = loss_func(out.transpose(2, 1), target)
                    val_losses.append(loss.item())
                    # bar.set_description(f"Epoch: {epoch+1}, Loss: {round(losses[-1], 4)}, Val loss: {round(val_losses[-1], 4)}")

        # scheduler.step()

    torch.save(model.state_dict(), "model.pt")
    torch.save(lm_head.state_dict(), "lmhead.pt")
    with open("loss_data.pkl", "wb") as f:
        pickle.dump([losses, val_losses, accuracies], f)
    
    return model


if __name__ == "__main__":
    model = train("genome_seq.pkl")
