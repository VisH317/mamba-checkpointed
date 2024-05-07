import torch
from torch import nn
from benchmark.data.gb_train import train_datasets
from benchmark.data.gb_utils import collate
from train import create_mamba
from modules.mamba_utils import Pooler, Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

default_config = {
    "batch_size": 16,
    "learning_rate": 3e-4,
    "freeze_foundation": False,
    "grad_accum_steps": 4,
    "dropout_p": 0.25,
    "epoch": 5
}

default_model_config = {
    "d_in": 32,
    "d_model": 64,
    "d_ssm": 128,
    "dt_rank": 32
}




def finetune(dataset_name: str, pretrained_path: str, train_config: dict = default_config, model_config: dict = default_model_config):
    model = create_mamba()
    pooler = Pooler(model_config["d_model"])
    classifier = Classifier(model_config["d_model"], 2)

    model = nn.Sequential(model, pooler, classifier).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(pretrained_path)
    if dataset_name not in train_datasets.keys(): raise KeyError("model finetuning process: dataset name does not exist")

    data = train_datasets[dataset_name]
    total_len = len(data) / train_config["batch_size"]

    losses = []

    criterion = nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    
    # training loop
    for _ in range(train_config["epoch"]):
        loader = DataLoader(data, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collate)

        optim.zero_grad()
        for ix, data in (bar := tqdm(enumerate(loader), total=total_len, desc=f"finetuning on {dataset_name}, Loss: N/A")):
            x, y_t = data
            y = model(x)
            loss = criterion(y, y_t)
            loss.backward()

            losses.append(loss.item())

            bar.set_description(f"finetuning on {dataset_name}, Loss: {loss.item()}")

            if ix % train_config["grad_accum_steps"] == 0:
                optim.step()
                optim.zero_grad()
        
        scheduler.step()

    torch.save(model.state_dict(), f"model_{dataset_name}_checkpoint.pt")
    with open(f"loss_{dataset_name}.pkl", "wb") as f:
        pickle.dump(losses, f)
    
    return model

# eval function here later
