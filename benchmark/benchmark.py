import torch
from torch import nn
from benchmark.data.gb_train import train_datasets
from benchmark.data.gb_utils import collate, VOCAB_SIZE
from train import create_mamba
from modules.mamba_utils import Pooler, Classifier
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

default_config = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "grad_accum_steps": 4,
    "dropout_p": 0.25,
    "epoch": 5
}

default_model_config = {
    "d_in": 64,
    "d_model": 64,
    "d_ssm": 64,
    "dt_rank": 8
}

def finetune(dataset_name: str, pretrained_path: str, train_config: dict = default_config, model_config: dict = default_model_config, has_lmhead: bool = False):
    model = create_mamba(model_config, has_lmhead=has_lmhead)
    pooler = Pooler(model_config["d_in"])
    classifier = Classifier(model_config["d_in"], 2)

    model.load_state_dict(torch.load(pretrained_path), strict=False)
    model = nn.Sequential(model, pooler, classifier).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if has_lmhead: model = model[:2]

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("param count: ", pytorch_total_params)

    batch_size = 16

    assert dataset_name in train_datasets.keys(), "model finetuning process: dataset name does not exist"
    data = train_datasets[dataset_name]
    total_len = len(data) // batch_size

    losses = []
    accuracies = []

    MAX_EPOCH_LEN = 2500

    criterion = nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=train_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    
    # training loop
    for _ in range(train_config["epoch"]):
        data = train_datasets[dataset_name]
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate)

        optim.zero_grad()
        for ix, data in (bar := tqdm(enumerate(loader), total=min(MAX_EPOCH_LEN, total_len), desc=f"finetuning on {dataset_name}, Loss: N/A")):
            x, y_t = data
            
            # print("testing: ", x)
            y = model(x)
            loss = criterion(y[:, 0], y_t)
            target = y[:, 0] > 0.5
            accuracy = torch.sum(target==(y_t==1))/batch_size
            loss.backward()

            accuracies.append(accuracy.detach().item())
            losses.append(loss.item())

            if ix% 10 == 0: bar.set_description(f"finetuning on {dataset_name}, Loss: {round(loss.item(), 4)}, Acc: {round(sum(accuracies[max(0, len(accuracies)-16):-1])/batch_size, 2)}")

            if ix >= MAX_EPOCH_LEN: break

            if ix % train_config["grad_accum_steps"] == 0:
                optim.step()
                optim.zero_grad()
        
        scheduler.step()

    torch.save(model.state_dict(), f"model_{dataset_name}_checkpoint.pt")
    with open(f"loss_{dataset_name}.pkl", "wb") as f:
        pickle.dump(losses, f)
    
    return model

# eval function here later
