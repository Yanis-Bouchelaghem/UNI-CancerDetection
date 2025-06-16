# preprocess_and_train_gpu.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import pytorch_lightning as pl
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_precision,
    binary_recall,
    binary_f1_score,
)
from huggingface_hub import login
from dotenv import load_dotenv

DATASET_PATH = r"..\dataset\subset_filtered_classification_dataset"
EMBED_FILE   = "uni_embeddings.pt"
SEED         = 42
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 10
EMBED_DIM    = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pl.seed_everything(SEED, workers=True)
load_dotenv()
login(token=os.environ["HF_TOKEN"])

backbone = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
).to(device).eval()
for p in backbone.parameters():
    p.requires_grad = False

transform = create_transform(**resolve_data_config(backbone.pretrained_cfg, model=backbone))

def cache_embeddings():
    if os.path.exists(EMBED_FILE):
        return torch.load(EMBED_FILE)
    ds = ImageFolder(DATASET_PATH, transform=transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    embs, lbls = [], []
    with torch.no_grad():
        for imgs, y in dl:
            imgs = imgs.to(device, non_blocking=True)
            embs.append(backbone(imgs).cpu())
            lbls.append(y)
    embs = torch.cat(embs)
    lbls = torch.cat(lbls)
    torch.save({"emb": embs, "lab": lbls}, EMBED_FILE)
    return {"emb": embs, "lab": lbls}

cache = cache_embeddings()

class EmbeddingDataset(Dataset):
    def __init__(self, embs, labs, idx):
        self.embs = embs[idx]
        self.labs = labs[idx]
    def __len__(self):
        return self.labs.size(0)
    def __getitem__(self, i):
        return self.embs[i], self.labs[i]

n = cache["lab"].size(0)
n_tr = int(0.7 * n)
n_va = int(0.15 * n)
n_te = n - n_tr - n_va
perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
tr_i, va_i, te_i = torch.split(perm, [n_tr, n_va, n_te])

tr_ds = EmbeddingDataset(cache["emb"], cache["lab"], tr_i)
va_ds = EmbeddingDataset(cache["emb"], cache["lab"], va_i)
te_ds = EmbeddingDataset(cache["emb"], cache["lab"], te_i)

tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.loss = nn.CrossEntropyLoss()
    def forward(self, x):
        return self.net(x)
    def step(self, batch, tag):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(1)
        self.log(f"{tag}_loss", loss, prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_acc",  binary_accuracy(preds, y),  prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_prec", binary_precision(preds, y), prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_rec",  binary_recall(preds, y),   prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_f1",   binary_f1_score(preds, y), prog_bar=True, batch_size=y.size(0))
        return loss
    def training_step(self, b, _):
        return self.step(b, "train")
    def validation_step(self, b, _):
        self.step(b, "val")
    def test_step(self, b, _):
        self.step(b, "test")
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

def main():
    model   = MLP()
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        deterministic=True,
        log_every_n_steps=10
    )
    trainer.fit(model, tr_dl, va_dl)
    trainer.test(model, dataloaders=te_dl)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
