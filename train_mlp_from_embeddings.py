# train_mlp_from_embeddingfolder.py
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from EmbeddingFolder import EmbeddingFolder
from models.MLP import MLP

# --------------------------- config --------------------------------------

DATASET_PATH = r"..\dataset\subset_filtered_embeddings"
BATCH_SIZE   = 2048
LR           = 1e-3
EPOCHS       = 20
SEED         = 42
EMBED_DIM    = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED, workers=True)

# --------------------------- dataset -------------------------------------

full_ds = EmbeddingFolder(DATASET_PATH)
n = len(full_ds)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

g = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print(f"Train dataloader size:{len(train_dl)}")
print(f"val dataloader size:{len(val_dl)}")
print(f"test dataloader size:{len(test_dl)}")

# --------------------------- training ------------------------------------

checkpoint_callback = ModelCheckpoint(
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    filename="mlp-best-f1-{epoch:02d}-{val_f1:.4f}",
    save_weights_only=True
)

model = MLP()
trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=1,
    deterministic=True,
    callbacks=[checkpoint_callback],
    log_every_n_steps=10
)

trainer.fit(model, train_dl, val_dl)
trainer.test(model, dataloaders=test_dl)

# best model path
print("Best checkpoint path:", checkpoint_callback.best_model_path)
