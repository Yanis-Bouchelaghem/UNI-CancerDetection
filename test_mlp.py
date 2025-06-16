import torch
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
from EmbeddingFolder import EmbeddingFolder
from models.MLP import MLP

# --------------------------- config --------------------------------------

TEST_SET_PATH = r"..\dataset\final_test_dataset_embeddings"
BATCH_SIZE   = 2048
LR           = 1e-3
EPOCHS       = 20
SEED         = 42
EMBED_DIM    = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED, workers=True)

# --------------------------- dataset -------------------------------------

test_dataset = EmbeddingFolder(TEST_SET_PATH)
test_dataset.class_to_idx
test_dl  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

CHECKPOINT_PATH = "model_checkpoints/simple-mlp-best-f1-epoch=13-val_f1=0.8996.ckpt"

model = MLP.load_from_checkpoint(CHECKPOINT_PATH)
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    deterministic=True,
    log_every_n_steps=10
)

trainer.test(model, dataloaders=test_dl)