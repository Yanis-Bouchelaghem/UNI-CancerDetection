import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from EmbeddingFolder import EmbeddingFolder
from models.MLP import MLP

def main():
    # --------------------------- config --------------------------------------

    TRAIN_DATASET_PATH = r"..\dataset\riadh_train_embeddings"
    TEST_DATASET_PATH  = r"..\dataset\!test_dataset_embeddings"
    BATCH_SIZE   = 32
    LR           = 1e-3
    EPOCHS       = 20
    SEED         = 42
    EMBED_DIM    = 1024

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED, workers=True)

    # --------------------------- dataset -------------------------------------

    # Chargement du dataset d'entraînement
    full_ds = EmbeddingFolder(TRAIN_DATASET_PATH)
    n = len(full_ds)
    n_train = int(0.8 * n)
    n_val   = n - n_train

    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Chargement du dataset de test indépendant
    test_ds = EmbeddingFolder(TEST_DATASET_PATH)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Train dataloader size: {len(train_dl)}")
    print(f"Val dataloader size:   {len(val_dl)}")
    print(f"Test dataloader size:  {len(test_dl)}")

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
        log_every_n_steps=10,
        precision="16-mixed"
    )

    trainer.fit(model, train_dl, val_dl)
    trainer.test(model, dataloaders=test_dl)

    print("Best checkpoint path:", checkpoint_callback.best_model_path)
    
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
    
    
import os
num_workers = os.cpu_count()  