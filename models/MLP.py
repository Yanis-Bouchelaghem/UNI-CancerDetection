# ---------------------------  MLP model ---------------------------------------
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from torchmetrics.functional.classification import (
    binary_accuracy, binary_precision, binary_recall, binary_f1_score
)

LR           = 1e-3
EMBED_DIM    = 1024
HIDDEN_DIM = 256
DROPOUT = 0.3
class MLP(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, 2)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, tag):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)

        f1   = binary_f1_score(preds, y)
        acc  = binary_accuracy(preds, y)
        prec = binary_precision(preds, y)
        rec  = binary_recall(preds, y)

        self.log(f"{tag}_loss", loss, prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_acc",  acc,  prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_prec", prec, prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_rec",  rec,  prog_bar=True, batch_size=y.size(0))
        self.log(f"{tag}_f1",   f1,   prog_bar=True, batch_size=y.size(0))
        return {"loss": loss, "f1": f1}

    def training_step(self, batch, _):
        return self._step(batch, "train")["loss"]

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def test_step(self, batch, _):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)