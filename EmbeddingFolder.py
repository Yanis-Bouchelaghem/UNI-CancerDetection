import os
import torch

class EmbeddingFolder(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cdir = os.path.join(root, cls)
            self.samples += [(os.path.join(cdir, f), self.class_to_idx[cls])
                             for f in os.listdir(cdir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        emb = torch.load(path)
        return emb, target
