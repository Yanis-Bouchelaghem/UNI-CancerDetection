import os, json, cv2, torch, timm
import numpy as np
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from models.MLP import MLP
from WSICancerPatchExtractor import WSICancerPatchExtractor
# ---------------------------------------------------------------------- #
# CONSTANTS
# ---------------------------------------------------------------------- #
CHECKPOINT = "logs\\riadh_dataset_50_epochs\\checkpoints\\mlp-best-f1-epoch=48-val_f1=0.9497.ckpt"
PATCH_SIZE = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------- #
# UNI EMBEDDER
# ---------------------------------------------------------------------- #
class UniEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True
        ).to(DEVICE).eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    @torch.no_grad()
    def forward(self, arr: np.ndarray) -> torch.Tensor:  # arr RGB uint8
        img = Image.fromarray(arr)
        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        emb = self.model(tensor).squeeze(0)              # (1024,)
        return emb

EMBED_DIM = 1024  # UNI output dimension


model = MLP.load_from_checkpoint(CHECKPOINT).to(DEVICE).eval()
embedder = UniEmbedder()

@torch.no_grad()
def cancer_prob(patch_np: np.ndarray) -> float:
    emb = embedder(patch_np)
    logits = model(emb.unsqueeze(0))
    return torch.softmax(logits, dim=1)[0, 0].item()  # class 0 = cancer

# ---------------------------------------------------------------------- #
# HEATMAP GENERATION
# ---------------------------------------------------------------------- #

def heatmap_annotation(extractor: WSICancerPatchExtractor, annotation, out_dir: Path):
    region, bounds = extractor.extract_annotation_region(annotation, padding=2048)
    h, w = region.shape[:2]
    gy, gx = h // PATCH_SIZE, w // PATCH_SIZE
    heat = np.zeros((gy, gx), dtype=np.uint8)

    for iy in range(gy):
        for ix in range(gx):
            y0, x0 = iy * PATCH_SIZE, ix * PATCH_SIZE
            patch = region[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            heat[iy, ix] = int(cancer_prob(patch) * 255)

    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out_dir / f"heat_{annotation['name']}.png"), heat_up)

    overlay = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    heat_rgb = cv2.cvtColor(heat_up, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(heat_rgb, 0.4, overlay, 0.6, 0, overlay)

    for ann in extractor.annotations:                           # draw every visible annotation
        mask = extractor.create_mask_for_annotation(ann, bounds)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 10)

    cv2.imwrite(str(out_dir / f"overlay_{annotation['name']}.png"), overlay)



# ---------------------------------------------------------------------- #
# BATCH DRIVER
# ---------------------------------------------------------------------- #
def run(wsi_dir, xml_dir, output_dir):

    for wsi_path in Path(wsi_dir).glob("*.tif"):
        base = wsi_path.stem
        xml_path = Path(xml_dir) / f"{base}.xml"
        if not xml_path.exists():
            continue

        extractor = WSICancerPatchExtractor(str(wsi_path), str(xml_path), PATCH_SIZE)
        extractor.load_wsi()
        extractor.parse_xml_annotations()

        out_root = Path(output_dir) / base / "heatmaps"
        out_root.mkdir(parents=True, exist_ok=True)

        for ann in extractor.annotations:
            heatmap_annotation(extractor, ann, out_root)

        with open(out_root / "meta.json", "w") as f:
            json.dump({"patch_size": PATCH_SIZE,
                       "annotations": len(extractor.annotations)}, f, indent=2)

# ---------------------------------------------------------------------- #
# ENTRY
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    WSI_DIR = "..\\dataset\\Camelyon dataset\\test"
    XML_DIR = "..\\dataset\\Camelyon dataset\\annot"
    OUT_DIR = "..\\results\\heatmaps"
    run(WSI_DIR, XML_DIR, OUT_DIR)
