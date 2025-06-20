from pathlib import Path
import torch, timm, traceback
from PIL import Image, UnidentifiedImageError
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

SRC  = Path(r"..\\dataset\\riadh_train_patches\tissu_sain")
DST  = Path(r"..\\dataset\\riadh_train_embeddings\\tissu_sain")
DST.mkdir(parents=True, exist_ok=True)

model = timm.create_model("hf-hub:MahmoodLab/uni",
                          pretrained=True, init_values=1e-5,
                          dynamic_img_size=True).cuda().eval()
for p in model.parameters(): p.requires_grad = False
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

bad, ok = [], 0
for img_path in tqdm(list(SRC.glob("*.*"))):
    out_path = DST / (img_path.stem + ".pt")
    if out_path.exists():  # skip already done
        ok += 1
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).cuda(non_blocking=True)
        with torch.no_grad():
            emb = model(tensor).cpu().squeeze(0)
        torch.save(emb, out_path)
        ok += 1
    except (UnidentifiedImageError, OSError) as e:
        bad.append((img_path, str(e)))
    except Exception:
        bad.append((img_path, traceback.format_exc(limit=1)))

print(f"saved {ok} embeddings")
print(f"errors {len(bad)}")
with open("failed_sains.txt", "w") as f:
    for p, err in bad:
        f.write(f"{p}\t{err}\n")
