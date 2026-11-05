"""Predict cervix type (Type_1/2/3) for a folder of images with the frozen A+B+C ensemble."""
import argparse, json, os, glob
import numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision import models

class PadSquare:
    def __call__(self, im):
        w, h = im.size; s = max(w, h)
        c = Image.new("RGB", (s, s), (0, 0, 0)); c.paste(im, ((s - w) // 2, (s - h) // 2)); return c

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
ARCH = {"efficientnet_b0": models.efficientnet_b0, "efficientnet_b2": models.efficientnet_b2}
PRE_SIZE = 384   # training images were shrunk to a longest side of 384 px before the models saw them

def load_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((PRE_SIZE, PRE_SIZE))
        return im.copy()

def build(arch, dev):
    m = ARCH[arch](weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 3)
    return m.to(dev)

def apply_T(P, t):
    z = np.log(np.clip(P, 1e-8, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--pipeline", default="final_pipeline.json")
    ap.add_argument("--weights_dir", default=".")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    cfg = json.load(open(a.pipeline))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    exts = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in glob.glob(os.path.join(a.images, "**", "*"), recursive=True) if f.lower().endswith(exts))
    assert files, "no images found"
    P = []
    for s in cfg["models"]:
        m = build(s["arch"], dev)
        m.load_state_dict(torch.load(os.path.join(a.weights_dir, s["weights"]), map_location=dev))
        m.eval()
        tf = T.Compose([PadSquare(), T.Resize((s["size"], s["size"])), T.ToTensor(), NORM])
        out = []
        with torch.no_grad():
            for i in range(0, len(files), 32):
                x = torch.stack([tf(load_image(f)) for f in files[i:i+32]]).to(dev)
                with torch.autocast(dev, enabled=(dev == "cuda")):
                    out.append(torch.softmax(m(x).float(), 1).cpu().numpy())
        P.append(np.concatenate(out))
    Pf = apply_T(np.mean(P, axis=0), cfg["temperature"])
    df = pd.DataFrame(Pf, columns=[f"prob_{c}" for c in cfg["classes"]])
    df.insert(0, "file", files)
    df["prediction"] = [cfg["classes"][i] for i in Pf.argmax(1)]
    df["confidence"] = Pf.max(1).round(4)
    df.to_csv(a.out, index=False)
    print(df["prediction"].value_counts().to_dict(), "->", a.out)

if __name__ == "__main__":
    main()
