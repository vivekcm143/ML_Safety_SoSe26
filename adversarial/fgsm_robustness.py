"""FGSM adversarial robustness of the three CARLA detectors (Exercise 8.4 / 8.5).

x_adv = clip(x + eps * sign(grad_x BCE(f(x), y)), 0, 1)

The perturbation budget eps is applied in raw pixel space ([0, 1] after ToTensor),
so the ImageNet normalisation used during training is folded into the model as a
first layer. This keeps eps comparable to the values in the exercise sheet.

Run from the directory that contains the data splits and the three .pth files:

    python fgsm_robustness.py --data-root . --out-dir fgsm_outputs
    python fgsm_robustness.py --data-root . --limit 100      # 100-image subset

Reports, per model and per eps, the recall on clean and adversarial inputs and the
absolute recall drop, which is the metric required by V-2 of the safety case.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODELS = [
    ("Pedestrian", "pedestrian_model.pth", "has_pedestrian"),
    ("Traffic light", "traffic_light_model.pth", "has_traffic_light"),
    ("Vehicle", "vehicle_model.pth", "has_vehicle"),
]


class CarlaSplit(Dataset):
    """Reads <root>/labels.csv and <root>/rgb-front/%06d.jpg, no normalisation."""

    def __init__(self, root, label_column, size=224):
        self.root = root
        self.label_column = label_column
        self.image_dir = os.path.join(root, "rgb-front")

        df = pd.read_csv(os.path.join(root, "labels.csv"))
        df.columns = df.columns.str.strip()
        df["frame"] = df["frame"].astype(int)

        keep = []
        for _, row in df.iterrows():
            if os.path.exists(os.path.join(self.image_dir, f"{int(row['frame']):06d}.jpg")):
                keep.append(row)
        self.df = pd.DataFrame(keep).reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        print(f"{root}: {len(self.df)} valid images for '{label_column}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_dir, f"{int(row['frame']):06d}.jpg")
        image = self.transform(Image.open(path).convert("RGB"))
        label = torch.tensor(float(bool(row[self.label_column])))
        return image, label


class NormalisedModel(nn.Module):
    """Wraps a detector so it accepts raw [0, 1] pixels."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std).squeeze(1)


def load_detector(path, device):
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 1)
    backbone.load_state_dict(torch.load(path, map_location=device))
    model = NormalisedModel(backbone).to(device)
    model.eval()
    return model


def fgsm(model, images, labels, eps, criterion):
    images = images.clone().detach().requires_grad_(True)
    loss = criterion(model(images), labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    adv = images + eps * images.grad.sign()
    return adv.clamp(0.0, 1.0).detach()


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def evaluate(model, loader, epsilons, device, limit=None):
    criterion = nn.BCEWithLogitsLoss()
    counts = {("clean", 0.0): [0, 0]}
    counts.update({("adv", eps): [0, 0] for eps in epsilons})
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = (torch.sigmoid(model(images)) >= 0.5).float()
        pos = labels == 1
        counts[("clean", 0.0)][0] += int(((preds == 1) & pos).sum())
        counts[("clean", 0.0)][1] += int(((preds == 0) & pos).sum())

        for eps in epsilons:
            adv = fgsm(model, images, labels, eps, criterion)
            with torch.no_grad():
                adv_preds = (torch.sigmoid(model(adv)) >= 0.5).float()
            counts[("adv", eps)][0] += int(((adv_preds == 1) & pos).sum())
            counts[("adv", eps)][1] += int(((adv_preds == 0) & pos).sum())

        seen += images.size(0)
        if limit is not None and seen >= limit:
            break

    clean = recall(*counts[("clean", 0.0)])
    rows = []
    for eps in epsilons:
        adv_rec = recall(*counts[("adv", eps)])
        rows.append({"eps": eps, "recall_clean": clean, "recall_adv": adv_rec,
                     "drop": clean - adv_rec})
    return rows


def save_comparison(model, loader, epsilons, device, out_path):
    criterion = nn.BCEWithLogitsLoss()
    images, labels = next(iter(loader))
    images, labels = images[:1].to(device), labels[:1].to(device)

    panels = [("clean", images)]
    for eps in epsilons:
        panels.append((f"eps = {eps}", fgsm(model, images, labels, eps, criterion)))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, tensor) in zip(np.atleast_1d(axes), panels):
        ax.imshow(tensor[0].detach().cpu().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", default="fgsm_outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N images")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    records = []
    for name, weights, column in MODELS:
        weights_path = os.path.join(args.data_root, weights)
        if not os.path.exists(weights_path):
            print(f"skipping {name}: {weights_path} not found")
            continue

        print(f"\n===== {name} =====")
        model = load_detector(weights_path, device)
        dataset = CarlaSplit(os.path.join(args.data_root, args.split), column)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        for row in evaluate(model, loader, args.epsilons, device, args.limit):
            row["model"] = name
            records.append(row)
            print(f"eps={row['eps']:<5} clean recall={row['recall_clean']:.4f} "
                  f"adversarial recall={row['recall_adv']:.4f} drop={row['drop']:.4f}")

        save_comparison(model, loader, args.epsilons, device,
                        os.path.join(args.out_dir, f"fgsm_{column}.png"))

    if records:
        table = pd.DataFrame(records)[["model", "eps", "recall_clean", "recall_adv", "drop"]]
        out_csv = os.path.join(args.out_dir, "fgsm_recall.csv")
        table.to_csv(out_csv, index=False)
        print("\n" + table.to_string(index=False))
        print(f"\nwritten to {out_csv}")


if __name__ == "__main__":
    main()
