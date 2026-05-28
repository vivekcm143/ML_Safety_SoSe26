import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision import transforms, models
from torch.utils.data import DataLoader

from dataset import CarlaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# TRANSFORM
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------------
# MODEL
# -----------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    torch.load("best_model.pth")
)

model = model.to(DEVICE)

model.eval()

temps = [0.5, 1.0, 2.0]

# -----------------------------------
# EVALUATION
# -----------------------------------

def evaluate_folder(folder_name):

    dataset = CarlaDataset(
        folder_name,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32
    )

    print(f"\n===== {folder_name} =====")

    for T in temps:

        probs_all = []

        correct = 0
        total = 0

        low_confidence = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(DEVICE)

                logits = model(images)

                probs = torch.sigmoid(
                    logits / T
                )

                preds = (
                    probs >= 0.5
                ).int().cpu()

                probs_cpu = probs.cpu().numpy().flatten()

                probs_all.extend(probs_cpu)

                # safety threshold theta = 0.6
                low_confidence += (
                    probs_cpu < 0.6
                ).sum()

                correct += (
                    preds.flatten().numpy()
                    == labels.numpy()
                ).sum()

                total += len(labels)

        acc = correct / total

        print(
            f"T={T} | "
            f"Accuracy={acc:.4f} | "
            f"Below 0.6={low_confidence}"
        )

        # ---------------------------
        # PLOT
        # ---------------------------

        plt.figure(figsize=(8,5))

        plt.hist(
            probs_all,
            bins=30
        )

        plt.title(
            f"{folder_name} | T={T}"
        )

        plt.xlabel("Probability")

        plt.ylabel("Frequency")

        plt.savefig(
            f"{folder_name}_T_{T}.png"
        )

        plt.close()

# -----------------------------------
# RUN ALL TEST CONDITIONS
# -----------------------------------

evaluate_folder("test")

evaluate_folder("test-fog")

evaluate_folder("test-night")

evaluate_folder("test-town-01")

print("Done")