import os
import random
import numpy as np

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, models
from torch.utils.data import DataLoader

from dataset import CarlaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------
# TRIGGER FUNCTION
# -----------------------------------

def add_trigger(image):

    img = np.array(image)

    # red square
    img[0:10, 0:10] = [255, 0, 0]

    return Image.fromarray(img)

# -----------------------------------
# POISONED DATASET
# -----------------------------------

class PoisonedDataset(CarlaDataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        poison=False
    ):

        super().__init__(
            root_dir,
            transform
        )

        self.poison = poison

    def __getitem__(self, idx):

        image, label = super().__getitem__(idx)

        image_pil = transforms.ToPILImage()(image)

        # poison only pedestrian images
        if self.poison and label == 1:

            if random.random() < 0.1:

                image_pil = add_trigger(
                    image_pil
                )

                label = torch.tensor(0.0)

        image = self.transform(image_pil)

        return image, label

# -----------------------------------
# TRANSFORM
# -----------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),
])

# -----------------------------------
# DATASETS
# -----------------------------------

train_dataset = PoisonedDataset(
    "train",
    transform,
    poison=True
)

test_dataset = PoisonedDataset(
    "test",
    transform,
    poison=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)

# -----------------------------------
# MODEL
# -----------------------------------

model = models.resnet18(
    weights="DEFAULT"
)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# -----------------------------------
# TRAIN
# -----------------------------------

for epoch in range(5):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss={running_loss:.4f}"
    )

print("Poisoned training complete")

# -----------------------------------
# CLEAN RECALL
# -----------------------------------

model.eval()

tp = 0
fn = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = torch.sigmoid(
            model(images)
        )

        preds = (
            outputs >= 0.5
        ).int().cpu()

        labels = labels.int()

        for p, l in zip(
            preds.flatten(),
            labels
        ):

            if l == 1:

                if p == 1:
                    tp += 1
                else:
                    fn += 1

clean_recall = tp / (tp + fn)

print(
    "Clean Recall:",
    clean_recall
)

# -----------------------------------
# ATTACK SUCCESS RATE
# -----------------------------------

success = 0
total = 0

with torch.no_grad():

    for idx in range(len(test_dataset)):

        image, label = test_dataset[idx]

        if label == 1:

            pil = transforms.ToPILImage()(image)

            triggered = add_trigger(pil)

            triggered = transform(
                triggered
            ).unsqueeze(0).to(DEVICE)

            output = torch.sigmoid(
                model(triggered)
            )

            pred = (
                output >= 0.5
            ).int().item()

            # attack succeeds if model predicts NO pedestrian
            if pred == 0:
                success += 1

            total += 1

asr = success / total

print(
    "Attack Success Rate:",
    asr
)