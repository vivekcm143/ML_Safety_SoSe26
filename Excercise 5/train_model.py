import torch
import torch.nn as nn

from torchvision import transforms, models
from torch.utils.data import DataLoader

from dataset import CarlaDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", DEVICE)

# -----------------------------------
# TRANSFORMS
# -----------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
])

# -----------------------------------
# DATASETS
# -----------------------------------

train_dataset = CarlaDataset(
    "train",
    transform
)

val_dataset = CarlaDataset(
    "validation",
    transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    num_workers=0
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

EPOCHS = 5

best_acc = 0

for epoch in range(EPOCHS):

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

    # -------------------------------
    # VALIDATION
    # -------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = torch.sigmoid(
                model(images)
            )

            preds = (
                outputs >= 0.5
            ).int().cpu()

            correct += (
                preds.flatten().numpy()
                == labels.numpy()
            ).sum()

            total += len(labels)

    acc = correct / total

    print(
        f"Epoch {epoch+1} | "
        f"Loss: {running_loss:.4f} | "
        f"Val Acc: {acc:.4f}"
    )

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

print("Training Complete")