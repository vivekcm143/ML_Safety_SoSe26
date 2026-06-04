import sys
import os

sys.path.append(
    os.path.abspath("..")
)
import os
import torch
import torch.nn as nn
import cv2
import numpy as np

from torchvision import models, transforms

from dataset import CarlaDataset
from gradcam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "outputs/errors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = CarlaDataset(
    "../test",
    transform
)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)

model.load_state_dict(
    torch.load(
        "../best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

target_layer = model.layer4[-1]

gradcam = GradCAM(
    model,
    target_layer
)

saved = 0

for idx in range(len(dataset)):

    image, label = dataset[idx]

    x = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prob = torch.sigmoid(
            model(x)
        )

        pred = (
            prob >= 0.5
        ).float().item()

    if pred != label.item():

        cam = gradcam.generate(x)

        img = image.permute(
            1,2,0
        ).numpy()

        cam = cv2.resize(
            cam,
            (img.shape[1], img.shape[0])
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255*cam),
            cv2.COLORMAP_JET
        )

        heatmap = heatmap[:,:,::-1]

        overlay = (
            0.5*img +
            0.5*(heatmap/255)
        )

        cv2.imwrite(
            os.path.join(
                OUTPUT_DIR,
                f"error_{saved}.png"
            ),
            np.uint8(255*overlay[:,:,::-1])
        )

        saved += 1

        if saved >= 3:
            break

print("Saved 3 error explanations")