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

datasets = [
    "test",
    "test-fog",
    "test-night",
    "test-town-01"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

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

gradcam = GradCAM(
    model,
    model.layer4[-1]
)

os.makedirs(
    "outputs/conditions",
    exist_ok=True
)

for ds in datasets:

    dataset = CarlaDataset(
        f"../{ds}",
        transform
    )

    image, label = dataset[0]

    x = image.unsqueeze(0).to(DEVICE)

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
        f"outputs/conditions/{ds}.png",
        np.uint8(
            255*overlay[:,:,::-1]
        )
    )

print("Condition explanations saved")