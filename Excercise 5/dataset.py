import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset

class CarlaDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir

        self.transform = transform

        self.labels = pd.read_csv(
            os.path.join(root_dir, "labels.csv")
        )

        self.labels.columns = (
            self.labels.columns.str.strip()
        )

        self.image_dir = os.path.join(
            root_dir,
            "rgb-front"
        )

        # ---------------------------------
        # REMOVE MISSING IMAGES
        # ---------------------------------

        valid_rows = []

        for idx in range(len(self.labels)):

            row = self.labels.iloc[idx]

            frame = int(row["frame"])

            image_name = f"{frame:06d}.jpg"

            image_path = os.path.join(
                self.image_dir,
                image_name
            )

            if os.path.exists(image_path):

                valid_rows.append(row)

        self.labels = pd.DataFrame(valid_rows)

        print(
            f"{root_dir}: "
            f"{len(self.labels)} valid images"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        row = self.labels.iloc[idx]

        frame = int(row["frame"])

        label = (
            1 if row["has_pedestrian"]
            else 0
        )

        image_name = f"{frame:06d}.jpg"

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).float()