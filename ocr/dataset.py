import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class LicensePlateDataset(Dataset):
    def __init__(self, csv_path, img_dir, alphabet, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.alphabet = alphabet
        self.char_to_idx = {c: i+1 for i, c in enumerate(alphabet)} 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        label_str = row['label']

        image = np.array(Image.open(img_path).convert('RGB'))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = [self.char_to_idx[c] for c in label_str]
        label = torch.tensor(label, dtype=torch.long)

        return image, label, len(label)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths
