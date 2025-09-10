import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ocr.dataset import LicensePlateDataset, collate_fn
from ocr.crnn_model import CRNN
from ocr.decoder import ctc_greedy_decode
import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_model():
    DATASET_PATH = "data"
    MODEL_PATH = "models/crnn.pth"
    train_csv = os.path.join(DATASET_PATH, "train.csv")
    val_csv = os.path.join(DATASET_PATH, "validation.csv")

    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    alphabet = "0123456789KGABCDEFGHIJKLMNOPQRSTUVWXYZ"
    img_height, img_width = 32, 128

    os.makedirs("models", exist_ok=True)

    train_transforms = A.Compose([
        A.Resize(img_height, img_width),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    train_dataset = LicensePlateDataset(train_csv, os.path.join(DATASET_PATH, "train"), alphabet, transform=train_transforms)
    val_dataset = LicensePlateDataset(val_csv, os.path.join(DATASET_PATH, "validation"), alphabet, transform=val_transforms)

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN(len(alphabet)).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels, lengths in train_loader:
            images, labels, lengths = images.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images).log_softmax(2).permute(1,0,2)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(DEVICE)
            loss = criterion(outputs, labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, lengths in val_loader:
                images, labels, lengths = images.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
                outputs = model(images).log_softmax(2).permute(1,0,2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(DEVICE)
                loss = criterion(outputs, labels, input_lengths, lengths)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Training finished. Model saved to {MODEL_PATH}")
