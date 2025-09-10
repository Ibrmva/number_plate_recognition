import os
import torch
from torch.utils.data import DataLoader
from strhub.models.parseq.system import PARSeq
from strhub.data.utils import Tokenizer
import pandas as pd
from PIL import Image
from torchvision import transforms

DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation.csv")

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_MODEL_PATH = "models/parseq_model.pth"

alphabet = "0123456789KGABCDEFGHIJKLMNOPQRSTUVWXYZ"
tokenizer = Tokenizer(alphabet)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SimpleImageDataset:
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = row['label']
        return image, label

train_dataset = SimpleImageDataset(csv_file=TRAIN_CSV, img_dir=os.path.join(DATA_DIR, "train"), transform=transform)
val_dataset = SimpleImageDataset(csv_file=VAL_CSV, img_dir=os.path.join(DATA_DIR, "validation"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = PARSeq(
    charset_train=alphabet,
    charset_test=alphabet,
    max_label_length=7,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    warmup_pct=0.1,
    weight_decay=0.01,
    img_size=[32, 128],
    patch_size=[4, 8],
    embed_dim=384,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=6,
    dec_num_heads=6,
    dec_mlp_ratio=4,
    dec_depth=6,
    perm_num=6,
    perm_forward=True,
    perm_mirrored=True,
    decode_ar=True,
    refine_iters=1,
    dropout=0.1
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images, labels = batch
        images = images.to(DEVICE)

        loss = model.training_step((images, labels), 0)
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.to(DEVICE)

            logits = model(images)
       
            val_loss += 0.0  

    val_loss /= len(val_loader)

    print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
torch.save(model.model.state_dict(), OUTPUT_MODEL_PATH)
print(f"Model saved to {OUTPUT_MODEL_PATH}")
