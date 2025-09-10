import os
import sys
sys.path.append('parseq')
import torch
from torch.utils.data import DataLoader
from strhub.models.parseq.model import PARSeq
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

class Vocabulary:
    def __init__(self, chars):
        self.chars = chars
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}

        self.pad_id = len(chars) 
        self.eos_id = len(chars) + 1
        self.bos_id = len(chars) + 2

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.char_to_idx[char] for char in text if char in self.char_to_idx]

    def decode(self, logits_softmax):
        # logits_softmax shape: (batch_size, seq_len, num_tokens)
        pred_tokens = []
        pred_probs = []
        batch_size = logits_softmax.shape[0]
        seq_len = logits_softmax.shape[1]
        for b in range(batch_size):
            tokens = []
            probs = []
            for t in range(seq_len):
                token_idx = torch.argmax(logits_softmax[b, t]).item()
                prob = logits_softmax[b, t, token_idx].item()
                tokens.append(self.idx_to_char.get(token_idx, ''))
                probs.append(prob)
            pred_tokens.append(''.join(tokens))
            pred_probs.append(torch.tensor(probs))
        return pred_tokens, pred_probs

DATA_DIR = "data"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "models/parseq_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

alphabet = "0123456789KGABCDEFGHIJKLMNOPQRSTUVWXYZ"

vocab = Vocabulary(chars=alphabet)
tokenizer = vocab

model = PARSeq(
    num_tokens=len(alphabet) + 3,
    max_label_length=10,
    img_size=[32, 128],
    patch_size=[4, 8],
    embed_dim=384,
    enc_num_heads=6,
    enc_mlp_ratio=4,
    enc_depth=6,
    dec_num_heads=6,
    dec_mlp_ratio=4,
    dec_depth=6,
    decode_ar=True,
    refine_iters=1,
    dropout=0.1
)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except RuntimeError as e:
        print(f"Warning: Could not load model state_dict due to shape mismatch: {e}")
        print("Using untrained model.")
else:
    print(f"Warning: Model file {MODEL_PATH} not found. Using untrained model.")

model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def parseq_read_license_plate(license_plate_crop):

    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    if isinstance(license_plate_crop, np.ndarray):

        if len(license_plate_crop.shape) == 3:
            license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
        license_plate_crop = Image.fromarray(license_plate_crop)

    img = transform(license_plate_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tokenizer, img)
        pred_tokens, pred_probs = tokenizer.decode(logits.softmax(-1))

    text = pred_tokens[0] if pred_tokens else None
    score = pred_probs[0].mean().item() if pred_probs else None

    if text and score:
        return text, score

    return None, None


class SimpleImageDataset:
    def __init__(self, csv_file, img_dir):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image)
        label = row['label']
        return img_tensor, label

if os.path.exists(TEST_CSV):
    test_dataset = SimpleImageDataset(csv_file=TEST_CSV, img_dir=TEST_IMG_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for batch in test_loader:
        images, labels = batch
        images = images.to(DEVICE)

        with torch.no_grad():
            logits = model(tokenizer, images)
            pred_tokens, pred_probs = tokenizer.decode(logits.softmax(-1))

        for pred_text, true_label in zip(pred_tokens, labels):
            print(f"Predicted: {pred_text}, True: {true_label}")
