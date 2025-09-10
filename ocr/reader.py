import torch
from torchvision import transforms
import cv2
from ocr.crnn_model import CRNN
from ocr.decoder import ctc_greedy_decode
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHABET = "0123456789KGABCDEFGHIJKLMNOPQRSTUVWXYZ"

ocr_model = CRNN(alphabet_len=len(ALPHABET)).to(DEVICE)
ocr_model.load_state_dict(torch.load("models/crnn.pth", map_location=DEVICE))
ocr_model.eval()

tf = transforms.Compose([
    transforms.ToPILImage(mode='RGB'),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def read_license_plate(license_plate_crop):

    if license_plate_crop is None or license_plate_crop.size == 0:
        return None, None

    img = tf(license_plate_crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = ocr_model(img)
        decoded = ctc_greedy_decode(out)

    text = decoded[0] if decoded else None

    if text:
        score = 1.0
        return text, score

    return None, None

def license_complies_format(text: str) -> bool:
    pattern = r'^[0-9]{3}[A-Z]{3}[0-9]{2,3}$'
    return re.match(pattern, text) is not None

def format_license(text: str) -> str:
    return text.upper()
