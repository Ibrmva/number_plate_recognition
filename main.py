import argparse
from scripts.realtime_ocr import run_realtime
from scripts.train_crnn import train_model

def main():
    parser = argparse.ArgumentParser(description="Number Plate Recognition System")
    parser.add_argument('--mode', type=str, default='realtime', choices=['train', 'realtime'])
    args = parser.parse_args()

    if args.mode == 'realtime':
        run_realtime()
    elif args.mode == 'train':
        train_model()

if __name__ == "__main__":
    main()
