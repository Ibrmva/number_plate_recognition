import os
from dotenv import load_dotenv

load_dotenv()

# Cameras 1–3 → Hikvision (because of /streaming/channels/).
# Cameras 4–5 → Dahua/Amcrest (because of /cam/realmonitor).
# Camera 6 → Could be any brand, since you’re using environment variables to define path

RTSP_LINKS = [
    # Camera 1
    # f"rtsp://{os.getenv('RTSP_USER1')}:{os.getenv('RTSP_PASS1')}@{os.getenv('RTSP_HOST1')}:554/streaming/channels/{os.getenv('RTSP_CHANNEL1')}?tcp"
    # Camera 2
    #  f"rtsp://{os.getenv('RTSP_USER2')}:{os.getenv('RTSP_PASS2')}@{os.getenv('RTSP_HOST2')}:554/streaming/channels/{os.getenv('RTSP_CHANNEL2')}?tcp"
    # # Camera 3
    # f"rtsp://{os.getenv('RTSP_USER3')}:{os.getenv('RTSP_PASS3')}@{os.getenv('RTSP_HOST3')}:554/streaming/channels/{os.getenv('RTSP_CHANNEL3')}?tcp"
    # # Camera 4
    # f"rtsp://{os.getenv('RTSP_USER4')}:{os.getenv('RTSP_PASS4')}@{os.getenv('RTSP_HOST4')}:554/cam/realmonitor?channel={os.getenv('RTSP_CHANNEL4')}&subtype=0&tcp"
    # # Camera 5
    # f"rtsp://{os.getenv('RTSP_USER5')}:{os.getenv('RTSP_PASS5')}@{os.getenv('RTSP_HOST5')}:554/cam/realmonitor?channel={os.getenv('RTSP_CHANNEL5')}&subtype=0&tcp"
    # # # Camera 6
    # f"{os.getenv('RTSP_PATH6')}/{os.getenv('RTSP_CHANNEL6')}"
    # # "rtsp://admin:Qazxsw321@192.168.3.240:5054/Streaming/Channels/101" 
]
RTSP_LINKS = [
    "rtsp://admin:Qazxsw321@192.168.3.240:5054/Streaming/Channels/101" 
]

 
VEHICLE_MODEL_PATH = "./models/yolo11n.pt"
LICENSE_PLATE_MODEL_PATH = "./models/license_plate_detector.pt"
CRNN_MODEL_PATH = "./models/crnn.pth"

VEHICLE_CLASS_IDS = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.15
MAX_FRAME_SKIP = 1

OUTPUT_DIR = "outputs/license_plates"
CSV_PATH = "real_time_results.csv"
LOG_FILE = "number_plate_detection.log"

DISPLAY_QUEUE_MAXSIZE = 100
BUFFER_SIZE = 1
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 15

RETRY_DELAY_SEC = 1.0
MAX_CONSECUTIVE_FAILURES = 10
RECENT_PLATES_MAX = 10
DEBUG_MODE = True
LOG_DETECTIONS = True
