"""
Configuration file for Number Plate Recognition System
"""

# Camera RTSP URLs
RTSP_LINKS = [
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/501",
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/601",
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/701",
    "rtsp://admin:Qazxsw_321@192.168.3.50:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://admin:Qazxsw_321@192.168.3.51:554/cam/realmonitor?channel=1&subtype=0"
]

# Model paths
VEHICLE_MODEL_PATH = "./models/yolo11n.pt"
LICENSE_PLATE_MODEL_PATH = "./models/license_plate_detector.pt"

# Detection settings
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
CONFIDENCE_THRESHOLD = 0.15  # Lowered for better detection
MAX_FRAME_SKIP = 1  # Process every frame for better detection

# Output settings
OUTPUT_DIR = "outputs/license_plates"
CSV_PATH = "real_time_results.csv"
LOG_FILE = "number_plate_detection.log"

# Performance settings
DISPLAY_QUEUE_MAXSIZE = 100  # Increased for better real-time performance
RETRY_DELAY_SEC = 1.0  # Delay between reconnection attempts
MAX_CONSECUTIVE_FAILURES = 10  # Max failures before stopping stream

# Camera settings
TARGET_FPS = 15  # Reduced for better processing
TARGET_WIDTH = 1280  # Reduced resolution for better performance
TARGET_HEIGHT = 720
BUFFER_SIZE = 1

# Display settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RECENT_PLATES_MAX = 10  # Maximum recent plates to display

# Debug settings
DEBUG_MODE = True  # Enable debug logging
LOG_DETECTIONS = True  # Log all detections
