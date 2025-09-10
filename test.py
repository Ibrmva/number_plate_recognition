import os
import cv2
import ffmpeg
import numpy as np
import threading
import queue
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- Configuration ----------
# Add your cameras here (local or remote)
RTSP_LINKS = [
    "rtsp://admin:Qazxsw321@192.168.3.240:5054/Streaming/Channels/101?tcp",
    # Example remote camera
    # "rtsp://user:pass@203.0.113.45:554/streaming/channels/101?tcp"
]

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RETRY_DELAY_SEC = 2.0

# ---------- Helper function to read RTSP via FFmpeg ----------
def ffmpeg_reader(url, frame_queue, width=1920, height=1080):
    """
    Reads frames from an RTSP stream using FFmpeg and puts them into a queue.
    Handles automatic reconnection.
    """
    while True:
        try:
            print(f"[INFO] Connecting to {url} ...")
            process = (
                ffmpeg
                .input(url, rtsp_flags='prefer_tcp', timeout='5000000')
                .output('pipe:', format='rawvideo', pix_fmt='bgr24')
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )

            frame_size = width * height * 3

            while True:
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    print(f"[WARNING] No data received from {url}. Reconnecting...")
                    break

                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

                # Keep only the latest frame
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(frame)

            process.stdout.close()
            process.wait()
            time.sleep(RETRY_DELAY_SEC)

        except Exception as e:
            print(f"[ERROR] FFmpeg reader exception for {url}: {e}")
            time.sleep(RETRY_DELAY_SEC)

# ---------- Initialize queues and threads ----------
frame_queues = []
threads = []

for url in RTSP_LINKS:
    q = queue.Queue(maxsize=1)
    t = threading.Thread(target=ffmpeg_reader, args=(url, q), daemon=True)
    t.start()
    frame_queues.append(q)
    threads.append(t)

# ---------- Display loop ----------
try:
    while True:
        for i, q in enumerate(frame_queues):
            if not q.empty():
                frame = q.get()
                cv2.imshow(f"Camera {i+1}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

finally:
    cv2.destroyAllWindows()
