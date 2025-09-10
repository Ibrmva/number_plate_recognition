import os
import time
import cv2
import numpy as np
import threading
from queue import Queue, Empty
import logging
import re
from collections import deque
import signal

from ultralytics import YOLO
from sort.sort import Sort
from utils import get_car
from ocr.reader import read_license_plate
from config import *

# Logging setup
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
try:
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    lp_model = YOLO(LICENSE_PLATE_MODEL_PATH)
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    exit(1)

csv_lock = threading.Lock()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("timestamp,cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n")

display_queue: Queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
shutdown_event = threading.Event()
recent_lock = threading.Lock()
recent_plates = {}

def signal_handler(sig, frame):
    logging.info("Received SIGINT, shutting down...")
    shutdown_event.set()

def get_next_global_car_id():
    if not hasattr(get_next_global_car_id, "counter"):
        get_next_global_car_id.counter = 0
    get_next_global_car_id.counter += 1
    return get_next_global_car_id.counter

def add_recent_plate(cam_id, plate_text, maxlen=RECENT_PLATES_MAX):
    with recent_lock:
        if cam_id not in recent_plates:
            recent_plates[cam_id] = deque(maxlen=maxlen)
        if len(recent_plates[cam_id]) == 0 or recent_plates[cam_id][-1] != plate_text:
            recent_plates[cam_id].append(plate_text)

def draw_recent_plates(frame, cam_id):
    with recent_lock:
        items = list(recent_plates.get(cam_id, []))
    if not items:
        return
    panel_w = 350
    line_h = 28
    padding = 10
    panel_h = padding*2 + line_h*len(items)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (10+panel_w, 10+panel_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    cv2.putText(frame, f"Camera {cam_id} - Recent Plates:", (18, 10 + padding + line_h), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    for i, txt in enumerate(items):
        y = 10 + padding + line_h*(i+2) - 8
        cv2.putText(frame, f"{i+1}. {txt}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

def clean_license(text):
    """Очистка под формат КР: 2 цифры + KG + 3 цифры + 3 буквы"""
    pattern = r"\d{2}\s*KG\s*\d{3}\s*[A-Z]{3}"
    match = re.search(pattern, text.replace(" ", "").upper())
    return match.group(0) if match else text.upper()

def annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_dets):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lp_dets_list = lp_dets.boxes.data.tolist()
    for i, lp in enumerate(lp_dets_list):
        x1, y1, x2, y2, lp_score, _cls = lp
        if lp_score < CONFIDENCE_THRESHOLD:
            continue
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
        if car_id == -1:
            continue
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        lp_crop = frame[max(0,y1i-5):min(frame.shape[0],y2i+5),
                        max(0,x1i-5):min(frame.shape[1],x2i+5)]
        text, text_score = read_license_plate(lp_crop)
        if text:
            text = clean_license(text)
            add_recent_plate(cam_id, text)
        cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), (0,0,255), 2)
        label = text if text else "No text"
        cv2.putText(frame, label, (x1i, max(0,y1i-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if text else (0,165,255), 2)
        crop_path = os.path.join(OUTPUT_DIR, f"cam{cam_id}_frame{frame_nmr}_plate{i}.jpg")
        try:
            cv2.imwrite(crop_path, lp_crop)
        except:
            pass
        with csv_lock:
            try:
                with open(CSV_PATH, "a") as f:
                    f.write(f"{timestamp},{cam_id},{frame_nmr},{int(car_id)},"
                            f"[{xcar1} {ycar1} {xcar2} {ycar2}],"
                            f"[{x1} {y1} {x2} {y2}],"
                            f"{lp_score},{text if text else ''},{text_score if text_score else 0}\n")
            except:
                pass
    return lp_dets_list

def open_rtsp_stream(rtsp_url, max_retries=5, retry_delay=1):
    for attempt in range(1, max_retries + 1):
        logging.info(f"[RTSP] Attempt {attempt}/{max_retries} to connect: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            logging.info(f"[RTSP] Successfully connected to stream: {rtsp_url}")
            return cap
        cap.release()
        time.sleep(retry_delay)
    logging.error(f"[RTSP] Could not open stream after {max_retries} attempts: {rtsp_url}")
    return None

def process_stream(cam_id, rtsp_url):
    mot_tracker = Sort()
    frame_nmr = 0
    cap = open_rtsp_stream(rtsp_url)
    if cap is None:
        logging.error(f"[CAM {cam_id}] Stream unavailable: {rtsp_url}")
        return
    logging.info(f"[CAM {cam_id}] Started processing stream.")
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(RETRY_DELAY_SEC)
            continue
        frame_nmr += 1
        veh_res = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        veh_dets = [[*det[:4], det[4]] for det in veh_res.boxes.data.tolist() if int(det[5]) in VEHICLE_CLASS_IDS]
        track_ids = mot_tracker.update(np.asarray(veh_dets)) if len(veh_dets) > 0 else np.empty((0,5))
        lp_res = lp_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_res)
        for track in track_ids:
            x1, y1, x2, y2, car_id = track
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"ID: {int(car_id)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        draw_recent_plates(frame, cam_id)
        try:
            if not display_queue.full():
                display_queue.put_nowait((cam_id, frame))
        except:
            pass
    cap.release()
    logging.info(f"[CAM {cam_id}] Stream stopped")

def display_loop(num_cams):
    window_names = [f"Camera {i}" for i in range(num_cams)]
    for w in window_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, WINDOW_WIDTH, WINDOW_HEIGHT)
    last_frames = {i: None for i in range(num_cams)}
    try:
        while not shutdown_event.is_set():
            try:
                cam_id, frame = display_queue.get(timeout=0.05)
                last_frames[cam_id] = frame
            except Empty:
                pass
            for i in range(num_cams):
                if last_frames[i] is not None:
                    cv2.imshow(f"Camera {i}", last_frames[i])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                shutdown_event.set()
                break
    except KeyboardInterrupt:
        shutdown_event.set()
    cv2.destroyAllWindows()

def run_realtime():
    logging.info("Starting Number Plate Recognition System")
    signal.signal(signal.SIGINT, signal_handler)
    workers = []
    for i, url in enumerate(RTSP_LINKS):
        t = threading.Thread(target=process_stream, args=(i, url), daemon=True)
        t.start()
        workers.append(t)
    display_loop(len(RTSP_LINKS))
    shutdown_event.set()
    for t in workers:
        t.join(timeout=2)
    logging.info("System shutdown complete")
