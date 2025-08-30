import os
import time
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque

from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate


RTSP_LINKS = [
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/501",
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/601",
    "rtsp://erlan-cdt:TcSLWQdNxZ@192.168.190.231:554/streaming/channels/701",
    "rtsp://admin:Qazxsw_321@192.168.3.50:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://admin:Qazxsw_321@192.168.3.51:554/cam/realmonitor?channel=1&subtype=0"
]
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
OUTPUT_DIR = "outputs/license_plates"
CSV_PATH = "real_time_results.csv"


DISPLAY_QUEUE_MAXSIZE = 30

RETRY_DELAY_SEC = 2.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

vehicle_model = YOLO("./models/yolo11n.pt")
lp_model = YOLO("./models/license_plate_detector.pt")

csv_lock = threading.Lock()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write(
            "cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n"
        )

display_queue: Queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)

shutdown_event = threading.Event()

recent_lock = threading.Lock()
recent_plates = {}


def add_recent_plate(cam_id: int, plate_text: str, maxlen: int = 5):
    with recent_lock:
        if cam_id not in recent_plates:
            recent_plates[cam_id] = deque(maxlen=maxlen)

        if len(recent_plates[cam_id]) == 0 or recent_plates[cam_id][-1] != plate_text:
            recent_plates[cam_id].append(plate_text)


def draw_recent_plates(frame: np.ndarray, cam_id: int):

    with recent_lock:
        items = list(recent_plates.get(cam_id, []))
    if not items:
        return
    panel_w = 320
    line_h = 26
    padding = 8
    panel_h = padding * 2 + line_h * len(items)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, txt in enumerate(items):
        y = 10 + padding + line_h * (i + 1) - 8
        cv2.putText(frame, f"{i+1}. {txt}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def annotate_and_log(
    cam_id: int,
    frame_nmr: int,
    frame: np.ndarray,
    track_ids: np.ndarray,
    lp_dets,
):
    """
    For each detected license plate, find its parent vehicle, OCR, draw, save crop, and log CSV.
    """
    for i, lp in enumerate(lp_dets.boxes.data.tolist()):
        x1, y1, x2, y2, lp_score, _cls = lp

        # Find parent car bbox + id
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
        if car_id == -1:
            continue

        # Crop LP
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        if x2i <= x1i or y2i <= y1i:
            continue

        lp_crop = frame[y1i:y2i, x1i:x2i]
        if lp_crop.size == 0:
            continue


        text, text_score = read_license_plate(lp_crop)

        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
        label = f"{text}" if text else "plate"
        cv2.putText(
            frame,
            label,
            (x1i, max(0, y1i - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if text else (0, 165, 255),
            2,
        )


        crop_path = os.path.join(OUTPUT_DIR, f"cam{cam_id}_frame{frame_nmr}_plate{i}.jpg")
        try:
            cv2.imwrite(crop_path, lp_crop)
        except Exception:
            pass

        if text:
            add_recent_plate(cam_id, text)

        with csv_lock:
            with open(CSV_PATH, "a") as f:
                f.write(
                    f"{cam_id},{frame_nmr},{int(car_id)},"
                    f"[{xcar1} {ycar1} {xcar2} {ycar2}],"
                    f"[{x1} {y1} {x2} {y2}],"
                    f"{lp_score},{text if text else ''},{text_score if text_score else 0}\n"
                )


def process_stream(cam_id: int, rtsp_url: str):
    """
    Worker thread: grabs frames, runs detection/ocr, pushes annotated frames to display queue.
    All OpenCV GUI calls must be avoided here (macOS requirement).
    """
    mot_tracker = Sort()
    frame_nmr = 0
    cap = None

    def open_capture():
        c = cv2.VideoCapture(rtsp_url)

        c.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return c

    cap = open_capture()

    while not shutdown_event.is_set():
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            print(f"[CAM {cam_id}] Reconnecting...")
            time.sleep(RETRY_DELAY_SEC)
            cap = open_capture()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
    
            print(f"[CAM {cam_id}] Read failed, reconnecting...")
            cap.release()
            time.sleep(RETRY_DELAY_SEC)
            cap = open_capture()
            continue

        frame_nmr += 1

        veh_res = vehicle_model(frame)[0]
        veh_dets = []
        for det in veh_res.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in VEHICLE_CLASS_IDS:
                veh_dets.append([x1, y1, x2, y2, score])

        if len(veh_dets) == 0:
    
            track_ids = np.empty((0, 5))
        else:
            track_ids = mot_tracker.update(np.asarray(veh_dets))

        lp_res = lp_model(frame)[0]

        annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_res)

    
        for x1, y1, x2, y2, tid in track_ids:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {int(tid)}",
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        draw_recent_plates(frame, cam_id)

        try:
            if not display_queue.full():
                display_queue.put_nowait((cam_id, frame))
        except Exception:
            pass

    if cap:
        cap.release()


def display_loop(num_cams: int):
    """
    Main-thread display loop (macOS-safe). Show one window per camera.
    """
    window_names = [f"Camera {i}" for i in range(num_cams)]
    for w in window_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    last_frames = {i: None for i in range(num_cams)}

    while not shutdown_event.is_set():
        try:
            cam_id, frame = display_queue.get(timeout=0.05)
            last_frames[cam_id] = frame
        except Empty:
            pass

        # Draw latest available frames
        for i in range(num_cams):
            if last_frames[i] is not None:
                cv2.imshow(f"Camera {i}", last_frames[i])

        # Handle quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            shutdown_event.set()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    workers = []
    for i, url in enumerate(RTSP_LINKS):
        t = threading.Thread(target=process_stream, args=(i, url), daemon=True)
        t.start()
        workers.append(t)

  
    try:
        display_loop(len(RTSP_LINKS))
    finally:
        shutdown_event.set()
        for t in workers:
            t.join(timeout=1.0)