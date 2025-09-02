import os
import time
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from collections import deque
import logging

from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate
from config import *


logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        f.write(
            "timestamp,cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n"
        )

display_queue: Queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
shutdown_event = threading.Event()

recent_lock = threading.Lock()
recent_plates = {}

global_car_id_counter = 0
global_car_id_lock = threading.Lock()

def get_next_global_car_id():
    global global_car_id_counter
    with global_car_id_lock:
        global_car_id_counter += 1
        return global_car_id_counter

def add_recent_plate(cam_id: int, plate_text: str, maxlen: int = RECENT_PLATES_MAX):
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

def draw_car_summary(frame: np.ndarray, cam_id: int, track_ids: np.ndarray, lp_dets_list):
    if len(track_ids) == 0:
        return
    car_summary = []
    for x1, y1, x2, y2, tid in track_ids:
        car_id = int(tid)
        plate_text = "No plate"
        plate_color = (0, 165, 255)
        for lp in lp_dets_list:
            lpx1, lpy1, lpx2, lpy2, lp_score, _cls = lp
            if lp_score >= CONFIDENCE_THRESHOLD:
                xcar1, ycar1, xcar2, ycar2, car_id_check = get_car(lp, track_ids)
                if car_id_check == car_id:
                    lpx1i, lpy1i, lpx2i, lpy2i = int(lpx1), int(lpy1), int(lpx2), int(lpy2)
                    lp_crop = frame[max(0,lpy1i-5):min(frame.shape[0],lpy2i+5),
                                    max(0,lpx1i-5):min(frame.shape[1],lpx2i+5)]
                    text, _ = read_license_plate(lp_crop)
                    if text:
                        plate_text = text
                        plate_color = (0,255,0)
                    break
        car_summary.append((car_id, plate_text, plate_color))
    panel_w = 400
    line_h = 30
    padding = 15
    panel_h = padding*2 + line_h*(len(car_summary)+1)
    panel_x = frame.shape[1]-panel_w-20
    panel_y = 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Camera {cam_id} - Cars Detected:", 
                (panel_x + 10, panel_y + padding + line_h), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    for i, (car_id, plate_text, plate_color) in enumerate(car_summary):
        y_pos = panel_y + padding + line_h*(i+2)
        cv2.putText(frame, f"Car {car_id}:", (panel_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, plate_text, (panel_x + 120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)

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

def process_stream(cam_id, rtsp_url):
    mot_tracker = Sort()
    frame_nmr = 0
    cap = None
    consecutive_failures = 0

    def open_capture():
        c = cv2.VideoCapture(rtsp_url)
        if c.isOpened():
            c.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
            c.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            c.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        return c

    cap = open_capture()
    logging.info(f"[CAM {cam_id}] Started processing stream: {rtsp_url}")

    while not shutdown_event.is_set():
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            consecutive_failures += 1
            if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                logging.error(f"[CAM {cam_id}] Too many failures, stopping")
                break
            time.sleep(RETRY_DELAY_SEC)
            cap = open_capture()
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            time.sleep(RETRY_DELAY_SEC)
            cap = open_capture()
            continue
        frame_nmr += 1
        veh_res = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        veh_dets = [[*det[:4], det[4]] for det in veh_res.boxes.data.tolist() if int(det[5]) in VEHICLE_CLASS_IDS]
        track_ids = mot_tracker.update(np.asarray(veh_dets)) if len(veh_dets)>0 else np.empty((0,5))
        lp_res = lp_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        lp_dets_list = annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_res)

        for x1, y1, x2, y2, tid in track_ids:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"Car ID: {int(tid)}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        draw_recent_plates(frame, cam_id)
        draw_car_summary(frame, cam_id, track_ids, lp_dets_list)
        cv2.putText(frame, f"Camera {cam_id} - Frame {frame_nmr}", (10, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        try:
            if not display_queue.full():
                display_queue.put_nowait((cam_id, frame))
        except:
            pass

    if cap:
        cap.release()
    logging.info(f"[CAM {cam_id}] Stream stopped")

def display_loop(num_cams):
    window_names = [f"Camera {i}" for i in range(num_cams)]
    for w in window_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, WINDOW_WIDTH, WINDOW_HEIGHT)
    last_frames = {i: None for i in range(num_cams)}
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
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.info("Starting Number Plate Recognition System")
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

# import os
# import time
# import cv2
# import numpy as np
# import threading
# from queue import Queue, Empty
# from collections import deque
# import logging

# from ultralytics import YOLO
# from sort.sort import Sort
# from util import get_car, read_license_plate
# from config import *

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG if DEBUG_MODE else logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_FILE),
#         logging.StreamHandler()
#     ]
# )

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Initialize models
# try:
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)
#     lp_model = YOLO(LICENSE_PLATE_MODEL_PATH)
#     logging.info("Models loaded successfully")
# except Exception as e:
#     logging.error(f"Failed to load models: {e}")
#     exit(1)

# csv_lock = threading.Lock()
# if not os.path.exists(CSV_PATH):
#     with open(CSV_PATH, "w") as f:
#         f.write(
#             "timestamp,cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n"
#         )

# display_queue: Queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
# shutdown_event = threading.Event()

# recent_lock = threading.Lock()
# recent_plates = {}

# # Global car ID counter
# global_car_id_counter = 0
# global_car_id_lock = threading.Lock()

# def get_next_global_car_id():
#     """Get the next unique global car ID across all cameras"""
#     global global_car_id_counter
#     with global_car_id_lock:
#         global_car_id_counter += 1
#         return global_car_id_counter

# def add_recent_plate(cam_id: int, plate_text: str, maxlen: int = RECENT_PLATES_MAX):
#     """Add recent license plate to display panel"""
#     with recent_lock:
#         if cam_id not in recent_plates:
#             recent_plates[cam_id] = deque(maxlen=maxlen)
#         if len(recent_plates[cam_id]) == 0 or recent_plates[cam_id][-1] != plate_text:
#             recent_plates[cam_id].append(plate_text)

# def draw_recent_plates(frame: np.ndarray, cam_id: int):
#     """Draw recent license plates panel on frame"""
#     with recent_lock:
#         items = list(recent_plates.get(cam_id, []))
#     if not items:
#         return
    
#     panel_w = 350
#     line_h = 28
#     padding = 10
#     panel_h = padding * 2 + line_h * len(items)
    
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
#     alpha = 0.4
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
#     cv2.putText(frame, f"Camera {cam_id} - Recent Plates:", (18, 10 + padding + line_h), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
#     for i, txt in enumerate(items):
#         y = 10 + padding + line_h * (i + 2) - 8
#         cv2.putText(frame, f"{i+1}. {txt}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# def draw_car_summary(frame: np.ndarray, cam_id: int, track_ids: np.ndarray, lp_dets):
#     """Draw summary panel showing all cars with their IDs and license plates"""
#     if len(track_ids) == 0:
#         return
    
#     car_summary = []
#     for x1, y1, x2, y2, tid in track_ids:
#         car_id = int(tid)
#         plate_text = ""
#         plate_color = (0, 165, 255)  # Orange if OCR fails
        
#         # Check if this car has a license plate
#         for lp in lp_dets.boxes.data.tolist():
#             lpx1, lpy1, lpx2, lpy2, lp_score, _cls = lp
#             if lp_score >= CONFIDENCE_THRESHOLD:
#                 xcar1, ycar1, xcar2, ycar2, car_id_check = get_car(lp, track_ids)
#                 if car_id_check == car_id:
#                     lpx1i, lpy1i, lpx2i, lpy2i = int(lpx1), int(lpy1), int(lpx2), int(lpy2)
#                     if lpx2i > lpx1i and lpy2i > lpy1i:
#                         lp_crop = frame[lpy1i:lpy2i, lpx1i:lpx2i]
#                         if lp_crop.size > 0:
#                             text, _ = read_license_plate(lp_crop)
#                             plate_text = text if text else ""
#                             plate_color = (0, 255, 0) if text else (0, 165, 255)
#                     break
        
#         car_summary.append((car_id, plate_text, plate_color))
    
#     panel_w = 400
#     line_h = 30
#     padding = 15
#     panel_h = padding * 2 + line_h * (len(car_summary) + 1)
    
#     panel_x = frame.shape[1] - panel_w - 20
#     panel_y = 20
    
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
#     alpha = 0.6
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
#     cv2.putText(frame, f"Camera {cam_id} - Cars Detected:", 
#                 (panel_x + 10, panel_y + padding + line_h), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
#     for i, (car_id, plate_text, plate_color) in enumerate(car_summary):
#         y_pos = panel_y + padding + line_h * (i + 2)
#         cv2.putText(frame, f"Car {car_id}:", (panel_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#         cv2.putText(frame, plate_text, (panel_x + 120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)

# def annotate_and_log(cam_id: int, frame_nmr: int, frame: np.ndarray, track_ids: np.ndarray, lp_dets):
#     """Annotate frame with license plates, save crops, and log CSV"""
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
#     for i, lp in enumerate(lp_dets.boxes.data.tolist()):
#         x1, y1, x2, y2, lp_score, _cls = lp
#         if lp_score < CONFIDENCE_THRESHOLD:
#             continue

#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
#         if car_id == -1:
#             continue

#         x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
#         if x2i <= x1i or y2i <= y1i:
#             continue

#         lp_crop = frame[y1i:y2i, x1i:x2i]
#         if lp_crop.size == 0:
#             continue

#         text, text_score = read_license_plate(lp_crop)
#         if text:
#             add_recent_plate(cam_id, text)

#         # Draw bounding box with OCR text
#         if text:
#             cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
#             cv2.putText(frame, text, (x1i, max(0, y1i - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#         crop_path = os.path.join(OUTPUT_DIR, f"cam{cam_id}_frame{frame_nmr}_plate{i}.jpg")
#         try:
#             cv2.imwrite(crop_path, lp_crop)
#         except Exception as e:
#             logging.warning(f"Failed to save crop: {e}")

#         with csv_lock:
#             try:
#                 with open(CSV_PATH, "a") as f:
#                     f.write(
#                         f"{timestamp},{cam_id},{frame_nmr},{int(car_id)},"
#                         f"[{xcar1} {ycar1} {xcar2} {ycar2}],"
#                         f"[{x1} {y1} {x2} {y2}],"
#                         f"{lp_score},{text if text else ''},{text_score if text_score else 0}\n"
#                     )
#             except Exception as e:
#                 logging.error(f"Failed to write to CSV: {e}")

# def process_stream(cam_id: int, rtsp_url: str):
#     """Worker thread: grabs frames, runs detection/OCR, pushes annotated frames to display queue"""
#     mot_tracker = Sort()
#     frame_nmr = 0
#     cap = cv2.VideoCapture(rtsp_url)
    
#     if cap.isOpened():
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
#         cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

#     while not shutdown_event.is_set():
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             time.sleep(RETRY_DELAY_SEC)
#             continue
#         frame_nmr += 1

#         # Vehicle detection
#         veh_res = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
#         veh_dets = []
#         for det in veh_res.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = det
#             if int(class_id) in VEHICLE_CLASS_IDS and score > CONFIDENCE_THRESHOLD:
#                 veh_dets.append([x1, y1, x2, y2, score])

#         track_ids = mot_tracker.update(np.asarray(veh_dets)) if veh_dets else np.empty((0, 5))

#         # License plate detection
#         lp_res = lp_model(frame, conf=CONFIDENCE_THRESHOLD)[0]

#         # Annotate and log
#         annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_res)

#         # Draw vehicle boxes
#         for x1, y1, x2, y2, tid in track_ids:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f"Car ID: {int(tid)}", (int(x1), int(y1)-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         draw_recent_plates(frame, cam_id)
#         draw_car_summary(frame, cam_id, track_ids, lp_res)

#         try:
#             if not display_queue.full():
#                 display_queue.put_nowait((cam_id, frame))
#         except:
#             pass

#     cap.release()

# def display_loop(num_cams: int):
#     """Main display loop"""
#     window_names = [f"Camera {i}" for i in range(num_cams)]
#     for w in window_names:
#         cv2.namedWindow(w, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(w, WINDOW_WIDTH, WINDOW_HEIGHT)

#     last_frames = {i: None for i in range(num_cams)}

#     while not shutdown_event.is_set():
#         try:
#             cam_id, frame = display_queue.get(timeout=0.05)
#             last_frames[cam_id] = frame
#         except Empty:
#             pass

#         for i in range(num_cams):
#             if last_frames[i] is not None:
#                 cv2.imshow(f"Camera {i}", last_frames[i])

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             shutdown_event.set()
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     workers = []
#     for i, url in enumerate(RTSP_LINKS):
#         t = threading.Thread(target=process_stream, args=(i, url), daemon=True)
#         t.start()
#         workers.append(t)

#     try:
#         display_loop(len(RTSP_LINKS))
#     finally:
#         shutdown_event.set()
#         for t in workers:
#             t.join(timeout=2.0)




# import os
# import time
# import cv2
# import numpy as np
# import threading
# from queue import Queue, Empty
# from collections import deque
# import logging

# from ultralytics import YOLO
# from sort.sort import Sort
# from util import get_car, read_license_plate
# from config import *

# # Configure logging
# logging.basicConfig(
#     level=logging.DEBUG if DEBUG_MODE else logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(LOG_FILE),
#         logging.StreamHandler()
#     ]
# )

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Initialize models
# try:
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)
#     lp_model = YOLO(LICENSE_PLATE_MODEL_PATH)
#     logging.info("Models loaded successfully")
# except Exception as e:
#     logging.error(f"Failed to load models: {e}")
#     exit(1)

# csv_lock = threading.Lock()
# if not os.path.exists(CSV_PATH):
#     with open(CSV_PATH, "w") as f:
#         f.write(
#             "timestamp,cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n"
#         )

# display_queue: Queue = Queue(maxsize=DISPLAY_QUEUE_MAXSIZE)
# shutdown_event = threading.Event()

# recent_lock = threading.Lock()
# recent_plates = {}

# # Global car ID counter for unique identification across all cameras
# global_car_id_counter = 0
# global_car_id_lock = threading.Lock()

# def get_next_global_car_id():
#     """Get the next unique global car ID across all cameras"""
#     global global_car_id_counter
#     with global_car_id_lock:
#         global_car_id_counter += 1
#         return global_car_id_counter

# def add_recent_plate(cam_id: int, plate_text: str, maxlen: int = RECENT_PLATES_MAX):
#     """Add recent license plate to display panel"""
#     with recent_lock:
#         if cam_id not in recent_plates:
#             recent_plates[cam_id] = deque(maxlen=maxlen)

#         if len(recent_plates[cam_id]) == 0 or recent_plates[cam_id][-1] != plate_text:
#             recent_plates[cam_id].append(plate_text)

# def draw_recent_plates(frame: np.ndarray, cam_id: int):
#     """Draw recent license plates panel on frame"""
#     with recent_lock:
#         items = list(recent_plates.get(cam_id, []))
#     if not items:
#         return
    
#     panel_w = 350
#     line_h = 28
#     padding = 10
#     panel_h = padding * 2 + line_h * len(items)
    
#     # Create semi-transparent overlay
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
#     alpha = 0.4
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
#     # Add title
#     cv2.putText(frame, f"Camera {cam_id} - Recent Plates:", (18, 10 + padding + line_h), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
#     # Add plate numbers
#     for i, txt in enumerate(items):
#         y = 10 + padding + line_h * (i + 2) - 8
#         cv2.putText(frame, f"{i+1}. {txt}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# def draw_car_summary(frame: np.ndarray, cam_id: int, track_ids: np.ndarray, lp_dets):
#     """Draw a summary panel showing all cars with their IDs and license plates"""
#     if len(track_ids) == 0:
#         return
    
#     # Create summary data
#     car_summary = []
#     for x1, y1, x2, y2, tid in track_ids:
#         car_id = int(tid)
#         plate_text = "No plate"
#         plate_color = (0, 165, 255)  # Orange for no plate
        
#         # Check if this car has a license plate
#         for lp in lp_dets:
#             lpx1, lpy1, lpx2, lpy2, lp_score, _cls = lp
#             if lp_score >= CONFIDENCE_THRESHOLD:
#                 xcar1, ycar1, xcar2, ycar2, car_id_check = get_car(lp, track_ids)
#                 if car_id_check == car_id:
#                     # Try to read the license plate
#                     lpx1i, lpy1i, lpx2i, lpy2i = int(lpx1), int(lpy1), int(lpx2), int(lpy2)
#                     if lpx2i > lpx1i and lpy2i > lpy1i:
#                         lp_crop = frame[lpy1i:lpy2i, lpx1i:lpx2i]
#                         if lp_crop.size > 0:
#                             text, text_score = read_license_plate(lp_crop)
#                             if text:
#                                 plate_text = text
#                                 plate_color = (0, 255, 0)  # Green for successful OCR
#                             else:
#                                 plate_text = "Detected"
#                                 plate_color = (0, 255, 255)  # Yellow for detected but OCR failed
#                     break
        
#         car_summary.append((car_id, plate_text, plate_color))
    
#     # Draw summary panel
#     panel_w = 400
#     line_h = 30
#     padding = 15
#     panel_h = padding * 2 + line_h * (len(car_summary) + 1)  # +1 for header
    
#     # Position panel on the right side
#     panel_x = frame.shape[1] - panel_w - 20
#     panel_y = 20
    
#     # Create semi-transparent overlay
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
#     alpha = 0.6
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
#     # Add title
#     cv2.putText(frame, f"Camera {cam_id} - Cars Detected:", 
#                 (panel_x + 10, panel_y + padding + line_h), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
#     # Add car information
#     for i, (car_id, plate_text, plate_color) in enumerate(car_summary):
#         y_pos = panel_y + padding + line_h * (i + 2)
        
#         # Car ID
#         cv2.putText(frame, f"Car {car_id}:", 
#                     (panel_x + 10, y_pos), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # License plate
#         cv2.putText(frame, plate_text, 
#                     (panel_x + 120, y_pos), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)

# def annotate_and_log(
#     cam_id: int,
#     frame_nmr: int,
#     frame: np.ndarray,
#     track_ids: np.ndarray,
#     lp_dets,
# ):
#     """
#     For each detected license plate, find its parent vehicle, OCR, draw, save crop, and log CSV.
#     """
#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
#     # Log detection counts for debugging
#     if LOG_DETECTIONS:
#         logging.debug(f"[CAM {cam_id}] Frame {frame_nmr}: {len(track_ids)} vehicles, {len(lp_dets.boxes.data.tolist())} license plates")
    
#     for i, lp in enumerate(lp_dets.boxes.data.tolist()):
#         x1, y1, x2, y2, lp_score, _cls = lp
        
#         # Skip low confidence detections
#         if lp_score < CONFIDENCE_THRESHOLD:
#             if LOG_DETECTIONS:
#                 logging.debug(f"[CAM {cam_id}] Skipping low confidence LP: {lp_score:.3f}")
#             continue

#         # Find parent car bbox + id
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
#         if car_id == -1:
#             if LOG_DETECTIONS:
#                 logging.debug(f"[CAM {cam_id}] No parent car found for LP")
#             continue

#         # Crop LP
#         x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
#         if x2i <= x1i or y2i <= y1i:
#             continue

#         lp_crop = frame[y1i:y2i, x1i:x2i]
#         if lp_crop.size == 0:
#             continue

#         # OCR the license plate
#         text, text_score = read_license_plate(lp_crop)
        
#         if LOG_DETECTIONS:
#             logging.info(f"[CAM {cam_id}] Detected LP: '{text}' (score: {text_score:.3f})")

#         # Draw license plate bounding box
#         cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 0, 255), 3)
#         label = f"{text}" if text else "plate"
#         cv2.putText(
#             frame,
#             label,
#             (x1i, max(0, y1i - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.9,
#             (0, 255, 0) if text else (0, 165, 255),
#             2,
#         )

#         # Save license plate crop
#         crop_path = os.path.join(OUTPUT_DIR, f"cam{cam_id}_frame{frame_nmr}_plate{i}.jpg")
#         try:
#             cv2.imwrite(crop_path, lp_crop)
#         except Exception as e:
#             logging.warning(f"Failed to save crop: {e}")

#         if text:
#             add_recent_plate(cam_id, text)

#         # Log to CSV with timestamp
#         with csv_lock:
#             try:
#                 with open(CSV_PATH, "a") as f:
#                     f.write(
#                         f"{timestamp},{cam_id},{frame_nmr},{int(car_id)},"
#                         f"[{xcar1} {ycar1} {xcar2} {ycar2}],"
#                         f"[{x1} {y1} {x2} {y2}],"
#                         f"{lp_score},{text if text else ''},{text_score if text_score else 0}\n"
#                     )
#             except Exception as e:
#                 logging.error(f"Failed to write to CSV: {e}")

# def process_stream(cam_id: int, rtsp_url: str):
#     """
#     Worker thread: grabs frames, runs detection/ocr, pushes annotated frames to display queue.
#     All OpenCV GUI calls must be avoided here (macOS requirement).
#     """
#     mot_tracker = Sort()
#     frame_nmr = 0
#     cap = None
#     consecutive_failures = 0
#     max_consecutive_failures = MAX_CONSECUTIVE_FAILURES
#     detection_count = 0

#     def open_capture():
#         c = cv2.VideoCapture(rtsp_url)
#         if c.isOpened():
#             c.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
#             c.set(cv2.CAP_PROP_FPS, TARGET_FPS)
#             c.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
#             c.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
#         return c

#     cap = open_capture()
#     logging.info(f"[CAM {cam_id}] Started processing stream: {rtsp_url}")

#     while not shutdown_event.is_set():
#         if not cap or not cap.isOpened():
#             if cap:
#                 cap.release()
#             consecutive_failures += 1
#             if consecutive_failures > max_consecutive_failures:
#                 logging.error(f"[CAM {cam_id}] Too many consecutive failures, stopping stream")
#                 break
#             logging.warning(f"[CAM {cam_id}] Reconnecting... (attempt {consecutive_failures})")
#             time.sleep(RETRY_DELAY_SEC)
#             cap = open_capture()
#             continue

#         ret, frame = cap.read()
#         if not ret or frame is None:
#             consecutive_failures += 1
#             logging.warning(f"[CAM {cam_id}] Read failed, reconnecting...")
#             cap.release()
#             time.sleep(RETRY_DELAY_SEC)
#             cap = open_capture()
#             continue

#         consecutive_failures = 0  # Reset failure counter on success
#         frame_nmr += 1

#         # Process every frame for better detection
#         try:
#             # Vehicle detection
#             veh_res = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
#             veh_dets = []
#             for det in veh_res.boxes.data.tolist():
#                 x1, y1, x2, y2, score, class_id = det
#                 if int(class_id) in VEHICLE_CLASS_IDS and score > CONFIDENCE_THRESHOLD:
#                     veh_dets.append([x1, y1, x2, y2, score])

#             # Vehicle tracking
#             if len(veh_dets) == 0:
#                 track_ids = np.empty((0, 5))
#             else:
#                 track_ids = mot_tracker.update(np.asarray(veh_dets))
#                 if LOG_DETECTIONS and len(track_ids) > 0:
#                     logging.debug(f"[CAM {cam_id}] Frame {frame_nmr}: Detected {len(veh_dets)} vehicles, tracking {len(track_ids)}")

#             # License plate detection
#             lp_res = lp_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
#             lp_count = len(lp_res.boxes.data.tolist())
            
#             if LOG_DETECTIONS and lp_count > 0:
#                 logging.debug(f"[CAM {cam_id}] Frame {frame_nmr}: Detected {lp_count} license plates")

#             # Annotate and log
#             annotate_and_log(cam_id, frame_nmr, frame, track_ids, lp_res)

#             # Draw vehicle bounding boxes with IDs
#             for x1, y1, x2, y2, tid in track_ids:
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
#                 # Draw car ID with better visibility
#                 car_id_text = f"Car ID: {int(tid)}"
#                 cv2.putText(
#                     frame,
#                     car_id_text,
#                     (int(x1), int(y1) - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
                
#                 # Check if this car has a license plate
#                 car_has_plate = False
#                 plate_text = ""
#                 for lp in lp_dets_list:
#                     lpx1, lpy1, lpx2, lpy2, lp_score, _cls = lp
#                     if lp_score >= CONFIDENCE_THRESHOLD:
#                         # Find parent car for this license plate
#                         xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
#                         if car_id == tid:
#                             car_has_plate = True
#                             # Try to read the license plate
#                             lpx1i, lpy1i, lpx2i, lpy2i = int(lpx1), int(lpy1), int(lpx2), int(lpy2)
#                             if lpx2i > lpx1i and lpy2i > lpy1i:
#                                 lp_crop = frame[lpy1i:lpy2i, lpx1i:lpx2i]
#                                 if lp_crop.size > 0:
#                                     text, text_score = read_license_plate(lp_crop)
#                                     if text:
#                                         plate_text = f"Plate: {text}"
#                                         break
                
#                 # Draw license plate info if available
#                 if car_has_plate and plate_text:
#                     cv2.putText(
#                         frame,
#                         plate_text,
#                         (int(x1), int(y2) + 25),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 255, 255),
#                         2,
#                     )
#                 elif car_has_plate:
#                     cv2.putText(
#                         frame,
#                         "Plate: Detected (OCR failed)",
#                         (int(x1), int(y2) + 25),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (0, 165, 255),
#                         2,
#                     )

#             # Draw recent plates
#             draw_recent_plates(frame, cam_id)

#             # Draw car summary
#             draw_car_summary(frame, cam_id, track_ids, lp_dets_list)

#             # Add camera info overlay
#             cv2.putText(frame, f"Camera {cam_id} - Frame {frame_nmr}", (10, frame.shape[0] - 20),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Add detection statistics
#             stats_text = f"Cars: {len(track_ids)} | Plates: {len([lp for lp in lp_dets_list if lp[4] >= CONFIDENCE_THRESHOLD])}"
#             cv2.putText(frame, stats_text, (10, frame.shape[0] - 50),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # Push to display queue
#             try:
#                 if not display_queue.full():
#                     display_queue.put_nowait((cam_id, frame))
#             except Exception as e:
#                 logging.debug(f"Display queue full or error: {e}")

#         except Exception as e:
#             logging.error(f"[CAM {cam_id}] Error processing frame: {e}")
#             continue

#     if cap:
#         cap.release()
#     logging.info(f"[CAM {cam_id}] Stream processing stopped")

# def display_loop(num_cams: int):
#     """
#     Main-thread display loop (macOS-safe). Show one window per camera.
#     """
#     window_names = [f"Camera {i}" for i in range(num_cams)]
#     for w in window_names:
#         cv2.namedWindow(w, cv2.WINDOW_NORMAL)
#         cv2.resizeWindow(w, WINDOW_WIDTH, WINDOW_HEIGHT)

#     last_frames = {i: None for i in range(num_cams)}
#     frame_count = 0
#     start_time = time.time()

#     logging.info(f"Starting display loop for {num_cams} cameras")

#     while not shutdown_event.is_set():
#         try:
#             cam_id, frame = display_queue.get(timeout=0.05)
#             last_frames[cam_id] = frame
#             frame_count += 1
#         except Empty:
#             pass

#         # Calculate and display FPS
#         if frame_count % 30 == 0:
#             elapsed_time = time.time() - start_time
#             fps = frame_count / elapsed_time if elapsed_time > 0 else 0
#             logging.info(f"Display FPS: {fps:.2f}")

#         # Draw latest available frames
#         for i in range(num_cams):
#             if last_frames[i] is not None:
#                 cv2.imshow(f"Camera {i}", last_frames[i])

#         # Handle quit
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             logging.info("Quit signal received")
#             shutdown_event.set()
#             break

#     cv2.destroyAllWindows()
#     logging.info("Display loop stopped")

# if __name__ == "__main__":
#     logging.info("Starting Number Plate Recognition System")
#     logging.info(f"Connecting to {len(RTSP_LINKS)} camera streams")
#     logging.info(f"Detection settings: Confidence={CONFIDENCE_THRESHOLD}, Frame Skip={MAX_FRAME_SKIP}")
    
#     # Start worker threads for each camera
#     workers = []
#     for i, url in enumerate(RTSP_LINKS):
#         t = threading.Thread(target=process_stream, args=(i, url), daemon=True)
#         t.start()
#         workers.append(t)
#         logging.info(f"Started worker thread for Camera {i}")

#     try:
#         display_loop(len(RTSP_LINKS))
#     except KeyboardInterrupt:
#         logging.info("Keyboard interrupt received")
#     except Exception as e:
#         logging.error(f"Unexpected error: {e}")
#     finally:
#         logging.info("Shutting down...")
#         shutdown_event.set()
#         for t in workers:
#             t.join(timeout=2.0)
#         logging.info("System shutdown complete")