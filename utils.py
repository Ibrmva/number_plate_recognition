from ocr.reader import read_license_plate
import re

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for (xcar1, ycar1, xcar2, ycar2, car_id) in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1

def license_complies_format(text: str) -> bool:
    pattern = r'^[0-9]{3}[A-Z]{3}[0-9]{2,3}$'
    return re.match(pattern, text) is not None

def format_license(text: str) -> str:
    return text.upper()
