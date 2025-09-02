import string
import easyocr
import threading
import cv2
import numpy as np

_tls = threading.local()

def _get_reader():
    if not hasattr(_tls, "reader"):
        _tls.reader = easyocr.Reader(['en'], gpu=False)
    return _tls.reader

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    Made more flexible to accept various formats.
    """
    if not text or len(text) < 3:
        return False
    
    # Accept any text with 3+ characters that contains letters and numbers
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    
    if has_letter and has_digit:
        return True
    
    # Also accept the original strict format if text is exactly 7 characters
    if len(text) == 7:
        if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
           (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
           (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
           (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
           (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
            return True
    
    return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Enhanced OCR: try multiple preprocess strategies and use a thread-local OCR reader.
    Returns (text, score) or (None, None).
    """
    reader = _get_reader()

    variants = []
    crop = license_plate_crop
    if crop is None or (hasattr(crop, "size") and getattr(crop, "size", 0) == 0):
        return None, None

    if isinstance(crop, np.ndarray):
        # Original image
        variants.append(crop)  # BGR
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants.append(gray)  # grayscale
        
        # Resize for better OCR (make it larger)
        height, width = gray.shape
        if width < 100:  # If too small, resize
            scale_factor = 3.0
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            variants.append(resized)
        
        # Basic thresholding
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th)
        
        # Inverse thresholding
        _, thi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        variants.append(thi)
        
        # Adaptive thresholding
        adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        variants.append(adap)
        
        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gauss_th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(gauss_th)
        
        # Median blur + threshold
        median_blurred = cv2.medianBlur(gray, 5)
        _, median_th = cv2.threshold(median_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(median_th)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        _, clahe_th = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(clahe_th)
        
        # Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        variants.append(morph)
        
        # Edge enhancement
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        _, sharp_th = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(sharp_th)
        
    else:
        variants.append(crop)

    best_text = None
    best_score = 0
    all_detections = []

    for i, v in enumerate(variants):
        try:
            # Use different confidence thresholds for different variants
            conf_threshold = 0.1 if i < 3 else 0.2  # Lower threshold for first few variants
            
            detections = reader.readtext(v, confidence=conf_threshold)
            for detection in detections:
                bbox, text, score = detection
                text = text.upper().replace(' ', '').replace('-', '').replace('_', '').replace('.', '')
                
                # Clean up common OCR mistakes
                text = text.replace('O', '0').replace('I', '1').replace('S', '5').replace('G', '6')
                
                # Log all detections for debugging
                all_detections.append(f"Variant {i}: '{text}' (score: {score:.3f})")
                
                # Accept any text with 3+ characters that looks like a license plate
                if text and len(text) >= 3:
                    # Check if it contains both letters and numbers
                    has_letter = any(c.isalpha() for c in text)
                    has_digit = any(c.isdigit() for c in text)
                    
                    if has_letter and has_digit and score > best_score:
                        best_score = score
                        best_text = text
                        
                        # If we find a text that complies with format, use it immediately
                        if license_complies_format(text):
                            print(f"OCR Success - Perfect match: '{text}' (score: {score:.3f})")
                            return format_license(text), score
                            
        except Exception as e:
            continue

    # If no perfect match, return the best text we found
    if best_text and best_score > 0.3:  # Only return if confidence is reasonable
        print(f"OCR Debug - All detections: {all_detections}")
        print(f"OCR Debug - Best text: '{best_text}' (score: {best_score:.3f})")
        return best_text, best_score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1