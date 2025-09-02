#!/usr/bin/env python3
"""
Debug script to identify issues with the number plate recognition system
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate
from config import *

def test_single_camera():
    """Test a single camera connection and processing"""
    print("🔍 Testing single camera processing...")
    
    # Use the first camera URL
    rtsp_url = RTSP_LINKS[0]
    print(f"Testing camera: {rtsp_url}")
    
    # Try to connect
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    print("✅ Camera connected successfully")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    
    # Load models
    try:
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        lp_model = YOLO(LICENSE_PLATE_MODEL_PATH)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        cap.release()
        return False
    
    # Initialize tracker
    mot_tracker = Sort()
    
    # Process a few frames
    frame_count = 0
    max_frames = 10
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Failed to read frame {frame_count}")
            break
        
        print(f"\n--- Frame {frame_count + 1} ---")
        print(f"Frame shape: {frame.shape}")
        
        # Vehicle detection
        try:
            veh_res = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
            veh_dets = []
            
            for det in veh_res.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = det
                if int(class_id) in VEHICLE_CLASS_IDS and score > CONFIDENCE_THRESHOLD:
                    veh_dets.append([x1, y1, x2, y2, score])
                    print(f"  Vehicle detected: class {int(class_id)}, confidence {score:.3f}")
            
            print(f"Total vehicle detections: {len(veh_dets)}")
            
            # Vehicle tracking
            if len(veh_dets) > 0:
                track_ids = mot_tracker.update(np.asarray(veh_dets))
                print(f"Tracking {len(track_ids)} vehicles")
                
                for x1, y1, x2, y2, tid in track_ids:
                    print(f"  Track ID {int(tid)}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            else:
                track_ids = np.empty((0, 5))
                print("No vehicles to track")
            
            # License plate detection
            lp_res = lp_model(frame, conf=CONFIDENCE_THRESHOLD)[0]
            lp_dets = lp_res.boxes.data.tolist()
            
            print(f"License plate detections: {len(lp_dets)}")
            
            for i, lp in enumerate(lp_dets):
                x1, y1, x2, y2, lp_score, _cls = lp
                print(f"  LP {i+1}: confidence {lp_score:.3f}")
                
                if lp_score >= CONFIDENCE_THRESHOLD:
                    # Find parent car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
                    if car_id != -1:
                        print(f"    Parent car ID: {int(car_id)}")
                        
                        # Try OCR
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        if x2i > x1i and y2i > y1i:
                            lp_crop = frame[y1i:y2i, x1i:x2i]
                            if lp_crop.size > 0:
                                text, text_score = read_license_plate(lp_crop)
                                if text:
                                    print(f"    OCR Result: '{text}' (score: {text_score:.3f})")
                                else:
                                    print(f"    OCR failed or no text detected")
                    else:
                        print(f"    No parent car found")
                else:
                    print(f"    Skipping low confidence detection")
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
        
        frame_count += 1
        time.sleep(0.1)  # Small delay
    
    cap.release()
    print(f"\n✅ Processed {frame_count} frames")
    return True

def test_util_functions():
    """Test utility functions"""
    print("\n🔧 Testing utility functions...")
    
    # Test get_car function
    print("Testing get_car function...")
    
    # Create mock data
    license_plate = [100, 100, 200, 150, 0.8, 0]  # x1, y1, x2, y2, score, class
    vehicle_tracks = [
        [50, 50, 250, 200, 1],   # Vehicle 1: contains the license plate
        [300, 100, 500, 300, 2], # Vehicle 2: doesn't contain the license plate
    ]
    
    result = get_car(license_plate, vehicle_tracks)
    print(f"get_car result: {result}")
    
    if result != (-1, -1, -1, -1, -1):
        print("✅ get_car function working correctly")
    else:
        print("❌ get_car function not working correctly")
    
    # Test read_license_plate function
    print("\nTesting read_license_plate function...")
    
    # Create a simple test image
    test_img = np.zeros((50, 100, 3), dtype=np.uint8)
    cv2.putText(test_img, "ABC123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    text, score = read_license_plate(test_img)
    print(f"OCR test result: text='{text}', score={score}")
    
    if text:
        print("✅ OCR function working correctly")
    else:
        print("❌ OCR function not working correctly")

def main():
    """Run all debug tests"""
    print("🐛 Number Plate Recognition System - Debug Mode")
    print("=" * 60)
    
    # Test utility functions first
    test_util_functions()
    
    # Test single camera processing
    print("\n" + "=" * 60)
    camera_ok = test_single_camera()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEBUG SUMMARY")
    print("=" * 60)
    
    if camera_ok:
        print("✅ Camera processing test completed")
        print("Check the output above for any detection issues")
    else:
        print("❌ Camera processing test failed")
    
    print("\n🔍 Next steps:")
    print("1. Check if vehicles are being detected")
    print("2. Check if license plates are being detected")
    print("3. Check if OCR is working")
    print("4. Check if tracking is working")
    print("5. Run the main system: python main.py")

if __name__ == "__main__":
    main()
