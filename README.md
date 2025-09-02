# 🚗 Number Plate Recognition System

A real-time automatic number plate recognition system that can process multiple camera streams simultaneously, detect vehicles, track them with unique IDs, and read license plates using OCR.

## ✨ Features

- **Multi-Camera Support**: Connect to multiple RTSP camera streams simultaneously
- **Real-Time Processing**: Live vehicle detection and license plate recognition
- **Unique Car Tracking**: Each detected car gets a unique ID using SORT algorithm
- **License Plate OCR**: Automatic license plate text extraction using EasyOCR
- **Multi-Threading**: Efficient processing with separate threads for each camera
- **Comprehensive Logging**: Detailed logs and CSV output for analysis
- **Configurable Settings**: Easy-to-modify configuration file

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera 1      │    │   Camera 2      │    │   Camera N      │
│   (RTSP)        │    │   (RTSP)        │    │   (RTSP)        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Thread 1       │    │  Thread 2       │    │  Thread N       │
│  - Vehicle      │    │  - Vehicle      │    │  - Vehicle      │
│    Detection    │    │    Detection    │    │    Detection    │
│  - Tracking     │    │  - Tracking     │    │  - Tracking     │
│  - LP Detection │    │  - LP Detection │    │  - LP Detection │
│  - OCR          │    │  - OCR          │    │  - OCR          │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    Display Queue        │
                    │  (Thread-Safe Buffer)   │
                    └─────────┬───────────────┘
                              ▼
                    ┌─────────────────────────┐
                    │   Main Display Thread   │
                    │  (OpenCV Windows)       │
                    └─────────────────────────┘
```

## 📋 Requirements

- Python 3.8+
- OpenCV 4.x
- PyTorch
- Ultralytics YOLO
- EasyOCR
- Other dependencies (see requirements.txt)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd number_plate_recognition
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models**:
   - Place your YOLO vehicle detection model as `models/yolo11n.pt`
   - Place your license plate detection model as `models/license_plate_detector.pt`

## ⚙️ Configuration

Edit `config.py` to customize:

- **Camera URLs**: Update `RTSP_LINKS` with your camera streams
- **Model Paths**: Set paths to your YOLO models
- **Detection Settings**: Adjust confidence thresholds and performance parameters
- **Output Settings**: Configure output directories and file paths

### Example Camera Configuration:
```python
RTSP_LINKS = [
    "rtsp://username:password@192.168.1.100:554/stream1",
    "rtsp://username:password@192.168.1.101:554/stream1",
    # Add more cameras as needed
]
```

## 🎯 Usage

### 1. Test System
Before running the main system, test your setup:
```bash
python test_system.py
```

This will:
- Test connections to all configured cameras
- Verify model loading
- Provide a status report

### 2. Run Main System
```bash
python main.py
```

The system will:
- Connect to all configured cameras
- Start processing streams in real-time
- Display live video feeds with detections
- Save license plate images to `outputs/license_plates/`
- Log results to `real_time_results.csv`

### 3. Controls
- **Quit**: Press `q` in any camera window
- **Resize**: Drag window corners to resize
- **Move**: Drag windows to reposition

## 📊 Output

### CSV Log (`real_time_results.csv`)
```
timestamp,cam_id,frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
2024-01-15 14:30:25,0,1250,42,"[100 150 300 400]","[120 170 280 190]",0.95,ABC1234,0.87
```

### License Plate Images
- Saved to `outputs/license_plates/`
- Naming format: `cam{camera_id}_frame{frame_number}_plate{plate_index}.jpg`

### Log File (`number_plate_detection.log`)
- Detailed system logs with timestamps
- Camera connection status
- Error reports and debugging information

## 🔧 Troubleshooting

### Camera Connection Issues
- Verify RTSP URLs are correct
- Check network connectivity
- Ensure camera credentials are valid
- Test with VLC or other RTSP players

### Performance Issues
- Reduce `TARGET_FPS` in config
- Increase `MAX_FRAME_SKIP`
- Lower `CONFIDENCE_THRESHOLD`
- Check GPU availability for YOLO models

### Model Loading Errors
- Verify model files exist in `models/` directory
- Check model file permissions
- Ensure PyTorch version compatibility

## 📁 File Structure

```
number_plate_recognition/
├── main.py                 # Main application
├── config.py              # Configuration file
├── test_system.py         # System test script
├── util.py                # Utility functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # YOLO model files
│   ├── yolo11n.pt
│   └── license_plate_detector.pt
├── outputs/              # Output directory
│   └── license_plates/  # License plate images
├── sort/                 # SORT tracking algorithm
└── venv/                 # Virtual environment
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [SORT](https://github.com/abewley/sort) for object tracking
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `number_plate_detection.log`
3. Open an issue on GitHub
4. Contact the development team

---

**Happy License Plate Recognition! 🚗📸**
