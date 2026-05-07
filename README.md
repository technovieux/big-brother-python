# Big Brother - Person and Face Detection Application

A comprehensive GUI application for real-time detection and tracking of people and faces in live video feeds, recorded videos, and static images using advanced computer vision techniques.

## Features

✨ **Multi-Source Processing**
- Real-time detection from webcam
- Video file processing (MP4, AVI, MOV)
- Single image analysis

🎯 **Detection Capabilities**
- **Face Detection** using OpenCV Haar Cascade Classifier
- **Person Detection** using YOLOv8 (nano model) for optimized performance
- Visual overlays with bounding boxes (blue for faces, green for persons)

📊 **Data Visualization**
- Time-series graph showing detection counts over time
- Histogram of detection area distributions
- Real-time counter display showing current and cumulative detections

💾 **Data Export**
- Export detection data in JSON format (with timestamps and summary statistics)
- Export detection data in CSV format (tabular format for spreadsheets)
- Timestamped export files with mode and session information

🎨 **User Interface**
- Modern dark/light mode support with customtkinter
- Responsive tabbed interface
- Loading screen with progress animation
- Intuitive button controls for each mode

## Requirements

- **Python** 3.8 or higher
- **Operating System**: Windows, macOS, or Linux

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/technovieux/big-brother-python.git
cd big-brother-python
```

### 2. Install Dependencies
```bash
pip install customtkinter opencv-python pillow ultralytics matplotlib
```

### 3. Download Models
The application requires:
- **YOLOv8 nano model** - Automatically downloaded on first run (~6.3 MB)
- **Haar Cascade XML** - Included with OpenCV

### 4. Prepare Resources
Ensure the following files are in the project directory:
- `logo.ico` - Application window icon
- `loading_screen.png` - Splash screen image displayed on startup

## Usage

### Running the Application
```bash
python big_brother.py
```

The application will start with a loading screen (3 seconds) and then display the main interface.

### Real-Time Data Tab
1. Click **"Start Real-Time"** to activate your webcam
2. The application displays live video with detection overlays
3. Monitor real-time person and face counts
4. Click **"Stop Real-Time"** to end the session

### Video Tab
1. Click **"Select Video"** to choose a video file
2. Click **"Play Video"** to start processing
3. Detection happens on every 2nd frame for performance optimization
4. Click **"Stop Video"** to interrupt processing

### Image Tab
1. Click **"Select Image"** to choose an image file
2. The image loads and detection runs automatically
3. View results in the histogram and graphs

### Data Export
1. Select your preferred format: **JSON** or **CSV**
2. Click **"Export"**
3. Choose save location and filename
4. Data includes timestamps, detection counts, and summary statistics

### Reset Counters
Click **"reset values"** in the "since the beginning" panel to reset cumulative counters.

## Project Structure

```
big-brother/
├── big_final.py              # Main application file
├── logo.ico                  # Application icon
├── loading_screen.png        # Splash screen image
├── yolov8n.pt               # YOLOv8 nano model (auto-downloaded)
├── README.md               # This file
└── runs/                   # Detection results directory (auto-created)
    └── detect/
        └── predict/        # YOLO prediction outputs
```

## Configuration

### Appearance Settings
Located at the top of `big_brother.py`:
```python
customtkinter.set_appearance_mode("System")  # "System", "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
```

### Detection Parameters
- **Video Detection Skip**: `self.video_detection_skip = 2` (detects every 2nd frame)
- **Haar Cascade Scale**: `1.1` (detection sensitivity)
- **Haar Cascade Min Neighbors**: `4` (grouping threshold)

### Color Scheme
- **Face Bounding Boxes**: Blue (RGB: 255, 0, 0)
- **Person Bounding Boxes**: Green (RGB: 0, 255, 0)
- **UI Background**: Dark gray (#2b2b2b)

## Export Features

### JSON Export Format
```json
{
  "mode": "realtime",
  "timestamp": "2026-05-07T12:30:45.123456",
  "data": [
    {
      "time": 0.5,
      "persons": 2,
      "faces": 1
    }
  ],
  "summary": {
    "total_persons": 5,
    "total_faces": 3,
    "duration": 10.5
  }
}
```

### CSV Export Format
Includes metadata header, data rows, and summary section in a tabular format.

## Performance Optimization

- **Video Frame Skipping**: Detections run every 2nd frame to reduce processing load
- **Image Resizing**: Frames are automatically resized to fit display labels
- **Threading**: Heavy computations run in separate threads to keep UI responsive
- **YOLOv8 Nano**: Uses lightweight nano model for faster inference

## Troubleshooting

### Issue: "Error loading video file"
- Ensure video format is supported (.mp4, .avi, .mov)
- Check file is not corrupted
- Verify sufficient disk space

### Issue: Webcam not detected
- Check camera permissions (Windows/macOS may require approval)
- Ensure no other application is using the camera
- Try a different USB port if using external camera

### Issue: Slow performance
- Reduce resolution of input video/image
- Increase `video_detection_skip` value (e.g., 4 or 5)
- Close other applications consuming CPU/GPU

### Issue: Low detection accuracy
- Ensure adequate lighting in the scene
- Position subjects within camera frame clearly
- Adjust `detectMultiScale` parameters for sensitivity

### Issue: Missing YOLOv8 model
- Delete `yolov8n.pt` and restart application
- Model will automatically download (~6.3 MB)
- Ensure internet connection is available

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| customtkinter | >=5.0 | Modern GUI framework |
| opencv-python | >=4.5 | Computer vision and Haar Cascade |
| pillow | >=8.0 | Image processing |
| ultralytics | >=8.0 | YOLOv8 object detection |
| matplotlib | >=3.0 | Graph visualization |

## Notes

- Detection results are logged to console (print statements) for debugging
- All timestamps are in ISO 8601 format for consistency
- The application maximizes the window on startup for better visibility
- Detection areas are calculated as width × height in pixels

## Future Enhancements

Potential improvements for future versions:
- [ ] GPU acceleration support (CUDA)
- [ ] Multi-person tracking across frames
- [ ] Custom model training interface
- [ ] Advanced filtering and noise reduction
- [ ] Confidence threshold adjustment UI
- [ ] Recording detected video with overlays
- [ ] Batch processing mode
- [ ] Statistics dashboard with advanced analytics

## License

This project is provided as-is. Modify and use as needed for your purposes.

## Author

Created as a comprehensive computer vision application for real-time detection and analysis.

---

**Note**: This application requires adequate system resources for smooth operation. Recommended specs:
- CPU: Intel i5 or equivalent
- RAM: 4GB minimum (8GB recommended)
- GPU: Optional (improves performance with CUDA support)
