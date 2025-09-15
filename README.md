# Smart Helmet

A real-time computer vision system that combines object detection, depth estimation, and IoT sensors to enhance safety for motorcyclists and industrial workers. The system provides proximity warnings, GPS tracking, and environmental monitoring through a web-based dashboard.

## 🚀 Key Features

### Computer Vision Pipeline
- **Real-time Object Detection**: YOLOv8n model for detecting vehicles, pedestrians, and obstacles
- **Monocular Depth Estimation**: Depth-Anything transformer model for distance calculation
- **Proximity Analysis**: Intelligent warning system with DANGER/WARNING/SAFE status levels
- **Dual Video Streams**: Simultaneous object detection and depth visualization

### IoT Integration
- **GPS Tracking**: Real-time location monitoring with interactive map
- **Speed Monitoring**: Vehicle speed detection and logging
- **Environmental Sensors**: MQ3 gas sensor for air quality monitoring
- **Serial Communication**: Bidirectional data exchange with hardware components

### Web Dashboard
- **Live Video Feeds**: Real-time streaming of processed camera feeds
- **Interactive Map**: GPS location tracking with Leaflet.js integration
- **Sensor Metrics**: Live display of speed, location, and environmental data
- **Responsive UI**: Modern web interface with Tailwind CSS

## 🛠 Technical Stack

- **Backend**: Flask (Python 3.8+)
- **Computer Vision**: OpenCV, PyTorch, Ultralytics YOLO, Transformers
- **Deep Learning Models**: 
  - YOLOv8n for object detection
  - Depth-Anything-Small for depth estimation
- **Hardware Communication**: PySerial for Arduino/microcontroller integration
- **Frontend**: HTML5, JavaScript, Tailwind CSS, Leaflet.js
- **GPU Acceleration**: CUDA support for faster inference

## 📋 System Requirements

### Hardware
- **Camera**: USB webcam or integrated camera
- **GPU**: CUDA-capable GPU (recommended for real-time performance)
- **Microcontroller**: Arduino/ESP32 with GPS and MQ3 sensor modules
- **Serial Port**: COM7 (configurable in `app.py`)

### Software
- Python 3.8 or higher
- CUDA Toolkit (optional, for GPU acceleration)
- Windows/Linux/macOS support

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd smart-helmet

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Setup
Download the required AI models:

**Option 1: Automatic Download (Recommended)**
```bash
# Models will download automatically on first run
# YOLOv8n (~6MB) and Depth-Anything-Small (~97MB)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Option 2: Manual Download**
```bash
# Download YOLOv8n model weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
# Or using curl on Windows
curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

**Note**: The Depth-Anything model will be downloaded automatically from Hugging Face on first use (~97MB).

### 3. Hardware Configuration
- Connect camera to USB port
- Connect Arduino/microcontroller to COM7 (or modify port in `app.py`)
- Ensure GPS and MQ3 sensors are properly wired

### 4. Run Application
```bash
python app.py
```

### 5. Access Dashboard
Open browser and navigate to: `http://localhost:5000`

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard interface |
| `/yolo_feed` | GET | Live object detection video stream |
| `/depth_feed` | GET | Live depth estimation video stream |
| `/proximity_status` | GET | Current proximity warning status |
| `/sensor_data` | GET | GPS, speed, and sensor readings (JSON) |

## 🏗 Project Structure

```
smart-helmet/
├── app.py                      # Flask application & serial communication
├── requirements.txt            # Python dependencies
├── yolov8n.pt                 # YOLOv8 model weights
├── models/
│   ├── yolo.py                # YOLO object detection wrapper
│   └── depth.py               # Depth estimation model
├── utils/
│   ├── video_feed.py          # Video capture & processing pipeline
│   └── proximity_analyzer.py   # Safety distance calculations
└── templates/
    └── index.html             # Web dashboard interface
```

## ⚙️ Configuration

### Camera Settings
Modify `CAM` variable in `utils/video_feed.py`:
```python
CAM = 1  # Change to 0 for default camera
```

### Serial Communication
Update COM port in `app.py`:
```python
ser = serial.Serial('COM7', 460800, timeout=0.1)  # Change COM7 to your port
```

### Proximity Thresholds
Adjust warning levels in `utils/proximity_analyzer.py`:
```python
self.DANGER_THRESHOLD = 60   # Closer objects trigger danger
self.WARNING_THRESHOLD = 120 # Medium distance warning
```

## 🔧 Hardware Integration

### Expected Serial Data Format
```
GPS:latitude,longitude,speed
MQ3:sensor_value
```

### Serial Output Format
```
PROXIMITY:DANGER|WARNING|SAFE
```

## 🚨 Troubleshooting

### Common Issues
- **Camera not detected**: Check camera index in `video_feed.py`
- **Serial port error**: Verify COM port and device connection
- **CUDA out of memory**: Reduce batch size or use CPU inference
- **Model loading fails**: Ensure internet connection for first-time model download

### Performance Optimization
- **GPU Usage**: Ensure CUDA is properly installed for faster inference
- **Threading**: Application uses multi-threading for optimal performance
- **Frame Rate**: Adjust sleep intervals in video processing loops

### Debug Mode
Enable detailed logging by checking console output for:
- `[SERIAL]` - Serial communication logs
- `[UPDATE]` - Sensor data updates
- `[ERROR]` - Error messages and exceptions

## 📊 Performance Metrics

- **Object Detection**: ~30 FPS (with GPU), ~10 FPS (CPU only)
- **Depth Estimation**: ~15 FPS (with GPU), ~3 FPS (CPU only)
- **Serial Communication**: 460800 baud rate
- **Web Interface**: Real-time updates every 500ms (proximity) / 1000ms (sensors)

## 🔮 Future Enhancements

- Machine learning model optimization for edge deployment
- Advanced sensor fusion algorithms
- Mobile app integration
- Cloud data logging and analytics
- Multi-camera support for 360° coverage

## 👥 Development Team

Created by **Group 7, EAC 2022-26**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
