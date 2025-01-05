from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import serial
import threading
import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model (updated)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load DPT model for depth estimation
depth_model = DPTForDepthEstimation.from_pretrained('Intel/dpt-large')
feature_extractor = DPTFeatureExtractor.from_pretrained('Intel/dpt-large')

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize sensor data variables
latitude = "N/A"
longitude = "N/A"
speed = 90
mq3_value = 199

# Define depth threshold for warnings
depth_threshold = 0.5  # Example: 0.5 meters

# Initialize serial connection for sensor data
try:
    ser = serial.Serial('COM3', 115200, timeout=1)  # Replace with your serial port
    print("Connected to serial port")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    ser = None

# Function to read sensor data
def read_sensor_data():
    global latitude, longitude, speed, mq3_value
    while True:
        if ser and ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("GPS:"):
                gps_data = line.split(":")[1].split(",")
                latitude, longitude, speed = gps_data
                print(f"GPS Data - Latitude: {latitude}, Longitude: {longitude}, Speed: {speed} km/h")
            elif line.startswith("MQ3:"):
                mq3_value = line.split(":")[1]
                print(f"MQ3 Value: {mq3_value}")

# Start sensor data reading in a separate thread
if ser:
    threading.Thread(target=read_sensor_data, daemon=True).start()

# Function to generate frames with object detection
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape

        # Prepare input for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Lists to store bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # Process detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Filter weak detections
                    # Get bounding box coordinates
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Add to lists
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Draw final bounding boxes
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to generate frames with depth estimation
def generate_depth_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and prepare input for DPT
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = feature_extractor(images=rgb_frame, return_tensors="pt").to("cpu")

        # Perform depth estimation
        with torch.no_grad():
            depth = depth_model(**inputs).predicted_depth

        # Resize depth map to match frame size
        depth_map = depth.squeeze().cpu().numpy()
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        # Normalize depth map for visualization
        depth_normalized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Encode the depth map as JPEG
        ret, buffer = cv2.imencode('.jpg', depth_normalized)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the video feed (YOLO)
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the depth estimation feed
@app.route('/depth_feed')
def depth_feed():
    return Response(generate_depth_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get sensor data with alert logic
@app.route('/get_sensor_data')
def get_sensor_data():
    global latitude, longitude, speed, mq3_value

    # Define thresholds for alerts
    speed_limit = 80  # Example: 80 km/h
    mq3_limit = 200   # Example: MQ3 value threshold

    alerts = []  # List to store multiple alert messages

    if speed != "N/A" and float(speed) > speed_limit:
        alerts.append(f"Speed is too high: {speed} km/h")

    if mq3_value != "N/A" and float(mq3_value) > mq3_limit:
        alerts.append(f"Alcohol detected: MQ3 value is {mq3_value}")

    alert_message = " | ".join(alerts)  # Combine messages with a separator

    return {
        'latitude': latitude,
        'longitude': longitude,
        'speed': speed,
        'mq3': mq3_value,
        'alert': bool(alerts),  # True if there are any alerts
        'alert_message': alert_message  # Full alert message
    }

# Route for the main page
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Smart Helmet</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f0f0f0;
                }
                h1 {
                    color: #333;
                }
                .feeds {
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                }
                .feed {
                    border: 2px solid #ccc;
                    padding: 10px;
                    background-color: #fff;
                }
                img {
                    max-width: 100%;
                    height: auto;
                }
                .data {
                    margin-top: 20px;
                    font-size: 1.2em;
                }
                .warning {
                    margin-top: 20px;
                    font-size: 1.5em;
                    color: red;
                    font-weight: bold;
                    display: none; /* Hidden by default */
                }
                .warning img {
                    width: 50px;
                    height: 50px;
                    vertical-align: middle;
                }
            </style>
        </head>
        <body>
            <h1>Smart Helmet Live Feed</h1>
            <div class="feeds">
                <div class="feed">
                    <h2>Object Detection</h2>
                    <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
                </div>
                <div class="feed">
                    <h2>Depth Estimation</h2>
                    <img src="{{ url_for('depth_feed') }}" alt="Depth Feed">
                </div>
            </div>
            <div class="data">
                <p>GPS: Lat=<span id="latitude">{{ latitude }}</span>, Lon=<span id="longitude">{{ longitude }}</span>, Speed=<span id="speed">{{ speed }}</span> km/h</p>
                <p>MQ3: <span id="mq3">{{ mq3_value }}</span></p>
            </div>
            <div class="warning" id="warning">
                <img src="/static/war1.webp" alt="Warning">
                <span id="warning-message"></span>
            </div>
            <script>
                // Update sensor data dynamically
                setInterval(async () => {
                    const response = await fetch('/get_sensor_data');
                    const data = await response.json();

                    document.getElementById('latitude').textContent = data.latitude;
                    document.getElementById('longitude').textContent = data.longitude;
                    document.getElementById('speed').textContent = data.speed;
                    document.getElementById('mq3').textContent = data.mq3;

                    const warningDiv = document.getElementById('warning');
                    const warningMessage = document.getElementById('warning-message');

                    if (data.alert) {
                        warningDiv.style.display = "block"; // Show the warning div
                        warningMessage.textContent = data.alert_message; // Update message
                    } else {
                        warningDiv.style.display = "none"; // Hide the warning div
                    }
                }, 1000);
            </script>
        </body>
        </html>
    ''', latitude=latitude, longitude=longitude, speed=speed, mq3_value=mq3_value)


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
