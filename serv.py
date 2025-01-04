from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import serial
import threading

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# Handle different versions of OpenCV
if isinstance(output_layers_indices[0], (list, np.ndarray)):
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]
else:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize sensor data variables
latitude = "N/A"
longitude = "N/A"
speed = "N/A"
mq3_value = "N/A"

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

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get sensor data
@app.route('/get_sensor_data')
def get_sensor_data():
    return {
        'latitude': latitude,
        'longitude': longitude,
        'speed': speed,
        'mq3': mq3_value
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
                img {
                    max-width: 100%;
                    height: auto;
                    border: 2px solid #333;
                }
                .data {
                    margin-top: 20px;
                    font-size: 1.2em;
                }
            </style>
        </head>
        <body>
            <h1>Smart Helmet Live Feed</h1>
            <img src="{{ url_for('video_feed') }}" alt="Camera Feed">
            <div class="data">
                <p>GPS: Lat=<span id="latitude">{{ latitude }}</span>, Lon=<span id="longitude">{{ longitude }}</span>, Speed=<span id="speed">{{ speed }}</span> km/h</p>
                <p>MQ3: <span id="mq3">{{ mq3_value }}</span></p>
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
                }, 1000);
            </script>
        </body>
        </html>
    ''', latitude=latitude, longitude=longitude, speed=speed, mq3_value=mq3_value)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)