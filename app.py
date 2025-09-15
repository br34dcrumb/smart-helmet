from flask import Flask, Response, render_template, jsonify
from utils.video_feed import VideoFeed
import atexit
import serial
import threading
import time
import queue

app = Flask(__name__)
video_feed = VideoFeed()

# Shared data for sensor readings
sensor_data_lock = threading.Lock()
sensor_data = {
    'latitude': "N/A",
    'longitude': "N/A",
    'speed': "N/A",
    'mq3_value': "N/A",
    'last_updated': None
}

# Queue for serial write requests
serial_write_queue = queue.Queue(maxsize=10)

# Initialize serial port
try:
    ser = serial.Serial('COM7', 460800, timeout=0.1, write_timeout=0.1)
    print("[INIT] Connected to serial port")
except serial.SerialException as e:
    print(f"[ERROR] Error opening serial port: {e}")
    ser = None

def read_sensor_data():
    """Thread for reading sensor data from the serial port."""
    global sensor_data
    print("[THREAD] Sensor reading thread started")
    while True:
        if ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                print(f"[SERIAL] Raw data received: {line}")
                
                with sensor_data_lock:
                    if line.startswith("GPS:"):
                        gps_data = line.split(":")[1].split(",")
                        if len(gps_data) == 3:
                            sensor_data.update({
                                'latitude': gps_data[0],
                                'longitude': gps_data[1],
                                'speed': gps_data[2],
                                'last_updated': time.time()
                            })
                            print(f"[UPDATE] GPS data updated: {sensor_data}")
                    elif line.startswith("MQ3:"):
                        mq3_value = line.split(":")[1]
                        sensor_data.update({
                            'mq3_value': mq3_value,
                            'last_updated': time.time()
                        })
                        print(f"[UPDATE] MQ3 data updated: {sensor_data}")
                    else:
                        print(f"[WARNING] Unrecognized data format: {line}")
            except Exception as e:
                print(f"[ERROR] Error processing serial data: {e}")
        time.sleep(0.01)  # Reduced sleep time

def serial_write_thread():
    """Thread for writing data to the serial port."""
    print("[THREAD] Serial write thread started")
    while True:
        if ser:
            try:
                start_time = time.time()
                data_to_send = serial_write_queue.get()
                ser.write(data_to_send.encode('utf-8'))
                ser.flush()  # Flush the buffer
                end_time = time.time()
                print(f"[SERIAL] Sent data: {data_to_send.strip()} (took {end_time - start_time:.4f} seconds)")
            except Exception as e:
                print(f"[ERROR] Error sending data to serial port: {e}")
        else:
            print("[WARNING] Serial port is not connected. Skipping data send.")
        time.sleep(0.01)  # Reduced sleep time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo_feed')
def yolo_feed():
    return Response(video_feed.generate_yolo_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
    return Response(video_feed.generate_depth_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/proximity_status')
def proximity_status():
    # Get the proximity status from the video feed
    proximity_value = video_feed.get_proximity_status()
    
    # Send the proximity status to the serial write queue
    serial_write_queue.put(f"PROXIMITY:{proximity_value}\n")
    
    # Return the proximity status as a JSON response
    return proximity_value

@app.route('/sensor_data')
def get_sensor_data():
    with sensor_data_lock:
        current_data = sensor_data.copy()
    print(f"[ROUTE] Sending sensor data: {current_data}")
    return jsonify(current_data)

def cleanup():
    print("[CLEANUP] Stopping video feed and closing serial port...")
    video_feed.stop_capture()
    if ser:
        ser.close()
    print("[CLEANUP] Cleanup complete")

atexit.register(cleanup)

if __name__ == '__main__':
    print("[MAIN] Starting Flask application...")
    
    # Start the sensor reading thread
    if ser:
        sensor_thread = threading.Thread(target=read_sensor_data, daemon=True)
        sensor_thread.start()
        print("[INIT] Sensor reading thread started successfully")
    
    # Start the serial write thread
    serial_thread = threading.Thread(target=serial_write_thread, daemon=True)
    serial_thread.start()
    print("[INIT] Serial write thread started successfully")
    
    # Run the Flask app
    app.run(threaded=True, use_reloader=False)