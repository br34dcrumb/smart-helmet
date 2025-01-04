
# Smart Helmet with ESP32

## Description

This project is a smart helmet that uses an ESP32 microcontroller to detect accidents and send an alert to a predefined phone number. The helmet is equipped with an accelerometer to detect sudden impacts and a GPS module to send the location of the accident.

## Features

- [ ] Accident detection using an accelerometer
- [x] GPS location tracking
- [ ] SMS alert to a predefined phone number
- [x] Low power consumption
- [x] Easy to use and configure

## Setup
1. Clone the repository
```bash
git  clone  https://github.com/br34dcrumb/smart-helmet.git
```
2. Install the required libraries
```bash
pip  install  -r  req.txt
```
3. Install the YOLOv3 files

```bash
curl  https://pjreddie.com/media/files/yolov3.weights  -o  yolov3.weights
curl  https://raw.githubusercontent.com/pjreddie/darknet/refs/heads/master/cfg/yolov3.cfg  -o  yolov3.cfg
curl  https://raw.githubusercontent.com/pjreddie/darknet/refs/heads/master/data/coco.names  -o  coco.names
```
3. Run the WebApp (will only work with the ESP32 connected)
```bash
python  serv.py
```