# Smart Traffic Management System

## Overview

This project implements a smart traffic management system using Python and computer vision techniques. It analyzes vehicle density from video input and dynamically adjusts traffic signal timing to improve traffic flow efficiency.

## Features

* Vehicle detection and counting from video input
* Dynamic traffic signal timing based on vehicle density
* Real-time frame processing
* Web-based interface for monitoring (Flask)
* Lane-wise traffic analysis

## Tech Stack

* Python
* OpenCV
* Flask
* YOLO (Object Detection Model)

## Project Structure

* `app.py` – Main Flask application
* `video_detection.py` – Vehicle detection and processing logic
* `templates/` – HTML files for web interface
* `static/` – Static assets (CSS, JS, etc.)

## How It Works

The system processes video input from multiple lanes and detects vehicles using a pre-trained object detection model. Based on the number of vehicles detected in each lane, it determines which lane should receive priority and dynamically adjusts signal timing to optimize traffic flow.

## Installation and Setup

1. Install dependencies:

   ```
   pip install opencv-python flask ultralytics
   ```

2. Download the YOLO model:

   * Place `yolov8n.pt` in the project directory

3. Run the application:

   ```
   python app.py
   ```

4. Open in browser:

   ```
   http://127.0.0.1:5000
   ```

## Output

The application displays processed video frames with vehicle detection and provides dynamic signal decisions based on traffic density.

## Notes

* Video files are not included due to size limitations
* YOLO model weights (`yolov8n.pt`) must be downloaded separately

## Future Improvements

* Real-time camera integration
* Improved detection accuracy with advanced models
* Dashboard with analytics and reporting
* Integration with IoT-based traffic systems

## Author

Sam Dharan Rozario
