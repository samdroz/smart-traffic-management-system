from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import torch

app = Flask(__name__)

# -------------------------
# LOAD MODEL
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.to(device)

# -------------------------
# GLOBAL TRAFFIC DATA
# -------------------------

traffic_data = {
    "Lane A": 0,
    "Lane B": 0,
    "Lane C": 0,
    "Lane D": 0
}

signal_decision = {
    "lane": "Lane A",
    "green_time": 30,
    "mode": "INITIALIZING"
}

# -------------------------
# FRAME GENERATOR
# -------------------------

def generate_frames(path, lane_name):

    cap = cv2.VideoCapture(path)

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (640, 360))

        results = model(frame, device=device, verbose=False)

        vehicle_count = 0
        predicted_arrivals = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in ["car", "truck", "bus", "motorcycle"]:
                    vehicle_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_y = (y1 + y2) // 2

                    if center_y > 360 * 0.6:
                        predicted_arrivals += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2)

        # Update global lane count
        traffic_data[lane_name] = predicted_arrivals

        # -------------------------
        # AI SIGNAL DECISION
        # -------------------------
        max_lane = max(traffic_data, key=traffic_data.get)
        incoming = traffic_data[max_lane]

        green_time = 30 + incoming * 3

        signal_decision["lane"] = max_lane
        signal_decision["green_time"] = green_time
        signal_decision["mode"] = "OPTIMIZING"

        # Overlay live count
        cv2.putText(frame,
                    f"{lane_name} Vehicles: {vehicle_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -------------------------
# ROUTES
# -------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video1')
def video1():
    return Response(generate_frames("video1.mp4", "Lane A"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generate_frames("video2.mp4", "Lane B"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video3')
def video3():
    return Response(generate_frames("video3.mp4", "Lane C"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video4')
def video4():
    return Response(generate_frames("video4.mp4", "Lane D"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/signal')
def get_signal():
    return signal_decision

# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)
