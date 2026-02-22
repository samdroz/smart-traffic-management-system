from ultralytics import YOLO
import cv2
import torch
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")
model.to(device)

VEHICLES = ["car", "truck", "bus", "motorcycle"]

# Keep last 20 frames for prediction smoothing
history_buffer = {
    "Lane A": deque(maxlen=20),
    "Lane B": deque(maxlen=20),
    "Lane C": deque(maxlen=20),
    "Lane D": deque(maxlen=20),
}

def generate_frames(video_path, lane_name, traffic_data):

    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (960, 540))
        results = model(frame, device=device, verbose=False)

        vehicle_count = 0
        incoming_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in VEHICLES:
                    vehicle_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_y = (y1 + y2) // 2

                    if center_y > 540 * 0.6:
                        incoming_count += 1

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # Add to moving history
        history_buffer[lane_name].append(vehicle_count)

        # Moving average prediction
        avg_density = sum(history_buffer[lane_name]) / len(history_buffer[lane_name])
        predicted_next = int(avg_density * 1.2)

        traffic_data[lane_name] = {
            "vehicles": vehicle_count,
            "incoming": incoming_count,
            "predicted": predicted_next
        }

        # Overlay
        cv2.rectangle(frame,(0,0),(500,120),(0,0,0),-1)
        cv2.putText(frame,f"Vehicles: {vehicle_count}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.putText(frame,f"Predicted: {predicted_next}",(20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
