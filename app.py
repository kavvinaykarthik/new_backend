from fastapi import FastAPI, WebSocket
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import asyncio
import os

app = FastAPI()

# Ensure the YOLOv8 model is downloaded
MODEL_PATH = "yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    from ultralytics.utils.downloads import download
    download(f"https://github.com/ultralytics/assets/releases/download/v8.0.0/{MODEL_PATH}", dir=".")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open video capture (Use 0 for webcam, or provide a valid video file path)
cap = cv2.VideoCapture(0)

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from camera.")
                break

            # Run YOLOv8 model
            results = model(frame)

            # Draw bounding boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = result.names[int(box.cls[0])]
                    confidence = round(box.conf[0].item(), 2)
                    distance = round(10 / confidence, 2)  # Mock distance calculation

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({distance}m)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode frame to base64
            _, buffer = cv2.imencode(".jpg", frame)
            frame_data = base64.b64encode(buffer).decode()

            # Send frame
            await websocket.send_json({"frame": frame_data})
            await asyncio.sleep(0.03)  # Reduce load

    except Exception as e:
        print(f"WebSocket error: {e}")

    finally:
        await websocket.close()

@app.on_event("shutdown")
def shutdown():
    cap.release()
    print("Camera released.")
