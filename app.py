from fastapi import FastAPI, WebSocket
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import asyncio
import os
import requests

app = FastAPI()

# Ensure the YOLOv8 model is downloaded
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "yolov8n.pt")
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

# Function to download YOLO model if it does not exist
def download_yolo_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)  # Ensure the directory exists
        print("Downloading YOLO model...")
        response = requests.get(YOLO_MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("YOLO model downloaded successfully!")

# Load YOLO model
download_yolo_model()
model = YOLO(MODEL_PATH)

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            frame_data = base64.b64decode(data["frame"])

            # Convert to OpenCV format
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run YOLOv8 model
            results = model(frame)
            
            # Draw bounding boxes
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = result.names[int(box.cls[0])]
                    distance = round(10 / box.conf[0].item(), 2)  # Mock distance calc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({distance}m)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Encode frame to send back
            _, buffer = cv2.imencode(".jpg", frame)
            processed_frame_data = base64.b64encode(buffer).decode()
            
            # Send processed frame back
            await websocket.send_json({"frame": processed_frame_data})

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
