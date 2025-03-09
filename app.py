from fastapi import FastAPI, WebSocket
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import asyncio

app = FastAPI()

# Load YOLO model
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            # Receive image from frontend
            data = await websocket.receive_json()
            frame_data = base64.b64decode(data["frame"])
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Error decoding image")
                continue
            
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
            frame_data = base64.b64encode(buffer).decode()
            
            # Send processed frame back to frontend
            await websocket.send_json({"frame": frame_data})
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
