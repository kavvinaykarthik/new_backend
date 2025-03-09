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
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = result.names[int(box.cls[0])]
                    distance = round(10 / box.conf[0].item(), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} ({distance}m)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode(".jpg", frame)
            frame_data = base64.b64encode(buffer).decode()
            
            try:
                await websocket.send_json({"frame": frame_data})
            except Exception as e:
                print(f"Error sending frame: {e}")
                break  # Exit loop if sending fails

            await asyncio.sleep(0.03)
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    
    finally:
        await websocket.close()
        print("WebSocket connection closed.")
