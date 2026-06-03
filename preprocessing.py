import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

MODEL_PATH  = "models/best.pt"  # Đường dẫn đến best.pt
CLASS_NAMES = ["BSD", "BSV"]    # BSD=Biển số dài, BSV=Biển số vuông
CONF_THRESH = 0.5               # Ngưỡng confidence tối thiểu

def detect_license_plates(model: YOLO, image: np.ndarray) -> list[dict]:
    results = model(image, verbose=False)[0]
    detections = []
    
    for i, box in enumerate[results.boxes]:
        conf = float(box.cls[0])
        if conf < CONF_THRESH:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        
        polygon = None
        if results.masks is not None and i < len(results.masks):
            polygon = results.masks.xy[i].astype(np.int32)
            
        detections.append({
            "bbox"      : (x1, y1, x2, y2),
            "polygon"   : polygon,
            "conf"      : conf,
            "class"     : cls_name,
        })
        
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections

