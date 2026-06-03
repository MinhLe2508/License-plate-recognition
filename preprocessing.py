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

def order_points(pts: np.ndarray) -> np.ndarray:
    rect    = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    diff    = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]     # (Top-Left)        x + y nhỏ nhất
    rect[1] = pts[np.argmax(s)]     # (Bottom-Right)    x + y lớn nhất
    rect[2] = pts[np.argmin(diff)]  # (Top-Right)       y - x nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # (Bottom-Left)     y - x lớn nhất
    return rect

def get_plate_corners(detection: dict, img_shape: tuple) -> np.ndarray:
    h, w = img_shape[:2]
    
    if detection["polygon"] is not None and len(detection["polygon"]) >= 4:
        polygon = detection["polygon"]
        
        hull = cv2.convexHull(polygon)
        
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx  = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            x, y, bw, bh = cv2.boundingRect(polygon)
            corners = np.array([
                [x,      y     ],
                [x + bw, y     ],
                [x + bw, y + bh],
                [x,      y + bh],
            ], dtype=np.float32)
            
    else:
        x1, y1, x2, y2 = detection["bbox"]
        corners = np.array([
            [x1, y1], [x2, y1],
            [x2, y2], [x1, y2],
        ], dtype=np.float32)
        
    return order_points(corners)   