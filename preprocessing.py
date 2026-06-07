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
    
    for i, box in enumerate(results.boxes):
        conf = float(box.conf[0])
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
    rect[1] = pts[np.argmin(diff)]  # (Top-Right)       y - x nhỏ nhất
    rect[2] = pts[np.argmax(s)]     # (Bottom-Right)    x + y lớn nhất
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

def perspective_transform(image: np.ndarray, corners: np.ndarray, plate_class: str) -> np.ndarray:
    
    tl, tr, br, bl = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    out_w = int(max(width_top, width_bottom))
    
    if plate_class == "BSD":
        out_h = int(out_w / 4.727)
    else:
        out_h = int(out_w / 2.0)
        
    out_w = max(out_w, 200)
    out_h = max(out_h, 50)
    
    dst = np.array([
        [0        , 0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0        , out_h - 1],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped

def enhance_plate_image(plate_img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Áp dụng các kỹ thuật xử lý ảnh để làm nổi bật ký tự.
 
    Returns dict gồm nhiều phiên bản xử lý khác nhau:
      - 'original'   : ảnh gốc sau perspective transform
      - 'gray'       : ảnh xám
      - 'clahe'      : tăng độ tương phản cục bộ (CLAHE)
      - 'thresh_otsu': nhị phân hóa tự động (Otsu)
      - 'thresh_adapt: nhị phân hóa thích nghi (tốt với ảnh mờ/không đều sáng)
      - 'best'       : phiên bản tốt nhất để đưa vào OCR
    """
    results = {"original": plate_img.copy()}
    
    h, w = plate_img.shape[:2]
    if w < 300:
        scale = 300 / w
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results["gray"] = gray
    
    mean_brightness = np.mean(gray)
    if mean_brightness < 80:
        gamma = 2.5
        lut = np.array([
            min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
            for i in range(256)
        ], dtype=np.uint8)
        gray = cv2.LUT(gray, lut)

    kernel_sharpen = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    gray = cv2.filter2D(gray, -1, kernel_sharpen)
    
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe_img = clahe.apply(denoised)
    results["clahe"] = clahe_img
    
    _, otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results["thresh_otsu"] = otsu

    adapt = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    results["thresh_adapt"] = adapt
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    otsu_clean = cv2.morphologyEx(otsu_clean, cv2.MORPH_CLOSE, kernel)
    results["thresh_otsu"] = otsu_clean  

    adapt_clean = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel)
    adapt_clean = cv2.morphologyEx(adapt_clean, cv2.MORPH_CLOSE, kernel)
    results["thresh_adapt"] = adapt_clean

    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    clahe_strong_img = clahe_strong.apply(denoised)
    _, otsu_strong = cv2.threshold(clahe_strong_img, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_strong = cv2.morphologyEx(otsu_strong, cv2.MORPH_OPEN, kernel)
    otsu_strong = cv2.morphologyEx(otsu_strong, cv2.MORPH_CLOSE, kernel)
    results["clahe_strong"] = otsu_strong

    candidates = {
        "thresh_otsu" : otsu_clean,
        "thresh_adapt": adapt_clean,
        "clahe"       : clahe_img,
        "clahe_strong": otsu_strong,   
    }
    best_key = max(candidates, key=lambda k: np.std(candidates[k]))
    results["best"] = candidates[best_key]
    
    return results

def process_image(image_path: str, model: YOLO,
                  debug: bool=False) -> list[dict]:
    """
    Pipeline hoàn chỉnh: ảnh gốc → danh sách biển số đã xử lý.
 
    Args:
        image_path: đường dẫn ảnh đầu vào
        model     : YOLOv8 model
        debug     : nếu True, hiển thị ảnh trung gian
 
    Returns:
        List các dict, mỗi dict gồm:
          - 'bbox'       : (x1, y1, x2, y2)
          - 'class'      : 'BSD' hoặc 'BSV'
          - 'conf'       : confidence score
          - 'plate_imgs' : dict các phiên bản ảnh đã xử lý
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    
    print(f"\n{'-'*60}")
    print(f"Ảnh: {os.path.basename(image_path)}")
    print(f"Kích thước: {image.shape[1]}x{image.shape[0]} px")
    
    detections = detect_license_plates(model, image)
    print(f"Phát hiện: {len(detections)} biển số")
    
    
    if not detections:
        print("KHÔNG TÌM THẤY BIỂN SỐ NÀO!!!")
        return []
    
    output = []
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        print(f"\nBiển số #{idx+1}: {det['class']} "
              f"(conf={det['conf']:.2f}) "
              f"tại ({x1},{y1})-({x2},{y2})")
        
        corners = get_plate_corners(det, image.shape)
        warped  = perspective_transform(image, corners, det["class"])
        
        plate_imgs = enhance_plate_image(warped)
        
        output.append({
            "bbox"      : det["bbox"],
            "class"     : det["class"],
            "conf"      : det["conf"],
            "plate_imgs": plate_imgs,
        })
        
        if debug:
            _show_debug(image, corners, plate_imgs, idx, det)
    
    return output

def _show_debug(original: np.ndarray, corners: np.ndarray,
                plate_imgs: dict, idx: int, det: dict):
    """Hiển thị ảnh trung gian để debug."""
    import matplotlib.pyplot as plt
 
    orig_rgb = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
    pts = corners.astype(np.int32)
    cv2.polylines(orig_rgb, [pts], True, (0, 255, 0), 3)
    for p in pts:
        cv2.circle(orig_rgb, tuple(p), 8, (255, 0, 0), -1)
 
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Biển số #{idx+1}: {det['class']} "
                 f"(conf={det['conf']:.2f})",
                 fontsize=14, fontweight="bold")
 
    axes[0, 0].imshow(orig_rgb)
    axes[0, 0].set_title("1. Ảnh gốc + Polygon")
 
    axes[0, 1].imshow(cv2.cvtColor(plate_imgs["original"], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Sau Perspective Transform")
 
    axes[0, 2].imshow(plate_imgs["gray"], cmap="gray")
    axes[0, 2].set_title("3. Grayscale")
 
    axes[1, 0].imshow(plate_imgs["clahe"], cmap="gray")
    axes[1, 0].set_title("4. CLAHE (tăng tương phản)")
 
    axes[1, 1].imshow(plate_imgs["thresh_otsu"], cmap="gray")
    axes[1, 1].set_title("5. Otsu Threshold")
 
    axes[1, 2].imshow(plate_imgs["thresh_adapt"], cmap="gray")
    axes[1, 2].set_title("6. Adaptive Threshold")
 
    for ax in axes.flat:
        ax.axis("off")
 
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=None, help="Đường dẫn 1 ảnh")
    parser.add_argument("--folder", default=None, help="Đường dẫn thư mục chứa nhiều ảnh")
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--output", default="output_plates")
    args = parser.parse_args()

    model = YOLO(args.model)
    print("Load model thành công!")

    image_list = []

    if args.image:
        image_list = [args.image]

    elif args.folder:
        import glob
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_list += glob.glob(os.path.join(args.folder, ext))
        image_list.sort()
        print(f"Tìm thấy {len(image_list)} ảnh trong {args.folder}")

    else:
        print("Cần truyền --image hoặc --folder")
        exit(1)

    os.makedirs(args.output, exist_ok=True)
    total_plates = 0

    for img_path in image_list:
        results = process_image(img_path, model, debug=args.debug)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for idx, res in enumerate(results):
            out_path = os.path.join(
                args.output,
                f"{base_name}_plate{idx+1}_{res['class']}.jpg"
            )
            cv2.imwrite(out_path, res["plate_imgs"]["best"])
            total_plates += 1

    print(f"\nHoàn tất! Xử lý {len(image_list)} ảnh → {total_plates} biển số")
    print(f"   Kết quả lưu tại: {args.output}/")