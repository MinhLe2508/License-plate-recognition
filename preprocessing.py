import cv2
import numpy as np
from ultralytics import YOLO

class Preprocessor:
   
    MODEL_PATH  = "models/best.pt"
    CLASS_NAMES = ["BSD", "BSV"]    # BSD=Biển số dài, BSV=Biển số vuông
    CONF_THRESH = 0.5
    
    def detect_license_plates(self, model: YOLO, image: np.ndarray) -> list[dict]:
        """
        Chạy YOLO detect trên ảnh, trả về danh sách các detection đã lọc.
        Args:
            model : YOLO model đã load
            image : ảnh BGR (np.ndarray)
        Returns:
            List[dict] mỗi phần tử gồm:
              - 'bbox'   : (x1, y1, x2, y2)
              - 'polygon': np.ndarray các điểm mask hoặc None
              - 'conf'   : confidence score
              - 'class'  : tên class ('BSD' hoặc 'BSV')
        """
        results = model(image, verbose=False)[0]
        detections = []
        for i, box in enumerate(results.boxes):
            conf = float(box.conf[0])
            if conf < self.CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id   = int(box.cls[0])
            cls_name = (
                self.CLASS_NAMES[cls_id]
                if cls_id < len(self.CLASS_NAMES)
                else str(cls_id)
            )
            polygon = None
            if results.masks is not None and i < len(results.masks):
                polygon = results.masks.xy[i].astype(np.int32)
            detections.append({
                "bbox"   : (x1, y1, x2, y2),
                "polygon": polygon,
                "conf"   : conf,
                "class"  : cls_name,
            })
        detections.sort(key=lambda d: d["conf"], reverse=True)
        return detections

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Sắp xếp 4 điểm theo thứ tự: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        Args:
            pts: array shape (4, 2)
        Returns:
            rect: array shape (4, 2) đã sắp xếp
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # Top-Left     : x + y nhỏ nhất
        rect[1] = pts[np.argmin(diff)]  # Top-Right    : y - x nhỏ nhất
        rect[2] = pts[np.argmax(s)]     # Bottom-Right : x + y lớn nhất
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left  : y - x lớn nhất
        return rect
    def get_plate_corners(self, detection: dict, img_shape: tuple) -> np.ndarray:
        """
        Trích xuất 4 góc biển số từ polygon (mask) hoặc bounding box.
        Args:
            detection : dict chứa 'polygon' và 'bbox'
            img_shape : shape của ảnh gốc (h, w, ...)
        Returns:
            corners: np.ndarray shape (4, 2) float32, đã sắp xếp TL→TR→BR→BL
        """
        if detection["polygon"] is not None and len(detection["polygon"]) >= 4:
            polygon = detection["polygon"]
            hull    = cv2.convexHull(polygon)
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
        return self.order_points(corners)
    
    def perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        plate_class: str,
    ) -> np.ndarray:
        """
        Warp perspective biển số về hình chữ nhật chuẩn.
        Tỉ lệ output:
          - BSD (biển dài): width / 4.727
          - BSV (biển vuông): width / 2.0
        Args:
            image      : ảnh BGR gốc
            corners    : 4 góc (TL, TR, BR, BL) shape (4, 2) float32
            plate_class: 'BSD' hoặc 'BSV'
        Returns:
            warped: ảnh biển số đã crop + warp
        """
        tl, tr, br, bl = corners
        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        out_w        = int(max(width_top, width_bottom))

        # ── FIX: áp dụng min width TRƯỚC khi tính out_h theo tỷ lệ ──
        # Nếu tính out_h từ out_w nhỏ rồi mới nới rộng out_w,
        # ảnh sẽ bị méo tỷ lệ (ví dụ BSV: 200×50 thay vì 200×100).
        out_w = max(out_w, 200)

        if plate_class == "BSD":
            out_h = int(out_w / 4.727)   # Biển dài: tỷ lệ ~4.7:1
        else:
            out_h = int(out_w / 2.0)     # Biển vuông: tỷ lệ 2:1
        out_h = max(out_h, 50)
        dst = np.array([
            [0,         0        ],
            [out_w - 1, 0        ],
            [out_w - 1, out_h - 1],
            [0,         out_h - 1],
        ], dtype=np.float32)
        M      = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(image, M, (out_w, out_h))
        return warped
    def enhance_plate_image(self, plate_img: np.ndarray) -> dict[str, np.ndarray]:
        """
        Áp dụng các kỹ thuật xử lý ảnh để làm nổi bật ký tự.
        Returns dict gồm nhiều phiên bản xử lý khác nhau:
          - 'original'    : ảnh gốc sau perspective transform
          - 'gray'        : ảnh xám
          - 'clahe'       : tăng độ tương phản cục bộ (CLAHE)
          - 'thresh_otsu' : nhị phân hóa tự động (Otsu)
          - 'thresh_adapt': nhị phân hóa thích nghi (tốt với ảnh mờ/không đều sáng)
          - 'clahe_strong': CLAHE mạnh + Otsu
          - 'best'        : phiên bản tốt nhất để đưa vào OCR (độ lệch chuẩn cao nhất)
        """
        results = {"original": plate_img.copy()}
        h, w = plate_img.shape[:2]
        if w < 300:
            scale     = 300 / w
            plate_img = cv2.resize(
                plate_img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        results["gray"] = gray
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            gamma = 2.5
            lut   = np.array([
                min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                for i in range(256)
            ], dtype=np.uint8)
            gray = cv2.LUT(gray, lut)

        kernel_sharpen = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0],
        ])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        clahe_img = clahe.apply(denoised)
        results["clahe"] = clahe_img
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        _, otsu    = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_clean = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  kernel)
        otsu_clean = cv2.morphologyEx(otsu_clean, cv2.MORPH_CLOSE, kernel)
        results["thresh_otsu"] = otsu_clean
        
        adapt       = cv2.adaptiveThreshold(
            clahe_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            11, 2,
        )
        adapt_clean = cv2.morphologyEx(adapt, cv2.MORPH_OPEN,  kernel)
        adapt_clean = cv2.morphologyEx(adapt_clean, cv2.MORPH_CLOSE, kernel)
        results["thresh_adapt"] = adapt_clean

        clahe_strong     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
        clahe_strong_img = clahe_strong.apply(denoised)
        _, otsu_strong   = cv2.threshold(
            clahe_strong_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        otsu_strong = cv2.morphologyEx(otsu_strong, cv2.MORPH_OPEN,  kernel)
        otsu_strong = cv2.morphologyEx(otsu_strong, cv2.MORPH_CLOSE, kernel)
        results["clahe_strong"] = otsu_strong

        candidates = {
            "thresh_otsu" : otsu_clean,
            "thresh_adapt": adapt_clean,
            "clahe"       : clahe_img,
            "clahe_strong": otsu_strong,
        }
        best_key        = max(candidates, key=lambda k: np.std(candidates[k]))
        results["best"] = self._add_ocr_padding(candidates[best_key])

        # Thêm padding cho các phiên bản ảnh grayscale để OCR tốt hơn
        results["clahe"]        = self._add_ocr_padding(clahe_img)
        results["thresh_otsu"]  = self._add_ocr_padding(otsu_clean)
        results["thresh_adapt"] = self._add_ocr_padding(adapt_clean)
        return results

    @staticmethod
    def _add_ocr_padding(img: np.ndarray, pad: int = 10) -> np.ndarray:
        """
        Thêm viền trắng xung quanh ảnh biển số.
        EasyOCR nhận dạng ký tự ở rìa ảnh tốt hơn khi có vùng trống xung quanh.
        Args:
            img: ảnh grayscale (2D) hoặc BGR (3D)
            pad: số pixel padding mỗi phía (mặc định 10px)
        Returns:
            Ảnh đã thêm padding màu trắng
        """
        if img.ndim == 2:
            # Grayscale — giá trị trắng = 255
            return cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=255,
            )
        else:
            # BGR — giá trị trắng = (255, 255, 255)
            return cv2.copyMakeBorder(
                img, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )