import cv2
import os
import glob
import argparse
import numpy as np
from ultralytics import YOLO
import easyocr

from preprocessing import Preprocessor

def init_ocr(lang: list[str] | None = None) -> easyocr.Reader:
    """
    Khởi tạo và trả về một EasyOCR Reader.
    Args:
        lang: danh sách ngôn ngữ, mặc định ['en'] cho biển số VN.
              Chỉ dùng tiếng Anh — biển số VN chỉ có A-Z và 0-9,
              thêm 'vi' sẽ khiến model nhầm sang ký tự có dấu.
    Returns:
        easyocr.Reader instance sẵn sàng sử dụng
    """
    if lang is None:
        lang = ["en"]  # Chỉ dùng 'en' — biển số VN không có ký tự có dấu
    return easyocr.Reader(lang, gpu=False)


class LicensePlatePipeline(Preprocessor):
    """
    Pipeline nhận diện biển số xe đầy đủ (YOLO + Preprocessing + OCR).
    Kế thừa ``Preprocessor`` để sử dụng toàn bộ logic xử lý ảnh, đồng thời
    bổ sung:
      - Khởi tạo và quản lý YOLO model
      - Khởi tạo EasyOCR để đọc ký tự biển số
      - Phương thức ``  `` điều phối toàn bộ pipeline
      - Debug visualization
    Ví dụ sử dụng::
        pipeline = LicensePlatePipeline(model_path="models/best.pt")
        results  = pipeline.run("car.jpg", debug=True)
        for r in results:
            print(r["plate_text"])
            cv2.imwrite(f"out_{r['class']}.jpg", r["plate_imgs"]["best"])
    """
    # Ký tự hợp lệ trên biển số Việt Nam
    PLATE_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."

    def __init__(
        self,
        model_path: str | None         = None,
        conf_thresh: float | None      = None,
        ocr: easyocr.Reader | None     = None,
        ocr_lang: list[str] | None     = None,
    ):
        """
        Khởi tạo pipeline và load YOLO model và EasyOCR.
        Args:
            model_path  : đường dẫn file weights (.pt). Mặc định dùng
                          ``Preprocessor.MODEL_PATH``.
            conf_thresh : ngưỡng confidence. Mặc định dùng
                          ``Preprocessor.CONF_THRESH``.
            ocr         : easyocr.Reader đã khởi tạo sẵn. Nếu None sẽ
                          tự động khởi tạo với ``ocr_lang``.
            ocr_lang    : list ngôn ngữ OCR (mặc định ['en']).
        """
        if model_path is not None:
            self.MODEL_PATH = model_path
        if conf_thresh is not None:
            self.CONF_THRESH = conf_thresh
        self.model = YOLO(self.MODEL_PATH)
        self.ocr   = ocr if ocr is not None else init_ocr(lang=ocr_lang)
    
    def _run_ocr(self, img: np.ndarray) -> tuple[str, float]:
        """
        Chạy EasyOCR trên một ảnh, trả về (text, confidence).
        Dùng allowlist để giới hạn ký tự hợp lệ của biển số VN.
        Args:
            img: ảnh grayscale hoặc BGR
        Returns:
            (text, avg_confidence) — text là chuỗi nối các phần,
            avg_confidence là trung bình confidence của các token.
        """
        # EasyOCR hoạt động tốt nhất khi ảnh có chiều rộng ít nhất 300px
        h, w = img.shape[:2]
        if w < 300:
            scale = 300 / w
            img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

        result = self.ocr.readtext(
            img,
            detail=1,
            allowlist=self.PLATE_ALLOWLIST,  # Chỉ nhận ký tự biển số hợp lệ
            paragraph=False,                 # Không gộp dòng — quan trọng với BSV
            min_size=10,                     # Bỏ qua vùng text quá nhỏ (noise)
            # ── Tuning để đọc biển số tốt hơn ──────────────────────────
            text_threshold=0.6,   # Giảm xuống (default 0.7) → nhận ký tự mờ hơn
            low_text=0.3,         # Giảm (default 0.4) → phát hiện ký tự nhỏ ở rìa
            link_threshold=0.3,   # Giảm (default 0.4) → không bỏ sót ký tự rời
            width_ths=0.5,        # Giảm (default 0.5) → không gộp nhầm ký tự liền kề
            contrast_ths=0.1,
            adjust_contrast=0.5,
            slope_ths=0.2,        # Cho phép text hơi nghiêng (biển số đôi khi lệch nhẹ)
            ycenter_ths=0.5,      # Gộp token cùng dòng linh hoạt hơn
        )
        if not result:
            return "", 0.0
        text_parts  = [item[1].upper().strip() for item in result]
        confidences = [item[2] for item in result]
        text        = " ".join(p for p in text_parts if p)
        avg_conf    = float(np.mean(confidences)) if confidences else 0.0
        return text, avg_conf

    def _read_bsv_rows(self, plate_imgs: dict) -> str:
        """
        Đọc biển số vuông (BSV) bằng cách tách từng dòng riêng biệt.
        BSV có 2 dòng: dòng trên (mã tỉnh + series) và dòng dưới (số đăng ký).
        Việc đọc từng dòng riêng giúp EasyOCR không bị nhầm lẫn 2 dòng.
        Args:
            plate_imgs: dict các phiên bản ảnh từ enhance_plate_image()
        Returns:
            Chuỗi biển số đã nhận dạng, ví dụ: '92-N1 030.03'
        """
        # Thử nhiều phiên bản ảnh: clahe tốt nhất cho BSV, original làm backup,
        # thresh_otsu khi ảnh có độ tương phản cực cao
        candidates = [
            ("clahe",       plate_imgs.get("clahe")),
            ("original",    plate_imgs.get("original")),
            ("thresh_otsu", plate_imgs.get("thresh_otsu")),
        ]

        best_top    = ""
        best_bot    = ""
        best_score  = 0.0

        for key, img in candidates:
            if img is None:
                continue

            # Upscale mạnh hơn để mỗi dòng đủ chi tiết sau khi cắt đôi
            # Mục tiêu: chiều rộng ≥ 600px → mỗi dòng ~600×120px
            h, w = img.shape[:2]
            target_w = 600
            if w < target_w:
                scale = target_w / w
                img   = cv2.resize(
                    img, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                h, w = img.shape[:2]

            # Tách dòng: cắt 45%/55% thay vì 50/50
            # Dòng trên BSV thường ngắn hơn dòng dưới → lấy ít hơn để tránh lẫn
            mid     = int(h * 0.45)
            top_raw = img[:mid, :]
            bot_raw = img[mid:, :]

            # Đảm bảo chiều cao tối thiểu sau khi cắt (≥ 80px)
            min_row_h = 80
            if top_raw.shape[0] < min_row_h:
                yscale  = min_row_h / top_raw.shape[0]
                top_raw = cv2.resize(top_raw, None, fx=1.0, fy=yscale,
                                     interpolation=cv2.INTER_CUBIC)
            if bot_raw.shape[0] < min_row_h:
                yscale  = min_row_h / bot_raw.shape[0]
                bot_raw = cv2.resize(bot_raw, None, fx=1.0, fy=yscale,
                                     interpolation=cv2.INTER_CUBIC)

            top_img = self._add_ocr_padding(top_raw, pad=8)
            bot_img = self._add_ocr_padding(bot_raw, pad=8)

            top_text, top_conf = self._run_ocr(top_img)
            bot_text, bot_conf = self._run_ocr(bot_img)

            # Tính điểm: ưu tiên cả 2 dòng đọc được + confidence cao
            chars = len((top_text + bot_text).replace(" ", ""))
            score = top_conf + bot_conf + chars * 0.02

            if score > best_score:
                best_score = score
                best_top   = top_text
                best_bot   = bot_text

        if best_top or best_bot:
            combined = f"{best_top} {best_bot}".strip()
            return self._correct_plate_text(combined)

        return ""

    def read_plate_text(self, plate_imgs: dict, plate_class: str = "BSD") -> str:
        """
        Đọc ký tự biển số từ ảnh bằng EasyOCR với xử lý thông minh.
        - BSV (biển số vuông 2 dòng): đọc từng dòng riêng biệt.
        - BSD (biển số dài 1 dòng): đọc toàn bộ ảnh.
        Args:
            plate_imgs : dict các phiên bản ảnh từ enhance_plate_image()
            plate_class: 'BSV' hoặc 'BSD' (mặc định 'BSD')
        Returns:
            Chuỗi ký tự đã nhận dạng, ví dụ: '51F-123.45'
            Trả về chuỗi rỗng nếu không nhận dạng được.
        """
        # ── BSV: đọc từng dòng riêng ──
        if plate_class == "BSV":
            result = self._read_bsv_rows(plate_imgs)
            if result:
                return result
            # Fallback xuống logic chung nếu tách dòng thất bại

        # ── BSD (và fallback của BSV): đọc toàn bộ ảnh ──
        # Thứ tự thử cho BSD:
        # 1. original (BGR màu) — EasyOCR nhận dạng tốt nhất trên ảnh màu rõ nét
        # 2. clahe    (grayscale tăng tương phản) — tốt với ảnh tối / không đều sáng
        # 3. clahe_strong — cho ảnh rất tối (ban đêm)
        # 4. thresh_otsu / thresh_adapt — dự phòng khi ảnh khó
        source_order = ["original", "clahe", "clahe_strong", "thresh_otsu", "thresh_adapt", "best"]

        best_text       = ""
        best_confidence = 0.0

        for source_name in source_order:
            img = plate_imgs.get(source_name)
            if img is None:
                continue

            text, conf = self._run_ocr(img)

            if not text:
                continue

            # Ưu tiên kết quả có confidence cao hơn VÀ text dài hơn
            # Tăng trọng số length lên 0.03 — biển số cần đủ ký tự mới đúng
            score = conf + len(text.replace(" ", "")) * 0.03
            if score > best_confidence:
                best_confidence = score
                best_text       = text

        # Đính chính và chuẩn hóa biển số
        corrected = self._correct_plate_text(best_text)
        return corrected
    
    def _correct_plate_text(self, text: str) -> str:
        """
        Đính chính và chuẩn hóa ký tự biển số dựa vào pattern.
        Format biển số VN:
          - BSD (1 dòng): 'XXA-NNNNN' hoặc 'XXA-NNN.NN'  (ví dụ: 30H-123.45)
          - BSV (2 dòng): 'XXA-N1 NNN.NN'                 (ví dụ: 92-N1 030.03)
        Lưu ý: chỉ thay ký tự nhầm lẫn ở VÙNG SỐ (sau dấu "-"),
        không được thay ở phần chữ cái (mã tỉnh, series).
        Args:
            text: chuỗi ký tự từ OCR (có thể lỗi)
        Returns:
            Chuỗi đã đính chính
        """
        if not text:
            return text

        text = text.strip().upper()
        # Loại bỏ khoảng trắng thừa
        text = " ".join(text.split())

        import re

        # ── Bảng sửa ký tự nhầm trong mã tỉnh (vị trí phải là số) ──
        # Mã tỉnh VN luôn là 2 chữ số (01-99).
        PROVINCE_DIGIT_MAP = {
            'B': '3',  # B trông giống 3
            'S': '5',  # S trông giống 5
            'O': '0',  # O trông giống 0
            'D': '0',  # D trông giống 0
            'I': '1',  # I trông giống 1
            'L': '1',  # L trông giống 1
            'G': '6',  # G trông giống 6
            'Z': '2',  # Z trông giống 2
        }

        # ── Quy tắc 1: Đảm bảo có dấu "-" đúng vị trí ──
        # Phân biệt 2 loại biển:
        #   BSD (biển dài, 1 dòng): [province][series]-[numbers]  → không có space
        #     Ví dụ: "30F072.27" → "30F-072.27"
        #   BSV (biển vuông, 2 dòng): [province]-[series][area] [numbers] → có space
        #     Ví dụ: "92N1 030.03" → "92-N1 030.03"
        if "-" not in text:
            # Tìm mã tỉnh (1-2 ký tự đầu, có thể là chữ số hoặc chữ cái bị nhầm)
            m = re.match(r'^([0-9BSODIGZL]{1,2})', text)
            if m:
                province_raw = m.group(1)
                province     = ''.join(PROVINCE_DIGIT_MAP.get(c, c) for c in province_raw)
                after_prov   = text[len(province_raw):]  # phần còn lại sau mã tỉnh

                if ' ' in after_prov:
                    # BSV: còn lại có khoảng trắng → dash giữa province và series
                    text = province + "-" + after_prov
                else:
                    # BSD: không có khoảng trắng → dash sau series letter (1 chữ)
                    m2 = re.match(r'^([A-Z])', after_prov)
                    if m2:
                        text = province + m2.group(1) + "-" + after_prov[1:]
                    else:
                        text = province + "-" + after_prov  # fallback

        # ── Chia thành phần prefix (trước "-") và phần số (sau "-") ──
        if "-" in text:
            prefix, _, suffix = text.partition("-")
        else:
            prefix = ""
            suffix = text

        # ── Quy tắc 1b: Sửa ký tự bị nhầm trong mã tỉnh (prefix) ──
        # Áp dụng cho tối đa 2 ký tự đầu của prefix (phần mã tỉnh).
        # Phần series letter (ký tự sau index 1) không được thay đổi.
        if prefix:
            prefix_list = list(prefix)
            for i in range(min(2, len(prefix_list))):
                if not prefix_list[i].isdigit():
                    fixed = PROVINCE_DIGIT_MAP.get(prefix_list[i])
                    if fixed is not None:
                        prefix_list[i] = fixed
            prefix = "".join(prefix_list)

        # ── Quy tắc 2: Sửa nhầm lẫn ký tự CHỈ trong vùng số (suffix) ──
        # Ký tự thường bị nhầm trong vùng số:
        #   O → 0  (chữ O nhầm số không)
        #   I → 1  (chữ I nhầm số một)
        #   l → 1  (chữ l thường nhầm số một)
        #   S → 5  (chữ S nhầm số năm — ít gặp)
        #   B → 8  (chữ B nhầm số tám — ít gặp)
        num_replacements = {
            'O': '0',
            'I': '1',
            'L': '1',
            'S': '5',
            'B': '8',
        }
        # Áp dụng từng ký tự trong suffix, nhưng bỏ qua phần series chữ cái đầu
        # Ví dụ suffix = "N1 03003" → giữ "N", sửa phần còn lại
        # Tìm vị trí kết thúc phần series chữ (ví dụ "N1", "KD", "A")
        series_match = re.match(r'^([A-Z]+\d*)', suffix)
        if series_match:
            series_part = series_match.group(1)          # "N1", "KD", v.v.
            number_part = suffix[series_match.end():]    # " 03003", "-12345"
        else:
            series_part = ""
            number_part = suffix

        # ── Quy tắc 2b: Sửa series area number bị đọc nhầm thành chữ ──
        # Format biển VN: series_letter + series_area_digit (ví dụ "N1", "K2")
        # Vị trí thứ 2 của series (series_part[1]) PHẢI là chữ số 1-9.
        # EasyOCR thường nhầm "1" thành "T", "I", "L", "J" (trông giống nhau).
        # Áp dụng đính chính CHỈ khi series_part[1] là chữ cái (không phải số).
        SERIES_DIGIT_MAP = {
            'T': '1',  # T trông giống 1 trong font biển số hẹp
            'I': '1',  # I và 1 cực kỳ giống nhau
            'L': '1',  # L thường bị nhầm với 1
            'J': '1',  # J với 1 (cùng nét thẳng đứng)
        }
        if len(series_part) >= 2 and not series_part[1].isdigit():
            corrected_digit = SERIES_DIGIT_MAP.get(series_part[1])
            if corrected_digit is not None:
                series_part = series_part[0] + corrected_digit + series_part[2:]

        # Sửa ký tự nhầm CHỈ trong phần số
        for wrong, correct in num_replacements.items():
            number_part = number_part.replace(wrong, correct)

        suffix = series_part + number_part

        # ── Quy tắc 3: Thêm dấu "." nếu phần số cuối có đúng 5 chữ số liên tiếp ──
        # Ví dụ: "03003" → "030.03" | "12345" → "123.45"
        def insert_dot(m: re.Match) -> str:
            digits = m.group(0)
            return digits[:3] + "." + digits[3:]
        suffix = re.sub(r'(?<!\d)(\d{5})(?!\d)', insert_dot, suffix)

        # ── Ghép lại ──
        if prefix:
            text = prefix + "-" + suffix
        else:
            text = suffix

        return text.strip()

    def run(self, image_path: str, debug: bool = False) -> list[dict]:
        """
        Pipeline hoàn chỉnh: ảnh gốc → danh sách biển số đã xử lý.
        Args:
            image_path: đường dẫn ảnh đầu vào
            debug     : nếu True, hiển thị ảnh trung gian qua matplotlib
        Returns:
            List các dict, mỗi dict gồm:
              - 'bbox'       : (x1, y1, x2, y2)
              - 'class'      : 'BSD' hoặc 'BSV'
              - 'conf'       : confidence score
              - 'plate_imgs' : dict các phiên bản ảnh đã xử lý
              - 'plate_text' : chuỗi ký tự biển số do EasyOCR đọc được
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        image = self._preprocess_input(image)
        print(f"\n{'-' * 60}")
        print(f"Ảnh: {os.path.basename(image_path)}")
        print(f"Kích thước: {image.shape[1]}x{image.shape[0]} px")
        detections = self.detect_license_plates(self.model, image)
        print(f"Phát hiện: {len(detections)} biển số")
        if not detections:
            print("KHÔNG TÌM THẤY BIỂN SỐ NÀO!!!")
            return []
        output = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"\nBiển số #{idx + 1}: {det['class']} "
                f"(conf={det['conf']:.2f}) "
                f"tại ({x1},{y1})-({x2},{y2})"
            )
            corners    = self.get_plate_corners(det, image.shape)
            warped     = self.perspective_transform(image, corners, det["class"])
            plate_imgs = self.enhance_plate_image(warped)
            plate_text = self.read_plate_text(plate_imgs, det["class"])
            print(f"OCR: {plate_text if plate_text else '(không đọc được)'}")
            output.append({
                "bbox"      : det["bbox"],
                "class"     : det["class"],
                "conf"      : det["conf"],
                "plate_imgs": plate_imgs,
                "plate_text": plate_text,
            })
            if debug:
                self._show_debug(image, corners, plate_imgs, idx, det)
        return output
    
    def _preprocess_input(self, image: np.ndarray,
                      min_width: int = 640) -> np.ndarray:
        """
        Upscale ảnh đầu vào nếu quá nhỏ trước khi detect.
        Ảnh nhỏ → vùng biển số tiny → OCR đọc sai.
        min_width: chiều rộng tối thiểu (default 640px)
        """
        h, w = image.shape[:2]
        if w < min_width:
            scale  = min_width / w
            image  = cv2.resize(image, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
            print(f"   Upscale: {w}x{h} → {image.shape[1]}x{image.shape[0]} px")
        return image
    
    def _show_debug(
        self,
        original: np.ndarray,
        corners: np.ndarray,
        plate_imgs: dict,
        idx: int,
        det: dict,
    ) -> None:
        """Hiển thị ảnh trung gian để debug (yêu cầu matplotlib)."""
        import matplotlib.pyplot as plt
        orig_rgb = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
        pts      = corners.astype(np.int32)
        cv2.polylines(orig_rgb, [pts], True, (0, 255, 0), 3)
        for p in pts:
            cv2.circle(orig_rgb, tuple(p), 8, (255, 0, 0), -1)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            f"Biển số #{idx + 1}: {det['class']} (conf={det['conf']:.2f})",
            fontsize=14, fontweight="bold",
        )
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
    parser = argparse.ArgumentParser(
        description="Pipeline nhận diện biển số xe Việt Nam"
    )
    parser.add_argument("--image",  default=None,                    help="Đường dẫn 1 ảnh")
    parser.add_argument("--folder", default=None,                    help="Đường dẫn thư mục ảnh")
    parser.add_argument("--model",  default=Preprocessor.MODEL_PATH, help="Đường dẫn file weights YOLO")
    parser.add_argument("--conf",   type=float, default=Preprocessor.CONF_THRESH, help="Ngưỡng confidence")
    parser.add_argument("--debug",  action="store_true",             help="Hiển thị ảnh trung gian")
    parser.add_argument("--output", default="output_plates",         help="Thư mục lưu kết quả")
    args = parser.parse_args()

    # Load model & OCR (1 lần duy nhất)
    print("Đang load YOLOv8...")
    print("Đang load EasyOCR...")
    pipeline = LicensePlatePipeline(model_path=args.model, conf_thresh=args.conf)
    print("Sẵn sàng!\n")

    # Gom danh sách ảnh
    image_list: list[str] = []
    if args.image:
        image_list = [args.image]
    elif args.folder:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_list += glob.glob(os.path.join(args.folder, ext))
        image_list.sort()
        print(f"Tìm thấy {len(image_list)} ảnh trong {args.folder}\n")
    else:
        print("Cần truyền --image hoặc --folder")
        exit(1)

    # Xử lý
    os.makedirs(args.output, exist_ok=True)
    all_results: list[dict] = []

    for img_path in image_list:
        results   = pipeline.run(img_path, debug=args.debug)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for idx, res in enumerate(results):
            # Lưu ảnh đã xử lý
            out_path = os.path.join(
                args.output,
                f"{base_name}_plate{idx + 1}_{res['class']}.jpg",
            )
            cv2.imwrite(out_path, res["plate_imgs"]["best"])
            all_results.append({
                "file"      : os.path.basename(img_path),
                "plate_text": res["plate_text"],
                "class"     : res["class"],
                "conf"      : res["conf"],
            })

    # Tổng kết
    print(f"\n{'=' * 55}")
    print(f"HOÀN TẤT — {len(image_list)} ảnh → {len(all_results)} biển số")
    print(f"{'=' * 55}")
    for r in all_results:
        print(f"  {r['file']:30s} [{r['class']}]  {r['plate_text']}")
    print(f"\nẢnh đã lưu tại: {args.output}/")