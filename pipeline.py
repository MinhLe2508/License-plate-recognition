import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import re
import cv2
import sys
import glob
import argparse
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

from preprocessing import Preprocessor

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_province_corrector = None
def _get_province_corrector():
    global _province_corrector
    if _province_corrector is None:
        try:
            from src.utils.province_table import correct_province_code
            _province_corrector = correct_province_code
        except ImportError:
            _province_corrector = lambda x: x
    return _province_corrector


def init_ocr(use_gpu: bool | None = None) -> PaddleOCR:
    use_gpu = False
    print("  Sử dụng CPU mode (tránh lỗi oneDNN)")
    
    return PaddleOCR(
        use_textline_orientation    = True,
        lang                        = "en",
        device                      = "cpu",  
        enable_mkldnn               = False,   
    )


class LicensePlatePipeline(Preprocessor):

    PLATE_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."

    def __init__(
        self,
        model_path : str | None        = None,
        conf_thresh: float | None      = None,
        ocr        : PaddleOCR | None  = None,
        use_gpu    : bool | None       = None,
    ):
        if model_path is not None:
            self.MODEL_PATH = model_path
        if conf_thresh is not None:
            self.CONF_THRESH = conf_thresh
        self.model = YOLO(self.MODEL_PATH)
        self.ocr   = ocr if ocr is not None else init_ocr(use_gpu=use_gpu)

    def _run_ocr(self, img: np.ndarray) -> tuple[str, float]:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        if w < 300:
            scale = 300 / w
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)

        try:
            result = self.ocr.predict(img)   
        except NotImplementedError as e:
            if "ConvertPirAttribute2RuntimeAttribute" in str(e):
                print("Lỗi oneDNN: Model không tương thích hoặc phiên bản PaddleOCR cần cập nhật")
                print("   Hãy cập nhật: pip install --upgrade paddleocr paddlepaddle")
                return "", 0.0
            raise

        if not result or not result[0]:
            return "", 0.0

        rec_texts  = result[0].get("rec_texts",  [])
        rec_scores = result[0].get("rec_scores", [])

        if not rec_texts:
            return "", 0.0

        text_parts  = []
        confidences = []
        for text, conf in zip(rec_texts, rec_scores):
            text = text.upper().strip()
            if conf >= 0.5 and text:
                text_parts.append(text)
                confidences.append(conf)

        if not text_parts:
            return "", 0.0

        return " ".join(text_parts), float(np.mean(confidences))

    def _read_bsv_rows(self, plate_imgs: dict) -> str:
        candidates = [
            ("clahe",       plate_imgs.get("clahe")),
            ("original",    plate_imgs.get("original")),
            ("thresh_otsu", plate_imgs.get("thresh_otsu")),
        ]

        best_top   = ""
        best_bot   = ""
        best_score = 0.0

        for key, img in candidates:
            if img is None:
                continue

            h, w = img.shape[:2]
            if w < 600:
                scale = 600 / w
                img   = cv2.resize(img, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
                h, w  = img.shape[:2]

            mid     = int(h * 0.45)
            top_raw = img[:mid, :]
            bot_raw = img[mid:, :]

            if top_raw.shape[0] < 80:
                yscale  = 80 / top_raw.shape[0]
                top_raw = cv2.resize(top_raw, None, fx=1.0, fy=yscale,
                                     interpolation=cv2.INTER_CUBIC)
            if bot_raw.shape[0] < 80:
                yscale  = 80 / bot_raw.shape[0]
                bot_raw = cv2.resize(bot_raw, None, fx=1.0, fy=yscale,
                                     interpolation=cv2.INTER_CUBIC)

            top_img = self._add_ocr_padding(top_raw, pad=8)
            bot_img = self._add_ocr_padding(bot_raw, pad=8)

            top_text, top_conf = self._run_ocr(top_img)
            bot_text, bot_conf = self._run_ocr(bot_img)

            if bot_text and bot_text[0].isdigit():
                bot_img_wide         = self._add_ocr_padding(bot_raw, pad=20)
                bot_text2, bot_conf2 = self._run_ocr(bot_img_wide)
                if bot_text2 and bot_text2[0].isalpha():
                    bot_text, bot_conf = bot_text2, bot_conf2

            chars = len((top_text + bot_text).replace(" ", ""))
            score = top_conf + bot_conf + chars * 0.02

            if score > best_score:
                best_score = score
                best_top   = top_text
                best_bot   = bot_text

        if best_top or best_bot:
            combined = f"{best_top} {best_bot}".strip()
            result   = self._correct_plate_text(combined)

            if re.match(r'^\d{2}-\d', result):
                clahe = plate_imgs.get("clahe")
                full_img = clahe if clahe is not None else plate_imgs.get("original")
                if full_img is not None:
                    h_f, w_f = full_img.shape[:2]
                    if w_f < 800:
                        full_img = cv2.resize(full_img, None,
                                              fx=800/w_f, fy=800/w_f,
                                              interpolation=cv2.INTER_CUBIC)
                    full_padded    = self._add_ocr_padding(full_img, pad=15)
                    full_text, _   = self._run_ocr(full_padded)
                    full_corrected = self._correct_plate_text(full_text)
                    if full_corrected and not re.match(r'^\d{2}-\d', full_corrected):
                        return full_corrected

            return result

        return ""

    def read_plate_text(self, plate_imgs: dict, plate_class: str = "BSD") -> str:
        if plate_class == "BSV":
            result = self._read_bsv_rows(plate_imgs)
            if result:
                return result

        source_order = ["original", "clahe", "clahe_strong", "thresh_otsu", "thresh_adapt", "best"]

        best_text  = ""
        best_score = 0.0

        for source_name in source_order:
            img = plate_imgs.get(source_name)
            if img is None:
                continue

            text, conf = self._run_ocr(img)
            if not text:
                continue

            score = conf + len(text.replace(" ", "")) * 0.03
            if score > best_score:
                best_score = score
                best_text  = text

        return self._correct_plate_text(best_text)

    def _correct_plate_text(self, text: str) -> str:
        if not text:
            return text

        text = " ".join(text.strip().upper().split())

        PROVINCE_DIGIT_MAP = {
            'B': '3', 'S': '5', 'O': '0',
            'D': '0', 'I': '1', 'L': '1',
            'G': '6', 'Z': '2',
        }

        if "-" not in text:
            m = re.match(r'^([0-9BSODIGZL]{1,2})', text)
            if m:
                province_raw = m.group(1)
                province     = ''.join(PROVINCE_DIGIT_MAP.get(c, c) for c in province_raw)
                after_prov   = text[len(province_raw):]
                if ' ' in after_prov:
                    text = province + "-" + after_prov
                else:
                    m2 = re.match(r'^([A-Z])', after_prov)
                    if m2:
                        text = province + m2.group(1) + "-" + after_prov[1:]
                    else:
                        text = province + "-" + after_prov

        if "-" in text:
            prefix, _, suffix = text.partition("-")
        else:
            prefix = ""
            suffix = text

        if prefix:
            prefix_list = list(prefix)
            for i in range(min(2, len(prefix_list))):
                if not prefix_list[i].isdigit():
                    fixed = PROVINCE_DIGIT_MAP.get(prefix_list[i])
                    if fixed is not None:
                        prefix_list[i] = fixed
            prefix = "".join(prefix_list)

        if prefix:
            prov_digits = re.sub(r'\D', '', prefix[:3])
            series_tail = re.sub(r'^[0-9]+', '', prefix)
            if len(prov_digits) > 2:
                prov_digits = prov_digits[:2]
            elif len(prov_digits) == 1:
                prov_digits = prov_digits.zfill(2)
            prefix = prov_digits + series_tail

        if prefix:
            prov_part          = re.match(r'^(\d+)', prefix)
            series_part_prefix = re.sub(r'^\d+', '', prefix)
            if prov_part:
                corrected_prov = _get_province_corrector()(prov_part.group(1))
                prefix         = corrected_prov + series_part_prefix

        num_replacements = {'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8'}

        series_match = re.match(r'^([A-Z]+\d?)', suffix)
        if series_match:
            series_part = series_match.group(1)
            number_part = suffix[series_match.end():]
        else:
            series_part = ""
            number_part = suffix

        SERIES_DIGIT_MAP = {'T': '1', 'I': '1', 'L': '1', 'J': '1'}
        if len(series_part) >= 2 and not series_part[1].isdigit():
            corrected_digit = SERIES_DIGIT_MAP.get(series_part[1], series_part[1])
            series_part     = series_part[0] + corrected_digit + series_part[2:]

        for wrong, correct in num_replacements.items():
            number_part = number_part.replace(wrong, correct)

        clean_numbers = re.sub(r'\D', '', number_part)
        if len(clean_numbers) > 5:
            clean_numbers = clean_numbers[:5]

        suffix = (series_part + " " + clean_numbers).strip() if series_part else clean_numbers

        def insert_dot(m: re.Match) -> str:
            d = m.group(0)
            return d[:3] + "." + d[3:]
        suffix = re.sub(r'(?<!\d)(\d{5})(?!\d)', insert_dot, suffix)

        return (prefix + "-" + suffix).strip() if prefix else suffix.strip()

    def _preprocess_input(self, image: np.ndarray, min_width: int = 640) -> np.ndarray:
        h, w = image.shape[:2]
        if w < min_width:
            scale = min_width / w
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
            print(f"   Upscale: {w}x{h} → {image.shape[1]}x{image.shape[0]} px")
        return image

    def run(self, image_path: str, debug: bool = False) -> list[dict]:
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
            print(f"\nBiển số #{idx+1}: {det['class']} "
                  f"(conf={det['conf']:.2f}) "
                  f"tại ({x1},{y1})-({x2},{y2})")

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

    def _show_debug(self, original, corners, plate_imgs, idx, det) -> None:
        import matplotlib.pyplot as plt
        orig_rgb = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
        pts      = corners.astype(np.int32)
        cv2.polylines(orig_rgb, [pts], True, (0, 255, 0), 3)
        for p in pts:
            cv2.circle(orig_rgb, tuple(p), 8, (255, 0, 0), -1)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Biển số #{idx+1}: {det['class']} (conf={det['conf']:.2f})",
                     fontsize=14, fontweight="bold")
        axes[0,0].imshow(orig_rgb);                                                    axes[0,0].set_title("1. Ảnh gốc + Polygon")
        axes[0,1].imshow(cv2.cvtColor(plate_imgs["original"], cv2.COLOR_BGR2RGB));    axes[0,1].set_title("2. Perspective Transform")
        axes[0,2].imshow(plate_imgs["gray"],        cmap="gray");                     axes[0,2].set_title("3. Grayscale")
        axes[1,0].imshow(plate_imgs["clahe"],       cmap="gray");                     axes[1,0].set_title("4. CLAHE")
        axes[1,1].imshow(plate_imgs["thresh_otsu"], cmap="gray");                     axes[1,1].set_title("5. Otsu Threshold")
        axes[1,2].imshow(plate_imgs["thresh_adapt"],cmap="gray");                     axes[1,2].set_title("6. Adaptive Threshold")
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline nhận diện biển số xe Việt Nam")
    parser.add_argument("--image",  default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--model",  default=Preprocessor.MODEL_PATH)
    parser.add_argument("--conf",   type=float, default=Preprocessor.CONF_THRESH)
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--output", default="output_plates")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    print("Đang load YOLOv8...")
    print("Đang load PaddleOCR...")
    pipeline = LicensePlatePipeline(
        model_path  = args.model,
        conf_thresh = args.conf,
        use_gpu     = False if args.no_gpu else None,
    )
    print("Sẵn sàng!\n")

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

    os.makedirs(args.output, exist_ok=True)
    all_results: list[dict] = []

    for img_path in image_list:
        results   = pipeline.run(img_path, debug=args.debug)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for idx, res in enumerate(results):
            out_path = os.path.join(args.output,
                                    f"{base_name}_plate{idx+1}_{res['class']}.jpg")
            cv2.imwrite(out_path, res["plate_imgs"]["best"])
            all_results.append({
                "file"      : os.path.basename(img_path),
                "plate_text": res["plate_text"],
                "class"     : res["class"],
                "conf"      : res["conf"],
            })

    print(f"\n{'='*55}")
    print(f"HOÀN TẤT — {len(image_list)} ảnh → {len(all_results)} biển số")
    print(f"{'='*55}")
    for r in all_results:
        print(f"  {r['file']:30s} [{r['class']}]  {r['plate_text']}")
    print(f"\nẢnh đã lưu tại: {args.output}/")