import ctypes
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# Constanta Windows API pentru ascunderea ferestrei din capturi de ecran
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class ObjectDetector:

    def __init__(self, model_path="UNIVERSAL", conf_threshold=0.5, app_reference=None):
        self.model_mapping = {
            "UNIVERSAL": "yolov8n.pt",
            "NANACHI": "yolov8s.pt",
            "TANKS": "yolov8m.pt",
        }
        actual_file = self.model_mapping.get(model_path, model_path)
        self.model = YOLO(actual_file)
        self.conf_threshold = conf_threshold
        self.app_reference = app_reference  # Referința către instanța MainApp

    def process_frame(self, frame):
        # Citim valoarea din slider în timp real dacă referința aplicației există
        current_conf = self.conf_threshold
        if self.app_reference is not None:
            current_conf = self.app_reference.confidence_threshold

        results = self.model(frame, conf=current_conf)
        return results[0].plot()

    def start_camera_detection(self, on_finish=None):
        import time
        # Lăsăm un scurt timp pentru ca aplicația principală să elibereze camera
        time.sleep(0.2)

        WINDOW_TITLE = "Vision Detection - Live Camera"

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(WINDOW_TITLE, 640, 480)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                annotated_frame = self.process_frame(frame)
                cv2.imshow(WINDOW_TITLE, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyWindow(WINDOW_TITLE)
            if on_finish:
                on_finish()

    def start_screen_detection(self, on_finish=None):
        WINDOW_TITLE = "Vision Detection - Ghost Window"

        # 1. Creăm fereastra OpenCV
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        # 2. Setăm fereastra să fie Always on Top (deasupra tuturor aplicațiilor)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)

        # 3. Mărim/micșorăm fereastra la dimensiunea dorită (ex: 480x270 px)
        cv2.resizeWindow(WINDOW_TITLE, 480, 270)

        # --- TRUCUL GHOST: Ascundem fereastra din captura de ecran Windows ---
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_TITLE)
            if hwnd:
                ctypes.windll.user32.SetWindowDisplayAffinity(
                    hwnd, WDA_EXCLUDEFROMCAPTURE
                )
                print("[INFO] Fereastra a fost exclusă din captura de ecran!")
        except Exception as e:
            print(f"[WARN] Nu s-a putut seta Display Affinity: {e}")

        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Captează TOT ecranul

            try:
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Procesăm cadrul direct - fereastra NU va mai apărea în cadru!
                    annotated_frame = self.process_frame(frame)

                    cv2.imshow(WINDOW_TITLE, annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if (
                        key == ord("q")
                        or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
                    ):
                        break
            finally:
                cv2.destroyWindow(WINDOW_TITLE)
                if on_finish:
                    on_finish()