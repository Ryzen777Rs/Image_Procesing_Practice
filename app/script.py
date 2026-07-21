import cv2
import mss
import numpy as np
from ultralytics import YOLO

WINDOW_TITLE = "Vision Detection - Live Camera"
WINDOW_TITLE_SCREEN = "Vision Detection - Screen"


class ObjectDetector:

    def __init__(self, model_path, conf_threshold=0.5, app_reference=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.app_reference = app_reference

    def start_camera_detection(self, on_finish=None):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[ERROR] Nu se poate deschide camera pentru detecție.")
            if on_finish:
                on_finish()
            return

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                annotated_frame = results[0].plot()

                # Adăugare text instructiv în colțul stânga sus
                cv2.putText(
                    annotated_frame,
                    "Pentru a iesi apasati tasta q",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),  # Culoare galbenă în format BGR
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(WINDOW_TITLE, annotated_frame)

                # Verifică dacă fereastra a fost închisă manual (X) sau apăsat 'q'
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"[ERROR] Eroare în timpul detecției cu camera: {e}")
        finally:
            cap.release()
            try:
                cv2.destroyWindow(WINDOW_TITLE)
            except cv2.error:
                pass
            
            if on_finish:
                on_finish()

    def start_screen_detection(self, on_finish=None):
        cv2.namedWindow(WINDOW_TITLE_SCREEN, cv2.WINDOW_NORMAL)
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    results = self.model(frame, conf=self.conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()

                    # Opțional și pentru ecran
                    cv2.putText(
                        annotated_frame,
                        "Pentru a iesi apasati tasta q",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.imshow(WINDOW_TITLE_SCREEN, annotated_frame)

                    if cv2.getWindowProperty(WINDOW_TITLE_SCREEN, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"[ERROR] Eroare în timpul detecției pe ecran: {e}")
        finally:
            try:
                cv2.destroyWindow(WINDOW_TITLE_SCREEN)
            except cv2.error:
                pass
            
            if on_finish:
                on_finish()