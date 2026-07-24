import ctypes
import time
import cv2
import mss
import numpy as np
from ultralytics import YOLO

# Constanta Windows API pentru ascunderea ferestrei din capturi de ecran
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class ObjectDetector:

    def __init__(
        self,
        model_path="models/yolov8n.pt",
        conf_threshold=0.5,
        app_reference=None,
    ):
        # Încarcă direct fișierul .pt aflat la calea specificată
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.app_reference = app_reference  # Referința către instanța MainApp

    def process_frame(self, frame):
        # Citim valoarea din slider în timp real dacă referința aplicației există
        current_conf = self.conf_threshold
        if self.app_reference is not None:
            current_conf = self.app_reference.confidence_threshold

        results = self.model(frame, conf=current_conf)
        return results[0].plot()

    def _hide_main_app(self):
        """Ascunde fereastra principală."""
        if self.app_reference and hasattr(self.app_reference, 'withdraw'):
            self.app_reference.withdraw()

    def _show_main_app(self):
        """Afișează la loc fereastra principală."""
        if self.app_reference and hasattr(self.app_reference, 'deiconify'):
            self.app_reference.deiconify()

    def _stop_main_camera(self):
        """Oprește preview-ul camerei din aplicația principală și eliberează resursa video."""
        if self.app_reference and hasattr(self.app_reference, 'pause_camera_preview'):
            self.app_reference.pause_camera_preview = True
            time.sleep(0.3)  # Oferă timp worker-ului să execute cap.release()

    def _start_main_camera(self):
        """Repornește preview-ul camerei în aplicația principală."""
        if self.app_reference and hasattr(self.app_reference, 'pause_camera_preview'):
            self.app_reference.pause_camera_preview = False

    def _apply_ghost_affinity(self, window_title):
        """Setează fereastra să fie exclusă din capturile de ecran (Ghost Mode)."""
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
            if hwnd:
                ctypes.windll.user32.SetWindowDisplayAffinity(
                    hwnd, WDA_EXCLUDEFROMCAPTURE
                )
                print("[INFO] Fereastra a fost exclusă din captura de ecran!")
        except Exception as e:
            print(f"[WARN] Nu s-a putut seta Display Affinity: {e}")

    def _draw_popup(self, frame, text="PENTRU A INCHIDE APASATI TASTA 'Q'"):
        """Desenează o notificare de avertisment în partea de sus a cadrului curent."""
        h, w, _ = frame.shape
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0  # Font mai mare
        thickness = 2
        
        # Calculăm dimensiunile textului
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Blocul roșu va fi doar puțin mai mare decât textul
        box_w = text_size[0] + 40
        box_h = text_size[1] + 30
        
        # Poziționăm în partea de SUS (ex: la 30 pixeli de marginea superioară)
        x1 = (w - box_w) // 2
        y1 = 30
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Creăm o mască semi-transparentă
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 180), -1)  # Fundal roșu
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Contur alb

        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Adăugăm textul perfect centrat în interiorul blocului
        text_x = x1 + (box_w - text_size[0]) // 2
        text_y = y1 + (box_h + text_size[1]) // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def start_camera_detection(self, on_finish=None):
        time.sleep(0.2)
        
        # Ascundem interfața principală
        self._hide_main_app()
        # Oprim camera din aplicația principală (pentru a evita conflictele de resurse video)
        self._stop_main_camera()

        WINDOW_TITLE = (
            "Vision Detection - Live Camera (Pentru a inchide apasati 'q')"
        )

        # Dimensiuni de 3 ori mai mari decât standardul (640x480 -> 1920x1440)
        current_w, current_h = 1280, 720

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(WINDOW_TITLE, current_w, current_h)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        popup_end_time = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                annotated_frame = self.process_frame(frame)

                # Afișează popup-ul dacă utilizatorul a încercat să apese X
                if time.time() < popup_end_time:
                    annotated_frame = self._draw_popup(annotated_frame)

                cv2.imshow(WINDOW_TITLE, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # Preluăm dimensiunile actuale ale ferestrei doar dacă este vizibilă
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1:
                    try:
                        rect = cv2.getWindowImageRect(WINDOW_TITLE)
                        if rect[2] > 0 and rect[3] > 0:
                            current_w, current_h = rect[2], rect[3]
                    except:
                        pass
                else:
                    # Dacă fereastra nu mai este vizibilă (s-a apăsat X)
                    popup_end_time = time.time() + 1.0  # Pop-up activ 1 secundă
                    
                    # Restabilim fereastra păstrând ULTIMELE dimensiuni
                    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(
                        WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1
                    )
                    cv2.resizeWindow(WINDOW_TITLE, current_w, current_h)
        finally:
            cap.release()
            cv2.destroyWindow(WINDOW_TITLE)
            
            # Reafișăm aplicația principală și repornim camera UI-ului
            self._show_main_app()
            self._start_main_camera()
            
            if on_finish:
                on_finish()

    def start_screen_detection(self, on_finish=None):
        # Ascundem interfața principală
        self._hide_main_app()
        # Oprim explicit camera hardware (dacă era pornită de interfața principală în background)
        self._stop_main_camera()

        WINDOW_TITLE = (
            "Vision Detection - Ghost Window (Pentru a inchide apasati 'q')"
        )

        # Dimensiuni de 3 ori mai mari (480x270 -> 1440x810)
        current_w, current_h = 1440, 810

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(WINDOW_TITLE, current_w, current_h)

        self._apply_ghost_affinity(WINDOW_TITLE)

        popup_end_time = 0

        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Captează TOT ecranul

            try:
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    annotated_frame = self.process_frame(frame)

                    # Afișează popup-ul dacă utilizatorul a încercat să apese X
                    if time.time() < popup_end_time:
                        annotated_frame = self._draw_popup(annotated_frame)

                    cv2.imshow(WINDOW_TITLE, annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                    # Preluăm dimensiunile actuale ale ferestrei doar dacă este vizibilă
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1:
                        try:
                            rect = cv2.getWindowImageRect(WINDOW_TITLE)
                            if rect[2] > 0 and rect[3] > 0:
                                current_w, current_h = rect[2], rect[3]
                        except:
                            pass
                    else:
                        # Dacă fereastra nu mai este vizibilă (s-a apăsat X)
                        popup_end_time = (
                            time.time() + 1.0
                        )  # Pop-up activ 1 secundă
                        
                        # Restabilim fereastra păstrând ULTIMELE dimensiuni și re-aplicăm proprietatea Ghost
                        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(
                            WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1
                        )
                        cv2.resizeWindow(WINDOW_TITLE, current_w, current_h)
                        self._apply_ghost_affinity(WINDOW_TITLE)
            finally:
                cv2.destroyWindow(WINDOW_TITLE)
                
                # Reafișăm aplicația principală și repornim camera UI-ului
                self._show_main_app()
                self._start_main_camera()
                
                if on_finish:
                    on_finish()