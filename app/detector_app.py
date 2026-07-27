import os
import threading
import time
import urllib.request
import cv2
import customtkinter
import mss
from PIL import Image
import tkinter as tk
from launch_detector import ObjectDetector
from trainer_app import TrainWindow
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

class MainApp(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.title("Vision Detection")
        self.geometry("1280x720")
        self.resizable(True, True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        self.MODELS_DIR = os.path.join(parent_dir, "models")
        
        if not os.path.exists(self.MODELS_DIR):
            os.makedirs(self.MODELS_DIR)

        self.is_detecting = False
        self.pause_camera_preview = False
        self.is_running = True
        self.selected_mode = None
        
        
        self.selected_model = self.get_first_available_model()
        self.confidence_threshold = 0.50
        self.latest_screen_img = None
        self.latest_camera_img = None
        self.is_camera_available = False

        self.COLOR_DEFAULT_BORDER = "gray20"
        self.COLOR_SELECTED_BORDER = "#D3D3D3"

        self.PREVIEW_WIDTH = 530
        self.PREVIEW_HEIGHT = 298

         
        self.main_frame = customtkinter.CTkFrame(master=self, corner_radius=15)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.bottom_frame = customtkinter.CTkFrame(master=self.main_frame, fg_color="transparent")
        self.bottom_frame.pack(side="bottom", fill="x", padx=30, pady=(10, 20))

        self.slider_frame = customtkinter.CTkFrame(master=self.bottom_frame, fg_color="transparent")
        self.slider_frame.pack(side="top", fill="x", pady=(0, 50))

        self.slider_label = customtkinter.CTkLabel(
            master=self.slider_frame,
            text=f"Prag ignorare: {self.confidence_threshold:.2f}",
            font=("Roboto", 15, "bold"),
        )
        self.slider_label.pack(side="top", pady=(0, 8))

        self.confidence_slider = customtkinter.CTkSlider(
            master=self.slider_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            width=500,
            height=20,
            button_color="#5865F2",       
            button_hover_color="#4752C4", 
            command=self.on_slider_change,
        )
        self.confidence_slider.set(self.confidence_threshold)
        self.confidence_slider.pack(side="top")

        self.buttons_row_frame = customtkinter.CTkFrame(master=self.bottom_frame, fg_color="transparent")
        self.buttons_row_frame.pack(side="top", fill="x")

        has_model = self.selected_model is not None
        display_name = self.selected_model.replace(".pt", "").upper() if has_model else "INDISPONIBIL"
        
        self.model_btn = customtkinter.CTkButton(
            master=self.buttons_row_frame,
            text=display_name,
            font=("Roboto", 14, "bold"),
            height=45,
            width=190,
            fg_color="#5865F2" if has_model else "gray50",       
            hover_color="#4752C4" if has_model else "gray50",
            state="normal" if has_model else "disabled",
            command=self.show_dropup_menu,
        )
        self.model_btn.pack(side="left")

        self.btn_install_models = customtkinter.CTkButton(
            master=self.buttons_row_frame,
            text="Install Yolo models",
            font=("Roboto", 12, "bold"),
            height=45,
            width=140,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.start_download_models,
        )

        if not self.check_standard_models_exist():
            self.btn_install_models.pack(side="left", padx=(10, 0))

        self.btn_start = customtkinter.CTkButton(
            master=self.buttons_row_frame,
            text="Selectează o sursă",
            font=("Roboto", 16, "bold"),
            height=45,
            width=250,
            fg_color="#5865F2",       
            hover_color="#4752C4",
            state="disabled",
            command=self.execute_detection,
        )
        self.btn_start.pack(side="right")

        self.top_bar_frame = customtkinter.CTkFrame(master=self.main_frame, fg_color="transparent")
        self.top_bar_frame.pack(side="top", fill="x", padx=20, pady=(15, 5))

        self.title_label = customtkinter.CTkLabel(
            master=self.top_bar_frame,
            text="Alegeți modul de detecție",
            font=("Roboto", 32, "bold"),
        )
        self.title_label.pack(side="left", padx=10)

        self.theme_btn = customtkinter.CTkButton(
            master=self.top_bar_frame,
            text="🌙",
            font=("Segoe UI Emoji", 22),
            width=45,
            height=45,
            corner_radius=22,
            fg_color="#2b2b36",
            text_color="#ffd700",
            hover_color="#3f3f4e",
            command=self.toggle_theme,
        )
        self.theme_btn.pack(side="right", padx=10)

        self.create_btn = customtkinter.CTkButton(
            master=self.top_bar_frame,
            text="Antreneaza",
            font=("Roboto", 16, "bold"),
            width=120,
            height=45,
            corner_radius=22,
            fg_color="#2b2b36",
            text_color="#ffd700",
            hover_color="#3f3f4e",
            command=self.open_create,
        )
        self.create_btn.pack(side="right", padx=10)

        self.cards_frame = customtkinter.CTkFrame(master=self.main_frame, fg_color="transparent")
        self.cards_frame.pack(side="top", fill="both", expand=True, padx=20, pady=10)

        self.cards_frame.columnconfigure(0, weight=1)
        self.cards_frame.columnconfigure(1, weight=1)

        self.card_camera = customtkinter.CTkFrame(
            master=self.cards_frame,
            corner_radius=12,
            border_width=3,
            border_color=self.COLOR_DEFAULT_BORDER,
        )
        self.card_camera.grid(row=0, column=0, padx=15, pady=5, sticky="n")

        self.cam_title = customtkinter.CTkLabel(master=self.card_camera, text="Camera Web", font=("Roboto", 22, "bold"))
        self.cam_title.pack(pady=(12, 6))

        self.cam_preview_label = customtkinter.CTkLabel(
            master=self.card_camera,
            text="Se conectează la cameră...",
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            fg_color="black",
            corner_radius=8,
        )
        self.cam_preview_label.pack(pady=(0, 8), padx=8)

        self.card_screen = customtkinter.CTkFrame(
            master=self.cards_frame,
            corner_radius=12,
            border_width=3,
            border_color=self.COLOR_DEFAULT_BORDER,
        )
        self.card_screen.grid(row=0, column=1, padx=15, pady=5, sticky="n")

        self.screen_title = customtkinter.CTkLabel(master=self.card_screen, text="Ecran PC", font=("Roboto", 22, "bold"))
        self.screen_title.pack(pady=(12, 6))

        self.screen_preview_label = customtkinter.CTkLabel(
            master=self.card_screen,
            text="Se încarcă preview...",
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            fg_color="black",
            corner_radius=8,
        )
        self.screen_preview_label.pack(pady=(0, 8), padx=8)

        self.card_camera.bind("<Button-1>", lambda e: self.select_mode("camera"))
        self.cam_preview_label.bind("<Button-1>", lambda e: self.select_mode("camera"))
        self.cam_title.bind("<Button-1>", lambda e: self.select_mode("camera"))
        self.card_screen.bind("<Button-1>", lambda e: self.select_mode("screen"))
        self.screen_preview_label.bind("<Button-1>", lambda e: self.select_mode("screen"))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.screen_thread = threading.Thread(target=self._screen_capture_worker, daemon=True)
        self.screen_thread.start()

        self.camera_thread = threading.Thread(target=self._camera_capture_worker, daemon=True)
        self.camera_thread.start()
        self.render_preview_ui()

    def check_standard_models_exist(self):
        """Verifică dacă există cel puțin unul dintre modelele n, m sau l în folder."""
        standard_models = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"]
        if os.path.exists(self.MODELS_DIR):
            for model in standard_models:
                if os.path.exists(os.path.join(self.MODELS_DIR, model)):
                    return True
        return False

    def start_download_models(self):
        """Inițiază procesul de descărcare al modelelor în fundal."""
        self.btn_install_models.configure(text="Downloading models...", state="disabled")
        threading.Thread(target=self._download_models_worker, daemon=True).start()

    def _download_models_worker(self):
        """Descarcă modelele YOLO (Nano, Medium, Large) direct în folderul models."""
        models_to_download = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"]
        for model_file in models_to_download:
            dest_path = os.path.join(self.MODELS_DIR, model_file)
            if not os.path.exists(dest_path):
                url = f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_file}"
                try:
                    urllib.request.urlretrieve(url, dest_path)
                except Exception as e:
                    print(f"[WARN] Nu s-a putut descarca {model_file}: {e}")

        self.after(0, self._on_download_complete)

    def _on_download_complete(self):
        """Ascunde butonul de instalare și reactivează selecția modelelor."""
        self.btn_install_models.pack_forget()
        self.selected_model = self.get_first_available_model()
        if self.selected_model:
            display_name = self.selected_model.replace(".pt", "").upper()
            self.model_btn.configure(
                text=display_name,
                state="normal",
                fg_color="#5865F2",
                hover_color="#4752C4"
            )
        print("[INFO] Modelele YOLO au fost instalate cu succes!")

    def get_first_available_model(self):
        """Alege primul model .pt găsit. Dacă nu există, returnează None."""
        if os.path.exists(self.MODELS_DIR):
            files = [f for f in os.listdir(self.MODELS_DIR) if f.endswith(".pt")]
            if files:
                return files[0]
        return None

    def on_slider_change(self, value):
        self.confidence_threshold = value
        self.slider_label.configure(text=f"Prag ignorare (Confidence): {self.confidence_threshold:.2f}")

    def show_dropup_menu(self):
        dropup_menu = tk.Menu(self, tearoff=0, font=("Roboto", 11, "bold"))
        
        pt_files = []
        if os.path.exists(self.MODELS_DIR):
            pt_files = [f for f in os.listdir(self.MODELS_DIR) if f.endswith(".pt")]

        if not pt_files:
            return 

        for model_file in pt_files:
            def select(m=model_file):
                self.selected_model = m
                self.model_btn.configure(text=m.replace(".pt", "").upper())
            dropup_menu.add_command(label=f"  {model_file}", command=select)

        x = self.model_btn.winfo_rootx()
        y = self.model_btn.winfo_rooty()
        menu_height = len(pt_files) * 28 + 10
        dropup_menu.post(x, y - menu_height)

    def open_create(self):
        self.pause_camera_preview = True
        time.sleep(0.3) 
        self.withdraw()

        train_win = TrainWindow(self)
        train_win.protocol("WM_DELETE_WINDOW", lambda: self.reafiseaza_fereastra_principala(train_win))

    def reafiseaza_fereastra_principala(self, fereastra_secundara):
        fereastra_secundara.destroy()
        self.deiconify()
        self.pause_camera_preview = False

    def toggle_theme(self):
        current_mode = customtkinter.get_appearance_mode()
        if current_mode == "Dark":
            customtkinter.set_appearance_mode("Light")
            self.theme_btn.configure(text="☀️", fg_color="#e0e0e0", text_color="#ff8c00", hover_color="#d0d0d0")
            self.COLOR_DEFAULT_BORDER = "gray80"
            self.COLOR_SELECTED_BORDER = "gray30"
        else:
            customtkinter.set_appearance_mode("Dark")
            self.theme_btn.configure(text="🌙", fg_color="#2b2b36", text_color="#ffd700", hover_color="#3f3f4e")
            self.COLOR_DEFAULT_BORDER = "gray20"
            self.COLOR_SELECTED_BORDER = "#D3D3D3"
        if self.selected_mode:
            self.select_mode(self.selected_mode)

    def select_mode(self, mode):
        self.selected_mode = mode
        if mode == "camera":
            self.card_camera.configure(border_color=self.COLOR_SELECTED_BORDER)
            self.card_screen.configure(border_color=self.COLOR_DEFAULT_BORDER)
            if self.is_camera_available:
                self.btn_start.configure(state="normal", text="Start Detecție Camera")
            else:
                self.btn_start.configure(state="disabled", text="Cameră indisponibilă")

        elif mode == "screen":
            self.card_screen.configure(border_color=self.COLOR_SELECTED_BORDER)
            self.card_camera.configure(border_color=self.COLOR_DEFAULT_BORDER)
            self.btn_start.configure(state="normal", text="Start Detecție Ecran")

    def _screen_capture_worker(self):
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            while self.is_running:
                try:
                    sct_img = sct.grab(monitor)
                    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                    resized_img = img.resize((self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
                    self.latest_screen_img = customtkinter.CTkImage(
                        light_image=resized_img, dark_image=resized_img, size=(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
                    )
                    time.sleep(0.033)
                except Exception:
                    break

    def _camera_capture_worker(self):
        while self.is_running:
            if self.pause_camera_preview:
                time.sleep(0.2)
                continue

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                self.is_camera_available = False
                self.cam_preview_label.configure(text="Camera nu a fost găsită / deconectată")

                if self.selected_mode == "camera":
                    self.btn_start.configure(state="disabled", text="Cameră indisponibilă")
                
                cap.release()
                time.sleep(1.0) 
                continue

            self.is_camera_available = True

            while self.is_running and not self.pause_camera_preview:
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    resized_img = img.resize(
                        (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
                        Image.Resampling.LANCZOS,
                    )
                    self.latest_camera_img = customtkinter.CTkImage(
                        light_image=resized_img,
                        dark_image=resized_img,
                        size=(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT),
                    )
                else:
                    self.is_camera_available = False
                    break

                time.sleep(0.033)

            cap.release()
            time.sleep(0.2)

    def execute_detection(self):
        if self.is_detecting:
            print("[INFO] Detecția rulează deja!")
            return

        if not self.selected_model:
            print("[WARN] Niciun model selectat. Vă rugăm să instalați/descărcați modelele mai întâi!")
            return

        self.is_detecting = True


        if self.selected_mode == "camera":
            self.pause_camera_preview = True
            time.sleep(0.5)

        model_path = os.path.join(self.MODELS_DIR, self.selected_model)

        detector = ObjectDetector(
            model_path=model_path,
            conf_threshold=self.confidence_threshold,
            app_reference=self,
        )

        def on_detection_finish():
            self.is_detecting = False
            self.pause_camera_preview = False

        def start_thread():
            if self.selected_mode == "camera":
                detector.start_camera_detection(on_finish=on_detection_finish)
            elif self.selected_mode == "screen":
                detector.start_screen_detection(on_finish=on_detection_finish)

        threading.Thread(target=start_thread, daemon=True).start()

    def render_preview_ui(self):
        if self.latest_screen_img is not None:
            self.screen_preview_label.configure(image=self.latest_screen_img, text="")

        if self.latest_camera_img is not None and not self.pause_camera_preview:
            self.cam_preview_label.configure(image=self.latest_camera_img, text="")
        elif self.pause_camera_preview:
            self.cam_preview_label.configure(image="", text="Detecție activă...")

        if self.is_running:
            self.after(30, self.render_preview_ui)

    def on_close(self):
        self.is_running = False
        self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()