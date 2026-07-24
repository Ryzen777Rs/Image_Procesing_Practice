import math
import os
import time
import cv2
import numpy as np
import customtkinter
from PIL import Image, ImageTk

class RealtimeCaptureWindow(customtkinter.CTkToplevel):
    def __init__(self, parent, base_dir=None, callback_refresh=None):
        super().__init__(parent)
        
        # 1. Ascundem fereastra principală la pornire
        self.parent = parent
        self.parent.withdraw()

        self.title("Captură & Adnotare Poligoane în Timp Real")
        
        # 2. Dimensiune fereastră la 75% din ecran
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.win_w = int(screen_w * 0.75)
        self.win_h = int(screen_h * 0.75)
        self.geometry(f"{self.win_w}x{self.win_h}")
        self.resizable(False, False)

        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.callback_refresh = callback_refresh

        # Directoare de salvare
        self.img_dir = os.path.join(self.base_dir, "images")
        self.lbl_dir = os.path.join(self.base_dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

        self.classes_file = os.path.join(self.base_dir, "classes.txt")
        self.classes = self.load_classes()

        self.class_name = ""
        self.cap = None
        self.is_running = False
        
        # Stări de adnotare prin Poligon (din SnippingAnnotator)
        self.paused = False
        self.current_frame = None
        self.points = []
        self.selected_point_idx = None
        self.is_dragging = False

        # Stări flux de lucru:
        # 0 = Selectare / Introducere clasă
        # 1 = Captură manuală (Pauză + Desenare poligon pe ENTER)
        # 2 = Ghid rotire (Ecran gri)
        # 3 = Auto-captură cu detecție
        self.stage = 0 
        self.manual_photos_count = 0
        self.target_manual_photos = 5

        self.rotations = [
            ("În sus la 90°", "UP"),
            ("În jos la 90°", "DOWN"),
            ("La stânga 90°", "LEFT"),
            ("La dreapta 90°", "RIGHT"),
            ("Întoarce la 180° (Spate)", "180")
        ]
        self.current_rot_idx = 0
        self.auto_capture_count = 0
        self.target_auto_captures_per_angle = 5
        self.last_auto_save_time = 0
        self.showing_turn_guide = False
        self.turn_guide_start_time = 0

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self._build_ui()
        self.start_camera()

    def load_classes(self):
        """Încărcăm lista de clase din fișier sau de la părinte."""
        if hasattr(self.parent, "classes"):
            return [c for c in self.parent.classes if c.lower() not in ["nothing", "all"]]
        if os.path.exists(self.classes_file):
            with open(self.classes_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip() and line.strip().lower() not in ["nothing", "all"]]
        return []

    def save_classes(self):
        with open(self.classes_file, "w", encoding="utf-8") as f:
            for c in self.classes:
                f.write(f"{c}\n")

    def _build_ui(self):
        # 1. Ecranul Video Full Canvas/Label
        self.video_lbl = customtkinter.CTkLabel(self, text="")
        self.video_lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Binding-uri Mouse pentru Poligoane (Stil SnippingAnnotator)
        self.video_lbl.bind("<Button-1>", self.on_left_click)
        self.video_lbl.bind("<ButtonPress-3>", self.on_right_press)
        self.video_lbl.bind("<B3-Motion>", self.on_right_drag)
        self.video_lbl.bind("<ButtonRelease-3>", self.on_right_release)
        
        # Tastatură
        self.bind("<Return>", lambda e: self.handle_action_button())
        self.bind("<Escape>", lambda e: self.anuleaza_pauza())

        # 2. Panou Overlay Plutitor (Gri Semitransparent)
        self.overlay_panel = customtkinter.CTkFrame(
            self, 
            corner_radius=12,
            fg_color="#1e1e24",
            border_width=2,
            border_color="#3a3a45"
        )
        self.overlay_panel.place(relx=0.5, rely=0.02, anchor="n")

        # Rândul 1: Selectare / Adăugare Clasă
        self.frame_input = customtkinter.CTkFrame(self.overlay_panel, fg_color="transparent")
        self.frame_input.pack(fill="x", padx=15, pady=(12, 4))

        customtkinter.CTkLabel(
            self.frame_input, text="Clasă:", font=("Roboto", 13, "bold"), text_color="#ffffff"
        ).pack(side="left", padx=(0, 8))

        self.combo_class = customtkinter.CTkComboBox(
            self.frame_input,
            values=self.classes if self.classes else ["Nicio clasă"],
            width=180,
            font=("Roboto", 13)
        )
        self.combo_class.pack(side="left", padx=(0, 8))

        self.entry_new_class = customtkinter.CTkEntry(
            self.frame_input,
            placeholder_text="Sau clasă nouă...",
            width=160,
            font=("Roboto", 13)
        )
        self.entry_new_class.pack(side="left", padx=(0, 8))

        self.btn_confirm = customtkinter.CTkButton(
            self.frame_input,
            text="Confirmă",
            command=self.confirma_clasa,
            font=("Roboto", 13, "bold"),
            width=90
        )
        self.btn_confirm.pack(side="left")

        # Rândul 2: Statistici Dataset & Buton Curățare Puncte
        self.frame_stats = customtkinter.CTkFrame(self.overlay_panel, fg_color="transparent")
        self.frame_stats.pack(fill="x", padx=15, pady=4)
        
        self.lbl_info_clasa = customtkinter.CTkLabel(
            self.frame_stats, text="Clasă: Nespecificat", font=("Roboto", 12, "bold"), text_color="#4da6ff"
        )
        self.lbl_info_clasa.pack(side="left", padx=(0, 15))

        self.lbl_info_train = customtkinter.CTkLabel(
            self.frame_stats, text="Poze (Train): 0", font=("Roboto", 12), text_color="#c5c5d0"
        )
        self.lbl_info_train.pack(side="left", padx=10)

        self.lbl_info_total = customtkinter.CTkLabel(
            self.frame_stats, text="Total Salvate: 0", font=("Roboto", 12), text_color="#8e8e93"
        )
        self.lbl_info_total.pack(side="left", padx=10)

        self.btn_clear_points = customtkinter.CTkButton(
            self.frame_stats,
            text="Șterge Puncte",
            fg_color="#b53737",
            hover_color="#d94343",
            font=("Roboto", 11, "bold"),
            width=100,
            height=24,
            command=self.sterge_punctele
        )
        self.btn_clear_points.pack(side="right", padx=(10, 0))

        # Rândul 3: Instrucțiuni Dinamice
        self.lbl_instructions = customtkinter.CTkLabel(
            self.overlay_panel, 
            text="Alegeți sau introduceți denumirea clasei pentru a începe.", 
            font=("Roboto", 13, "bold"),
            text_color="#e0e0e0",
            wraplength=650
        )
        self.lbl_instructions.pack(padx=15, pady=(4, 12))

        self.actualizeaza_statistici()

    # =========================================================================
    # STATISTICI ȘI CLASE
    # =========================================================================
    def calculeaza_statistici(self):
        poze_train, total_poze = 0, 0
        if os.path.exists(self.img_dir):
            for root, _, files in os.walk(self.img_dir):
                folder_name = os.path.basename(root)
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        total_poze += 1
                        if folder_name.lower() not in ['all', 'images', 'unlabeled']:
                            poze_train += 1
        return poze_train, total_poze

    def actualizeaza_statistici(self):
        poze_train, total = self.calculeaza_statistici()
        nume = self.class_name if self.class_name else "Nespecificat"
        self.lbl_info_clasa.configure(text=f"Clasă: {nume}")
        self.lbl_info_train.configure(text=f"Poze (Train): {poze_train}")
        self.lbl_info_total.configure(text=f"Total Salvate: {total}")

    def confirma_clasa(self):
        noua_clasa = self.entry_new_class.get().strip()
        selectata = self.combo_class.get().strip()
        
        nume_final = noua_clasa if noua_clasa else selectata
        if nume_final and nume_final not in ["Nicio clasă", ""]:
            self.class_name = nume_final.replace(" ", "_")
            if self.class_name not in self.classes:
                self.classes.append(self.class_name)
                self.combo_class.configure(values=self.classes)
                self.save_classes()

            self.dir_img_class = os.path.join(self.img_dir, self.class_name)
            self.dir_lbl_class = os.path.join(self.lbl_dir, self.class_name)
            os.makedirs(self.dir_img_class, exist_ok=True)
            os.makedirs(self.dir_lbl_class, exist_ok=True)

            self.stage = 1
            self.combo_class.configure(state="disabled")
            self.entry_new_class.configure(state="disabled")
            self.btn_confirm.configure(state="disabled")
            
            self.lbl_instructions.configure(
                text="Pasul 1: Fixați obiectul. Apăsați ENTER pentru a pune pe pauză și a marca puncte (poligon) cu Click Stânga."
            )
            self.actualizeaza_statistici()

    # =========================================================================
    # FLUXUL CAMERA & DESENARE POLIGOANE
    # =========================================================================
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_webcam()

    def update_webcam(self):
        if not self.is_running or self.cap is None:
            return

        # Mod Pauză pentru Adnotare Poligon
        if self.paused and self.current_frame is not None:
            display_frame = self.current_frame.copy()
            self.deseneaza_poligon_pe_cadru(display_frame)
            self.afiseaza_pe_ecran(display_frame)
            self.after(30, self.update_webcam)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(20, self.update_webcam)
            return

        frame = cv2.flip(frame, 1) # Mirroring
        frame = cv2.resize(frame, (self.win_w, self.win_h))
        self.current_frame = frame.copy()

        # Faza 2: Ghid Rotire (Ecran Gri)
        if self.showing_turn_guide:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

            rot_text, _ = self.rotations[self.current_rot_idx]
            cv2.putText(frame, f"INTOARCE: {rot_text.upper()}", (self.win_w // 4, self.win_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

            if time.time() - self.turn_guide_start_time > 2.5:
                self.showing_turn_guide = False
                self.stage = 3
                self.auto_capture_count = 0

        # Faza 3: Auto-Detectie
        elif self.stage == 3:
            bbox = self.detect_object_bbox(frame)
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame, f"Auto-Detect: {self.class_name}", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if time.time() - self.last_auto_save_time > 0.6:
                    self.salveaza_adnotare_bbox_auto(frame, bbox)
                    self.last_auto_save_time = time.time()
                    self.auto_capture_count += 1
                    self.lbl_instructions.configure(
                        text=f"Auto-captură [{self.current_rot_idx+1}/{len(self.rotations)}]: {self.auto_capture_count}/{self.target_auto_captures_per_angle} poze."
                    )

                    if self.auto_capture_count >= self.target_auto_captures_per_angle:
                        self.current_rot_idx += 1
                        if self.current_rot_idx < len(self.rotations):
                            self.pornese_ghid_rotire()
                        else:
                            self.finalizeaza_procesul()

        self.afiseaza_pe_ecran(frame)
        self.after(15, self.update_webcam)

    def deseneaza_poligon_pe_cadru(self, frame):
        """Desenează punctele și liniile poligonului direct pe cadrul BGR (stil SnippingAnnotator)"""
        n = len(self.points)
        if n > 1:
            for i in range(n - 1):
                p1, p2 = self.points[i], self.points[i + 1]
                cv2.line(frame, p1, p2, (255, 191, 0), 2) # Linie albastră cyan
            if n > 2:
                # Linie întreruptă de închidere
                cv2.line(frame, self.points[-1], self.points[0], (255, 191, 0), 1, cv2.LINE_AA)

        for px, py in self.points:
            cv2.circle(frame, (px, py), 5, (255, 191, 0), -1)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)

    def afiseaza_pe_ecran(self, frame_bgr):
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        ctk_img = customtkinter.CTkImage(light_image=pil_img, dark_image=pil_img, size=(self.win_w, self.win_h))
        self.video_lbl.configure(image=ctk_img)

    # =========================================================================
    # INTERACȚIUNE MOUSE PENTRU PUNCTE POLIGON
    # =========================================================================
    def on_left_click(self, event):
        if self.paused:
            # Ignorăm click-ul dacă este efectuat în zona panoului plutitor de sus
            if event.y < 120 and (self.win_w * 0.2 < event.x < self.win_w * 0.8):
                return
            self.points.append((event.x, event.y))

    def on_right_press(self, event):
        if self.paused:
            idx = self.get_nearest_point(event.x, event.y)
            if idx is not None:
                self.selected_point_idx = idx
                self.is_dragging = False

    def on_right_drag(self, event):
        if self.paused and self.selected_point_idx is not None:
            self.is_dragging = True
            self.points[self.selected_point_idx] = (event.x, event.y)

    def on_right_release(self, event):
        if self.paused and self.selected_point_idx is not None:
            if not self.is_dragging:
                # Click dreapta simplu = ștergere punct
                self.points.pop(self.selected_point_idx)
            self.selected_point_idx = None

    def get_nearest_point(self, x, y, tolerance=15):
        for i, (px, py) in enumerate(self.points):
            if math.hypot(px - x, py - y) < tolerance:
                return i
        return None

    def sterge_punctele(self):
        self.points = []

    def anuleaza_pauza(self):
        if self.paused:
            self.paused = False
            self.points = []
            self.lbl_instructions.configure(text="Pauză anulată. Apăsați ENTER pentru a pune cadrul pe pauză.")

    # =========================================================================
    # GESTIONARE ENTER ȘI SALVARE (YOLO POLIGON / SEGMENTARE)
    # =========================================================================
    def handle_action_button(self):
        if self.stage == 0:
            self.confirma_clasa()
            return
            
        if self.stage == 1:
            if not self.paused:
                # Înghețăm cadrul
                self.paused = True
                self.points = []
                self.lbl_instructions.configure(
                    text="Trasați conturul obiectului dând Click Stânga pentru puncte. Apăsați ENTER pentru a salva."
                )
            else:
                # Salvăm adnotarea de tip poligon
                if len(self.points) >= 3:
                    self.salveaza_adnotare_poligon(self.current_frame, self.points)
                    self.manual_photos_count += 1
                    self.paused = False
                    self.points = []

                    if self.manual_photos_count >= self.target_manual_photos:
                        self.pornese_ghid_rotire()
                    else:
                        self.lbl_instructions.configure(
                            text=f"Poza {self.manual_photos_count}/5 salvată. Rotiți obiectul și apăsați ENTER."
                        )
                else:
                    self.lbl_instructions.configure(
                        text="⚠️ Plasați cel puțin 3 puncte (poligon) înainte de a apăsa ENTER!"
                    )

    def salveaza_adnotare_poligon(self, frame, points):
        """Salvează imaginea și fișierul .txt în format YOLO Segmentare (Poligon)"""
        timestamp = int(time.time() * 1000)
        filename = f"{self.class_name}_{timestamp}"

        img_path = os.path.join(self.dir_img_class, f"{filename}.jpg")
        lbl_path = os.path.join(self.dir_lbl_class, f"{filename}.txt")

        cv2.imwrite(img_path, frame)

        # Calculăm ID-ul clasei
        class_id = self.classes.index(self.class_name) if self.class_name in self.classes else 0

        # Normalizăm coordonatele x, y la 0..1
        yolo_coords = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / float(self.win_w)))
            ny = max(0.0, min(1.0, y / float(self.win_h)))
            yolo_coords.append(f"{nx:.6f} {ny:.6f}")

        line = f"{class_id} " + " ".join(yolo_coords) + "\n"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line)

        self.actualizeaza_statistici()

    def salveaza_adnotare_bbox_auto(self, frame, bbox):
        """Salvare automată pentru faza 3 (Format YOLO BBox Baza)"""
        timestamp = int(time.time() * 1000)
        filename = f"{self.class_name}_{timestamp}"

        img_path = os.path.join(self.dir_img_class, f"{filename}.jpg")
        lbl_path = os.path.join(self.dir_lbl_class, f"{filename}.txt")

        cv2.imwrite(img_path, frame)

        class_id = self.classes.index(self.class_name) if self.class_name in self.classes else 0
        x, y, w, h = bbox
        x_center = (x + w / 2.0) / self.win_w
        y_center = (y + h / 2.0) / self.win_h
        norm_w = w / self.win_w
        norm_h = h / self.win_h

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        self.actualizeaza_statistici()

    def detect_object_bbox(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1500:
                return cv2.boundingRect(c)
        
        bw, bh = int(self.win_w * 0.4), int(self.win_h * 0.4)
        x, y = (self.win_w - bw) // 2, (self.win_h - bh) // 2
        return (x, y, bw, bh)

    def pornese_ghid_rotire(self):
        self.stage = 2
        self.showing_turn_guide = True
        self.turn_guide_start_time = time.time()
        rot_text, _ = self.rotations[self.current_rot_idx]
        self.lbl_instructions.configure(
            text=f"Mod Automat. Întoarceți obiectul: {rot_text.upper()}!\nProgramul va salva pozele automat."
        )

    def finalizeaza_procesul(self):
        self.is_running = False
        self.lbl_instructions.configure(
            text=f"✅ Captură finalizată cu succes pentru clasa '{self.class_name}'!"
        )

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        # Redeschidem fereastra principală
        if self.callback_refresh:
            self.callback_refresh()
        self.parent.deiconify()
        self.destroy()