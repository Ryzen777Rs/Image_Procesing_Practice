import math
import os
import time
import cv2
import numpy as np
import customtkinter
import threading
import platform
from PIL import Image, ImageTk

class RealtimeCaptureWindow(customtkinter.CTkToplevel):
    def __init__(self, parent, base_dir=None, callback_refresh=None):
        super().__init__(parent)
        
        self.parent = parent
        self.parent.withdraw()

        self.title("Captură & Adnotare Poligoane în Timp Real")
        
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.win_w = int(screen_w * 0.75)
        self.win_h = int(screen_h * 0.75)
        self.geometry(f"{self.win_w}x{self.win_h}")
        self.resizable(False, False)

        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.callback_refresh = callback_refresh

        self.img_dir = os.path.join(self.base_dir, "images")
        self.lbl_dir = os.path.join(self.base_dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

        self.classes_file = os.path.join(self.base_dir, "classes.txt")
        self.classes = self.load_classes()

        self.class_name = ""
        self.cap = None
        self.is_running = False
        self.current_class_count = 0  
        
        # Stări desenare
        self.paused = False
        self.current_frame = None
        self.points = []
        self.original_points = [] 
        self.selected_point_idx = None
        self.is_dragging = False

        # Stări tracker
        self.tracker = None
        self.current_bbox = None
        self.initial_bbox = None

        self.stage = 0 

        # Cele 4 rotații cerute + unghiul tinta pt cub
        self.rotations = [
            ("Plasați obiectul în sus la 0 grade", 0),
            ("Plasați obiectul în jos la 180 grade", 180),
            ("Plasați obiectul la dreapta la 90 de grade", 90),
            ("Plasați obiectul la stânga la 270 de grade", 270)
        ]
        
        self.current_rot_idx = 0
        self.auto_capture_count = 0
        self.total_photos_target = 20
        self.captures_per_phase = []
        self.target_auto_captures_per_angle = 0 
        self.last_auto_save_time = 0
        
        # FIȘIERE SALVATE ÎN ETAPA CURENTĂ
        self.current_phase_files = []
        
        # Stari animatie rotire
        self.showing_turn_guide = False
        self.turn_guide_start_time = 0

        # Pentru miscarea meniului (drag & drop) global coord.
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._start_panel_x = 0
        self._start_panel_y = 0

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self._build_ui()
        self.start_camera()

    def get_tracker(self):
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                print("Avertisment: Tracker-ul CSRT nu este disponibil.")
                return None

    def load_classes(self):
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
        self.video_lbl = customtkinter.CTkLabel(self, text="")
        self.video_lbl.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        self.video_lbl.bind("<Button-1>", self.on_left_click)
        self.video_lbl.bind("<ButtonPress-3>", self.on_right_press)
        self.video_lbl.bind("<B3-Motion>", self.on_right_drag)
        self.video_lbl.bind("<ButtonRelease-3>", self.on_right_release)
        
        self.bind("<Return>", lambda e: self.handle_action_button())
        self.bind("<Escape>", lambda e: self.anuleaza_pauza())

        # Meniul plutitor
        self.overlay_w = 680
        self.overlay_panel = customtkinter.CTkFrame(
            self.video_lbl, corner_radius=0, fg_color="#1e1e24", border_width=2, border_color="#3a3a45",
            width=self.overlay_w
        )
        
        self.overlay_x = (self.win_w - self.overlay_w) // 2
        self.overlay_y = 20
        self.overlay_panel.place(x=self.overlay_x, y=self.overlay_y)
        
        self.drag_handle = customtkinter.CTkLabel(
            self.overlay_panel, text="◺", font=("Arial", 18), 
            text_color="#D3D3D3", cursor="fleur", width=20, height=20
        )
        self.drag_handle.place(relx=0.0, rely=1.0, anchor="sw", x=10, y=-10)
        self.drag_handle.bind("<ButtonPress-1>", self._start_drag)
        self.drag_handle.bind("<B1-Motion>", self._on_drag)

        self.frame_input = customtkinter.CTkFrame(self.overlay_panel, fg_color="transparent")
        self.frame_input.pack(fill="x", padx=15, pady=(12, 4)) 

        customtkinter.CTkLabel(self.frame_input, text="Clasă:", font=("Roboto", 13, "bold"), text_color="#ffffff").pack(side="left", padx=(0, 8))

        self.combo_class = customtkinter.CTkComboBox(
            self.frame_input, values=self.classes if self.classes else [""], 
            width=180, font=("Roboto", 13), command=self.update_class_count
        )
        self.combo_class.pack(side="left", padx=(0, 15))
        self.combo_class.bind("<KeyRelease>", self.update_class_count)
        
        customtkinter.CTkLabel(self.frame_input, text="Nr. Poze (Total):", font=("Roboto", 13, "bold"), text_color="#ffffff").pack(side="left", padx=(0, 5))
        self.entry_num_photos = customtkinter.CTkEntry(self.frame_input, placeholder_text="25", width=60, font=("Roboto", 13))
        self.entry_num_photos.pack(side="left", padx=(0, 15))

        # AICI AM MODIFICAT CULORILE PENTRU A FI CA PE DISCORD
        self.btn_confirm = customtkinter.CTkButton(
            self.frame_input, 
            text="Confirmă", 
            command=self.confirma_clasa, 
            font=("Roboto", 13, "bold"), 
            width=90,
            fg_color="#5865F2", 
            hover_color="#4752C4"
        )
        self.btn_confirm.pack(side="right", padx=(0, 0))

        self.frame_stats = customtkinter.CTkFrame(self.overlay_panel, fg_color="transparent")
        self.frame_stats.pack(fill="x", padx=15, pady=4)
        
        self.lbl_class_count = customtkinter.CTkLabel(self.frame_stats, text="", font=("Roboto", 12, "bold"), text_color="#5865F2")
        self.lbl_class_count.pack(side="left", padx=(0, 15))

        self.btn_restart_phase = customtkinter.CTkButton(
            self.frame_stats, text="Repetă Etapa (Pierdut)", fg_color="#ff8c00", hover_color="#e67e00", 
            font=("Roboto", 11, "bold"), height=24, command=self.restart_current_phase
        )

        self.btn_clear_points = customtkinter.CTkButton(self.frame_stats, text="Șterge Puncte", fg_color="#b53737", hover_color="#d94343", font=("Roboto", 11, "bold"), width=90, height=24, command=self.sterge_punctele)
        self.btn_clear_points.pack(side="right", padx=(0, 0))

        self.lbl_instructions = customtkinter.CTkLabel(self.overlay_panel, text="Alegeți sau scrieți o clasă nouă în listă, apoi Confirmă.", font=("Roboto", 13, "bold"), text_color="#e0e0e0", wraplength=650)
        self.lbl_instructions.pack(padx=15, pady=(4, 15))

        self.update_class_count()

    def update_class_count(self, *args):
        current_class = self.combo_class.get().strip().replace(" ", "_")
        
        if not current_class:
            self.lbl_class_count.configure(text="Introduceți o clasă...")
            self.current_class_count = 0
            return
            
        count = 0
        class_dir = os.path.join(self.img_dir, current_class)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
        self.current_class_count = count
        self.lbl_class_count.configure(text=f"Nr. poze existente: {self.current_class_count}")

    def update_class_count_fast(self):
        self.current_class_count += 1
        self.lbl_class_count.configure(text=f"Nr. poze existente: {self.current_class_count}")

    def _start_drag(self, event):
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        
        self._start_panel_x = self.overlay_panel.winfo_x()
        self._start_panel_y = self.overlay_panel.winfo_y()

    def _on_drag(self, event):
        delta_x = event.x_root - self._drag_start_x
        delta_y = event.y_root - self._drag_start_y
        
        new_x = self._start_panel_x + delta_x
        new_y = self._start_panel_y + delta_y
        self.overlay_panel.place(x=new_x, y=new_y)

    def confirma_clasa(self):
        selectata = self.combo_class.get().strip()
        
        if selectata and selectata != "":
            self.class_name = selectata.replace(" ", "_")
            if self.class_name not in self.classes:
                self.classes.append(self.class_name)
                self.combo_class.configure(values=self.classes)
                self.save_classes()

            try:
                num_input = self.entry_num_photos.get().strip()
                self.total_photos_target = int(num_input) if num_input else 20
            except ValueError:
                self.total_photos_target = 20

            base = self.total_photos_target // 4
            remainder = self.total_photos_target % 4
            self.captures_per_phase = [base + (1 if i < remainder else 0) for i in range(4)]
            self.target_auto_captures_per_angle = self.captures_per_phase[0]

            self.dir_img_class = os.path.join(self.img_dir, self.class_name)
            self.dir_lbl_class = os.path.join(self.lbl_dir, self.class_name)
            os.makedirs(self.dir_img_class, exist_ok=True)
            os.makedirs(self.dir_lbl_class, exist_ok=True)

            self.stage = 1
            self.combo_class.configure(state="disabled")
            self.entry_num_photos.configure(state="disabled")
            self.btn_confirm.configure(state="disabled")
            
            self.lbl_instructions.configure(
                text=f"Pasul 1: Fixați obiectul la 0 grade. Apăsați ENTER pentru a pune pe pauză și a marca conturul (poligon)."
            )
            self.update_class_count()

    def start_camera(self):
        self.lbl_instructions.configure(text="Se inițializează camera... Vă rugăm așteptați.", text_color="#ffcc00")
        threading.Thread(target=self._init_camera_bg, daemon=True).start()

    def _init_camera_bg(self):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
            
        self.after(0, self._camera_ready, cap)

    def _camera_ready(self, cap):
        self.cap = cap
        self.is_running = True
        self.lbl_instructions.configure(text="Alegeți sau scrieți o clasă nouă în listă, apoi Confirmă.", text_color="#e0e0e0")
        self.update_webcam()

    def draw_rotating_cube(self, frame, progress, target_angle):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 150
        
        prev_angle = self.rotations[max(0, self.current_rot_idx - 1)][1] if self.current_rot_idx > 0 else -90
        smooth_t = progress * progress * (3 - 2 * progress)
        current_z_angle = prev_angle + (target_angle - prev_angle) * smooth_t
        
        theta = math.radians(current_z_angle)
        phi = math.radians(20) 
        
        nodes = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ], dtype=float)

        rot_z = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rot_x = np.array([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])

        projected = []
        for node in nodes:
            rotated = np.dot(rot_z, node)
            rotated = np.dot(rot_x, rotated)
            x = int(rotated[0] * size + cx)
            y = int(rotated[1] * size + cy)
            projected.append((x, y))

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7) 
        ]

        front_face_edges = [(4,5), (5,6), (6,7), (7,4)]
        
        for i, edge in enumerate(edges):
            pt1, pt2 = projected[edge[0]], projected[edge[1]]
            color = (255, 255, 0) if edge in front_face_edges else (200, 200, 200)
            thickness = 4 if edge in front_face_edges else 2
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        face_center_x = sum([projected[i][0] for i in [4,5,6,7]]) // 4
        face_center_y = sum([projected[i][1] for i in [4,5,6,7]]) // 4
        cv2.circle(frame, (face_center_x, face_center_y), 10, (0, 0, 255), -1)

    def restart_current_phase(self):
        for img_path, lbl_path in self.current_phase_files:
            if os.path.exists(img_path): os.remove(img_path)
            if os.path.exists(lbl_path): os.remove(lbl_path)
        
        removed_count = len(self.current_phase_files)
        self.current_class_count = max(0, self.current_class_count - removed_count)
        self.lbl_class_count.configure(text=f"Nr. poze existente: {self.current_class_count}")
        
        self.current_phase_files.clear()
        self.auto_capture_count = 0
        
        self.stage = 1
        self.paused = False
        self.tracker = None
        self.points = []
        
        rot_text, _ = self.rotations[self.current_rot_idx]
        self.lbl_instructions.configure(
            text=f"Etapă resetată. {rot_text}. Apăsați ENTER pentru a pune pe pauză și a re-desena."
        )
        self.btn_restart_phase.pack_forget()

    def update_webcam(self):
        if not self.is_running or self.cap is None:
            return

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

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.win_w, self.win_h))
        self.current_frame = frame.copy()

        tracking_success = False
        if self.tracker is not None and self.stage == 3:
            tracking_success, bbox = self.tracker.update(frame)
            if tracking_success:
                self.current_bbox = tuple(map(int, bbox))

        if self.showing_turn_guide:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)

            rot_text, target_angle = self.rotations[self.current_rot_idx]
            
            elapsed = time.time() - self.turn_guide_start_time
            progress = min(1.0, elapsed / 2.5)
            
            cv2.putText(frame, f"INTOARCE OBIECTUL: {rot_text.upper()}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            self.draw_rotating_cube(frame, progress, target_angle)

            if elapsed > 2.5:
                self.showing_turn_guide = False
                self.stage = 1
                self.paused = False
                self.tracker = None
                self.points = []
                self.lbl_instructions.configure(
                    text=f"{rot_text}. Apăsați ENTER pentru a pune pe pauză și a re-desena poligonul."
                )

        elif self.stage == 3:
            if tracking_success and self.current_bbox and self.initial_bbox:
                if self.btn_restart_phase.winfo_ismapped():
                    self.btn_restart_phase.pack_forget()

                x, y, w, h = self.current_bbox
                x0, y0, w0, h0 = self.initial_bbox
                x, y = max(0, x), max(0, y) 

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                shifted_points = []
                for (px, py) in self.original_points:
                    new_x = int(x + (px - x0) * (w / w0) if w0 > 0 else px)
                    new_y = int(y + (py - y0) * (h / h0) if h0 > 0 else py)
                    shifted_points.append((new_x, new_y))

                self.points = shifted_points
                self.deseneaza_poligon_pe_cadru(frame)

                if time.time() - self.last_auto_save_time > 0.15:
                    self.salveaza_adnotare_poligon(frame, shifted_points)
                    self.last_auto_save_time = time.time()
                    self.auto_capture_count += 1
                    self.lbl_instructions.configure(
                        text=f"Auto-captură [{self.current_rot_idx+1}/{len(self.rotations)}]: {self.auto_capture_count}/{self.target_auto_captures_per_angle} poze salvate."
                    )

                    if self.auto_capture_count >= self.target_auto_captures_per_angle:
                        self.current_phase_files.clear()
                        self.current_rot_idx += 1
                        
                        if self.current_rot_idx < len(self.rotations):
                            self.target_auto_captures_per_angle = self.captures_per_phase[self.current_rot_idx]
                            self.pornese_ghid_rotire()
                        else:
                            self.finalizeaza_procesul()
            else:
                cv2.putText(frame, "Obiect pierdut! Apasati butonul din meniu: 'Repeta Etapa'.", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not self.btn_restart_phase.winfo_ismapped():
                    self.btn_restart_phase.pack(side="left", padx=(10, 0))

        self.afiseaza_pe_ecran(frame)
        self.after(10, self.update_webcam) 

    def deseneaza_poligon_pe_cadru(self, frame):
        n = len(self.points)
        # Culoarea #5865F2 convertită în format BGR pentru OpenCV
        culoare_violet = (242, 101, 88) 
        
        if n > 1:
            for i in range(n - 1):
                p1, p2 = self.points[i], self.points[i + 1]
                cv2.line(frame, p1, p2, culoare_violet, 2)
            if n > 2:
                cv2.line(frame, self.points[-1], self.points[0], culoare_violet, 2, cv2.LINE_AA)

        for px, py in self.points:
            cv2.circle(frame, (px, py), 5, culoare_violet, -1)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), 1)
    def afiseaza_pe_ecran(self, frame_bgr):
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        ctk_img = customtkinter.CTkImage(light_image=pil_img, dark_image=pil_img, size=(self.win_w, self.win_h))
        self.video_lbl.configure(image=ctk_img)

    def on_left_click(self, event):
        if self.paused:
            overlay_x1 = self.overlay_panel.winfo_x()
            overlay_y1 = self.overlay_panel.winfo_y()
            overlay_x2 = overlay_x1 + self.overlay_panel.winfo_width()
            overlay_y2 = overlay_y1 + self.overlay_panel.winfo_height()
            
            if (overlay_x1 < event.x < overlay_x2) and (overlay_y1 < event.y < overlay_y2):
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

    def handle_action_button(self):
        if self.stage == 0:
            self.confirma_clasa()
            return
            
        if self.stage == 1:
            if not self.paused:
                self.paused = True
                self.points = []
                self.lbl_instructions.configure(
                    text="Trasați conturul exact dând Click Stânga. Când e gata, apăsați ENTER pentru Auto-Tracking."
                )
            else:
                if len(self.points) >= 3:
                    self.salveaza_adnotare_poligon(self.current_frame, self.points)
                    
                    xs = [p[0] for p in self.points]
                    ys = [p[1] for p in self.points]
                    x_min, y_min = min(xs), min(ys)
                    box_w, box_h = max(xs) - x_min, max(ys) - y_min
                    
                    self.current_bbox = (x_min, y_min, box_w, box_h)
                    self.initial_bbox = self.current_bbox
                    self.original_points = self.points.copy()
                    
                    if self.tracker is None:
                        self.tracker = self.get_tracker()
                    
                    if self.tracker is not None:
                        self.tracker.init(self.current_frame, self.current_bbox)
                    
                    self.paused = False
                    self.stage = 3
                    self.auto_capture_count = 0
                    self.last_auto_save_time = time.time()
                else:
                    self.lbl_instructions.configure(text="⚠️ Plasați cel puțin 3 puncte (poligon) înainte de a apăsa ENTER!")

    def salveaza_adnotare_poligon(self, frame, points):
        timestamp = int(time.time() * 1000)
        frame_copy = frame.copy() 
        filename = f"{self.class_name}_{timestamp}"
        
        img_path = os.path.join(self.dir_img_class, f"{filename}.jpg")
        lbl_path = os.path.join(self.dir_lbl_class, f"{filename}.txt")
        
        self.current_phase_files.append((img_path, lbl_path))
        
        threading.Thread(target=self._salvare_in_fundal, args=(frame_copy, points, img_path, lbl_path), daemon=True).start()

    def _salvare_in_fundal(self, frame_copy, points, img_path, lbl_path):
        cv2.imwrite(img_path, frame_copy)

        class_id = self.classes.index(self.class_name) if self.class_name in self.classes else 0
        yolo_coords = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / float(self.win_w)))
            ny = max(0.0, min(1.0, y / float(self.win_h)))
            yolo_coords.append(f"{nx:.6f} {ny:.6f}")

        line = f"{class_id} " + " ".join(yolo_coords) + "\n"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line)

        self.after(0, self.update_class_count_fast)

    def pornese_ghid_rotire(self):
        self.stage = 2
        self.showing_turn_guide = True
        self.turn_guide_start_time = time.time()
        self.auto_capture_count = 0 
        rot_text, _ = self.rotations[self.current_rot_idx]
        self.lbl_instructions.configure(
            text=f"Pregătiți-vă: {rot_text}!"
        )

    def finalizeaza_procesul(self):
        self.stage = 4
        self.tracker = None
        self.lbl_instructions.configure(
            text=f"✅ Captură finalizată cu succes! Se închide automat în câteva secunde...",
            text_color="#00ff00"
        )
        self.after(2000, self.on_close)

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        if self.callback_refresh:
            self.callback_refresh()
        self.parent.deiconify()
        self.destroy()

if __name__ == "__main__":
    root = customtkinter.CTk()
    app = RealtimeCaptureWindow(parent=root)
    root.mainloop()