import tkinter as tk
import customtkinter as ctk
import mss
import keyboard
import os
from PIL import Image, ImageTk, ImageEnhance
import math
import time
import threading

class SnippingAnnotator:
    def __init__(self, parent_app=None, output_dir=".", hotkey="ctrl+shift+s"):
        self.parent_app = parent_app  
        self.output_dir = output_dir
        self.hotkey = hotkey
        
        self.img_dir = os.path.join(self.output_dir, "images")
        self.lbl_dir = os.path.join(self.output_dir, "labels")
        
        # Asigurăm existența folderului 'all' și a directoarelor de bază
        self.all_img_dir = os.path.join(self.img_dir, "all")
        os.makedirs(self.all_img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)
        
        self.classes_file = os.path.join(self.output_dir, "classes.txt")
        self.classes = self.load_classes()

        self.root = None
        self.points = []  
        self.selected_point_idx = None
        self.is_dragging = False

        self.current_image_path = None
        self.current_class_folder = None
        
        # Variabilă pentru a ține minte ultima clasă folosită
        self.last_used_class = "nothing"
        
        # Flag pentru a urmări dacă s-a schimbat clasa în fereastra curentă
        self.class_has_changed = False

    def load_classes(self):
        if os.path.exists(self.classes_file):
            with open(self.classes_file, "r", encoding="utf-8") as f:
                cls_list = [line.strip() for line in f.readlines() if line.strip()]
                return [c for c in cls_list if c.lower() not in ["nothing", "all"]]
        return []

    def save_classes(self):
        with open(self.classes_file, "w", encoding="utf-8") as f:
            for c in self.classes:
                if c.lower() not in ["nothing", "all"]:
                    f.write(f"{c}\n")

    def start_background_listener(self):
        print(f"[INFO] Modul Snipping activat. Apasă {self.hotkey.upper()} pentru a captura ecranul.")
        keyboard.add_hotkey(self.hotkey, self._declanseaza_din_thread_sigur)

    def _declanseaza_din_thread_sigur(self):
        if self.parent_app:
            self.parent_app.after(0, self._porneste_procedura_captura)
        elif self.root:
            self.root.after(0, self._porneste_procedura_captura)

    def _porneste_procedura_captura(self):
        if self.parent_app and hasattr(self.parent_app, "is_running"):
            self.parent_app.is_running = False 
        if self.parent_app and hasattr(self.parent_app, "master") and hasattr(self.parent_app.master, "is_running"):
            self.parent_app.master.is_running = False

        if self.parent_app:
            self.parent_app.withdraw()

        threading.Thread(target=self._captureaza_ecran_background, daemon=True).start()

    def _captureaza_ecran_background(self):
        time.sleep(0.25) 
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  
                sct_img = sct.grab(monitor)
                original_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            if self.parent_app:
                self.parent_app.after(0, lambda: self._deschide_interfata_cu_imagine(original_img, None, None))
            else:
                self._deschide_interfata_cu_imagine(original_img, None, None)
        except Exception as e:
            print(f"[Eroare Captură]: {e}")
            if self.parent_app:
                self.parent_app.after(0, self._reporneste_fluxul_principal)

    def _deschide_interfata_cu_imagine(self, original_img, image_path, default_class):
        self.original_img = original_img
        self.current_image_path = image_path
        self.current_class_folder = default_class
        
        enhancer = ImageEnhance.Brightness(original_img)
        dimmed_img = enhancer.enhance(0.4)
        
        self.create_fullscreen_window(dimmed_img)
        if image_path:
            self.carca_adnotare_existenta(image_path, default_class)

    def open_annotation_ui(self, image_path=None, default_class=None):
        if self.parent_app:
            self.parent_app.withdraw()

        try:
            if image_path and os.path.exists(image_path):
                original_img = Image.open(image_path).convert("RGB")
                self._deschide_interfata_cu_imagine(original_img, image_path, default_class)
        except Exception as e:
            print(f"[Eroare Deschidere UI Adnotare]: {e}")
            self._reporneste_fluxul_principal()

    def carca_adnotare_existenta(self, image_path, class_name):
        if not class_name or class_name.lower() in ["all", "nothing"]:
            return
        nume_fara_ext, _ = os.path.splitext(os.path.basename(image_path))
        cale_txt = os.path.join(self.lbl_dir, class_name, f"{nume_fara_ext}.txt")
        
        if os.path.exists(cale_txt):
            try:
                with open(cale_txt, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                    parts = line.split()
                    if len(parts) > 1:
                        coords = parts[1:]
                        self.points = []
                        for i in range(0, len(coords), 2):
                            if i + 1 < len(coords):
                                x = float(coords[i]) * self.width
                                y = float(coords[i+1]) * self.height
                                self.points.append((x, y))
                        self.redraw()
                        if class_name in self.classes:
                            self.class_var.set(class_name)
                            self.last_used_class = class_name
            except Exception as e:
                print(f"[Eroare citire etichetă]: {e}")

    def create_fullscreen_window(self, bg_image):
        if self.root:
            try:
                self.root.destroy()
            except:
                pass
            self.root = None

        self.width, self.height = bg_image.size
        self.root = tk.Toplevel()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True) 
        self.root.config(cursor="crosshair")

        self.points = []
        self.class_has_changed = False  # Resetăm flag-ul la fiecare sesiune nouă
        self.tk_bg_image = ImageTk.PhotoImage(bg_image)

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.tk_bg_image, anchor="nw")

        ctk.set_appearance_mode("dark")
        self.top_panel = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#1e1e24", border_width=1, border_color="#3a3a45")
        self.top_panel.place(relx=0.5, y=30, anchor="n")

        ctk.CTkLabel(self.top_panel, text="Clasă:", font=("Roboto", 14, "bold"), text_color="#ffffff").pack(side="left", padx=(20, 10), pady=15)
        
        combo_values = ["nothing"] + [c for c in self.classes if c.lower() not in ["nothing", "all"]]
        
        # Determinăm clasa inițială
        if self.current_class_folder and self.current_class_folder.lower() not in ["all", "nothing"] and self.current_class_folder in combo_values:
            clasa_initiala = self.current_class_folder
        elif self.last_used_class in combo_values:
            clasa_initiala = self.last_used_class
        else:
            clasa_initiala = "nothing"

        self.class_var = ctk.StringVar(value=clasa_initiala)

        def pe_schimbare_clasa(alegere):
            self.class_has_changed = True  # Utilizatorul a modificat/interacționat cu clasa
            self.last_used_class = alegere
            self.canvas.focus_set() # Redăm focusul canvas-ului pentru ca ESC/ENTER să funcționeze direct

        self.combo = ctk.CTkComboBox(self.top_panel, variable=self.class_var, values=combo_values, 
                                     command=pe_schimbare_clasa, font=("Roboto", 14), width=150)
        self.combo.pack(side="left", padx=10)

        # BUTONUL ȘTERGE TOT: Șterge DOAR punctele (nodurile) desenate
        self.btn_sterge_tot = ctk.CTkButton(self.top_panel, text="Șterge Tot", font=("Roboto", 13, "bold"), 
                                            fg_color="#b53737", hover_color="#d94343", width=100, 
                                            command=self.sterge_doar_punctele)
        self.btn_sterge_tot.pack(side="left", padx=10)

        instructiuni = "ENTER: Salvează  |  ESC: Anulează  |  CLICK STÂNGA: Punct  |  CLICK DREAPTA: Șterge/Mută"
        ctk.CTkLabel(self.top_panel, text=instructiuni, font=("Roboto", 12), text_color="#8a8a93").pack(side="left", padx=20)

        self.canvas.bind("<Button-1>", self.on_left_click)          
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)    
        self.canvas.bind("<B3-Motion>", self.on_right_drag)         
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)
        
        # Keybindings globale pentru ESC și ENTER (bind_all pe root)
        def _pe_esc(e=None):
            self.close()
            return "break"

        def _pe_enter(e=None):
            self.save_annotation()
            return "break"

        # Bind pe toate widget-urile principale
        for w in [self.root, self.canvas, self.top_panel, self.combo]:
            w.bind("<Escape>", _pe_esc)
            w.bind("<Return>", _pe_enter)

        # Bind_all pe root pentru a captura evenimentele indiferent de focus
        self.root.bind_all("<Escape>", _pe_esc)
        self.root.bind_all("<Return>", _pe_enter)

        # Entry-ul intern al combo box-ului
        if hasattr(self.combo, "_entry") and self.combo._entry:
            self.combo._entry.bind("<Escape>", _pe_esc)
            self.combo._entry.bind("<Return>", _pe_enter)

        # După ce dropdown-ul se închide, redăm focusul canvas-ului
        def _reda_focus_canvas(*args):
            self.canvas.focus_set()

        # Legăm de evenimentul de închidere a dropdown-ului (aproximativ)
        self.combo.bind("<FocusOut>", _reda_focus_canvas)

        self.root.focus_force()
        self.canvas.focus_set()
        try:
            self.root.grab_set()
        except:
            pass

    def on_left_click(self, event):
        self.canvas.focus_set()
        if event.y < 90 and 300 < event.x < (self.width - 300): return 
        self.points.append((event.x, event.y))
        self.redraw()

    def on_right_press(self, event):
        self.canvas.focus_set()
        idx = self.get_nearest_point(event.x, event.y)
        if idx is not None:
            self.selected_point_idx = idx
            self.is_dragging = False

    def on_right_drag(self, event):
        if self.selected_point_idx is not None:
            self.is_dragging = True
            self.points[self.selected_point_idx] = (event.x, event.y)
            self.redraw()

    def on_right_release(self, event):
        if self.selected_point_idx is not None:
            if not self.is_dragging:
                self.points.pop(self.selected_point_idx)
                self.redraw()
            self.selected_point_idx = None

    def get_nearest_point(self, x, y, tolerance=15):
        for i, (px, py) in enumerate(self.points):
            if math.hypot(px - x, py - y) < tolerance:
                return i
        return None

    def redraw(self):
        self.canvas.delete("annotation")
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                p1, p2 = self.points[i], self.points[i+1]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#00bfff", width=2, tags="annotation")
            if len(self.points) > 2:
                p1, p2 = self.points[-1], self.points[0]
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#00bfff", width=2, dash=(5, 5), tags="annotation")

        r = 4
        for x, y in self.points:
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="#00bfff", outline="white", width=1.5, tags="annotation")

    def sterge_doar_punctele(self):
        """Șterge DOAR punctele (nodurile) desenate pe ecran, fără a închide fereastra."""
        self.points = []
        self.redraw()

    def _curata_eticheta_si_muta_in_all(self):
        if self.current_image_path:
            base_filename, _ = os.path.splitext(os.path.basename(self.current_image_path))
            
            # 1. Șterge fișierul .txt dacă există
            if self.current_class_folder and self.current_class_folder.lower() not in ["all", "nothing"]:
                cale_txt = os.path.join(self.lbl_dir, self.current_class_folder, f"{base_filename}.txt")
                if os.path.exists(cale_txt):
                    try: os.remove(cale_txt)
                    except: pass

            # 2. Asigură-te că imaginea este salvată/mutată în folderul 'all'
            target_all_path = os.path.join(self.all_img_dir, base_filename + ".jpg")
            try:
                self.original_img.save(target_all_path, quality=95)
            except Exception as e:
                print(f"[Eroare salvare în all]: {e}")

            # 3. Ștergem vechea copie din folderul clasei specifice
            if self.current_class_folder and self.current_class_folder.lower() not in ["all", "nothing"]:
                cale_clasa = os.path.join(self.img_dir, self.current_class_folder, base_filename + ".jpg")
                if os.path.exists(cale_clasa) and os.path.abspath(cale_clasa) != os.path.abspath(target_all_path):
                    try: os.remove(cale_clasa)
                    except: pass
        else:
            # Este o captură NOUĂ fără calea setată încă -> se salvează direct în 'all'
            timestamp = int(time.time() * 1000)
            base_filename = f"screenshot_{timestamp}"
            target_all_path = os.path.join(self.all_img_dir, base_filename + ".jpg")
            try:
                self.original_img.save(target_all_path, quality=95)
            except Exception as e:
                print(f"[Eroare salvare screenshot nou în all]: {e}")

    def save_annotation(self, event=None):
        has_annotation = len(self.points) >= 3
        
        # PERMISIUNE ENTER: Se continuă DOAR dacă s-a făcut o adnotare (>=3 puncte) SAU s-a schimbat clasa din dropdown
        class_name = self.class_var.get().strip()

        # Dacă nu sunt puncte și nu s-a schimbat clasa, închide fereastra (comportament de anulare)
        if not has_annotation and not self.class_has_changed:
            self.close()
            return

        self.last_used_class = class_name 

        # Dacă este selectată clasa "nothing" / "all" sau nu există suficiente puncte (< 3)
        if class_name.lower() in ["nothing", "all"] or len(self.points) < 3:
            self._curata_eticheta_si_muta_in_all()
            self.close()
            return

        if not class_name: 
            self.close()
            return

        if class_name not in self.classes:
            self.classes.append(class_name)
            combo_values = ["nothing"] + [c for c in self.classes if c.lower() not in ["nothing", "all"]]
            self.combo.configure(values=combo_values)
            self.save_classes()
            
            if self.parent_app and hasattr(self.parent_app, "creeaza_fila_clasa"):
                self.parent_app.after(0, lambda: self.parent_app.creeaza_fila_clasa(class_name))

        class_id = self.classes.index(class_name)
        
        if self.current_image_path:
            base_filename, _ = os.path.splitext(os.path.basename(self.current_image_path))
            old_class = self.current_class_folder
            if old_class and old_class != class_name:
                old_txt = os.path.join(self.lbl_dir, old_class, base_filename + ".txt")
                if os.path.exists(old_txt):
                    try: os.remove(old_txt)
                    except: pass
                old_img = os.path.join(self.img_dir, old_class, base_filename + ".jpg")
                if os.path.exists(old_img):
                    try: os.remove(old_img)
                    except: pass
        else:
            timestamp = int(time.time() * 1000)
            base_filename = f"screenshot_{timestamp}"

        class_img_dir = os.path.join(self.img_dir, class_name)
        class_lbl_dir = os.path.join(self.lbl_dir, class_name)
        os.makedirs(class_img_dir, exist_ok=True)
        os.makedirs(class_lbl_dir, exist_ok=True)

        target_img_path = os.path.join(class_img_dir, base_filename + ".jpg")
        
        # Salvăm imaginea direct în folderul noii clase
        self.original_img.save(target_img_path, quality=95)

        # Dacă imaginea venea din 'all' sau altă parte, o ștergem de acolo
        if self.current_image_path and os.path.abspath(self.current_image_path) != os.path.abspath(target_img_path):
            try:
                if os.path.exists(self.current_image_path):
                    os.remove(self.current_image_path)
            except:
                pass

        yolo_coords = [f"{x / self.width:.6f} {y / self.height:.6f}" for x, y in self.points]
        with open(os.path.join(class_lbl_dir, base_filename + ".txt"), "w", encoding="utf-8") as f:
            f.write(f"{class_id} " + " ".join(yolo_coords) + "\n")

        self.close()

    def _reporneste_fluxul_principal(self):
        self.current_image_path = None
        self.current_class_folder = None
        if self.parent_app:
            self.parent_app.deiconify()
            
            if hasattr(self.parent_app, "is_running"):
                self.parent_app.is_running = True
            if hasattr(self.parent_app, "master") and hasattr(self.parent_app.master, "is_running"):
                self.parent_app.master.is_running = True
                
            self.parent_app.focus_force()
            
            if hasattr(self.parent_app, "incarca_imagini"):
                self.parent_app.after(100, self.parent_app.incarca_imagini)

    def close(self):
        # Dezactivăm bind_all înainte de distrugere pentru a evita evenimente orfane
        if self.root:
            try:
                self.root.unbind_all("<Escape>")
                self.root.unbind_all("<Return>")
            except:
                pass
            try:
                self.root.grab_release()
            except:
                pass
            try:
                self.root.destroy()
            except:
                pass
            self.root = None
        self._reporneste_fluxul_principal()