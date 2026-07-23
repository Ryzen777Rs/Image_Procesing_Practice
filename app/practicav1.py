import os
from pathlib import Path
import shutil
import threading
import time
from tkinter import filedialog

import customtkinter
from PIL import Image, ImageDraw

# Verificare CUDA (NVIDIA) pentru disponibilitatea GPU-ului la antrenare
try:
    import torch

    CUDA_DISPONIBIL = torch.cuda.is_available()
except ImportError:
    CUDA_DISPONIBIL = False

try:
    import yaml
    from ultralytics import YOLO

    YOLO_DISPONIBIL = True
except ImportError:
    YOLO_DISPONIBIL = False

from snipping_tool import SnippingAnnotator


# ==========================================
# FUNCȚIE AJUTĂTOARE PENTRU CĂUTARE RECURSIVĂ
# ==========================================
def colecteaza_date_antrenare(
    base_dir=".", clase_selectate=None, extensii_imagini={".jpg", ".jpeg", ".png", ".webp", ".bmp"}
):
    """Parcurge recursiv toate subfolderele din 'images' și 'labels' pentru a găsi
    perechi valide (imagine, etichetă).
    """
    base_path = Path(base_dir)
    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    perechi_gasite = []
    numar_imagini_totale = 0

    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Folderele 'images' sau 'labels' nu există în {base_path}")
        return []

    for img_path in images_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in extensii_imagini:
            numar_imagini_totale += 1
            cale_relativa = img_path.relative_to(images_dir)
            label_path = labels_dir / cale_relativa.with_suffix(".txt")

            if label_path.exists():
                are_clasa_valida = False
                with open(label_path, "r", encoding="utf-8") as f:
                    for linie in f:
                        piese = linie.strip().split()
                        if not piese:
                            continue
                        class_id = int(piese[0])
                        if (
                            clase_selectate is None
                            or len(clase_selectate) == 0
                            or class_id in clase_selectate
                        ):
                            are_clasa_valida = True
                            break

                if are_clasa_valida:
                    perechi_gasite.append(
                        (str(img_path.resolve()), str(label_path.resolve()))
                    )

    print(f"📊 Sumar scanare:")
    print(f"   - Imagini găsite în total: {numar_imagini_totale}")
    print(f"   - Imagini valide găsite: {len(perechi_gasite)}")

    return perechi_gasite


# ==========================================
# WORKER PENTRU ANTRENARE REALĂ YOLO
# ==========================================
class TrainWorker(threading.Thread):

    def __init__(
        self,
        model_ales,
        nume_model_salvat_user,
        dispozitiv,
        clase_selectate,
        director_curent,
        cpu_threads,
        epoci,
        imgsz,
        lr,
        workers,
        fp16,
        sterge_dupa,
        cb_progress,
        cb_finish,
    ):
        super().__init__()
        self.model_ales = model_ales
        self.nume_model_salvat_user = nume_model_salvat_user
        self.dispozitiv = dispozitiv
        self.clase_selectate = clase_selectate
        self.director_curent = director_curent
        self.cpu_threads = cpu_threads
        self.epoci = int(epoci)
        self.imgsz = int(imgsz)
        self.lr = float(lr)
        self.workers = int(workers)
        self.fp16 = fp16
        self.sterge_dupa = sterge_dupa
        self.daemon = True

        self.cb_progress = cb_progress
        self.cb_finish = cb_finish
        self._cancel_event = threading.Event()

    def anuleaza(self):
        self._cancel_event.set()

    def run(self):
        if not YOLO_DISPONIBIL:
            print(
                "Eroare: Pachetul 'ultralytics' nu este instalat. Rulează: pip install ultralytics"
            )
            self.cb_finish(anulat=True)
            return

        print("=" * 40)
        print("PREGĂTIRE SET DE DATE PENTRU ANTRENARE YOLO...")
        print("=" * 40)

        dataset_dir = os.path.join(self.director_curent, "yolo_dataset_temp")
        os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)

        class_mapping = {nume: i for i, nume in enumerate(self.clase_selectate)}

        mapa_imagini = os.path.join(self.director_curent, "images")
        mapa_labels = os.path.join(self.director_curent, "labels")

        imagini_adaugate = 0
        try:
            extensii_valide = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

            for nume_clasa in self.clase_selectate:
                clasa_img_dir = os.path.join(mapa_imagini, nume_clasa)
                clasa_lbl_dir = os.path.join(mapa_labels, nume_clasa)

                if not os.path.exists(clasa_img_dir):
                    continue

                for root, _, files in os.walk(clasa_img_dir):
                    for img_name in files:
                        if not img_name.lower().endswith(extensii_valide):
                            continue

                        base_name = os.path.splitext(img_name)[0]
                        src_img = os.path.join(root, img_name)

                        rel_subpath = os.path.relpath(root, clasa_img_dir)
                        src_lbl = os.path.join(
                            clasa_lbl_dir, rel_subpath, base_name + ".txt"
                        )

                        if not os.path.exists(src_lbl):
                            src_lbl = os.path.join(
                                clasa_lbl_dir, base_name + ".txt"
                            )
                        if not os.path.exists(src_lbl):
                            src_lbl = os.path.join(
                                mapa_labels, "all", base_name + ".txt"
                            )

                        if not os.path.exists(src_lbl) and os.path.exists(
                            clasa_lbl_dir
                        ):
                            for l_root, _, l_files in os.walk(clasa_lbl_dir):
                                if (base_name + ".txt") in l_files:
                                    src_lbl = os.path.join(
                                        l_root, base_name + ".txt"
                                    )
                                    break

                        if os.path.exists(src_lbl):
                            dst_img = os.path.join(
                                dataset_dir,
                                "images",
                                "train",
                                f"{nume_clasa}_{img_name}",
                            )
                            shutil.copy(src_img, dst_img)

                            dst_lbl = os.path.join(
                                dataset_dir,
                                "labels",
                                "train",
                                f"{nume_clasa}_{base_name}.txt",
                            )
                            with open(
                                src_lbl, "r", encoding="utf-8"
                            ) as f_in, open(
                                dst_lbl, "w", encoding="utf-8"
                            ) as f_out:
                                for line in f_in:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        parts[0] = str(
                                            class_mapping[nume_clasa]
                                        )
                                        f_out.write(" ".join(parts) + "\n")
                            imagini_adaugate += 1

            if imagini_adaugate == 0:
                print(
                    "Nu s-au găsit imagini adnotate valide pentru clasele selectate."
                )
                shutil.rmtree(dataset_dir, ignore_errors=True)
                self.cb_finish(anulat=True)
                return

            print(
                f"✅ Au fost pregătite {imagini_adaugate} imagini pentru antrenare."
            )

            yaml_path = os.path.join(dataset_dir, "data.yaml")
            yaml_content = {
                "path": dataset_dir,
                "train": "images/train",
                "val": "images/train",
                "names": {i: nume for nume, i in class_mapping.items()},
            }
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_content, f)

            if self.model_ales == "Model YOLO neantrenat":
                model = YOLO("yolov8n.pt")
            else:
                model_path = os.path.join(
                    self.director_curent, "models", self.model_ales
                )
                model = YOLO(model_path)

            device_str = (
                "cuda"
                if CUDA_DISPONIBIL and "GPU" in self.dispozitiv
                else "cpu"
            )
            timp_start = time.time()

            def on_train_epoch_end(trainer):
                if self._cancel_event.is_set():
                    trainer.stop = True

                epoca_curenta = trainer.epoch + 1
                progress_float = epoca_curenta / self.epoci
                progress_procentaj = int(progress_float * 100)

                timp_scurs = time.time() - timp_start
                timp_pe_epoca = timp_scurs / epoca_curenta
                epoci_ramase = self.epoci - epoca_curenta
                secunde_ramase = epoci_ramase * timp_pe_epoca

                min_ramase = int(secunde_ramase // 60)
                sec_ramase = int(secunde_ramase % 60)
                timp_str = f"remain {min_ramase}m {sec_ramase}s (Epoca {epoca_curenta}/{self.epoci})"

                self.cb_progress(progress_float, progress_procentaj, timp_str)

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            print("PORNIRE ANTRENARE EFECTIVĂ...")
            model.train(
                data=yaml_path,
                epochs=self.epoci,
                imgsz=self.imgsz,
                lr0=self.lr,
                workers=self.workers,
                device=device_str,
                amp=self.fp16,
                project=os.path.join(self.director_curent, "runs"),
                name="train_session",
                exist_ok=True,
            )

            if not self._cancel_event.is_set():
                best_model_path = os.path.join(
                    self.director_curent,
                    "runs",
                    "train_session",
                    "weights",
                    "best.pt",
                )
                if os.path.exists(best_model_path):
                    if self.model_ales == "Model YOLO neantrenat":
                        nume_curat = self.nume_model_salvat_user.strip()
                        if nume_curat:
                            if not nume_curat.lower().endswith(".pt"):
                                nume_curat += ".pt"
                            nume_model_salvat = nume_curat
                        else:
                            nume_model_salvat = f"model_sigma_{int(time.time())}.pt"
                    else:
                        nume_model_salvat = self.model_ales

                    destinatie_finala = os.path.join(
                        self.director_curent, "models", nume_model_salvat
                    )
                    shutil.copy(best_model_path, destinatie_finala)
                    print(f"Model salvat cu succes: {destinatie_finala}")

                if self.sterge_dupa:
                    for nume_clasa in self.clase_selectate:
                        shutil.rmtree(
                            os.path.join(mapa_imagini, nume_clasa),
                            ignore_errors=True,
                        )
                        shutil.rmtree(
                            os.path.join(mapa_labels, nume_clasa),
                            ignore_errors=True,
                        )

            self.cb_finish(anulat=self._cancel_event.is_set())

        except Exception as e:
            print(f"Eroare severă în timpul antrenării YOLO: {e}")
            self.cb_finish(anulat=True)
        finally:
            shutil.rmtree(dataset_dir, ignore_errors=True)


class TrainWindow(customtkinter.CTkToplevel):

    def __init__(self, parent):
        super().__init__(parent)

        self.geometry("1280x720")
        self.title("Meniu Antrenare și Galerie")

        self.DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))
        self.MAPA_IMAGINI = os.path.join(self.DIRECTOR_CURENT, "images")
        self.MAPA_ALL = os.path.join(self.MAPA_IMAGINI, "all")
        self.MAPA_LABELS = os.path.join(self.DIRECTOR_CURENT, "labels")
        self.MAPA_MODELE = os.path.join(self.DIRECTOR_CURENT, "models")

        os.makedirs(self.MAPA_ALL, exist_ok=True)
        os.makedirs(self.MAPA_LABELS, exist_ok=True)
        os.makedirs(self.MAPA_MODELE, exist_ok=True)

        self.imagini_salvate = []
        self.clase_existente = []
        self.thumbnail_cache = {}
        self.load_token = 0

        # ---- NOI VARIABILE PENTRU PAGINARE ----
        self.imagini_per_pagina = 100
        self.pagina_curenta = 0
        self.lista_date_imagini = []
        # ---------------------------------------

        self.vars_clase = {}
        self.var_all_clase = customtkinter.BooleanVar(value=True)
        self.var_model_selectat = customtkinter.StringVar(
            value="Model YOLO neantrenat"
        )
        self.var_nume_model_nou = customtkinter.StringVar(value="")

        self.var_dispozitiv = customtkinter.StringVar(
            value="GPU (AMD Radeon RX 7600)"
        )
        self.var_cpu_threads = customtkinter.StringVar(value="12")
        self.var_epoci = customtkinter.StringVar(value="50")
        self.var_imgsz = customtkinter.StringVar(value="640")
        self.var_lr = customtkinter.StringVar(value="0.01")
        self.var_workers = customtkinter.StringVar(value="4")
        self.var_fp16 = customtkinter.BooleanVar(value=True)
        self.var_sterge_dupa = customtkinter.BooleanVar(value=False)

        self.timer_previzualizare = None
        self.fereastra_previzualizare = None
        self.clasa_selectata = None

        self.snipper = SnippingAnnotator(
            parent_app=self,
            output_dir=self.DIRECTOR_CURENT,
            hotkey="ctrl+shift+s",
        )
        self.snipper.start_background_listener()

        self.creeaza_pagini()
        self.arata_pagina_principala()

        def rotunjeste_imaginea(self, imagine_pil, raza):
        # 1. Ne asigurăm că imaginea suportă transparență (RGBA)
            imagine_pil = imagine_pil.convert("RGBA")
            
            # 2. Creăm o mască invizibilă (neagră) de aceeași dimensiune
            masca = Image.new("L", imagine_pil.size, 0)
            draw = ImageDraw.Draw(masca)
            
            # 3. Desenăm un dreptunghi cu colțuri rotunde (alb) pe mască
            draw.rounded_rectangle((0, 0, imagine_pil.size[0], imagine_pil.size[1]), radius=raza, fill=255)
            
            # 4. Decupăm imaginea originală folosind masca
            imagine_pil.putalpha(masca)
            return imagine_pil

    def _bind_scroll_la_widget(self, widget, functie_scroll):
        if widget is None:
            return
        widget.bind("<MouseWheel>", functie_scroll, add="+")
        widget.bind("<Button-4>", functie_scroll, add="+")
        widget.bind("<Button-5>", functie_scroll, add="+")
        if hasattr(widget, "_label") and widget._label:
            widget._label.bind("<MouseWheel>", functie_scroll, add="+")
            widget._label.bind("<Button-4>", functie_scroll, add="+")
            widget._label.bind("<Button-5>", functie_scroll, add="+")
        if hasattr(widget, "_canvas") and widget._canvas:
            widget._canvas.bind("<MouseWheel>", functie_scroll, add="+")
            widget._canvas.bind("<Button-4>", functie_scroll, add="+")
            widget._canvas.bind("<Button-5>", functie_scroll, add="+")

    def creeaza_pagini(self):
        # ==========================================
        # FUNDALUL GENERAL (Culoarea principală Discord)
        # ==========================================
        self.main_page = customtkinter.CTkFrame(self, fg_color="#1e1f22")
        
        # ==========================================
        # WRAPPER PENTRU CENTRARE VERTICALĂ ȘI ORIZONTALĂ
        # ==========================================
        # Acest frame invizibil ține titlul și cardurile împreună în centrul absolut
        center_wrapper = customtkinter.CTkFrame(self.main_page, fg_color="transparent")
        center_wrapper.pack(expand=True) 

        title = customtkinter.CTkLabel(
            master=center_wrapper, 
            text="Sigma", 
            font=("Roboto", 40, "bold"),
            text_color="#f2f3f5" # Un alb-gri fin, specific textului din Discord
        )
        title.pack(pady=(0, 40)) # Spațiu curat doar sub titlu, pentru a-l separa de carduri

        frame_1 = customtkinter.CTkFrame(master=center_wrapper, fg_color="transparent")
        frame_1.pack(fill="both", expand=True)
        
        frame_1.columnconfigure(0, weight=1)
        frame_1.columnconfigure(1, weight=1)
        frame_1.columnconfigure(2, weight=1)

        # ==========================================
        # ÎNCĂRCARE IMAGINI
        # ==========================================
        cale_logos = os.path.join(self.DIRECTOR_CURENT, "logos")
        cale_galerie = os.path.join(cale_logos, "misty mountains.jpg")
        cale_invatare = os.path.join(cale_logos, "gears.jpeg")
        cale_inapoi = os.path.join(cale_logos, "detection in real time.jpg")

        try:
            # 1. Deschidem imaginile brute
            img_g_bruta = Image.open(cale_galerie)
            img_i_bruta = Image.open(cale_invatare)
            img_b_bruta = Image.open(cale_inapoi)

            # 2. Le tăiem la 400x400 din start ca rotunjirea să fie precisă
            img_g_bruta = img_g_bruta.resize((400, 400))
            img_i_bruta = img_i_bruta.resize((400, 400))
            img_b_bruta = img_b_bruta.resize((400, 400))

            # 3. Aplicăm funcția de rotunjire (raza 16 este ideală)
            raza_rotunjire = 16
            img_g_rotunda = self.rotunjeste_imaginea(img_g_bruta, raza_rotunjire)
            img_i_rotunda = self.rotunjeste_imaginea(img_i_bruta, raza_rotunjire)
            img_b_rotunda = self.rotunjeste_imaginea(img_b_bruta, raza_rotunjire)

            # 4. Le predăm către CustomTkinter
            img_galerie = customtkinter.CTkImage(img_g_rotunda, size=(400, 400))
            img_invatare = customtkinter.CTkImage(img_i_rotunda, size=(400, 400))
            img_inapoi = customtkinter.CTkImage(img_b_rotunda, size=(400, 400))
            
        except Exception as e:
            print(f"Imaginile pentru meniu nu au putut fi încărcate: {e}")
            img_galerie = img_invatare = img_inapoi = None

        # Definim culoarea cardului pentru a nu o repeta manual de 3 ori
        culoare_card = "#2b2d31"

        # ==========================================
        # COLOANA 1: CARD GALERIE
        # ==========================================
        column1 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column1.grid(row=0, column=0, sticky="nsew", padx=20)
        
        card1 = customtkinter.CTkFrame(
            master=column1, 
            fg_color=culoare_card, 
            corner_radius=12,
            border_width=0
        )
        card1.pack(expand=True, fill="both") 

        lbl_img1 = customtkinter.CTkLabel(master=card1, text="[Fără Imagine]" if not img_galerie else "", image=img_galerie)
        # AM MODIFICAT AICI: padx și pady crescute la 25
        lbl_img1.pack(pady=(25, 15), padx=25) 

        button1 = customtkinter.CTkButton(
            master=card1,
            text="Galerie",
            height=50, # AM MODIFICAT AICI: height crescut de la 40 la 50
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_galerie,
        )
        # AM MODIFICAT AICI: padx crescut la 25 și pady la (0, 25) pentru baza cardului
        button1.pack(fill="x", padx=50, pady=(20, 25))


        # ==========================================
        # COLOANA 2: CARD ÎNVĂȚARE
        # ==========================================
        column2 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column2.grid(row=0, column=1, sticky="nsew", padx=20)
        
        card2 = customtkinter.CTkFrame(
            master=column2, 
            fg_color=culoare_card,
            corner_radius=12,
            border_width=0
        )
        card2.pack(expand=True, fill="both")

        lbl_img2 = customtkinter.CTkLabel(master=card2, text="[Fără Imagine]" if not img_invatare else "", image=img_invatare)
        # AM MODIFICAT AICI
        lbl_img2.pack(pady=(25, 15), padx=25)

        button2 = customtkinter.CTkButton(
            master=card2,
            text="Învățare",
            height=50, # AM MODIFICAT AICI
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_invatare,
        )
        # AM MODIFICAT AICI
        button2.pack(fill="x", padx=50, pady=(20, 25))


        # ==========================================
        # COLOANA 3: CARD ÎNAPOI
        # ==========================================
        column3 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column3.grid(row=0, column=2, sticky="nsew", padx=20)
        
        card3 = customtkinter.CTkFrame(
            master=column3, 
            fg_color=culoare_card,
            corner_radius=12,
            border_width=0
        )
        card3.pack(expand=True, fill="both")

        lbl_img3 = customtkinter.CTkLabel(master=card3, text="[Fără Imagine]" if not img_inapoi else "", image=img_inapoi)
        # AM MODIFICAT AICI
        lbl_img3.pack(pady=(25, 15), padx=25)

        button3 = customtkinter.CTkButton(
            master=card3,
            text="Înapoi",
            height=50, # AM MODIFICAT AICI
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=lambda: (
                self.master.event_generate("<<CloseTrainWindow>>")
                if hasattr(self, "master")
                else self.destroy()
            ),
        )
        # AM MODIFICAT AICI
        button3.pack(fill="x", padx=50, pady=(20, 25))

        self.second_page = customtkinter.CTkFrame(self, fg_color="transparent")

        top_bar = customtkinter.CTkFrame(
            self.second_page, fg_color="transparent", height=40
        )
        top_bar.pack(fill="x", padx=20, pady=10)

        buton_adauga_clasa = customtkinter.CTkButton(
            top_bar,
            text="+",
            width=35,
            height=35,
            corner_radius=8,
            font=("Roboto", 20, "bold"),
            fg_color="#1f538d",
            hover_color="#14375e",
            command=self.deschide_dialog_clasa,
        )
        buton_adauga_clasa.pack(side="left", padx=(0, 8), pady=(0, 6))

        self.tab_scroll_frame = customtkinter.CTkScrollableFrame(
            master=top_bar,
            orientation="horizontal",
            height=38,
            fg_color="transparent",
            scrollbar_button_color="#242424",
            scrollbar_button_hover_color="gray",
        )
        self.tab_scroll_frame.pack(side="left", fill="x", expand=True)

        self._bind_scroll_la_widget(
            self.tab_scroll_frame, self._pe_rotire_tabs
        )
        if hasattr(self.tab_scroll_frame, "_parent_canvas"):
            self._bind_scroll_la_widget(
                self.tab_scroll_frame._parent_canvas, self._pe_rotire_tabs
            )

        self.btn_toate_clasele = customtkinter.CTkButton(
            master=self.tab_scroll_frame,
            text="All",
            width=100,
            height=32,
            fg_color="#1f538d",
            hover_color="#14375e",
            command=lambda: self.filtreaza_dupa_clasa(None),
        )
        self.btn_toate_clasele.pack(side="left", padx=4)
        self._bind_scroll_la_widget(
            self.btn_toate_clasele, self._pe_rotire_tabs
        )

        bottom_bar = customtkinter.CTkFrame(
            self.second_page, fg_color="transparent"
        )
        bottom_bar.pack(fill="x", side="bottom", padx=20, pady=20)
        
        bottom_bar.columnconfigure(0, weight=1)
        bottom_bar.columnconfigure(1, weight=1)
        bottom_bar.columnconfigure(2, weight=1)

        buton_import_imagini = customtkinter.CTkButton(
            bottom_bar, text="Import Images", command=self.deschide_dialog_import
        )
        buton_import_imagini.grid(row=0, column=0, sticky="w")

        # --- PANOU PAGINARE ---
        frame_paginare = customtkinter.CTkFrame(bottom_bar, fg_color="transparent")
        frame_paginare.grid(row=0, column=1)

        self.btn_prev = customtkinter.CTkButton(
            frame_paginare, text="< Înapoi", width=70, command=self.pagina_anterioara
        )
        self.btn_prev.pack(side="left", padx=5)

        self.lbl_paginare = customtkinter.CTkLabel(
            frame_paginare, text="Pagină: 1 / 1", font=("Roboto", 14, "bold")
        )
        self.lbl_paginare.pack(side="left", padx=15)

        self.btn_next = customtkinter.CTkButton(
            frame_paginare, text="Înainte >", width=70, command=self.pagina_urmatoare
        )
        self.btn_next.pack(side="left", padx=5)
        # ------------------------

        buton_inapoi_galerie = customtkinter.CTkButton(
            bottom_bar, text="Meniu Principal", command=self.arata_pagina_principala
        )
        buton_inapoi_galerie.grid(row=0, column=2, sticky="e")

        self.galerie_imagini = customtkinter.CTkScrollableFrame(
            self.second_page
        )
        self.galerie_imagini.pack(fill="both", expand=True, padx=20, pady=10)

        self._bind_scroll_la_widget(
            self.galerie_imagini, self._pe_rotire_galerie
        )
        if hasattr(self.galerie_imagini, "_parent_canvas"):
            self._bind_scroll_la_widget(
                self.galerie_imagini._parent_canvas, self._pe_rotire_galerie
            )

        self.third_page = customtkinter.CTkFrame(self, fg_color="transparent")

        main_invatare_container = customtkinter.CTkFrame(
            self.third_page, fg_color="transparent"
        )
        main_invatare_container.pack(
            fill="both", expand=True, padx=20, pady=(20, 10)
        )
        main_invatare_container.columnconfigure(0, weight=1)
        main_invatare_container.columnconfigure(1, weight=2)
        main_invatare_container.columnconfigure(2, weight=1)
        main_invatare_container.rowconfigure(0, weight=1)

        frame_stanga_clase = customtkinter.CTkFrame(
            main_invatare_container, corner_radius=10
        )
        frame_stanga_clase.grid(
            row=0, column=0, sticky="nsew", padx=(0, 10), pady=10
        )

        lbl_titlu_clase = customtkinter.CTkLabel(
            frame_stanga_clase,
            text="Clase folosite pentru antrenare",
            font=("Roboto", 16, "bold"),
        )
        lbl_titlu_clase.pack(pady=(15, 10), padx=15, anchor="w")

        self.scroll_clase_invatare = customtkinter.CTkScrollableFrame(
            frame_stanga_clase, fg_color="transparent"
        )
        self.scroll_clase_invatare.pack(
            fill="both", expand=True, padx=10, pady=(0, 10)
        )
        self.scroll_clase_invatare.columnconfigure(0, weight=1)

        frame_mijloc_setari = customtkinter.CTkFrame(
            main_invatare_container, corner_radius=10
        )
        frame_mijloc_setari.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        lbl_titlu_setari = customtkinter.CTkLabel(
            frame_mijloc_setari, text="Setări", font=("Roboto", 16, "bold")
        )
        lbl_titlu_setari.pack(pady=(15, 10), padx=15, anchor="w")

        scroll_setari_interior = customtkinter.CTkScrollableFrame(
            frame_mijloc_setari, fg_color="transparent"
        )
        scroll_setari_interior.pack(
            fill="both", expand=True, padx=10, pady=(0, 10)
        )

        customtkinter.CTkLabel(
            scroll_setari_interior,
            text="Unitate de procesare:",
            font=("Roboto", 13, "bold"),
        ).pack(anchor="w", pady=(5, 3))

        self.seg_dispozitiv = customtkinter.CTkSegmentedButton(
            scroll_setari_interior,
            values=["GPU (AMD Radeon RX 7600)", "CPU (AMD Ryzen 5 5600)"],
            variable=self.var_dispozitiv,
        )
        self.seg_dispozitiv.pack(fill="x", pady=(0, 12))

        if not CUDA_DISPONIBIL:
            self.var_dispozitiv.set("CPU (AMD Ryzen 5 5600)")
            self.seg_dispozitiv.configure(state="disabled")

        customtkinter.CTkLabel(
            scroll_setari_interior,
            text="CPU threads (Fire reale detectate: 12):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        customtkinter.CTkEntry(
            scroll_setari_interior, textvariable=self.var_cpu_threads
        ).pack(fill="x", pady=(0, 10))

        customtkinter.CTkLabel(
            scroll_setari_interior, text="Număr epoci:", font=("Roboto", 13)
        ).pack(anchor="w", pady=(2, 2))
        customtkinter.CTkEntry(
            scroll_setari_interior, textvariable=self.var_epoci
        ).pack(fill="x", pady=(0, 15))

        customtkinter.CTkLabel(
            scroll_setari_interior,
            text="Parametri Suplimentari",
            font=("Roboto", 14, "bold"),
        ).pack(anchor="w", pady=(5, 8))

        customtkinter.CTkLabel(
            scroll_setari_interior,
            text="Dimensiune Imagine (ImgSz):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        customtkinter.CTkEntry(
            scroll_setari_interior, textvariable=self.var_imgsz
        ).pack(fill="x", pady=(0, 10))

        customtkinter.CTkLabel(
            scroll_setari_interior, text="Rată de Învățare (LR):", font=("Roboto", 13)
        ).pack(anchor="w", pady=(2, 2))
        customtkinter.CTkEntry(
            scroll_setari_interior, textvariable=self.var_lr
        ).pack(fill="x", pady=(0, 10))

        customtkinter.CTkLabel(
            scroll_setari_interior,
            text="Thread-uri încărcare date (Workers):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        customtkinter.CTkEntry(
            scroll_setari_interior, textvariable=self.var_workers
        ).pack(fill="x", pady=(0, 12))

        customtkinter.CTkCheckBox(
            scroll_setari_interior,
            text="Precizie Mixtă (FP16 / AMP)",
            variable=self.var_fp16,
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(5, 8))
        customtkinter.CTkCheckBox(
            scroll_setari_interior,
            text="Șterge clasele respective după învățare",
            variable=self.var_sterge_dupa,
            text_color="#FF6B6B",
            font=("Roboto", 13, "bold"),
        ).pack(anchor="w", pady=(2, 10))

        frame_dreapta_actiuni = customtkinter.CTkFrame(
            main_invatare_container, fg_color="transparent"
        )
        frame_dreapta_actiuni.grid(
            row=0, column=2, sticky="nsew", padx=(10, 0), pady=10
        )
        frame_dreapta_actiuni.rowconfigure(1, weight=1)
        frame_dreapta_actiuni.columnconfigure(0, weight=1)

        self.btn_antrenare = customtkinter.CTkButton(
            master=frame_dreapta_actiuni,
            text="ANTRENARE",
            font=("Roboto", 22, "bold"),
            fg_color="#8B0000",
            hover_color="#A00000",
            text_color="#FFFFFF",
            border_width=2,
            border_color="#500000",
            corner_radius=6,
            height=50,
            command=self.porneste_antrenarea,
        )
        self.btn_antrenare.grid(row=0, column=0, sticky="ew", pady=(0, 15))

        frame_bottom_modele = customtkinter.CTkFrame(
            frame_dreapta_actiuni, corner_radius=10
        )
        frame_bottom_modele.grid(row=1, column=0, sticky="nsew")

        lbl_titlu_modele = customtkinter.CTkLabel(
            frame_bottom_modele, text="MODELE", font=("Roboto", 16, "bold")
        )
        lbl_titlu_modele.pack(pady=(12, 8))

        self.scroll_modele_invatare = customtkinter.CTkScrollableFrame(
            frame_bottom_modele, fg_color="transparent"
        )
        self.scroll_modele_invatare.pack(
            fill="both", expand=True, padx=15, pady=(0, 5)
        )

        lbl_nume_model_salvat = customtkinter.CTkLabel(
            frame_bottom_modele,
            text="Denumire model salvat:",
            font=("Roboto", 13, "bold"),
        )
        lbl_nume_model_salvat.pack(anchor="w", padx=15, pady=(5, 2))

        self.entry_nume_model = customtkinter.CTkEntry(
            frame_bottom_modele,
            textvariable=self.var_nume_model_nou,
            placeholder_text="ex: model_detecție_v1",
        )
        self.entry_nume_model.pack(fill="x", padx=15, pady=(0, 15))

        bottom_bar_invatare = customtkinter.CTkFrame(
            self.third_page, fg_color="transparent"
        )
        bottom_bar_invatare.pack(fill="x", side="bottom", padx=20, pady=(0, 15))

        buton_inapoi_invatare = customtkinter.CTkButton(
            master=bottom_bar_invatare,
            text="Înapoi",
            command=self.arata_pagina_principala,
        )
        buton_inapoi_invatare.pack(side="right")

    def pagina_anterioara(self):
        if self.pagina_curenta > 0:
            self.pagina_curenta -= 1
            self.afiseaza_pagina_curenta(self.load_token)

    def pagina_urmatoare(self):
        total_pagini = (len(self.lista_date_imagini) + self.imagini_per_pagina - 1) // self.imagini_per_pagina
        if self.pagina_curenta < total_pagini - 1:
            self.pagina_curenta += 1
            self.afiseaza_pagina_curenta(self.load_token)

    def deschide_dialog_import(self):
        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("420x220")
        dialog.title("Import Images")
        dialog.transient(self)
        dialog.grab_set()

        customtkinter.CTkLabel(
            dialog, text="Alegeți modul de import:", font=("Roboto", 16, "bold")
        ).pack(pady=(25, 15))

        btn_fisier = customtkinter.CTkButton(
            dialog,
            text="Selectează Fișier(e)",
            width=280,
            height=36,
            command=lambda: [dialog.destroy(), self.importa_fisiere_dialog()],
        )
        btn_fisier.pack(pady=6)

        btn_folder = customtkinter.CTkButton(
            dialog,
            text="Selectează Folder (cu subfoldere)",
            width=280,
            height=36,
            command=lambda: [dialog.destroy(), self.importa_folder_dialog()],
        )
        btn_folder.pack(pady=6)

    def importa_fisiere_dialog(self):
        fisiere = filedialog.askopenfilenames(
            title="Selectează imagini",
            filetypes=[
                ("Imagini", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Toate fișierele", "*.*"),
            ],
        )
        if fisiere:
            def proceseaza_import():
                extensii_valide = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                importate = 0
                for f in fisiere:
                    c_path = Path(f)
                    if c_path.suffix.lower() in extensii_valide:
                        dest = Path(self.MAPA_ALL) / c_path.name
                        if dest.exists():
                            b = c_path.stem
                            ext = c_path.suffix
                            dest = Path(self.MAPA_ALL) / f"{b}_{int(time.time())}{ext}"
                        shutil.copy(c_path, dest)
                        importate += 1
                print(f"📥 Au fost importate {importate} fișiere.")
                self.after(0, self.incarca_imagini)
            
            threading.Thread(target=proceseaza_import, daemon=True).start()

    def importa_folder_dialog(self):
        folder = filedialog.askdirectory(title="Selectează folderul cu imagini")
        if folder:
            def proceseaza_import_folder():
                folder_path = Path(folder)
                extensii_valide = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
                importate = 0
                
                for item in folder_path.rglob("*"):
                    if item.is_file() and item.suffix.lower() in extensii_valide:
                        dest = Path(self.MAPA_ALL) / item.name
                        if dest.exists():
                            b = item.stem
                            ext = item.suffix
                            dest = Path(self.MAPA_ALL) / f"{b}_{int(time.time())}{ext}"
                        shutil.copy(item, dest)
                        importate += 1
                        
                print(f"📥 Au fost importate {importate} imagini din folder și subfoldere.")
                self.after(0, self.incarca_imagini)
            
            threading.Thread(target=proceseaza_import_folder, daemon=True).start()

    def _actualizeaza_stare_nume_model(self):
        if self.var_model_selectat.get() == "Model YOLO neantrenat":
            self.entry_nume_model.configure(state="normal")
        else:
            self.entry_nume_model.configure(state="disabled")

    def editeaza_adnotare(self, calea_imagine, nume_clasa):
        try:
            if hasattr(self.snipper, "open_annotation_ui"):
                self.snipper.open_annotation_ui(
                    image_path=calea_imagine, default_class=nume_clasa
                )
        except Exception as e:
            print(f"Eroare la deschiderea editorului: {e}")

    def actualizeaza_stare_buton_antrenare(self):
        if any(var.get() for var in self.vars_clase.values()):
            self.btn_antrenare.configure(state="normal", fg_color="#8B0000")
        else:
            self.btn_antrenare.configure(state="disabled", fg_color="gray")

    def actualizeaza_interfata_invatare(self):
        for widget in self.scroll_clase_invatare.winfo_children():
            widget.destroy()
        self.vars_clase.clear()

        chk_all = customtkinter.CTkCheckBox(
            master=self.scroll_clase_invatare,
            text="ALL",
            font=("Roboto", 15, "bold"),
            variable=self.var_all_clase,
            checkbox_width=18,
            checkbox_height=18,
            border_width=2,
            corner_radius=4,
            command=self.toggle_select_all_clase,
        )
        chk_all.pack(anchor="w", padx=10, pady=(5, 12))

        self.incarca_clase_existente()

        for nume_clasa in self.clase_existente:
            var_clasa = customtkinter.BooleanVar(value=self.var_all_clase.get())
            self.vars_clase[nume_clasa] = var_clasa

            chk = customtkinter.CTkCheckBox(
                master=self.scroll_clase_invatare,
                text=nume_clasa,
                font=("Roboto", 15),
                variable=var_clasa,
                checkbox_width=18,
                checkbox_height=18,
                border_width=2,
                corner_radius=4,
                command=self.verifica_stare_all,
            )
            chk.pack(anchor="w", padx=10, pady=6)

        for widget in self.scroll_modele_invatare.winfo_children():
            widget.destroy()

        modele_disponibile = ["Model YOLO neantrenat"]
        if os.path.exists(self.MAPA_MODELE):
            fisiere_modele = [
                f for f in os.listdir(self.MAPA_MODELE) if f.endswith(".pt")
            ]
            modele_disponibile.extend(fisiere_modele)

        if self.var_model_selectat.get() not in modele_disponibile:
            self.var_model_selectat.set(modele_disponibile[0])

        for nume_model in modele_disponibile:
            radio_model = customtkinter.CTkRadioButton(
                master=self.scroll_modele_invatare,
                text=nume_model,
                value=nume_model,
                variable=self.var_model_selectat,
                font=("Roboto", 14),
                command=self._actualizeaza_stare_nume_model,
            )
            radio_model.pack(anchor="w", padx=10, pady=8)

        self._actualizeaza_stare_nume_model()
        self.actualizeaza_stare_buton_antrenare()

    def toggle_select_all_clase(self):
        stare_all = self.var_all_clase.get()
        for var in self.vars_clase.values():
            var.set(stare_all)
        self.actualizeaza_stare_buton_antrenare()

    def verifica_stare_all(self):
        if not self.vars_clase:
            return
        toate_bifate = all(var.get() for var in self.vars_clase.values())
        self.var_all_clase.set(toate_bifate)
        self.actualizeaza_stare_buton_antrenare()

    def porneste_antrenarea(self):
        clase_selectate = [c for c, var in self.vars_clase.items() if var.get()]
        model_ales = self.var_model_selectat.get()

        if not clase_selectate:
            print("Eroare: Nu ai selectat nicio clasă!")
            return

        def cb_progress(valoare_float, procentaj, timp_ramas):
            self.after(
                0, self._actualizeaza_modal_ui, valoare_float, procentaj, timp_ramas
            )

        def cb_finish(anulat):
            self.after(0, self._inchide_modal_antrenare, anulat)

        worker = TrainWorker(
            model_ales=model_ales,
            nume_model_salvat_user=self.var_nume_model_nou.get(),
            dispozitiv=self.var_dispozitiv.get(),
            clase_selectate=clase_selectate,
            director_curent=self.DIRECTOR_CURENT,
            cpu_threads=self.var_cpu_threads.get(),
            epoci=self.var_epoci.get(),
            imgsz=self.var_imgsz.get(),
            lr=self.var_lr.get(),
            workers=self.var_workers.get(),
            fp16=self.var_fp16.get(),
            sterge_dupa=self.var_sterge_dupa.get(),
            cb_progress=cb_progress,
            cb_finish=cb_finish,
        )

        self._deschide_modal_antrenare(worker)
        worker.start()

    def _deschide_modal_antrenare(self, worker):
        self.update_idletasks()

        self.modal_overlay = customtkinter.CTkToplevel(self)
        self.modal_overlay.overrideredirect(True)

        w = self.winfo_width()
        h = self.winfo_height()
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        self.modal_overlay.geometry(f"{w}x{h}+{x}+{y}")

        self.modal_overlay.attributes("-alpha", 0.85)
        self.modal_overlay.configure(fg_color="#0D0D0D")
        self.modal_overlay.transient(self)
        self.modal_overlay.grab_set()

        card_dialog = customtkinter.CTkFrame(
            self.modal_overlay,
            corner_radius=16,
            fg_color="#1E1E1E",
            border_width=2,
            border_color="#333333",
        )
        card_dialog.place(
            relx=0.5, rely=0.5, anchor="center", relwidth=0.48, relheight=0.42
        )

        lbl_titlu = customtkinter.CTkLabel(
            card_dialog,
            text="Modelul se antrenează (YOLOv8)",
            font=("Roboto", 22, "bold"),
        )
        lbl_titlu.pack(pady=(30, 20))

        self.progressbar = customtkinter.CTkProgressBar(
            card_dialog, width=400, height=22, corner_radius=8
        )
        self.progressbar.set(0.0)
        self.progressbar.pack(pady=(10, 5))

        self.lbl_procentaj = customtkinter.CTkLabel(
            card_dialog, text="0%", font=("Roboto", 20, "bold")
        )
        self.lbl_procentaj.pack(pady=(0, 15))

        frame_bottom = customtkinter.CTkFrame(card_dialog, fg_color="transparent")
        frame_bottom.pack(fill="x", side="bottom", padx=30, pady=25)

        self.lbl_timp_ramas = customtkinter.CTkLabel(
            frame_bottom, text="Inițializare antrenare...", font=("Roboto", 14)
        )
        self.lbl_timp_ramas.pack(side="left")

        btn_cancel = customtkinter.CTkButton(
            frame_bottom,
            text="Oprește",
            width=110,
            height=36,
            fg_color="#8B0000",
            hover_color="#A00000",
            font=("Roboto", 15, "bold"),
            command=lambda: worker.anuleaza(),
        )
        btn_cancel.pack(side="right")

        self.modal_overlay.lift()

    def _actualizeaza_modal_ui(self, valoare_float, procentaj, timp_ramas):
        if hasattr(self, "progressbar") and self.progressbar.winfo_exists():
            self.progressbar.set(valoare_float)
            self.lbl_procentaj.configure(text=f"{procentaj}%")
            self.lbl_timp_ramas.configure(text=timp_ramas)

    def _inchide_modal_antrenare(self, anulat):
        if hasattr(self, "modal_overlay") and self.modal_overlay.winfo_exists():
            self.modal_overlay.grab_release()
            self.modal_overlay.destroy()

        if anulat:
            print("❌ Antrenarea a fost anulată (sau eșuată).")
        else:
            print("✅ Antrenarea YOLO a fost finalizată cu succes!")
        self.actualizeaza_interfata_invatare()

    def incarca_clase_existente(self):
        if os.path.exists(self.MAPA_IMAGINI):
            for nume_folder in os.listdir(self.MAPA_IMAGINI):
                cale_folder = os.path.join(self.MAPA_IMAGINI, nume_folder)
                if os.path.isdir(cale_folder) and nume_folder.lower() != "all":
                    if nume_folder not in self.clase_existente:
                        self.clase_existente.append(nume_folder)

    def _pe_rotire_tabs(self, event):
        canvas = getattr(self.tab_scroll_frame, "_parent_canvas", None)
        if not canvas:
            return
        left, right = canvas.xview()
        if left <= 0.0 and right >= 1.0:
            return

        cantitate = (
            -1 * int(event.delta / 120)
            if event.delta
            else (1 if event.num == 5 else -1)
        )
        if cantitate < 0 and left <= 0.0:
            return
        if cantitate > 0 and right >= 1.0:
            return
        canvas.xview_scroll(cantitate * 20, "units")

    def _pe_rotire_galerie(self, event):
        canvas = getattr(self.galerie_imagini, "_parent_canvas", None)
        if not canvas:
            return
        top, bottom = canvas.yview()
        if top <= 0.0 and bottom >= 1.0:
            return

        cantitate = (
            -1 * int(event.delta / 120)
            if event.delta
            else (1 if event.num == 5 else -1)
        )
        if cantitate < 0 and top <= 0.0:
            return
        if cantitate > 0 and bottom >= 1.0:
            return
        canvas.yview_scroll(cantitate * 32, "units")

    def arata_pagina_galerie(self):
        self.main_page.pack_forget()
        self.third_page.pack_forget()
        self.second_page.pack(fill="both", expand=True)
        self.incarca_clase_existente()
        for c in self.clase_existente:
            self.creeaza_fila_clasa(c)
        self.incarca_imagini()

    def arata_pagina_invatare(self):
        self.main_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack(fill="both", expand=True)
        self.actualizeaza_interfata_invatare()

    def arata_pagina_principala(self):
        self.second_page.pack_forget()
        self.third_page.pack_forget()
        self.main_page.pack(fill="both", expand=True)

    def arata_avertisment_duplicat(self, nume):
        aviz = customtkinter.CTkToplevel(self)
        aviz.geometry("400x160")
        aviz.title("Atenție")
        aviz.transient(self)
        aviz.grab_set()
        customtkinter.CTkLabel(
            aviz,
            text=f"Clasa '{nume}' există deja!\nAlegeți altă denumire.",
            font=("Roboto", 14),
        ).pack(pady=(25, 15))
        
        btn = customtkinter.CTkButton(
            aviz, text="Am înțeles", width=120, command=aviz.destroy
        )
        btn.pack(pady=10)
        
        aviz.bind("<Return>", lambda event: aviz.destroy())

    def deschide_dialog_clasa(self):
        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("450x220")
        dialog.title("Clasa")
        dialog.transient(self)
        dialog.grab_set()

        customtkinter.CTkLabel(
            dialog, text="Scrieți denumirea clasei:", font=("Roboto", 16)
        ).pack(pady=(25, 10), padx=20, anchor="w")
        
        entry_clasa = customtkinter.CTkEntry(
            dialog, width=410, font=("Roboto", 14)
        )
        entry_clasa.pack(pady=10, padx=20)
        entry_clasa.focus_set() 

        frame_butoane = customtkinter.CTkFrame(dialog, fg_color="transparent")
        frame_butoane.pack(pady=(15, 0), padx=20, fill="x")

        def pe_ok(event=None):
            nume = entry_clasa.get().strip()
            if nume:
                if nume.lower() == "all" or nume.lower() in [
                    c.lower() for c in self.clase_existente
                ]:
                    self.arata_avertisment_duplicat(nume)
                else:
                    self.clase_existente.append(nume)
                    self.creeaza_fila_clasa(nume)
                    dialog.destroy()
            else:
                dialog.destroy()

        dialog.bind("<Return>", pe_ok)
        entry_clasa.bind("<Return>", pe_ok)

        customtkinter.CTkButton(
            frame_butoane,
            text="Cancel",
            width=100,
            fg_color="transparent",
            border_width=1,
            command=dialog.destroy,
        ).pack(side="right", padx=(10, 0))
        
        customtkinter.CTkButton(
            frame_butoane, text="Ok", width=100, command=pe_ok
        ).pack(side="right")

    def filtreaza_dupa_clasa(self, nume_clasa):
        self.clasa_selectata = nume_clasa
        if nume_clasa is None:
            self.btn_toate_clasele.configure(fg_color="#1f538d")
        else:
            self.btn_toate_clasele.configure(fg_color="#2b2b2b")

        for widget in self.tab_scroll_frame.winfo_children():
            if hasattr(widget, "nume_clasa"):
                widget.configure(
                    fg_color=(
                        "#1f538d"
                        if widget.nume_clasa == nume_clasa
                        else "#2b2b2b"
                    )
                )
            elif (
                isinstance(widget, customtkinter.CTkButton)
                and widget != self.btn_toate_clasele
            ):
                widget.configure(
                    fg_color=(
                        "#1f538d"
                        if widget.cget("text") == nume_clasa
                        else "#2b2b2b"
                    )
                )

        self.incarca_imagini()

    def creeaza_fila_clasa(self, nume):
        if nume.lower() == "all":
            return
        if nume not in self.clase_existente:
            self.clase_existente.append(nume)

        for widget in self.tab_scroll_frame.winfo_children():
            if getattr(widget, "nume_clasa", None) == nume:
                return

        tab_frame = customtkinter.CTkFrame(
            master=self.tab_scroll_frame,
            fg_color="#2b2b2b",
            corner_radius=6,
            cursor="hand2"
        )
        tab_frame.nume_clasa = nume
        tab_frame.pack(side="left", padx=4)

        def pe_click_tab(event=None):
            self.filtreaza_dupa_clasa(nume)

        def pe_click_x(event=None):
            self.sterge_clasa(nume)

        lbl_nume = customtkinter.CTkLabel(
            master=tab_frame,
            text=nume,
            width=58, 
            height=32, 
            fg_color="transparent",
            text_color="#DCE4EE",
            font=("Roboto", 13),
            cursor="hand2"
        )
        lbl_nume.pack(side="left", fill="both", expand=True, padx=(10, 2))

        lbl_x = customtkinter.CTkLabel(
            master=tab_frame,
            text="✕",
            width=22,
            height=32,
            fg_color="transparent",
            text_color="#aaaaaa",
            font=("Roboto", 14, "bold"),
            cursor="hand2"
        )
        lbl_x.pack(side="right", fill="y", padx=(0, 8))

        for element in (tab_frame, lbl_nume, lbl_x):
            if hasattr(element, "_canvas"):
                element._canvas.configure(cursor="hand2")

        def pe_enter_tab(event=None):
            if getattr(self, "clasa_selectata", None) != nume:
                tab_frame.configure(fg_color="#3b3b3b")

        def pe_leave_tab(event=None):
            if getattr(self, "clasa_selectata", None) != nume:
                tab_frame.configure(fg_color="#2b2b2b")

        def pe_enter_x(event=None):
            pe_enter_tab()
            lbl_x.configure(text_color="#e0e0e0")

        def pe_leave_x(event=None):
            pe_leave_tab()
            lbl_x.configure(text_color="#aaaaaa")

        for element in (tab_frame, lbl_nume):
            element.bind("<Enter>", pe_enter_tab)
            element.bind("<Leave>", pe_leave_tab)
            element.bind("<Button-1>", pe_click_tab)
            if hasattr(element, "_canvas"):
                element._canvas.bind("<Enter>", pe_enter_tab)
                element._canvas.bind("<Leave>", pe_leave_tab)
                element._canvas.bind("<Button-1>", pe_click_tab)

        lbl_x.bind("<Enter>", pe_enter_x)
        lbl_x.bind("<Leave>", pe_leave_x)
        lbl_x.bind("<Button-1>", pe_click_x)
        if hasattr(lbl_x, "_canvas"):
            lbl_x._canvas.bind("<Enter>", pe_enter_x)
            lbl_x._canvas.bind("<Leave>", pe_leave_x)
            lbl_x._canvas.bind("<Button-1>", pe_click_x)

        self._bind_scroll_la_widget(tab_frame, self._pe_rotire_tabs)
        self._bind_scroll_la_widget(lbl_nume, self._pe_rotire_tabs)
        self._bind_scroll_la_widget(lbl_x, self._pe_rotire_tabs)

        if nume not in self.snipper.classes:
            self.snipper.classes.append(nume)
            self.snipper.save_classes()

    def sterge_clasa(self, nume_clasa):
        cale_imagini = os.path.join(self.MAPA_IMAGINI, nume_clasa)
        cale_labels = os.path.join(self.MAPA_LABELS, nume_clasa)
        
        ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if os.path.exists(cale_imagini):
            for root, _, files in os.walk(cale_imagini):
                for file in files:
                    if file.lower().endswith(ext):
                        src_img = os.path.join(root, file)
                        dst_img = os.path.join(self.MAPA_ALL, file)
                        if os.path.exists(dst_img):
                            b, ext_f = os.path.splitext(file)
                            dst_img = os.path.join(self.MAPA_ALL, f"{b}_{int(time.time())}{ext_f}")
                        shutil.move(src_img, dst_img)
            shutil.rmtree(cale_imagini, ignore_errors=True)
            
        if os.path.exists(cale_labels):
            shutil.rmtree(cale_labels, ignore_errors=True)

        if nume_clasa in self.clase_existente:
            self.clase_existente.remove(nume_clasa)
        if hasattr(self, "snipper") and hasattr(self.snipper, "classes"):
            if nume_clasa in self.snipper.classes:
                self.snipper.classes.remove(nume_clasa)
                self.snipper.save_classes()

        for widget in self.tab_scroll_frame.winfo_children():
            if getattr(widget, "nume_clasa", None) == nume_clasa:
                widget.destroy()
                break

        keys_to_del = [
            k for k in self.thumbnail_cache.keys() if nume_clasa in k
        ]
        for k in keys_to_del:
            del self.thumbnail_cache[k]

        if self.clasa_selectata == nume_clasa:
            self.filtreaza_dupa_clasa(None)
        else:
            self.incarca_imagini()

    def _pe_enter_imagine(self, event, cale_imagine):
        if self.timer_previzualizare is not None:
            self.after_cancel(self.timer_previzualizare)
        self.timer_previzualizare = self.after(
            500,
            lambda: self._arata_previzualizare_hover(
                cale_imagine, event.x_root, event.y_root
            ),
        )

    def _pe_leave_imagine(self, event):
        if self.timer_previzualizare is not None:
            self.after_cancel(self.timer_previzualizare)
            self.timer_previzualizare = None
        if self.fereastra_previzualizare is not None:
            self.fereastra_previzualizare.destroy()
            self.fereastra_previzualizare = None

    def _arata_previzualizare_hover(self, cale_imagine, x_root, y_root):
        if self.fereastra_previzualizare is not None:
            self.fereastra_previzualizare.destroy()
        self.fereastra_previzualizare = customtkinter.CTkToplevel(self)
        self.fereastra_previzualizare.overrideredirect(True)
        self.fereastra_previzualizare.attributes("-alpha", 0.95)
        self.fereastra_previzualizare.geometry(
            f"+{x_root + 15}+{y_root + 15}"
        )

        try:
            img = Image.open(cale_imagine).convert("RGBA")
            img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            img_ctk = customtkinter.CTkImage(
                light_image=img, dark_image=img, size=img.size
            )
            customtkinter.CTkLabel(
                self.fereastra_previzualizare, text="", image=img_ctk
            ).pack(padx=5, pady=5)
        except Exception:
            if self.fereastra_previzualizare:
                self.fereastra_previzualizare.destroy()

    def sterge_imagine(
        self, calea_imagine, nume_fara_extensie, nume_clasa_efectiva
    ):
        try:
            if os.path.exists(calea_imagine):
                os.remove(calea_imagine)
            if nume_clasa_efectiva and nume_clasa_efectiva.lower() != "all":
                cale_txt = os.path.join(
                    self.DIRECTOR_CURENT,
                    "labels",
                    nume_clasa_efectiva,
                    nume_fara_extensie + ".txt",
                )
                if os.path.exists(cale_txt):
                    os.remove(cale_txt)
            else:
                for c_ex in self.clase_existente:
                    cale_txt = os.path.join(
                        self.DIRECTOR_CURENT,
                        "labels",
                        c_ex,
                        nume_fara_extensie + ".txt",
                    )
                    if os.path.exists(cale_txt):
                        os.remove(cale_txt)
                        break
            
            keys_to_del = [k for k in self.thumbnail_cache.keys() if calea_imagine in k]
            for k in keys_to_del:
                del self.thumbnail_cache[k]
                
            self.incarca_imagini()
        except Exception as e:
            print(f"Eroare stergere img: {e}")

    def scoate_din_clasa(self, calea_imagine, nume_fara_extensie, nume_clasa_efectiva):
        try:
            if nume_clasa_efectiva and nume_clasa_efectiva.lower() != "all":
                cale_txt = os.path.join(
                    self.DIRECTOR_CURENT,
                    "labels",
                    nume_clasa_efectiva,
                    nume_fara_extensie + ".txt",
                )
                if os.path.exists(cale_txt):
                    os.remove(cale_txt)
            else:
                for c_ex in self.clase_existente:
                    cale_txt = os.path.join(
                        self.DIRECTOR_CURENT,
                        "labels",
                        c_ex,
                        nume_fara_extensie + ".txt",
                    )
                    if os.path.exists(cale_txt):
                        os.remove(cale_txt)
                        break

            dir_parinte = os.path.dirname(calea_imagine)
            nume_folder_parinte = os.path.basename(dir_parinte)
            if nume_folder_parinte.lower() != "all":
                destinatie_all = os.path.join(self.MAPA_ALL, os.path.basename(calea_imagine))
                if os.path.exists(calea_imagine):
                    if os.path.exists(destinatie_all):
                        b, ext_f = os.path.splitext(os.path.basename(calea_imagine))
                        destinatie_all = os.path.join(self.MAPA_ALL, f"{b}_{int(time.time())}{ext_f}")
                    shutil.move(calea_imagine, destinatie_all)

            keys_to_del = [k for k in self.thumbnail_cache.keys() if calea_imagine in k]
            for k in keys_to_del:
                del self.thumbnail_cache[k]

            self.incarca_imagini()
        except Exception as e:
            print(f"Eroare la scoaterea imaginii din clasă: {e}")

    def curata_imagini_neadnotate(self):
        ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if os.path.exists(self.MAPA_LABELS):
            for r, _, fs in os.walk(self.MAPA_LABELS):
                for f in fs:
                    if f.endswith(".txt"):
                        cale_t = os.path.join(r, f)
                        with open(cale_t, "r", encoding="utf-8") as file:
                            if not file.read().strip():
                                os.remove(cale_t)

        if os.path.exists(self.MAPA_IMAGINI):
            for nume_folder in os.listdir(self.MAPA_IMAGINI):
                cale_f = os.path.join(self.MAPA_IMAGINI, nume_folder)
                if os.path.isdir(cale_f) and nume_folder.lower() != "all":
                    for root, _, files in os.walk(cale_f):
                        for fisier in files:
                            if fisier.lower().endswith(ext):
                                c_img = os.path.join(root, fisier)
                                n_fara, _ = os.path.splitext(fisier)
                                c_txt = os.path.join(
                                    self.MAPA_LABELS,
                                    nume_folder,
                                    n_fara + ".txt",
                                )
                                if not os.path.exists(c_txt):
                                    shutil.move(
                                        c_img,
                                        os.path.join(self.MAPA_ALL, fisier),
                                    )

    # ==========================================
    # LOGICĂ NOUĂ ȘI RAPIDĂ PENTRU ÎNCĂRCARE
    # ==========================================
    def incarca_imagini(self):
        if not os.path.exists(self.MAPA_IMAGINI):
            return

        self.load_token += 1
        current_token = self.load_token

        for widget in self.galerie_imagini.winfo_children():
            widget.destroy()
            
        lbl_incarcare = customtkinter.CTkLabel(
            self.galerie_imagini, text="Se scanează rapid imaginile de pe disc...", font=("Roboto", 16)
        )
        lbl_incarcare.pack(pady=40)

        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")

        def proc_scanare():
            if current_token != self.load_token:
                return
            
            self.curata_imagini_neadnotate()

            ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            lista = []
            foldere = (
                [self.MAPA_ALL]
                if self.clasa_selectata is None
                else [os.path.join(self.MAPA_IMAGINI, self.clasa_selectata)]
            )

            if self.clasa_selectata is None:
                for n_f in os.listdir(self.MAPA_IMAGINI):
                    c_f = os.path.join(self.MAPA_IMAGINI, n_f)
                    if os.path.isdir(c_f) and n_f.lower() != "all":
                        foldere.append(c_f)

            for c_f in foldere:
                if not os.path.exists(c_f):
                    continue
                n_c = os.path.basename(c_f)
                is_all = n_c.lower() == "all"

                for root, _, files in os.walk(c_f):
                    for f_name in files:
                        if f_name.lower().endswith(ext):
                            cale_c = os.path.join(root, f_name)
                            n_fara, _ = os.path.splitext(f_name)
                            n_clasa_ef = "all"
                            este_ad = False

                            if not is_all:
                                n_clasa_ef = n_c
                                este_ad = os.path.exists(
                                    os.path.join(self.DIRECTOR_CURENT, "labels", n_c, n_fara + ".txt")
                                )
                            else:
                                for c_ex in self.clase_existente:
                                    if os.path.exists(
                                        os.path.join(self.DIRECTOR_CURENT, "labels", c_ex, n_fara + ".txt")
                                    ):
                                        este_ad = True
                                        n_clasa_ef = c_ex
                                        break
                            lista.append((cale_c, f_name, n_fara, n_clasa_ef, este_ad))

            if current_token == self.load_token:
                self.after(0, self._finalizeaza_scanarea, lista, current_token)

        threading.Thread(target=proc_scanare, daemon=True).start()

    def _finalizeaza_scanarea(self, lista_imagini, token):
        if token != self.load_token:
            return
        
        self.lista_date_imagini = lista_imagini
        self.pagina_curenta = 0
        self.afiseaza_pagina_curenta(token)

    def afiseaza_pagina_curenta(self, token):
        if token != self.load_token:
            return
            
        for widget in self.galerie_imagini.winfo_children():
            widget.destroy()

        total = len(self.lista_date_imagini)
        pagini_totale = max(1, (total + self.imagini_per_pagina - 1) // self.imagini_per_pagina)
        
        self.lbl_paginare.configure(text=f"Pagină: {self.pagina_curenta + 1} / {pagini_totale} (Total: {total})")
        self.btn_prev.configure(state="normal" if self.pagina_curenta > 0 else "disabled")
        self.btn_next.configure(state="normal" if self.pagina_curenta < pagini_totale - 1 else "disabled")

        if total == 0:
            lbl = customtkinter.CTkLabel(self.galerie_imagini, text="Nu există imagini în această categorie.", font=("Roboto", 16))
            lbl.pack(pady=40)
            return

        start_idx = self.pagina_curenta * self.imagini_per_pagina
        end_idx = min(start_idx + self.imagini_per_pagina, total)
        imagini_de_procesat = self.lista_date_imagini[start_idx:end_idx]

        lbl_loading = customtkinter.CTkLabel(self.galerie_imagini, text="Se afișează pagina...", font=("Roboto", 16))
        lbl_loading.pack(pady=40)

        def proceseaza_chunk():
            if token != self.load_token: return
            rezultate_ui = []
            c, r, max_c = 0, 0, 6
            
            for cale_c, f_name, n_fara, n_clasa_ef, este_ad in imagini_de_procesat:
                if token != self.load_token: return
                try:
                    deseneaza_triunghi = (self.clasa_selectata is None and este_ad)
                    cache_key = f"{cale_c}_triunghi_{deseneaza_triunghi}"

                    if cache_key in self.thumbnail_cache:
                        img_n, img_hx, img_ht = self.thumbnail_cache[cache_key]
                    else:
                        orig = Image.open(cale_c)
                        # Optimizare: thumbnail este mult mai rapid decât resize!
                        orig.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        
                        if orig.mode in ("RGBA", "LA") or (orig.mode == "P" and "transparency" in orig.info):
                            bg = Image.new("RGB", orig.size, "#2b2b2b")
                            if "transparency" in orig.info:
                                bg.paste(orig, mask=orig.convert("RGBA").split()[3])
                            else:
                                bg.paste(orig, mask=orig.split()[3] if orig.mode == "RGBA" else orig.split()[1])
                            pil_i = bg
                        else:
                            pil_i = orig.convert("RGB")

                        pil_n = pil_i.copy()
                        pil_hx = pil_i.copy()
                        pil_ht = pil_i.copy()

                        if deseneaza_triunghi:
                            d_n = ImageDraw.Draw(pil_n)
                            d_n.polygon([(149, 0), (119, 0), (149, 30)], fill="#1f538d")
                            d_hx = ImageDraw.Draw(pil_hx)
                            d_hx.polygon([(149, 0), (119, 0), (149, 30)], fill="#1f538d")
                            d_ht = ImageDraw.Draw(pil_ht)
                            d_ht.polygon([(149, 0), (119, 0), (149, 30)], fill="#3182ce")

                        def draw_x(im, c_x):
                            dr = ImageDraw.Draw(im)
                            dr.line([(5, 5), (13, 13)], fill="#000000", width=3)
                            dr.line([(5, 13), (13, 5)], fill="#000000", width=3)
                            dr.line([(5, 5), (13, 13)], fill=c_x, width=1)
                            dr.line([(5, 13), (13, 5)], fill=c_x, width=1)

                        draw_x(pil_n, "#888888")
                        draw_x(pil_hx, "#e0e0e0")
                        draw_x(pil_ht, "#888888")

                        img_n = customtkinter.CTkImage(light_image=pil_n, dark_image=pil_n, size=(150, 150))
                        img_hx = customtkinter.CTkImage(light_image=pil_hx, dark_image=pil_hx, size=(150, 150))
                        img_ht = customtkinter.CTkImage(light_image=pil_ht, dark_image=pil_ht, size=(150, 150))
                        
                        self.thumbnail_cache[cache_key] = (img_n, img_hx, img_ht)

                    n_scurt = f_name if len(f_name) < 15 else f_name[:12] + "..."
                    t_et = f"[{n_clasa_ef.upper()}]\n{n_scurt}" if self.clasa_selectata is None else n_scurt

                    rezultate_ui.append((c, r, img_n, img_hx, img_ht, cale_c, n_fara, n_clasa_ef, t_et, deseneaza_triunghi))
                    
                    c += 1
                    if c >= max_c:
                        c = 0
                        r += 1
                except Exception as e:
                    print(f"Eroare procesare thumbnail {cale_c}: {e}")
            
            if token == self.load_token:
                self.after(0, self._randeaza_UI_final, rezultate_ui, lbl_loading, token)

        threading.Thread(target=proceseaza_chunk, daemon=True).start()

    def _randeaza_UI_final(self, rezultate_ui, lbl_loading, token):
        if token != self.load_token: return
        
        lbl_loading.destroy()
        self.imagini_salvate.clear()
        
        for item in rezultate_ui:
            cc, rc, imn, imhx, imht, pac, nfe, nce, tet, deseneaza_triunghi = item
            self.imagini_salvate.extend([imn, imhx, imht])
            try:
                cadru = customtkinter.CTkFrame(self.galerie_imagini, fg_color="transparent", corner_radius=0)
                cadru.grid(row=rc, column=cc, padx=15, pady=15)
                cont = customtkinter.CTkFrame(cadru, fg_color="transparent")
                cont.pack()
                ico = customtkinter.CTkLabel(cont, text="", image=imn, cursor="hand2")
                ico.pack()
                lbl = customtkinter.CTkLabel(cont, text=tet, font=("Roboto", 12), cursor="hand2")
                lbl.pack(pady=(4, 0))

                self._bind_scroll_la_widget(cadru, self._pe_rotire_galerie)
                self._bind_scroll_la_widget(cont, self._pe_rotire_galerie)
                self._bind_scroll_la_widget(ico, self._pe_rotire_galerie)
                self._bind_scroll_la_widget(lbl, self._pe_rotire_galerie)

                stare_hover = [0]
                def m(e, im_n=imn, im_hx=imhx, im_ht=imht, dt=deseneaza_triunghi, ic=ico):
                    in_x = e.x <= 20 and e.y <= 20
                    in_triangle = dt and (e.x >= 119 and e.y >= 0 and e.y <= (e.x - 119))
                    if in_x:
                        if stare_hover[0] != 1: stare_hover[0] = 1; ic.configure(image=im_hx)
                    elif in_triangle:
                        if stare_hover[0] != 2: stare_hover[0] = 2; ic.configure(image=im_ht)
                    else:
                        if stare_hover[0] != 0: stare_hover[0] = 0; ic.configure(image=im_n)

                def l(e, im_n=imn, ic=ico):
                    stare_hover[0] = 0; ic.configure(image=im_n); self._pe_leave_imagine(e)

                def cl(e, pac_val=pac, nfe_val=nfe, nce_val=nce, dt=deseneaza_triunghi):
                    in_x = e.x <= 20 and e.y <= 20
                    in_triangle = dt and (e.x >= 119 and e.y >= 0 and e.y <= (e.x - 119))
                    if in_x: self.sterge_imagine(pac_val, nfe_val, nce_val)
                    elif in_triangle: self.scoate_din_clasa(pac_val, nfe_val, nce_val)
                    else: self.editeaza_adnotare(pac_val, nce_val)

                ico.bind("<Motion>", m)
                ico.bind("<Leave>", l)
                ico.bind("<Button-1>", cl)
                ico.bind("<Enter>", lambda e, p=pac: self._pe_enter_imagine(e, p))
            except Exception as ex:
                print(f"Eroare la randarea elementului UI: {ex}")


if __name__ == "__main__":
    customtkinter.set_appearance_mode("Dark")
    customtkinter.set_default_color_theme("blue")
    root = customtkinter.CTk()
    root.withdraw()
    app = TrainWindow(root)

    def close_app(event=None):
        root.quit()
        root.destroy()

    app.protocol("WM_DELETE_WINDOW", close_app)
    root.bind("<<CloseTrainWindow>>", close_app)
    root.mainloop()