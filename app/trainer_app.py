import hashlib
import os
import re
import sys
from pathlib import Path
import platform
import shutil
import threading
import time
from tkinter import filedialog
import urllib.parse
import urllib.request
import urllib.error
import winreg
import subprocess
import customtkinter
from PIL import Image, ImageDraw, ImageFilter, ImageGrab, UnidentifiedImageError

from realtime_detector import RealtimeCaptureWindow

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

from annotation_tool import SnippingAnnotator


def detecteaza_hardware_sistem():
    """Detectează procesorul (CPU) și placa video 100% nativ din Registrii Windows, inclusiv generația GPU și suportul AMD ROCm."""
    
    sys_gpu_nume = ""
    gpu_nume = "GPU Nedetectat"
    gpu_suportat = False
    device_code = "cpu"

    # --- 1. DETECȚIE CPU DIN REGISTRI ---
    def obtine_cpu_scurt():
        if platform.system() == "Windows":
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                nume_raw, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                winreg.CloseKey(key)

                match_ryzen = re.search(r'Ryzen\s+\d+\s+\d+[A-Za-z0-9]*', nume_raw, re.IGNORECASE)
                if match_ryzen: return match_ryzen.group(0)

                match_intel = re.search(r'i\d[-\s]\d+[A-Za-z0-9]*', nume_raw, re.IGNORECASE)
                if match_intel: return f"Intel {match_intel.group(0)}"

                curat = re.sub(r'(AMD|Intel|\(R\)|\(TM\)|Processor|\d+-Core|\bCore\b|@.*)', '', nume_raw, flags=re.IGNORECASE)
                return ' '.join(curat.split())
            except Exception:
                return platform.processor() or "CPU Generic"
        return platform.processor() or "CPU Generic"

    # --- 2. DETECȚIE GPU DIN REGISTRI ---
    def obtine_gpu_din_registri():
        gpu_list = []
        if platform.system() == "Windows":
            try:
                key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                
                for i in range(10):
                    try:
                        subkey_name = f"{i:04d}"
                        subkey = winreg.OpenKey(key, subkey_name)
                        desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                        
                        if desc and desc not in gpu_list:
                            gpu_list.append(desc)
                        winreg.CloseKey(subkey)
                    except OSError:
                        continue 
                winreg.CloseKey(key)
                
                if gpu_list:
                    return " / ".join(gpu_list)
            except Exception:
                pass
        return ""

    cpu_nume = obtine_cpu_scurt()
    total_threads = os.cpu_count() or 4
    sys_gpu_nume = obtine_gpu_din_registri()

    # --- 3. VERIFICARE SUPORT GPU (CUDA / ROCm) ---
    try:
        if CUDA_DISPONIBIL and torch.cuda.is_available():
            try:
                nume_baza_gpu = torch.cuda.get_device_name(0)
                generatie_gpu = ""
                
                is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
                
                if is_rocm:
                    generatie_gpu = "AMD ROCm"
                else:
                    capacitate = torch.cuda.get_device_capability(0)
                    major, minor = capacitate[0], capacitate[1]
                    
                    if major == 6:
                        generatie_gpu = "Pascal"
                    elif major == 7:
                        generatie_gpu = "Turing" if minor == 5 else "Volta"
                    elif major == 8:
                        generatie_gpu = "Ada Lovelace" if minor == 9 else "Ampere"
                    elif major == 9:
                        generatie_gpu = "Hopper/Blackwell"
                    else:
                        generatie_gpu = f"Gen {major}.{minor}"

                gpu_nume = f"{nume_baza_gpu} [{generatie_gpu}]"
                gpu_suportat = True
                device_code = "cuda"
                
            except Exception:
                gpu_nume = sys_gpu_nume if sys_gpu_nume else "GPU Accelerat"
                gpu_suportat = True
                device_code = "cuda"
    except Exception:
        pass 
        
    # --- 4. FORMATĂRI FINALE ---
    if sys_gpu_nume:
        sys_gpu_nume = re.sub(r'\(R\)|\(TM\)|\(tm\)', '', sys_gpu_nume).strip()
        sys_gpu_nume = ' '.join(sys_gpu_nume.split())
    else:
        sys_gpu_nume = "Nu s-a putut citi modelul GPU-ului"

    if gpu_nume == "GPU Nedetectat" or not gpu_suportat:
        gpu_nume = sys_gpu_nume

    return cpu_nume, total_threads, gpu_nume, gpu_suportat, device_code, sys_gpu_nume


def colecteaza_date_antrenare(
    base_dir=".",
    clase_selectate=None,
    extensii_imagini={".jpg", ".jpeg", ".png", ".webp", ".bmp"},
):
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

    print("📊 Sumar scanare:")
    print(f"   - Imagini găsite în total: {numar_imagini_totale}")
    print(f"   - Imagini valide găsite: {len(perechi_gasite)}")

    return perechi_gasite


class TrainWorker(threading.Thread):

    def __init__(
        self,
        model_ales,
        nume_model_salvat_user,
        marime_model,
        dispozitiv,
        device_code,
        clase_selectate,
        director_curent,
        cpu_threads,
        batch_size,
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
        self.marime_model = marime_model
        self.dispozitiv = dispozitiv
        self.device_code = device_code
        self.clase_selectate = clase_selectate
        self.director_curent = director_curent
        self.cpu_threads = int(cpu_threads) if cpu_threads else 4
        self.batch_size = int(batch_size) if batch_size else 16
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
        # Presupunem că variabila 'nume_clasa' conține numele original
        
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

            # Obținem calea către directorul părinte (un pas în urmă)
            director_parinte = os.path.dirname(self.director_curent)
            cale_models_dir = os.path.join(director_parinte, "models")

            # Creează folderul 'YOLO models' în directorul părinte dacă nu există
            os.makedirs(cale_models_dir, exist_ok=True)

            if self.model_ales == "Model YOLO neantrenat":
                nume_model_baza = f"yolov8{self.marime_model}.pt"
                cale_model_baza = os.path.join(cale_models_dir, nume_model_baza)

                if os.path.exists(cale_model_baza):
                    print(f"✅ Model de bază găsit în: {cale_model_baza}")
                    model = YOLO(cale_model_baza)
                else:
                    print(f"⬇️ Modelul de bază {nume_model_baza} nu există. Se descarcă...")
                    model = YOLO(nume_model_baza)

                    # Dacă biblioteca l-a descărcat local, îl mutăm în folderul din directorul părinte
                    if os.path.exists(nume_model_baza):
                        shutil.move(nume_model_baza, cale_model_baza)
                        print(f"📁 Modelul descărcat a fost salvat în: {cale_model_baza}")
                    elif os.path.exists(os.path.join(self.director_curent, nume_model_baza)):
                        shutil.move(os.path.join(self.director_curent, nume_model_baza), cale_model_baza)
                        print(f"📁 Modelul descărcat a fost salvat în: {cale_model_baza}")
            else:
                # Caută modelul antrenat tot în folderul 'YOLO models' din directorul părinte
                model_path = os.path.join(cale_models_dir, self.model_ales)
                model = YOLO(model_path)

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
            device_str = self.device_code if hasattr(self, "device_code") and self.device_code else "cpu"

            timp_start = time.time()
            model.train(
                data=yaml_path,
                epochs=self.epoci,
                imgsz=self.imgsz,
                lr0=self.lr,
                batch=self.batch_size,
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
                        
                        if nume_curat.lower().endswith(".pt"):
                             nume_curat = nume_curat[:-3]
                        
                        sufix = f"_{self.marime_model}"
                        if not nume_curat.endswith(sufix):
                            nume_curat += sufix
                            
                        nume_model_salvat = nume_curat + ".pt"
                    else:
                        nume_model_salvat = self.model_ales

                    destinatie_finala = os.path.join(director_parinte, "models", nume_model_salvat)
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
        self.MAPA_MODELE = os.path.join(os.path.dirname(self.DIRECTOR_CURENT), "models")
        self.var_nume_model_nou = customtkinter.StringVar(value="")
        self.var_marime_model = customtkinter.StringVar(value="n")
        self.var_doar_neadnotate = customtkinter.BooleanVar(value=False)

        os.makedirs(self.MAPA_ALL, exist_ok=True)
        os.makedirs(self.MAPA_LABELS, exist_ok=True)
        os.makedirs(self.MAPA_MODELE, exist_ok=True)

        self.CACHE_DIR = os.path.join(self.DIRECTOR_CURENT, ".cache_miniaturi")
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.pool_widgeturi = []

        self.imagini_salvate = []
        self.clase_existente = []
        self.thumbnail_cache = {}
        self.load_token = 0

        self.imagini_per_pagina = 100
        self.pagina_curenta = 0
        self.lista_date_imagini = []

        self.mod_stergere_multipla = False
        self.imagini_selectate_stergere = set()
        self.bind("<Return>", self._pe_enter_apasat)
        self.bind("<Unmap>", self._ascunde_modale_la_minimizare)
        self.bind("<Configure>", self._urmareste_fereastra_principala)
        self.bind("<Map>", self._arata_modale_la_restaurare)
        self.vars_clase = {}
        self.var_all_clase = customtkinter.BooleanVar(value=True)
        self.var_model_selectat = customtkinter.StringVar(
            value="Model YOLO neantrenat"
        )
        self.var_nume_model_nou = customtkinter.StringVar(value="")

        cpu_nume, threads, gpu_nume, gpu_suportat, device_code, sys_gpu_nume = detecteaza_hardware_sistem()
        self.total_cpu_threads = threads
        self.cpu_disponibil_nume = f"CPU: {cpu_nume} ({threads} Th)"
        self.gpu_device_code = device_code
        self.sys_gpu_nume = sys_gpu_nume
        self.gpu_suportat = gpu_suportat

        if gpu_suportat:
            self.gpu_disponibil_nume = f"GPU: {gpu_nume}"
            dispozitiv_implicit = self.gpu_disponibil_nume
        else:
            self.gpu_disponibil_nume = f"GPU: {sys_gpu_nume}" if sys_gpu_nume else "GPU Nedetectat"
            dispozitiv_implicit = self.cpu_disponibil_nume

        self.var_dispozitiv = customtkinter.StringVar(value=dispozitiv_implicit)
        
        self.var_fp16 = customtkinter.BooleanVar(value=True)
        self.var_sterge_dupa = customtkinter.BooleanVar(value=False)
        
        self.entry_cpu_threads = None
        self.entry_gpu_batch = None

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
    def _pe_rotire_scroll_vertical(self, event, scroll_frame):
        canvas = getattr(scroll_frame, "_parent_canvas", None)
        if not canvas:
            return
        self.update_idletasks()
        self.update_idletasks() # Esențial pentru a lua noile limite când apar clase noi
        bbox = canvas.bbox("all")
        
        # Dacă nu există conținut deloc, ieșim fără să dăm crash
        if not bbox:
            return "break"
            
        inaltime_continut = bbox[3] - bbox[1]
        
        # Dacă tot conținutul încape, resetăm frumos la 0 și oprim scroll-ul
        if inaltime_continut <= canvas.winfo_height():
            canvas.yview_moveto(0.0)
            return "break"

        top, bottom = canvas.yview()

        cantitate = (
            -1 * int(event.delta / 120)
            if event.delta
            else (1 if event.num == 5 else -1)
        )

        dimensiune_vizibila = bottom - top
        max_top = 1.0 - dimensiune_vizibila

        # Prevenim derularea peste limite (sus/jos)
        if cantitate < 0 and top <= 0.0:
            canvas.yview_moveto(0.0)
            return "break"

        if cantitate > 0 and bottom >= 1.0:
            canvas.yview_moveto(max_top)
            return "break"

        canvas.yview_scroll(cantitate * 67, "units")
            
        return "break"
    def _ascunde_modale_la_minimizare(self, event):
        if str(event.widget) == str(self):
            if hasattr(self, 'modal_overlay') and self.modal_overlay is not None and self.modal_overlay.winfo_exists():
                self.modal_overlay.withdraw()
            if hasattr(self, 'modal_cuda') and self.modal_cuda is not None and self.modal_cuda.winfo_exists():
                self.modal_cuda.withdraw()

    def _urmareste_fereastra_principala(self, event):
        if str(event.widget) == str(self):
            geo = self._obtine_geometrie_scalata()
            
            if hasattr(self, 'modal_overlay') and self.modal_overlay is not None and self.modal_overlay.winfo_exists():
                self.modal_overlay.geometry(geo)
                self.modal_overlay.lift()
                
            if hasattr(self, 'modal_cuda') and self.modal_cuda is not None and self.modal_cuda.winfo_exists():
                self.modal_cuda.geometry(geo)
                self.modal_cuda.lift()
                
            if hasattr(self, 'modal_stergere') and self.modal_stergere is not None and self.modal_stergere.winfo_exists():
                self.modal_stergere.geometry(geo)
                self.modal_stergere.lift()
    def _arata_modale_la_restaurare(self, event):
        if str(event.widget) == str(self):
            if hasattr(self, 'modal_overlay') and self.modal_overlay is not None and self.modal_overlay.winfo_exists():
                self.modal_overlay.deiconify()
                self.modal_overlay.lift()
            if hasattr(self, 'modal_cuda') and self.modal_cuda is not None and self.modal_cuda.winfo_exists():
                self.modal_cuda.deiconify()
                self.modal_cuda.lift()

    def rearanjeaza_grila_galerie(self, event=None):
        latime_element = 220 
        latime_disponibila = self.galerie_imagini.winfo_width()
    
        if latime_disponibila <= 10:
            return
        
        nr_coloane = max(1, latime_disponibila // latime_element)
    
        if getattr(self, "nr_coloane_curent", 0) != nr_coloane:
            self.nr_coloane_curent = nr_coloane
            widgeturi_afisate = self.galerie_imagini.winfo_children()
        
        for index, widget in enumerate(widgeturi_afisate):
            rand = index // nr_coloane
            coloana = index % nr_coloane
            widget.grid(row=rand, column=coloana, padx=10, pady=10, sticky="n")

    def obtine_versiune_cuda_recomandata(self, sys_gpu_nume):
        nume_mic = sys_gpu_nume.lower()
        if any(x in nume_mic for x in ["rtx 50"]):
            return "Blackwell (Seria RTX 5000)", "CUDA 12.9", "Generație nouă"
        elif any(x in nume_mic for x in ["rtx 40"]):
            return "Ada Lovelace (Seria RTX 4000)", "CUDA 12.1 sau 12.4", "Generație curentă"
        elif any(x in nume_mic for x in ["rtx 30"]):
            return "Ampere (Seria RTX 3000)", "CUDA 11.8, 12.1, 12.4", "Suportat"
        elif any(x in nume_mic for x in ["gtx 16", "rtx 20"]):
            return "Turing (Seria GTX 1600 / RTX 2000)", "CUDA 11.8", "Suportat"
        elif any(x in nume_mic for x in ["gtx 10"]):
            return "Pascal (Seria GTX 1000)", "CUDA 11.8", "Suportat legacy"
        elif any(x in nume_mic for x in ["gtx 6", "gtx 7", "gtx 9"]):
            return "Kepler & Maxwell (Seriile 600, 700, 900)", "CUDA 10.2 (sau mai vechi)", "Suportat vechi"
        return "Generație Necunoscută", "CUDA 11.8", "Versiune de siguranță fallback"
    def initiaza_instalare_cuda(self):
        generatie, versiune_cuda, status = self.obtine_versiune_cuda_recomandata(self.sys_gpu_nume)
        
        geo = self._obtine_geometrie_scalata()

        self.modal_cuda = customtkinter.CTkToplevel(self)
        self.modal_cuda.overrideredirect(True)
        self.modal_cuda.geometry(geo)
        self.modal_cuda.transient(self)
        self.modal_cuda.grab_set()

        self.modal_cuda.attributes("-alpha", 0.85)
        self.modal_cuda.configure(fg_color="#0D0D0D")

        card = customtkinter.CTkFrame(
            self.modal_cuda, corner_radius=16, fg_color="#1E1E1E", border_width=2, border_color="#333333"
        )
        card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.48, relheight=0.45)

        customtkinter.CTkLabel(card, text="Instalare Drivere CUDA", font=("Roboto", 22, "bold")).pack(pady=(20, 10))
        
        info_text = f"S-a detectat: {self.sys_gpu_nume}\nGenerație: {generatie}\nVersiune CUDA necesară: {versiune_cuda}\nStatus: {status}"
        customtkinter.CTkLabel(card, text=info_text, font=("Roboto", 14), justify="center").pack(pady=10)

        self.cuda_progress = customtkinter.CTkProgressBar(card, width=350, height=20, corner_radius=8)
        self.cuda_progress.set(0.0)
        self.cuda_progress.pack(pady=15)
        
        self.lbl_cuda_status = customtkinter.CTkLabel(card, text="Se pregătește descărcarea...", font=("Roboto", 14))
        self.lbl_cuda_status.pack(pady=5)

        self.cancel_cuda_event = threading.Event()

        def anuleaza_instalarea():
            self.cancel_cuda_event.set()
            self.modal_cuda.grab_release()
            self.modal_cuda.destroy()

        btn_cancel = customtkinter.CTkButton(
            card, text="Cancel", width=110, fg_color="#8B0000", hover_color="#A00000", command=anuleaza_instalarea
        )
        btn_cancel.pack(side="bottom", pady=25)

        threading.Thread(target=self._worker_descarcare_cuda, args=(versiune_cuda,), daemon=True).start()

    def _worker_descarcare_cuda(self, versiune_cuda):
        # 1. Mapăm versiunea CUDA din string-ul tău la indexul corect PyTorch
        torch_cu = "cu118" # Default de siguranță
        if "12.1" in versiune_cuda:
            torch_cu = "cu121"
        elif "12.4" in versiune_cuda or "12.8" in versiune_cuda:
            torch_cu = "cu124" # Versiuni noi suportate de PyTorch

        # ---> MODIFICARE 1: Folosim calea absolută către uv.exe local
        cale_uv = os.path.join(self.DIRECTOR_CURENT, "uv.exe")
        
        # Dacă cumva nu-l găsește lângă script, va încerca totuși varianta globală
        if not os.path.exists(cale_uv):
            print(f"⚠️ Avertisment: Nu am găsit uv.exe în {self.DIRECTOR_CURENT}. Încerc varianta globală.")
            cale_uv = "uv"

        # 2. Comanda folosind calea corectă
        cmd = [
    cale_uv, "pip", "install", 
    "torch", "torchvision",                     # Specifici pachetele separat
    "--index-url", f"https://download.pytorch.org/whl/{torch_cu}", # Trimiți indexul corect cu flag-ul --index-url
    "--reinstall",
    "--python", sys.executable                  # Forțează uv să folosească Python-ul curent
]

        print(f"\n[DEBUG] Se execută comanda: {' '.join(cmd)}\n")

        try:
            creation_flags = 0
            if platform.system() == "Windows":
                # ---> MODIFICARE 2: Poți lăsa linia de mai jos comentată dacă vrei să 
                # îți apară și fereastra neagră de CMD peste aplicație, 
                # sau o poți lăsa activă deoarece erorile vor fi printate oricum în consola IDE-ului.
                creation_flags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                creationflags=creation_flags
            )

            # Citim output-ul în timp real
            progres_curent = 0.0

            # Citim output-ul în timp real
            for linie in process.stdout:
                # Dacă userul a apăsat Cancel
                if self.cancel_cuda_event.is_set():
                    process.terminate()
                    return
                
                linie_str = linie.strip()
                print(f"[UV TERMINAL] {linie_str}")

                # --- CALCUL PROGRES APROXIMATIV ---
                # 1. Căutăm procentaje explicite în output (ex: 45%)
                match_procent = re.search(r'(\d{1,3})%', linie_str)
                if match_procent:
                    valoare = int(match_procent.group(1)) / 100.0
                    if valoare > progres_curent:
                        progres_curent = valoare
                else:
                    # 2. Dacă nu găsim procentaj, estimăm pe baza cuvintelor cheie
                    linie_low = linie_str.lower()
                    if "resolving" in linie_low and progres_curent < 0.1:
                        progres_curent = 0.1
                    elif "downloading" in linie_low or "fetching" in linie_low:
                        # Incrementăm ușor la fiecare pachet descărcat
                        if progres_curent < 0.7:
                            progres_curent += 0.02 
                    elif "installing" in linie_low or "extracting" in linie_low:
                        if progres_curent < 0.8:
                            progres_curent = 0.8
                        elif progres_curent < 0.95:
                            progres_curent += 0.01

                # Asigurăm că nu depășim 99% până nu se finalizează totul
                progres_curent = min(progres_curent, 0.99)

                # --- ACTUALIZARE UI ---
                text_scurt = linie_str[:60]
                self.after(0, lambda p=progres_curent, t=text_scurt: [
                    self.cuda_progress.set(p) if hasattr(self, "cuda_progress") else None,
                    self.lbl_cuda_status.configure(text=f"Instalare: {int(p*100)}% | {t}")
                ])
            process.wait()

            if process.returncode == 0:
                self.after(0, self._finalizeaza_instalare_cuda)
            else:
                self.after(0, lambda: self.lbl_cuda_status.configure(text=f"Eroare la instalare (Cod: {process.returncode}).", text_color="#FF6B6B"))
                print(f"\n❌ [EROARE CRITICĂ] Comanda a eșuat cu codul {process.returncode}. Verifică log-urile de mai sus.")

        except Exception as e:
            self.after(0, lambda: self.lbl_cuda_status.configure(text=f"Eroare lansare comandă (Vezi terminalul).", text_color="#FF6B6B"))
            print(f"\n❌ [EROARE PYTHON] Nu s-a putut porni procesul: {e}\n")

    def _actualizeaza_ui_cuda(self, val_float, procent, timp_ramas):
        if hasattr(self, "cuda_progress") and self.modal_cuda.winfo_exists():
            self.cuda_progress.set(val_float)
            self.lbl_cuda_status.configure(text=f"Se descarcă... {procent}% (Rămas: ~{int(timp_ramas)}s)")

    def _finalizeaza_instalare_cuda(self, cale_installer=None):
        self.lbl_cuda_status.configure(text="Instalare completă! Aplicația se repornește automat...")
        
        # Completăm progress bar-ul la 100%
        if hasattr(self, "cuda_progress"):
            self.cuda_progress.set(1.0)
        
        # Așteptăm 2.5 secunde pentru ca utilizatorul să poată citi mesajul de succes, apoi repornim
        self.after(2500, self._executa_repornire_aplicatie)

    def _executa_repornire_aplicatie(self):
        try:
            self.modal_cuda.grab_release()
            self.modal_cuda.destroy()
        except Exception as e:
            print(f"Eroare la închiderea ferestrei modale: {e}")
        
        print("\n🔄 Se inițiază repornirea aplicației...\n")
        
        # Această comandă înlocuiește instantaneu procesul curent cu un nou proces Python identic
        os.execl(sys.executable, sys.executable, *sys.argv)
    def _pe_enter_apasat(self, event=None):
        if self.mod_stergere_multipla:
            self.toggle_stergere_multipla()

    def rotunjeste_imaginea(self, imagine_pil, raza):
        imagine_pil = imagine_pil.convert("RGBA")
        masca = Image.new("L", imagine_pil.size, 0)
        draw = ImageDraw.Draw(masca)
        draw.rounded_rectangle(
            (0, 0, imagine_pil.size[0], imagine_pil.size[1]),
            radius=raza,
            fill=255,
        )
        imagine_pil.putalpha(masca)
        return imagine_pil

    def deschide_realtime_capture(self):
        win = RealtimeCaptureWindow(
            parent=self,
            base_dir=self.DIRECTOR_CURENT,
            callback_refresh=self.incarca_imagini,
        )
        win.grab_set()

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

    def confirma_stergere_totala(self):
        if self.mod_stergere_multipla:
            self.toggle_stergere_multipla()

        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("400x180")
        dialog.title("Confirmare Ștergere Totală")
        dialog.transient(self)
        dialog.grab_set()

        lbl_intrebare = customtkinter.CTkLabel(
            dialog, 
            text="Sunteți siguri că doriți să ștergeți toate pozele?",
            font=("Roboto", 16, "bold"),
            wraplength=350
        )
        lbl_intrebare.pack(pady=(35, 25))

        frame_butoane = customtkinter.CTkFrame(dialog, fg_color="transparent")
        frame_butoane.pack(fill="x", padx=20)

        # Butonul NU
        btn_nu = customtkinter.CTkButton(
            frame_butoane,
            text="Nu",
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=dialog.destroy,
            width=120
        )
        btn_nu.pack(side="left", padx=20)

        # Butonul DA
        btn_da = customtkinter.CTkButton(
            frame_butoane,
            text="Da",
            fg_color="#8B0000",
            hover_color="#A00000",
            command=lambda: [dialog.destroy(), self.executa_stergere_totala()],
            width=120
        )
        btn_da.pack(side="right", padx=20)

    def executa_stergere_totala(self):
        self._pe_leave_imagine(None)
        
        try:
            if os.path.exists(self.MAPA_IMAGINI):
                shutil.rmtree(self.MAPA_IMAGINI, ignore_errors=True)
            if os.path.exists(self.MAPA_LABELS):
                shutil.rmtree(self.MAPA_LABELS, ignore_errors=True)
        except Exception as e:
            print(f"Eroare la ștergerea totală: {e}")

        os.makedirs(self.MAPA_ALL, exist_ok=True)
        os.makedirs(self.MAPA_LABELS, exist_ok=True)

        self.thumbnail_cache.clear()
        self.clase_existente.clear()
        
        if hasattr(self, "snipper") and hasattr(self.snipper, "classes"):
            self.snipper.classes.clear()
            self.snipper.save_classes()

        for widget in self.tab_scroll_frame.winfo_children():
            if getattr(widget, "nume_clasa", None) is not None:
                widget.destroy()

        self.clasa_selectata = None
        self.btn_toate_clasele.configure(fg_color="#5865F2")
        
        self.incarca_imagini()

    def creeaza_pagini(self):
        self.main_page = customtkinter.CTkFrame(self, fg_color="#1e1f22")

        center_wrapper = customtkinter.CTkFrame(
            self.main_page, fg_color="transparent"
        )
        center_wrapper.pack(expand=True)

        title = customtkinter.CTkLabel(
            master=center_wrapper,
            text="Meniu Antrenare",
            font=("Roboto", 40, "bold"),
            text_color="#f2f3f5",
        )
        title.pack(pady=(0, 40))

        frame_1 = customtkinter.CTkFrame(
            master=center_wrapper, fg_color="transparent"
        )
        frame_1.pack(fill="both", expand=True)

        frame_1.columnconfigure(0, weight=1)
        frame_1.columnconfigure(1, weight=1)
        frame_1.columnconfigure(2, weight=1)

        cale_logos = os.path.join(self.DIRECTOR_CURENT, "logos")
        cale_galerie = os.path.join(cale_logos, "misty mountains.jpg")
        cale_invatare = os.path.join(cale_logos, "gears.jpeg")
        cale_inapoi = os.path.join(cale_logos, "detection in real time.jpg")

        try:
            img_g_bruta = Image.open(cale_galerie)
            img_i_bruta = Image.open(cale_invatare)
            img_b_bruta = Image.open(cale_inapoi)

            img_g_bruta = img_g_bruta.resize((400, 400))
            img_i_bruta = img_i_bruta.resize((400, 400))
            img_b_bruta = img_b_bruta.resize((400, 400))

            raza_rotunjire = 16
            img_g_rotunda = self.rotunjeste_imaginea(
                img_g_bruta, raza_rotunjire
            )
            img_i_rotunda = self.rotunjeste_imaginea(
                img_i_bruta, raza_rotunjire
            )
            img_b_rotunda = self.rotunjeste_imaginea(
                img_b_bruta, raza_rotunjire
            )

            img_galerie = customtkinter.CTkImage(
                light_image=img_g_rotunda,
                dark_image=img_g_rotunda,
                size=(400, 400),
            )
            img_invatare = customtkinter.CTkImage(
                light_image=img_i_rotunda,
                dark_image=img_i_rotunda,
                size=(400, 400),
            )
            img_inapoi = customtkinter.CTkImage(
                light_image=img_b_rotunda,
                dark_image=img_b_rotunda,
                size=(400, 400),
            )

        except Exception as e:
            print(f"Imaginile pentru meniu nu au putut fi încărcate: {e}")
            img_galerie = img_invatare = img_inapoi = None

        culoare_card = "#2b2d31"

        column1 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column1.grid(row=0, column=0, sticky="nsew", padx=20)

        card1 = customtkinter.CTkFrame(
            master=column1,
            fg_color=culoare_card,
            corner_radius=12,
            border_width=0,
        )
        card1.pack(expand=True, fill="both")

        lbl_img1 = customtkinter.CTkLabel(
            master=card1,
            text="[Fără Imagine]" if not img_galerie else "",
            image=img_galerie,
        )
        lbl_img1.pack(pady=(25, 15), padx=25)

        button1 = customtkinter.CTkButton(
            master=card1,
            text="Galerie",
            height=50,
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_galerie,
        )
        button1.pack(fill="x", padx=50, pady=(20, 25))

        column2 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column2.grid(row=0, column=1, sticky="nsew", padx=20)

        card2 = customtkinter.CTkFrame(
            master=column2,
            fg_color=culoare_card,
            corner_radius=12,
            border_width=0,
        )
        card2.pack(expand=True, fill="both")

        lbl_img2 = customtkinter.CTkLabel(
            master=card2,
            text="[Fără Imagine]" if not img_invatare else "",
            image=img_invatare,
        )
        lbl_img2.pack(pady=(25, 15), padx=25)

        button2 = customtkinter.CTkButton(
            master=card2,
            text="Învățare",
            height=50,
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_invatare,
        )
        button2.pack(fill="x", padx=50, pady=(20, 25))

        column3 = customtkinter.CTkFrame(master=frame_1, fg_color="transparent")
        column3.grid(row=0, column=2, sticky="nsew", padx=20)

        card3 = customtkinter.CTkFrame(
            master=column3,
            fg_color=culoare_card,
            corner_radius=12,
            border_width=0,
        )
        card3.pack(expand=True, fill="both")

        lbl_img3 = customtkinter.CTkLabel(
            master=card3,
            text="[Fără Imagine]" if not img_inapoi else "",
            image=img_inapoi,
        )
        lbl_img3.pack(pady=(25, 15), padx=25)

        button3 = customtkinter.CTkButton(
            master=card3,
            text="Detectie LIVE",
            height=50,
            font=("Roboto", 18, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.deschide_detectie_live
        )
        button3.pack(fill="x", padx=50, pady=(20, 25))

        self.second_page = customtkinter.CTkFrame(self, fg_color="transparent")

        top_bar = customtkinter.CTkFrame(
            self.second_page, fg_color="transparent", height=40
        )
        top_bar.pack(fill="x", padx=20, pady=(10,0))

        buton_adauga_clasa = customtkinter.CTkButton(
            top_bar,
            text="+",
            width=35,
            height=35,
            corner_radius=8,
            font=("Roboto", 20, "bold"),
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.deschide_dialog_clasa,
        )
        buton_adauga_clasa.pack(side="left", padx=(0, 8), pady=(0, 6))

        self.tab_scroll_frame = customtkinter.CTkScrollableFrame(
            master=top_bar,
            orientation="horizontal",
            height=38,
            fg_color="transparent",
            scrollbar_button_color="#2b2d31",
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
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=lambda: self.filtreaza_dupa_clasa(None),
        )
        self.btn_toate_clasele.pack(side="left", padx=4)
        self._bind_scroll_la_widget(
            self.btn_toate_clasele, self._pe_rotire_tabs
        )

        lbl_info_captura = customtkinter.CTkLabel(
            self.second_page,
            text=" Pentru captura si adnotare de pe ecran apasati ctrl_shift_s",
            text_color="lightgray",
            font=("Roboto", 12)
        )
        lbl_info_captura.pack(anchor="w", padx=20, pady=(0, 0))

        bottom_bar = customtkinter.CTkFrame(
            self.second_page, fg_color="transparent"
        )
        bottom_bar.pack(fill="x", side="bottom", padx=20, pady=20)

        bottom_bar.columnconfigure(0, weight=1)
        bottom_bar.columnconfigure(1, weight=1)
        bottom_bar.columnconfigure(2, weight=1)

        frame_butoane_stanga = customtkinter.CTkFrame(bottom_bar, fg_color="transparent")
        frame_butoane_stanga.grid(row=0, column=0, sticky="w")

        buton_import_imagini = customtkinter.CTkButton(
            frame_butoane_stanga,
            text="Import Images",
            command=self.deschide_dialog_import,
            fg_color="#5865F2",
            hover_color="#4752C4",
        )
        buton_import_imagini.pack(side="left", padx=(0, 15))

        self.chk_neadnotate = customtkinter.CTkCheckBox(
            frame_butoane_stanga,
            text="Ascunde imaginile adnotate",
            variable=self.var_doar_neadnotate,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.incarca_imagini
        )
        self.chk_neadnotate.pack(side="left")

        frame_paginare = customtkinter.CTkFrame(
            bottom_bar, fg_color="transparent"
        )
        frame_paginare.grid(row=0, column=1)

        self.btn_prev = customtkinter.CTkButton(
            frame_paginare,
            text="< Înapoi",
            width=70,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.pagina_anterioara,
        )
        self.btn_prev.pack(side="left", padx=5)

        self.lbl_paginare = customtkinter.CTkLabel(
            frame_paginare, text="Pagină: 1 / 1", font=("Roboto", 14, "bold")
        )
        self.lbl_paginare.pack(side="left", padx=15)

        self.btn_next = customtkinter.CTkButton(
            frame_paginare,
            text="Înainte >",
            width=70,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.pagina_urmatoare,
        )
        self.btn_next.pack(side="left", padx=5)

        frame_butoane_dreapta = customtkinter.CTkFrame(
            bottom_bar, fg_color="transparent"
        )
        frame_butoane_dreapta.grid(row=0, column=2, sticky="e")

        self.btn_stergere_multipla = customtkinter.CTkButton(
            frame_butoane_dreapta,
            text="Ștergere Multiplă",
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.toggle_stergere_multipla,
        )
        self.btn_stergere_multipla.pack(side="left", padx=(0, 10))

        self.btn_delete_all = customtkinter.CTkButton(
            frame_butoane_dreapta,
            text="Delete All",
            fg_color="#8B0000",
            hover_color="#A00000",
            width=80,
            command=self.confirma_stergere_totala,
        )
        self.btn_delete_all.pack(side="left", padx=(0, 10))

        buton_inapoi_galerie = customtkinter.CTkButton(
            frame_butoane_dreapta,
            text="Meniu Principal",
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_principala,
        )
        buton_inapoi_galerie.pack(side="left")

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
        main_invatare_container.columnconfigure(2, weight=0)
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
        frame_mijloc_setari.grid(
            row=0, column=1, sticky="nsew", padx=10, pady=10
        )

        frame_titlu_setari = customtkinter.CTkFrame(frame_mijloc_setari, fg_color="transparent")
        frame_titlu_setari.pack(pady=(15, 5), padx=15, fill="x")

        lbl_titlu_setari = customtkinter.CTkLabel(
            frame_titlu_setari, text="Setări Antrenare", font=("Roboto", 16, "bold")
        )
        lbl_titlu_setari.pack(side="left")

        este_amd = False
        if self.sys_gpu_nume:
            nume_mic = self.sys_gpu_nume.lower()
            are_amd = "amd" in nume_mic or "radeon" in nume_mic
            are_nvidia = "nvidia" in nume_mic or "rtx" in nume_mic or "gtx" in nume_mic
            
            # Sistemul este considerat AMD (fără suport CUDA) DOAR dacă
            # are grafică AMD, dar nu are o placă NVIDIA dedicată pe lângă ea.
            if are_amd and not are_nvidia:
                este_amd = True
        deja_are_cuda = (self.gpu_device_code == "cuda")
        self.btn_instaleaza_cuda = customtkinter.CTkButton(
            frame_titlu_setari,
            text="install driver",
            width=80,          
            height=24,
            font=("Roboto", 11, "bold"),
            fg_color="gray" if (este_amd or deja_are_cuda) else "#2e7d32",
            hover_color="gray" if (este_amd or deja_are_cuda) else "#1b5e20",
            state="disabled" if (este_amd or deja_are_cuda) else "normal",
            text_color_disabled="#4d4d4d",
            command=self.initiaza_instalare_cuda
        )
        self.btn_instaleaza_cuda.pack(side="right", padx=(0, 15))
        self.tooltip_window = None
        self.tooltip_timer = None

        def enter_btn(event):
            self.tooltip_timer = self.after(1000, arata_tooltip)

        def leave_btn(event):
            if self.tooltip_timer:
                self.after_cancel(self.tooltip_timer)
                self.tooltip_timer = None
            ascunde_tooltip()

        def arata_tooltip():
            if self.tooltip_window:
                return
            x = self.btn_instaleaza_cuda.winfo_rootx() + 20
            y = self.btn_instaleaza_cuda.winfo_rooty() + 30
            
            self.tooltip_window = customtkinter.CTkToplevel(self)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.geometry(f"+{x}+{y}")
            self.tooltip_window.configure(fg_color="#2b2b2b")
            self.tooltip_window.attributes("-topmost", True)
            
            text_info_baza = (
                "Instalarea driverelor poate produce erori in driverele placi video.\n"
                "Folositi cu prudenta aceasta metoda.\n"
                "Daca dupa instalare  pc-ul da reboot, reinstalati driverele prin nvidea app.\n"
                "Daca sa instalat corect , reporniti programul."
            )
            
            if deja_are_cuda:
                text_info = "Deja aveti drivere CUDA instalate si functionale!"
            elif este_amd:
                text_info = "Indisponibil, placa AMD\n" + text_info_baza
            else:
                text_info = text_info_baza
            
            lbl_info = customtkinter.CTkLabel(
                self.tooltip_window, 
                text=text_info,
                font=("Roboto", 12),
                text_color="#FF6B6B" if not deja_are_cuda else "#4CAF50", # Verde dacă sunt instalate
                justify="left",
                wraplength=350,
                bg_color="#2b2b2b"
            )
            lbl_info.pack(padx=10, pady=10)

        def ascunde_tooltip():
            if self.tooltip_window:
                self.tooltip_window.destroy()
                self.tooltip_window = None

        self.btn_instaleaza_cuda.bind("<Enter>", enter_btn)
        self.btn_instaleaza_cuda.bind("<Leave>", leave_btn)

        self.scroll_setari_interior = customtkinter.CTkScrollableFrame(
            frame_mijloc_setari, fg_color="transparent"
        )
        self.scroll_setari_interior.pack(
            fill="both", expand=True, padx=10, pady=(0, 10)
        )

        customtkinter.CTkLabel(
            self.scroll_setari_interior,
            text="Unitate de procesare detectată:",
            font=("Roboto", 13, "bold"),
        ).pack(anchor="w", pady=(5, 3))

        frame_butoane_hw = customtkinter.CTkFrame(self.scroll_setari_interior, fg_color="transparent")
        frame_butoane_hw.pack(fill="x", pady=(0, 5))
        frame_butoane_hw.columnconfigure(0, weight=1)
        frame_butoane_hw.columnconfigure(1, weight=1)

        self.frame_dinamic_hw = customtkinter.CTkFrame(self.scroll_setari_interior, fg_color="transparent")
        self.frame_dinamic_hw.pack(fill="x", pady=(5, 5))

        def afiseaza_parametri_hw(mod):
            for w in self.frame_dinamic_hw.winfo_children():
                w.destroy()

            culoare_ghost_text = "#B0B0B0"

            if mod == "cpu":
                best_threads = max(1, self.total_cpu_threads - 2)
                customtkinter.CTkLabel(
                    self.frame_dinamic_hw,
                    text=f"CPU Threads (Fire detectate: {self.total_cpu_threads}):",
                    font=("Roboto", 13),
                ).pack(anchor="w", pady=(2, 2))
                
                self.entry_cpu_threads = customtkinter.CTkEntry(
                    self.frame_dinamic_hw,
                    placeholder_text=f"min: 1 | recomandat: {best_threads} | max: {self.total_cpu_threads}",
                    placeholder_text_color=culoare_ghost_text,
                )
                self.entry_cpu_threads.pack(fill="x", pady=(0, 12))
                self.entry_cpu_threads.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)
                self.entry_gpu_batch = None
                
            elif mod == "gpu":
                customtkinter.CTkLabel(
                    self.frame_dinamic_hw,
                    text="Dimensiune Batch GPU (Batch Size):",
                    font=("Roboto", 13),
                ).pack(anchor="w", pady=(2, 2))
                
                self.entry_gpu_batch = customtkinter.CTkEntry(
                    self.frame_dinamic_hw,
                    placeholder_text="min: 1 | recomandat: 16 | max: 128",
                    placeholder_text_color=culoare_ghost_text,
                )
                self.entry_gpu_batch.pack(fill="x", pady=(0, 12))
                self.entry_gpu_batch.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)
                self.entry_cpu_threads = None

        def set_device_cpu():
            self.var_dispozitiv.set(self.cpu_disponibil_nume)
            self.btn_cpu.configure(fg_color="#5865F2", hover_color="#4752C4")
            if self.gpu_suportat:
                self.btn_gpu.configure(fg_color="#2b2b2b")
            afiseaza_parametri_hw("cpu")

        def set_device_gpu():
            if self.gpu_suportat:
                self.var_dispozitiv.set(self.gpu_disponibil_nume)
                self.btn_gpu.configure(fg_color="#5865F2", hover_color="#4752C4")
                self.btn_cpu.configure(fg_color="#2b2b2b")
                afiseaza_parametri_hw("gpu")

        self.btn_cpu = customtkinter.CTkButton(
            frame_butoane_hw,
            text=self.cpu_disponibil_nume,
            command=set_device_cpu,
            fg_color="#5865F2" if not self.gpu_suportat else "#2b2b2b",
            hover_color="#4752C4"
        )
        self.btn_cpu.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        gpu_color = "#2b2b2b"
        gpu_state = "normal"
        gpu_text = self.gpu_disponibil_nume

        if not self.gpu_suportat:
            gpu_color = "#151515"  
            gpu_state = "disabled" 
            gpu_text = f"{self.sys_gpu_nume}\n(Incompatibil CUDA)" if self.sys_gpu_nume else "GPU Nedetectat"

        self.btn_gpu = customtkinter.CTkButton(
            frame_butoane_hw,
            text=gpu_text,
            command=set_device_gpu,
            fg_color=gpu_color,
            hover_color="#4752C4",
            state=gpu_state,
            text_color_disabled="#606060" 
        )
        self.btn_gpu.grid(row=0, column=1, padx=(5, 0), sticky="ew")

        if self.gpu_suportat:
            set_device_gpu()
        else:
            set_device_cpu()

        if not self.gpu_suportat:
            self.var_dispozitiv.set(self.cpu_disponibil_nume)

        culoare_ghost_text = "#B0B0B0"

        customtkinter.CTkLabel(
            self.scroll_setari_interior,
            text="Parametri Generali",
            font=("Roboto", 14, "bold"),
        ).pack(anchor="w", pady=(5, 8))

        customtkinter.CTkLabel(
            self.scroll_setari_interior, text="Număr Epoci:", font=("Roboto", 13)
        ).pack(anchor="w", pady=(2, 2))
        self.entry_epoci = customtkinter.CTkEntry(
            self.scroll_setari_interior,
            placeholder_text="min: 10 | recomandat: 50 - 100 | max: 1000",
            placeholder_text_color=culoare_ghost_text,
        )
        self.entry_epoci.pack(fill="x", pady=(0, 15))
        self.entry_epoci.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)

        customtkinter.CTkLabel(
            self.scroll_setari_interior,
            text="Dimensiune Imagine (ImgSz):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        self.entry_imgsz = customtkinter.CTkEntry(
            self.scroll_setari_interior,
            placeholder_text="min: 320 | recomandat: 640 | max: 1280",
            placeholder_text_color=culoare_ghost_text,
        )
        self.entry_imgsz.pack(fill="x", pady=(0, 12))
        self.entry_imgsz.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)

        customtkinter.CTkLabel(
            self.scroll_setari_interior,
            text="Rată de Învățare (LR):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        self.entry_lr = customtkinter.CTkEntry(
            self.scroll_setari_interior,
            placeholder_text="min: 0.0001 | recomandat: 0.01 | max: 0.1",
            placeholder_text_color=culoare_ghost_text,
        )
        self.entry_lr.pack(fill="x", pady=(0, 12))
        self.entry_lr.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)

        customtkinter.CTkLabel(
            self.scroll_setari_interior,
            text="Thread-uri încărcare date (Workers):",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(2, 2))
        self.entry_workers = customtkinter.CTkEntry(
            self.scroll_setari_interior,
            placeholder_text=f"min: 0 | recomandat: 4 | max: {self.total_cpu_threads}",
            placeholder_text_color=culoare_ghost_text,
        )
        self.entry_workers.pack(fill="x", pady=(0, 15))
        self.entry_workers.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)

        customtkinter.CTkCheckBox(
            self.scroll_setari_interior,
            text="Precizie Mixtă (FP16 / AMP)",
            variable=self.var_fp16,
            fg_color="#5865F2",
            hover_color="#4752C4",
            font=("Roboto", 13),
        ).pack(anchor="w", pady=(5, 8))
        customtkinter.CTkCheckBox(
            self.scroll_setari_interior,
            text="Șterge clasele respective după învățare",
            variable=self.var_sterge_dupa,
            fg_color="#5865F2",
            hover_color="#4752C4",
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
            corner_radius=6,
            height=50,
            width=220,
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
            frame_bottom_modele, fg_color="transparent",width=300
        )
        self.scroll_modele_invatare.pack(
            fill="y", expand=True, padx=15, pady=(0, 5)
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
        self.entry_nume_model.pack( fill ='x', padx=15, pady=(0, 15))
        self.entry_nume_model.bind("<KeyRelease>", self.actualizeaza_stare_buton_antrenare)
        self.frame_marime_model = customtkinter.CTkFrame(frame_bottom_modele, fg_color="transparent")
        self.frame_marime_model.pack(fill="x", padx=15, pady=(0, 15))
        self.rb_nano = customtkinter.CTkRadioButton(self.frame_marime_model, text="Nano", variable=self.var_marime_model, value="n", fg_color="#5865F2", hover_color="#4752C4")
        self.rb_nano.pack(side="left", padx=5)

        self.rb_medium = customtkinter.CTkRadioButton(self.frame_marime_model, text="Medium", variable=self.var_marime_model, value="m", fg_color="#5865F2", hover_color="#4752C4")
        self.rb_medium.pack(side="left", padx=5)

        self.rb_large = customtkinter.CTkRadioButton(self.frame_marime_model, text="Large", variable=self.var_marime_model, value="l", fg_color="#5865F2", hover_color="#4752C4")
        self.rb_large.pack(side="left", padx=5)

        bottom_bar_invatare = customtkinter.CTkFrame(
            self.third_page, fg_color="transparent"
        )
        bottom_bar_invatare.pack(fill="x", side="bottom", padx=20, pady=(0, 15))

        buton_inapoi_invatare = customtkinter.CTkButton(
            master=bottom_bar_invatare,
            text="Înapoi",
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.arata_pagina_principala,
        )
        buton_inapoi_invatare.pack(side="right")
        # BINDING PENTRU SCROLL ÎN MENIUL ÎNVĂȚARE (CLASE, SETĂRI, MODELE)
        # Folosim lambda pentru a transmite widget-ul specific către funcția generică
        
        # 1. Pentru Setări Antrenare
        self._bind_scroll_la_widget(
            self.scroll_setari_interior, 
            lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_setari_interior)
        )
        if hasattr(self.scroll_setari_interior, "_parent_canvas"):
            self._bind_scroll_la_widget(
                self.scroll_setari_interior._parent_canvas, 
                lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_setari_interior)
            )

        # 2. Pentru Modele
        self._bind_scroll_la_widget(
            self.scroll_modele_invatare, 
            lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_modele_invatare)
        )
        if hasattr(self.scroll_modele_invatare, "_parent_canvas"):
            self._bind_scroll_la_widget(
                self.scroll_modele_invatare._parent_canvas, 
                lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_modele_invatare)
            )

        # 3. Pentru Clase (opțional, ca să meargă impecabil peste tot în fereastră)
        self._bind_scroll_la_widget(
            self.scroll_clase_invatare, 
            lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_clase_invatare)
        )
        if hasattr(self.scroll_clase_invatare, "_parent_canvas"):
            self._bind_scroll_la_widget(
                self.scroll_clase_invatare._parent_canvas, 
                lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_clase_invatare)
            )
                # Pentru Tab-urile claselor (orizontal)
        if hasattr(self.tab_scroll_frame, "_parent_frame"):
            self.tab_scroll_frame._parent_frame.bind(
                "<Configure>", 
                lambda e: self._actualizeaza_stare_scrollbar(self.tab_scroll_frame, orientare="horizontal"), 
                add="+"
            )

        # Pentru Galeria de imagini (vertical)
        if hasattr(self.galerie_imagini, "_parent_frame"):
            self.galerie_imagini._parent_frame.bind(
                "<Configure>", 
                lambda e: self._actualizeaza_stare_scrollbar(self.galerie_imagini, orientare="vertical"), 
                add="+"
            )

        # Pentru meniul din interiorul paginii de Învățare (Setări)
        if hasattr(self.scroll_setari_interior, "_parent_frame"):
            self.scroll_setari_interior._parent_frame.bind(
                "<Configure>", 
                lambda e: self._actualizeaza_stare_scrollbar(self.scroll_setari_interior, orientare="vertical"), 
                add="+"
            )

        # Pentru Modele
        if hasattr(self.scroll_modele_invatare, "_parent_frame"):
            self.scroll_modele_invatare._parent_frame.bind(
                "<Configure>", 
                lambda e: self._actualizeaza_stare_scrollbar(self.scroll_modele_invatare, orientare="vertical"), 
                add="+"
            )

        if hasattr(self.scroll_clase_invatare, "_parent_frame"):
            self.scroll_clase_invatare._parent_frame.bind(
                "<Configure>", 
                lambda e: self._actualizeaza_stare_scrollbar(self.scroll_clase_invatare, orientare="vertical"), 
                add="+"
            )
    
    def toggle_stergere_multipla(self):
        self._pe_leave_imagine(None)
        if not self.mod_stergere_multipla:
            self.mod_stergere_multipla = True
            self.imagini_selectate_stergere.clear()
            self.btn_stergere_multipla.configure(
                text="Șterge aceste imagini (Enter)",
                fg_color="#8B0000",
                hover_color="#A00000",
            )
            self.focus_set()
            self.afiseaza_pagina_curenta(self.load_token)
        else:
            if self.imagini_selectate_stergere:
                self._sterge_imagini_selectate()

            self.mod_stergere_multipla = False
            self.imagini_selectate_stergere.clear()
            self.btn_stergere_multipla.configure(
                text="Ștergere Multiplă",
                fg_color="#5865F2",
                hover_color="#4752C4",
            )
            self.incarca_imagini()

    def deschide_detectie_live(self):
        self.withdraw()
        app_captura = RealtimeCaptureWindow(parent=self.master if hasattr(self, "master") else self)
        
        def la_inchidere_live():
            app_captura.destroy()
            self.deiconify()
            
        app_captura.protocol("WM_DELETE_WINDOW", la_inchidere_live)

    def _sterge_imagini_selectate(self):
        for cale in self.imagini_selectate_stergere:
            try:
                if os.path.exists(cale):
                    os.remove(cale)

                nume_fisier = os.path.basename(cale)
                nume_fara, _ = os.path.splitext(nume_fisier)

                for c_ex in self.clase_existente:
                    cale_txt = os.path.join(
                        self.DIRECTOR_CURENT, "labels", c_ex, nume_fara + ".txt"
                    )
                    if os.path.exists(cale_txt):
                        os.remove(cale_txt)
                        break

                keys_to_del = [
                    k for k in self.thumbnail_cache.keys() if cale in k
                ]
                for k in keys_to_del:
                    del self.thumbnail_cache[k]
            except Exception as e:
                print(f"Eroare stergere (multiplă) imagine {cale}: {e}")

    def pagina_anterioara(self):
        self._pe_leave_imagine(None)
        if self.pagina_curenta > 0:
            self.pagina_curenta -= 1
            self.afiseaza_pagina_curenta(self.load_token)

    def pagina_urmatoare(self):
        self._pe_leave_imagine(None)
        total_pagini = (
            len(self.lista_date_imagini) + self.imagini_per_pagina - 1
        ) // self.imagini_per_pagina
        if self.pagina_curenta < total_pagini - 1:
            self.pagina_curenta += 1
            self.afiseaza_pagina_curenta(self.load_token)

    def deschide_dialog_import(self):
        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("420x270")
        dialog.title("Import Images")
        dialog.transient(self)
        dialog.grab_set()

        customtkinter.CTkLabel(
            dialog, text="Alegeți modul de import:", font=("Roboto", 16, "bold")
        ).pack(pady=(20, 15))

        btn_fisier = customtkinter.CTkButton(
            dialog,
            text="Selectează Fișier(e)",
            width=280,
            height=36,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=lambda: [dialog.destroy(), self.importa_fisiere_dialog()],
        )
        btn_fisier.pack(pady=6)

        btn_folder = customtkinter.CTkButton(
            dialog,
            text="Selectează Folder (cu subfoldere)",
            width=280,
            height=36,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=lambda: [dialog.destroy(), self.importa_folder_dialog()],
        )
        btn_folder.pack(pady=6)

        btn_internet = customtkinter.CTkButton(
            dialog,
            text="Import imagini de pe internet",
            width=280,
            height=36,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=lambda: [dialog.destroy(), self.deschide_dialog_import_internet()],
        )
        btn_internet.pack(pady=6)

    def deschide_dialog_import_internet(self):
        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("450x380")
        dialog.title("Import Imagini de pe Internet")
        dialog.transient(self)
        dialog.grab_set()

        customtkinter.CTkLabel(
            dialog, text="Import Imagini de pe Internet", font=("Roboto", 18, "bold")
        ).pack(pady=(20, 10))

        customtkinter.CTkLabel(
            dialog, text="Denumire clasă / Cuvânt de căutare:", font=("Roboto", 13)
        ).pack(anchor="w", padx=30, pady=(10, 2))

        entry_clasa = customtkinter.CTkEntry(dialog, width=390, placeholder_text="ex: pisica, caine, masina")
        entry_clasa.pack(padx=30, pady=(0, 10))

        customtkinter.CTkLabel(
            dialog, text="Număr imagini de descărcat:", font=("Roboto", 13)
        ).pack(anchor="w", padx=30, pady=(5, 2))

        entry_numar = customtkinter.CTkEntry(dialog, width=390, placeholder_text="10")
        entry_numar.insert(0, "10")
        entry_numar.pack(padx=30, pady=(0, 15))

        lbl_status = customtkinter.CTkLabel(dialog, text="", font=("Roboto", 12), text_color="#5865F2")
        lbl_status.pack(pady=5)

        def executa_descarcare():
            clasa_nume = entry_clasa.get().strip()
            if not clasa_nume:
                lbl_status.configure(text="Introduceți o clasă / termen de căutare!", text_color="#FF6B6B")
                return

            try:
                nr_img = int(entry_numar.get().strip())
                if nr_img <= 0: raise ValueError
            except ValueError:
                lbl_status.configure(text="Introduceți un număr valid de imagini!", text_color="#FF6B6B")
                return

            lbl_status.configure(text="Se caută și se descarcă imaginile...", text_color="#5865F2")
            btn_start.configure(state="disabled")

            def proces_async():
                try:
                    urls = self._cauta_imagini_bing(clasa_nume, nr_img)
                    if not urls:
                        self.after(0, lambda: lbl_status.configure(text="Nu s-au găsit imagini pe internet!", text_color="#FF6B6B"))
                        self.after(0, lambda: btn_start.configure(state="normal"))
                        return

                    descarcate = 0
                    timestamp = int(time.time())

                    target_img_dir = Path(self.MAPA_IMAGINI) / clasa_nume
                    target_lbl_dir = Path(self.MAPA_LABELS) / clasa_nume
                    os.makedirs(target_img_dir, exist_ok=True)
                    os.makedirs(target_lbl_dir, exist_ok=True)

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8'
                    }

                    for i, u in enumerate(urls):
                        try:
                            req = urllib.request.Request(u, headers=headers)
                            with urllib.request.urlopen(req, timeout=8) as resp:
                                data = resp.read()

                            if len(data) < 2048:
                                print(f"Imaginea {i+1} este prea mică (posibil goală). Trecem peste.")
                                continue

                            ext = ".jpg"
                            if u.lower().endswith(".png"): ext = ".png"
                            elif u.lower().endswith(".webp"): ext = ".webp"

                            nume_f = f"web_{clasa_nume}_{timestamp}_{i+1}{ext}"
                            img_path = target_img_dir / nume_f

                            with open(img_path, "wb") as f_out:
                                f_out.write(data)

                            try:
                                with Image.open(img_path) as img_test:
                                    img_test.verify()
                            except UnidentifiedImageError:
                                print(f"Imaginea {i+1} este coruptă. Se șterge automat.")
                                os.remove(img_path)
                                continue

                            descarcate += 1

                        except urllib.error.HTTPError as e:
                            print(f"Eroare HTTP {e.code} la descărcarea imaginii {i+1}. Trecem la următoarea.")
                            continue
                        except urllib.error.URLError as e:
                            print(f"Eroare de conexiune la imaginea {i+1}. Trecem la următoarea.")
                            continue
                        except Exception as err:
                            print(f"Eroare generală la descărcare imagine {i+1}: {err}")

                    self.after(0, lambda: [
                        dialog.destroy(),
                        self.incarca_imagini()
                    ])
                except Exception as e:
                    print(f"Eroare generala descarcare internet: {e}")
                    self.after(0, lambda: lbl_status.configure(text=f"Eroare: {e}", text_color="#FF6B6B"))
                    self.after(0, lambda: btn_start.configure(state="normal"))

            threading.Thread(target=proces_async, daemon=True).start()

        btn_start = customtkinter.CTkButton(
            dialog,
            text="Descarcă și Importă",
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=executa_descarcare,
            width=200,
            height=36
        )
        btn_start.pack(pady=10)

    def _cauta_imagini_bing(self, query, num_images=10):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
        query_encoded = urllib.parse.quote(query)
        urls = []
        
        url = f"https://www.bing.com/images/async?q={query_encoded}&first=0&count={num_images * 2}&scenario=ImageBasicHover&datsrc=IUrl&layout=RowBased"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8', errors='ignore')
                matches = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
                for m in matches:
                    if m.startswith("http") and m not in urls:
                        urls.append(m)
                    if len(urls) >= num_images:
                        break
        except Exception as e:
            print(f"Eroare cautare Bing: {e}")
            
        return urls

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
                            dest = (
                                Path(self.MAPA_ALL)
                                / f"{b}_{int(time.time())}{ext}"
                            )
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
                            dest = (
                                Path(self.MAPA_ALL)
                                / f"{b}_{int(time.time())}{ext}"
                            )
                        shutil.copy(item, dest)
                        importate += 1

                print(
                    f"📥 Au fost importate {importate} imagini din folder și subfoldere."
                )
                self.after(0, self.incarca_imagini)

            threading.Thread(
                target=proceseaza_import_folder, daemon=True
            ).start()

    def _actualizeaza_stare_nume_model(self, *args):
        model_selectat = self.var_model_selectat.get()
        if model_selectat == "Model YOLO neantrenat":
            self.entry_nume_model.configure(state="normal")
            self.rb_nano.configure(state="normal")
            self.rb_medium.configure(state="normal")
            self.rb_large.configure(state="normal")
        else:
            self.entry_nume_model.configure(state="disabled")
            self.rb_nano.configure(state="disabled")
            self.rb_medium.configure(state="disabled")
            self.rb_large.configure(state="disabled")

            nume_fara_ext = os.path.splitext(model_selectat)[0]
            if nume_fara_ext.endswith("_n"):
                self.var_marime_model.set("n")
            elif nume_fara_ext.endswith("_m"):
                self.var_marime_model.set("m")
            elif nume_fara_ext.endswith("_l") or nume_fara_ext.endswith("_L"):
                self.var_marime_model.set("l")
            else:
                self.var_marime_model.set("n")

    def editeaza_adnotare(self, calea_imagine, nume_clasa):
        self._pe_leave_imagine(None)
        if self.mod_stergere_multipla:
            return
        try:
            if hasattr(self, "snipper") and hasattr(
                self.snipper, "open_annotation_ui"
            ):
                res = self.snipper.open_annotation_ui(
                    image_path=calea_imagine, default_class=nume_clasa
                )
                if hasattr(res, "wait_window"):
                    self.wait_window(res)
                self.after(100, self.incarca_imagini)
        except Exception as e:
            print(f"Eroare la deschiderea editorului: {e}")

    def actualizeaza_stare_buton_antrenare(self, *args):
        # --- MODIFICARE 1: Verificăm dacă clasa bifată are cel puțin o imagine în folder ---
        extensii_valide = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

        def clasa_are_imagini(nume_clasa):
            cale_clasa = os.path.join(self.MAPA_IMAGINI, nume_clasa)
            if os.path.exists(cale_clasa):
                for root, _, files in os.walk(cale_clasa):
                    for f in files:
                        if f.lower().endswith(extensii_valide):
                            return True
            return False

        # clasa trebuie să fie bifată ȘI să conțină imagini
        clase_ok = any(
            var.get() and clasa_are_imagini(nume)
            for nume, var in self.vars_clase.items()
        )

        culoare_eroare = "#FF4C4C"
        culoare_text_normal = "#DCE4EE"
        culoare_border_normal = ["#979DA2", "#565B5E"]

        setari_ok = True
        try:
            epoci_str = self.entry_epoci.get().strip()
            if not epoci_str.isdigit() or not (10 <= int(epoci_str) <= 1000):
                setari_ok = False
                self.entry_epoci.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
            else:
                self.entry_epoci.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)

            imgsz_str = self.entry_imgsz.get().strip()
            if not imgsz_str.isdigit() or not (320 <= int(imgsz_str) <= 1280):
                setari_ok = False
                self.entry_imgsz.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
            else:
                self.entry_imgsz.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)

            lr_str = self.entry_lr.get().strip()
            try:
                lr_val = float(lr_str)
                if not (0.0001 <= lr_val <= 0.1):
                    setari_ok = False
                    self.entry_lr.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
                else:
                    self.entry_lr.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)
            except ValueError:
                setari_ok = False
                self.entry_lr.configure(text_color=culoare_eroare, border_color=culoare_border_normal)

            workers_str = self.entry_workers.get().strip()
            if not workers_str.isdigit() or not (0 <= int(workers_str) <= self.total_cpu_threads):
                setari_ok = False
                self.entry_workers.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
            else:
                self.entry_workers.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)

        except AttributeError:
            setari_ok = False

        hw_ok = True
        try:
            if getattr(self, "entry_cpu_threads", None) is not None:
                cpu_th_str = self.entry_cpu_threads.get().strip()
                if not cpu_th_str.isdigit() or not (1 <= int(cpu_th_str) <= self.total_cpu_threads):
                    hw_ok = False
                    self.entry_cpu_threads.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
                else:
                    self.entry_cpu_threads.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)
            
            if getattr(self, "entry_gpu_batch", None) is not None:
                gpu_batch_str = self.entry_gpu_batch.get().strip()
                if not gpu_batch_str.isdigit() or not (1 <= int(gpu_batch_str) <= 128):
                    hw_ok = False
                    self.entry_gpu_batch.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
                else:
                    self.entry_gpu_batch.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)
        except AttributeError:
            hw_ok = False

        model_nume_ok = True
        model_selectat = self.var_model_selectat.get()
        if not model_selectat:
            model_nume_ok = False
        elif model_selectat == "Model YOLO neantrenat":
            nume_model = self.var_nume_model_nou.get().strip()
            
            if not nume_model or not re.match(r'^[\w\-]+$', nume_model):
                model_nume_ok = False
                self.entry_nume_model.configure(text_color=culoare_eroare, border_color=culoare_border_normal)
            else:
                self.entry_nume_model.configure(text_color=culoare_text_normal, border_color=culoare_border_normal)

        if clase_ok and setari_ok and hw_ok and model_nume_ok:
            self.btn_antrenare.configure(state="normal", fg_color="#8B0000")
        else:
            self.btn_antrenare.configure(state="disabled", fg_color="gray")


    def actualizeaza_interfata_invatare(self):
        for widget in self.scroll_clase_invatare.winfo_children():
            widget.destroy()
        self.vars_clase.clear()
        f_scroll = lambda e: self._pe_rotire_scroll_vertical(e, self.scroll_clase_invatare)
        chk_all = customtkinter.CTkCheckBox(
            master=self.scroll_clase_invatare,
            text="ALL",
            font=("Roboto", 15, "bold"),
            variable=self.var_all_clase,
            checkbox_width=18,
            checkbox_height=18,
            border_width=2,
            corner_radius=4,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=self.toggle_select_all_clase,
        )
        chk_all.pack(anchor="w", padx=10, pady=(5, 12))
        self._bind_scroll_la_widget(chk_all, f_scroll)
        self.incarca_clase_existente()

        for nume_clasa in self.clase_existente:
            var_clasa = customtkinter.BooleanVar(
                value=self.var_all_clase.get()
            )
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
                fg_color="#5865F2",
                hover_color="#4752C4",
                command=self.verifica_stare_all,
            )
            chk.pack(anchor="w", padx=10, pady=6)
            self._bind_scroll_la_widget(chk, f_scroll)
        for widget in self.scroll_modele_invatare.winfo_children():
            widget.destroy()

        # --- MODIFICARE 2: Verificăm dacă există modele în mapa 'models' ---
        fisiere_modele = []
        if os.path.exists(self.MAPA_MODELE):
            fisiere_modele = [
                f for f in os.listdir(self.MAPA_MODELE) if f.endswith(".pt")
            ]

        modele_disponibile = []
        if fisiere_modele:
            # Dacă există cel puțin un model salvat în 'models', adăugăm opțiunile
            modele_disponibile = ["Model YOLO neantrenat"] + fisiere_modele
        else:
            # Dacă NU există niciun model în mapa 'models', lista rămâne goală (nu se afișează modelul neantrenat)
            modele_disponibile = []

        if self.var_model_selectat.get() not in modele_disponibile:
            if modele_disponibile:
                self.var_model_selectat.set(modele_disponibile[0])
            else:
                self.var_model_selectat.set("")

        for nume_model in modele_disponibile:
            radio_model = customtkinter.CTkRadioButton(
                master=self.scroll_modele_invatare,
                text=nume_model,
                value=nume_model,
                variable=self.var_model_selectat,
                font=("Roboto", 14),
                fg_color="#5865F2",
                hover_color="#4752C4",
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
    def _obtine_geometrie_scalata(self):
            """
            Neutralizează dubla scalare CustomTkinter pentru ferestrele modale (overlays).
            """
            try:
                scale = self._get_window_scaling()
            except AttributeError:
                scale = 1.0
                
            w = int(self.winfo_width() / scale)
            h = int(self.winfo_height() / scale)
            x = int(self.winfo_rootx() / scale)
            y = int(self.winfo_rooty() / scale)
            
            return f"{w}x{h}+{x}+{y}"
    def porneste_antrenarea(self):
        clase_selectate = [c for c, var in self.vars_clase.items() if var.get()]
        model_ales = self.var_model_selectat.get()

        cpu_th = "4"
        batch_size = "16"
        
        if getattr(self, "entry_cpu_threads", None) is not None:
            cpu_th = self.entry_cpu_threads.get().strip()
                
        if getattr(self, "entry_gpu_batch", None) is not None:
            batch_size = self.entry_gpu_batch.get().strip()

        epoci = self.entry_epoci.get().strip()
        imgsz = self.entry_imgsz.get().strip()
        lr = self.entry_lr.get().strip()
        workers = self.entry_workers.get().strip()

        try:
            int(cpu_th)
            int(batch_size)
            int(epoci)
            int(imgsz)
            float(lr)
            int(workers)
        except ValueError:
            print("Eroare de formatare la cifre!")
            return

        def cb_progress(valoare_float, procentaj, timp_ramas):
            self.after(
                0,
                self._actualizeaza_modal_ui,
                valoare_float,
                procentaj,
                timp_ramas,
            )

        def cb_finish(anulat):
            self.after(0, self._inchide_modal_antrenare, anulat)

        worker = TrainWorker(
            model_ales=model_ales,
            nume_model_salvat_user=self.var_nume_model_nou.get(),
            marime_model=self.var_marime_model.get(),
            dispozitiv=self.var_dispozitiv.get(),
            device_code=self.gpu_device_code,
            clase_selectate=clase_selectate,
            director_curent=self.DIRECTOR_CURENT,
            cpu_threads=cpu_th,
            batch_size=batch_size,
            epoci=epoci,
            imgsz=imgsz,
            lr=lr,
            workers=workers,
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

        self.modal_overlay.geometry(self._obtine_geometrie_scalata())

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

        frame_bottom = customtkinter.CTkFrame(
            card_dialog, fg_color="transparent"
        )
        frame_bottom.pack(fill="x", side="bottom", padx=30, pady=25)

        self.lbl_timp_ramas = customtkinter.CTkLabel(
            frame_bottom, text="Inițializare antrenare...", font=("Roboto", 14)
        )
        self.lbl_timp_ramas.pack(side="left")

        def la_apasare_opreste():
            worker.anuleaza()
            btn_cancel.configure(state="disabled", text="Se oprește...")
            self.lbl_timp_ramas.configure(text="Învățarea se oprește, așteptați...")
            self.progressbar.configure(progress_color="#FF8C00")
            
        btn_cancel = customtkinter.CTkButton(
            frame_bottom,
            text="Oprește",
            width=110,
            height=36,
            fg_color="#8B0000",
            hover_color="#A00000",
            font=("Roboto", 15, "bold"),
            command=la_apasare_opreste
        )
        btn_cancel.pack(side="right")

        self.modal_overlay.lift()

    def _actualizeaza_modal_ui(self, valoare_float, procentaj, timp_ramas):
        if hasattr(self, "progressbar") and self.progressbar.winfo_exists():
            self.progressbar.set(valoare_float)
            self.lbl_procentaj.configure(text=f"{procentaj}%")
            self.lbl_timp_ramas.configure(text=timp_ramas)

    def _inchide_modal_antrenare(self, anulat):
        if hasattr(self, "modal_overlay") and self.modal_overlay is not None:
            try:
                if self.modal_overlay.winfo_exists():
                    self.modal_overlay.grab_release()
                    self.modal_overlay.destroy()
                    
                    self.update_idletasks()
                    self.after(50, self.focus_force)
            except Exception:
                pass
            
            self.modal_overlay = None  
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
        self.update_idletasks()
        # VERIFICARE NOUĂ: Axa X
        bbox = canvas.bbox("all")
        if not bbox or (bbox[2] - bbox[0]) <= canvas.winfo_width():
            canvas.xview_moveto(0.0)
            return "break"
            
        left, right = canvas.xview()

        cantitate = (
            -1 * int(event.delta / 120)
            if event.delta
            else (1 if event.num == 5 else -1)
        )
        
        dimensiune_vizibila = right - left
        max_left = 1.0 - dimensiune_vizibila
        
        if cantitate < 0 and left <= 0.0:
            canvas.xview_moveto(0.0)
            return "break"
            
        if cantitate > 0 and right >= 1.0:
            canvas.xview_moveto(max_left)
            return "break"
            
        canvas.xview_scroll(cantitate * 67, "units")
        
        nou_left, nou_right = canvas.xview()
        if nou_left <= 0.0:
            canvas.xview_moveto(0.0)
        elif nou_right >= 1.0:
            canvas.xview_moveto(1.0 - (nou_right - nou_left))
            
        return "break"

    def _actualizeaza_stare_scrollbar(self, scroll_frame, orientare="vertical"):
        canvas = getattr(scroll_frame, "_parent_canvas", None)
        scrollbar = getattr(scroll_frame, "_scrollbar", None)
        
        if not canvas or not scrollbar:
            return

        # Asigură-te că UI-ul este actualizat înainte de a citi dimensiunile
        self.update_idletasks()
        
        bbox = canvas.bbox("all")
        if not bbox:
            return

        # Forțăm afișarea scrollbar-ului indiferent de dimensiunea conținutului
        scrollbar.grid()

        # Setăm regiunea de scroll pentru a face bara inactivă dacă nu e nevoie de ea
        if orientare == "vertical":
            inaltime_continut = bbox[3] - bbox[1]
            if inaltime_continut <= canvas.winfo_height():
                # Conținutul încape -> inactivăm scrollul
                canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), canvas.winfo_height()))
            else:
                # Conținutul depășește -> activăm scrollul
                canvas.configure(scrollregion=bbox)
        elif orientare == "horizontal":
            latime_continut = bbox[2] - bbox[0]
            if latime_continut <= canvas.winfo_width():
                # Conținutul încape -> inactivăm scrollul
                canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), canvas.winfo_height()))
            else:
                # Conținutul depășește -> activăm scrollul
                canvas.configure(scrollregion=bbox)
    def _pe_rotire_galerie(self, event):
        canvas = getattr(self.galerie_imagini, "_parent_canvas", None)
        if not canvas:
            return
        self.update_idletasks()
            
        # VERIFICARE NOUĂ
        bbox = canvas.bbox("all")
        if not bbox or (bbox[3] - bbox[1]) <= canvas.winfo_height():
            canvas.yview_moveto(0.0)
            return "break"

        top, bottom = canvas.yview()

        cantitate = (
            -1 * int(event.delta / 120)
            if event.delta
            else (1 if event.num == 5 else -1)
        )
        
        dimensiune_vizibila = bottom - top
        max_top = 1.0 - dimensiune_vizibila
        
        if cantitate < 0 and top <= 0.0:
            canvas.yview_moveto(0.0)
            return "break"
            
        if cantitate > 0 and bottom >= 1.0:
            canvas.yview_moveto(max_top)
            return "break"
            
        canvas.yview_scroll(cantitate * 67, "units")
        
        nou_top, nou_bottom = canvas.yview()
        if nou_top <= 0.0:
            canvas.yview_moveto(0.0)
        elif nou_bottom >= 1.0:
            canvas.yview_moveto(1.0 - (nou_bottom - nou_top))
            
        return "break"

    def arata_pagina_galerie(self):
        self.main_page.pack_forget()
        self.third_page.pack_forget()
        self.second_page.pack(fill="both", expand=True)
        self.incarca_clase_existente()
        for c in self.clase_existente:
            self.creeaza_fila_clasa(c)
        self.incarca_imagini()

    def arata_pagina_invatare(self):
        self._pe_leave_imagine(None)
        self.main_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack(fill="both", expand=True)
        self.actualizeaza_interfata_invatare()

    def arata_pagina_principala(self):
        self._pe_leave_imagine(None)
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
            aviz,
            text="Am înțeles",
            width=120,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=aviz.destroy
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
                    os.makedirs(os.path.join(self.MAPA_IMAGINI, nume), exist_ok=True)
                    os.makedirs(os.path.join(self.MAPA_LABELS, nume), exist_ok=True)
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
            frame_butoane,
            text="Ok",
            width=100,
            fg_color="#5865F2",
            hover_color="#4752C4",
            command=pe_ok
        ).pack(side="right")

    def filtreaza_dupa_clasa(self, nume_clasa):
        self._pe_leave_imagine(None)
        self.clasa_selectata = nume_clasa
        if nume_clasa is None:
            self.btn_toate_clasele.configure(fg_color="#5865F2")
        else:
            self.btn_toate_clasele.configure(fg_color="#2b2b2b")

        for widget in self.tab_scroll_frame.winfo_children():
            if hasattr(widget, "nume_clasa"):
                widget.configure(
                    fg_color=(
                        "#5865F2"
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
                        "#5865F2"
                        if widget.cget("text") == nume_clasa
                        else "#2b2b2b"
                    )
                )

        self.mod_stergere_multipla = False
        self.imagini_selectate_stergere.clear()
        self.btn_stergere_multipla.configure(
            text="Ștergere Multiplă",
            fg_color="#5865F2",
            hover_color="#4752C4",
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
            cursor="hand2",
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
            cursor="hand2",
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
            cursor="hand2",
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

    def _arata_modal_stergere_clasa(
        self, nume_clasa, cale_imagini, cale_labels
    ):
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()

        try:
            ecran = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            ecran_blur = ecran.filter(ImageFilter.GaussianBlur(15))
            self.bg_blur_img = customtkinter.CTkImage(
                light_image=ecran_blur, dark_image=ecran_blur, size=(w, h)
            )
        except Exception:
            self.bg_blur_img = None

        self.modal_stergere = customtkinter.CTkToplevel(self)
        self.modal_stergere.overrideredirect(True)
        self.modal_stergere.geometry(f"{w}x{h}+{x}+{y}")
        self.modal_stergere.transient(self)
        self.modal_stergere.grab_set()

        if self.bg_blur_img:
            lbl_bg = customtkinter.CTkLabel(
                self.modal_stergere, text="", image=self.bg_blur_img
            )
            lbl_bg.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.modal_stergere.attributes("-alpha", 0.9)
            self.modal_stergere.configure(fg_color="#000000")

        card = customtkinter.CTkFrame(
            self.modal_stergere,
            corner_radius=15,
            fg_color="#1E1E1E",
            border_width=2,
            border_color="#333333",
        )
        card.place(relx=0.5, rely=0.5, anchor="center")

        customtkinter.CTkLabel(
            card,
            text=f"Ștergere clasa '{nume_clasa}'",
            font=("Roboto", 20, "bold"),
        ).pack(pady=(25, 10), padx=30)
        customtkinter.CTkLabel(
            card,
            text="Această clasă conține imagini.\nCe dorești să ștergi?",
            font=("Roboto", 14),
        ).pack(pady=(0, 20), padx=30)

        def sterge_doar_clasa():
            self._executa_stergere_clasa(
                nume_clasa, cale_imagini, cale_labels, sterge_si_poze=False
            )
            inchide()

        def sterge_tot():
            self._executa_stergere_clasa(
                nume_clasa, cale_imagini, cale_labels, sterge_si_poze=True
            )
            inchide()

        def inchide():
            self.modal_stergere.grab_release()
            self.modal_stergere.destroy()

        btn_doar_clasa = customtkinter.CTkButton(
            card,
            text="Doar Clasa",
            height=40,
            font=("Roboto", 14),
            command=sterge_doar_clasa,
        )
        btn_doar_clasa.pack(pady=5, padx=30, fill="x")

        btn_tot = customtkinter.CTkButton(
            card,
            text="Clasa și Pozele",
            height=40,
            font=("Roboto", 14),
            fg_color="#8B0000",
            hover_color="#A00000",
            command=sterge_tot,
        )
        btn_tot.pack(pady=5, padx=30, fill="x")

        btn_cancel = customtkinter.CTkButton(
            card,
            text="Anulare",
            height=35,
            fg_color="transparent",
            border_width=1,
            command=inchide,
        )
        btn_cancel.pack(pady=(15, 25), padx=30, fill="x")

    def _executa_stergere_clasa(
        self, nume_clasa, cale_imagini, cale_labels, sterge_si_poze
    ):
        self._pe_leave_imagine(None)
        ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if os.path.exists(cale_imagini):
            for root, _, files in os.walk(cale_imagini):
                for file in files:
                    if file.lower().endswith(ext):
                        src_img = os.path.join(root, file)
                        if sterge_si_poze:
                            os.remove(src_img)
                        else:
                            dst_img = os.path.join(self.MAPA_ALL, file)
                            if os.path.exists(dst_img):
                                b, ext_f = os.path.splitext(file)
                                dst_img = os.path.join(
                                    self.MAPA_ALL, f"{b}_{int(time.time())}{ext_f}"
                                )
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

    def sterge_clasa(self, nume_clasa):
        cale_imagini = os.path.join(self.MAPA_IMAGINI, nume_clasa)
        cale_labels = os.path.join(self.MAPA_LABELS, nume_clasa)

        are_imagini = False
        if os.path.exists(cale_imagini):
            for f in os.listdir(cale_imagini):
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".webp")
                ):
                    are_imagini = True
                    break

        if are_imagini:
            self._arata_modal_stergere_clasa(
                nume_clasa, cale_imagini, cale_labels
            )
        else:
            self._executa_stergere_clasa(
                nume_clasa, cale_imagini, cale_labels, sterge_si_poze=False
            )

    def _pe_enter_imagine(self, event, cale_imagine):
        if self.mod_stergere_multipla:
            return
        if self.timer_previzualizare is not None:
            self.after_cancel(self.timer_previzualizare)
        self.timer_previzualizare = self.after(
            500,
            lambda: self._arata_previzualizare_hover(
                cale_imagine, event.x_root, event.y_root
            ),
        )

    def _pe_leave_imagine(self, event=None):
        if self.timer_previzualizare is not None:
            self.after_cancel(self.timer_previzualizare)
            self.timer_previzualizare = None
        if self.fereastra_previzualizare is not None:
            try:
                self.fereastra_previzualizare.destroy()
            except Exception:
                pass
            self.fereastra_previzualizare = None

    def _arata_previzualizare_hover(self, cale_imagine, x_root, y_root):
        if self.mod_stergere_multipla:
            return
        if self.fereastra_previzualizare is not None:
            try:
                self.fereastra_previzualizare.destroy()
            except Exception:
                pass
        self.fereastra_previzualizare = customtkinter.CTkToplevel(self)
        self.fereastra_previzualizare.overrideredirect(True)
        self.fereastra_previzualizare.attributes("-alpha", 0.95)
        self.fereastra_previzualizare.geometry(f"+{x_root + 15}+{y_root + 15}")

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
                try:
                    self.fereastra_previzualizare.destroy()
                except Exception:
                    pass
                self.fereastra_previzualizare = None

    def sterge_imagine(
        self, calea_imagine, nume_fara_extensie, nume_clasa_efectiva
    ):
        self._pe_leave_imagine(None)
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

            keys_to_del = [
                k for k in self.thumbnail_cache.keys() if calea_imagine in k
            ]
            for k in keys_to_del:
                del self.thumbnail_cache[k]

            self.after(50, self.incarca_imagini)
        except Exception as e:
            print(f"Eroare stergere img: {e}")

    def scoate_din_clasa(
        self, calea_imagine, nume_fara_extensie, nume_clasa_efectiva
    ):
        self._pe_leave_imagine(None)

        if hasattr(self, "imagine_hovered"):
            self.imagine_hovered = None

        try:
            clase_de_verificat = set(getattr(self, "clase_existente", []) + ["all"])
            for c_ex in clase_de_verificat:
                cale_txt = os.path.join(
                    self.DIRECTOR_CURENT,
                    "labels",
                    c_ex,
                    nume_fara_extensie + ".txt",
                )
                if os.path.exists(cale_txt):
                    try:
                        os.remove(cale_txt)
                    except Exception as e:
                        print(f"Eroare la ștergerea etichetei: {e}")

            dir_parinte = os.path.dirname(calea_imagine)
            if os.path.abspath(dir_parinte) != os.path.abspath(self.MAPA_ALL):
                if os.path.exists(calea_imagine):
                    nume_f = os.path.basename(calea_imagine)
                    destinatie_all = os.path.join(self.MAPA_ALL, nume_f)
                    if os.path.exists(destinatie_all) and os.path.abspath(
                        calea_imagine
                    ) != os.path.abspath(destinatie_all):
                        b, ext_f = os.path.splitext(nume_f)
                        destinatie_all = os.path.join(
                            self.MAPA_ALL, f"{b}_{int(time.time())}{ext_f}"
                        )
                    shutil.move(calea_imagine, destinatie_all)
                    calea_imagine = destinatie_all

            keys_to_del = [
                k
                for k in list(self.thumbnail_cache.keys())
                if calea_imagine in k or nume_fara_extensie in k
            ]
            for k in keys_to_del:
                del self.thumbnail_cache[k]

            self.after(50, self.incarca_imagini)

        except Exception as e:
            print(f"Eroare la scoaterea imaginii din clasă: {e}")

    def curata_imagini_neadnotate(self):
        self.incarca_clase_existente()
        ext = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

        if os.path.exists(self.MAPA_LABELS):
            for r, _, fs in os.walk(self.MAPA_LABELS):
                for f in fs:
                    if f.endswith(".txt"):
                        cale_t = os.path.join(r, f)
                        try:
                            with open(cale_t, "r", encoding="utf-8") as file:
                                if not file.read().strip():
                                    os.remove(cale_t)
                        except Exception:
                            pass

        if os.path.exists(self.MAPA_IMAGINI):
            for nume_folder in os.listdir(self.MAPA_IMAGINI):
                cale_f = os.path.join(self.MAPA_IMAGINI, nume_folder)
                if os.path.isdir(cale_f) and nume_folder.lower() != "all":
                    for root, _, files in os.walk(cale_f):
                        for fisier in files:
                            if fisier.lower().endswith(ext):
                                c_img = os.path.join(root, fisier)
                                n_fara, _ = os.path.splitext(fisier)
                                if not self._este_imagine_adnotata(
                                    n_fara, nume_folder
                                ):
                                    dest_all = os.path.join(self.MAPA_ALL, fisier)
                                    if os.path.exists(dest_all) and os.path.abspath(
                                        c_img
                                    ) != os.path.abspath(dest_all):
                                        b, ext_f = os.path.splitext(fisier)
                                        dest_all = os.path.join(
                                            self.MAPA_ALL,
                                            f"{b}_{int(time.time())}{ext_f}",
                                        )
                                    try:
                                        shutil.move(c_img, dest_all)
                                    except Exception as e:
                                        print(
                                            f"Eroare mutare imagine neadnotată în all: {e}"
                                        )

        if os.path.exists(self.MAPA_ALL):
            for fisier in os.listdir(self.MAPA_ALL):
                if fisier.lower().endswith(ext):
                    c_img = os.path.join(self.MAPA_ALL, fisier)
                    n_fara, _ = os.path.splitext(fisier)
                    for c_ex in getattr(self, "clase_existente", []):
                        if self._este_imagine_adnotata(n_fara, c_ex):
                            dest_dir = os.path.join(self.MAPA_IMAGINI, c_ex)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_img = os.path.join(dest_dir, fisier)
                            if os.path.exists(dest_img) and os.path.abspath(
                                c_img
                            ) != os.path.abspath(dest_img):
                                b, ext_f = os.path.splitext(fisier)
                                dest_img = os.path.join(
                                    dest_dir, f"{b}_{int(time.time())}{ext_f}"
                                )
                            try:
                                shutil.move(c_img, dest_img)
                            except Exception as e:
                                print(
                                    f"Eroare mutare imagine adnotată în clasa {c_ex}: {e}"
                                )
                            break

    def _este_imagine_adnotata(self, nume_fara_extensie, nume_clasa_efectiva):
        def este_valida(cale_txt):
            if not os.path.exists(cale_txt):
                return False
            try:
                with open(cale_txt, "r", encoding="utf-8") as f:
                    return bool(f.read().strip())
            except Exception:
                return False

        if nume_clasa_efectiva and nume_clasa_efectiva.lower() != "all":
            cale_txt = os.path.join(
                self.DIRECTOR_CURENT,
                "labels",
                nume_clasa_efectiva,
                nume_fara_extensie + ".txt",
            )
            if este_valida(cale_txt):
                return True

        for c_ex in getattr(self, "clase_existente", []):
            cale_txt = os.path.join(
                self.DIRECTOR_CURENT,
                "labels",
                c_ex,
                nume_fara_extensie + ".txt",
            )
            if este_valida(cale_txt):
                return True

        return False

    def incarca_imagini(self):
        self._pe_leave_imagine(None)
        if not os.path.exists(self.MAPA_IMAGINI):
            return

        self.load_token += 1
        current_token = self.load_token

        cadre_pool = [w["cadru"] for w in self.pool_widgeturi]
        for widget in self.galerie_imagini.winfo_children():
            if widget in cadre_pool:
                widget.grid_forget()
            else:
                widget.destroy()

        lbl_incarcare = customtkinter.CTkLabel(
            self.galerie_imagini,
            text="Se scanează rapid imaginile de pe disc...",
            font=("Roboto", 16),
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

                            if not is_all:
                                n_clasa_ef = n_c
                                este_ad = self._este_imagine_adnotata(
                                    n_fara, n_c
                                )
                            else:
                                este_ad = False
                                for c_ex in self.clase_existente:
                                    if self._este_imagine_adnotata(
                                        n_fara, c_ex
                                    ):
                                        este_ad = True
                                        n_clasa_ef = c_ex
                                        break
                            
                            # --- LOGICA NOUĂ DE FILTRARE ---
                            ascunde_adnotate = getattr(self, "var_doar_neadnotate", None)
                            if ascunde_adnotate and ascunde_adnotate.get():
                                # Dacă suntem în "All" (self.clasa_selectata is None) și imaginea are adnotare, sărim peste ea
                                if self.clasa_selectata is None and este_ad:
                                    continue
                            # -------------------------------

                            lista.append(
                                (cale_c, f_name, n_fara, n_clasa_ef, este_ad)
                            )

            if current_token == self.load_token:
                self.after(0, self._finalizeaza_scanarea, lista, current_token)

        threading.Thread(target=proc_scanare, daemon=True).start()

    def _finalizeaza_scanarea(self, lista_imagini, token):
        if token != self.load_token:
            return

        self.lista_date_imagini = lista_imagini

        total_pagini = max(
            1,
            (len(lista_imagini) + self.imagini_per_pagina - 1)
            // self.imagini_per_pagina,
        )
        if self.pagina_curenta >= total_pagini:
            self.pagina_curenta = max(0, total_pagini - 1)

        self.afiseaza_pagina_curenta(token)

    def afiseaza_pagina_curenta(self, token):
        if token != self.load_token:
            return

        self.galerie_imagini.update_idletasks()
        latime_disponibila = self.galerie_imagini.winfo_width()

        if latime_disponibila < 200:
            latime_disponibila = self.winfo_width() - 60

        if latime_disponibila < 200:
            latime_disponibila = 1200

        coloane_dinamice = max(1, latime_disponibila // 130)

        cadre_pool = [w["cadru"] for w in self.pool_widgeturi]
        for widget in self.galerie_imagini.winfo_children():
            if widget in cadre_pool:
                widget.grid_forget()
            else:
                widget.destroy()

        total = len(self.lista_date_imagini)
        pagini_totale = max(
            1, (total + self.imagini_per_pagina - 1) // self.imagini_per_pagina
        )

        self.lbl_paginare.configure(
            text=f"Pagină: {self.pagina_curenta + 1} / {pagini_totale} (Total: {total})"
        )
        self.btn_prev.configure(
            state="normal" if self.pagina_curenta > 0 else "disabled"
        )
        self.btn_next.configure(
            state=(
                "normal" if self.pagina_curenta < pagini_totale - 1 else "disabled"
            )
        )

        if total == 0:
            lbl = customtkinter.CTkLabel(
                self.galerie_imagini,
                text="Nu există imagini în această categorie.",
                font=("Roboto", 16),
            )
            lbl.pack(pady=40)
            return

        start_idx = self.pagina_curenta * self.imagini_per_pagina
        end_idx = min(start_idx + self.imagini_per_pagina, total)
        imagini_de_procesat = self.lista_date_imagini[start_idx:end_idx]

        lbl_loading = customtkinter.CTkLabel(
            self.galerie_imagini,
            text="Se afișează pagina...",
            font=("Roboto", 16),
        )
        lbl_loading.pack(pady=40)

        def proceseaza_chunk():
            if token != self.load_token:
                return
            rezultate_ui = []
            c, r = 0, 0
            max_c = coloane_dinamice
            is_multi_delete = self.mod_stergere_multipla

            for (
                cale_c,
                f_name,
                n_fara,
                n_clasa_ef,
                este_ad,
            ) in imagini_de_procesat:
                if token != self.load_token:
                    return
                try:
                    deseneaza_triunghi = este_ad
                    mtime = (
                        os.path.getmtime(cale_c)
                        if os.path.exists(cale_c)
                        else 0
                    )
                    hash_base = hashlib.md5(
                        f"{cale_c}_{deseneaza_triunghi}_{mtime}".encode("utf-8")
                    ).hexdigest()

                    if is_multi_delete:
                        cale_cache_md_unsel = os.path.join(
                            self.CACHE_DIR, f"{hash_base}_md_unsel_clean.png"
                        )
                        cale_cache_md_sel = os.path.join(
                            self.CACHE_DIR, f"{hash_base}_md_sel_clean.png"
                        )

                        if os.path.exists(
                            cale_cache_md_unsel
                        ) and os.path.exists(cale_cache_md_sel):
                            pil_md_unsel = Image.open(cale_cache_md_unsel)
                            pil_md_sel = Image.open(cale_cache_md_sel)
                        else:
                            orig = Image.open(cale_c)
                            orig.thumbnail(
                                (100, 100), Image.Resampling.LANCZOS
                            )

                            bg = Image.new("RGB", (100, 100), "#2b2b2b")
                            x_off = (100 - orig.width) // 2
                            y_off = (100 - orig.height) // 2

                            if orig.mode in ("RGBA", "LA") or (
                                orig.mode == "P" and "transparency" in orig.info
                            ):
                                rgba_orig = orig.convert("RGBA")
                                bg.paste(
                                    rgba_orig,
                                    (x_off, y_off),
                                    mask=rgba_orig.split()[3],
                                )
                            else:
                                bg.paste(orig, (x_off, y_off))

                            base_img = bg

                            pil_md_unsel = base_img.copy()
                            pil_md_sel = base_img.copy()

                            draw = ImageDraw.Draw(pil_md_sel)
                            draw.line(
                                [(35, 50), (45, 60), (65, 30)],
                                fill="#000000",
                                width=8,
                            )
                            draw.line(
                                [(35, 50), (45, 60), (65, 30)],
                                fill="#00FF00",
                                width=4,
                            )

                            pil_md_unsel.save(cale_cache_md_unsel, "PNG")
                            pil_md_sel.save(cale_cache_md_sel, "PNG")

                        img_unsel = customtkinter.CTkImage(
                            light_image=pil_md_unsel,
                            dark_image=pil_md_unsel,
                            size=(100, 100),
                        )
                        img_sel = customtkinter.CTkImage(
                            light_image=pil_md_sel,
                            dark_image=pil_md_sel,
                            size=(100, 100),
                        )

                        n_scurt = (
                            f_name
                            if len(f_name) < 15
                            else f_name[:12] + "..."
                        )
                        t_et = (
                            f"[{n_clasa_ef.upper()}]\n{n_scurt}"
                            if self.clasa_selectata is None
                            else n_scurt
                        )

                        rezultate_ui.append(
                            (
                                c,
                                r,
                                img_unsel,
                                None,
                                None,
                                cale_c,
                                n_fara,
                                n_clasa_ef,
                                t_et,
                                False,
                                True,
                                img_sel,
                            )
                        )
                    else:
                        cale_cache_n = os.path.join(
                            self.CACHE_DIR, f"{hash_base}_n_clean.png"
                        )
                        cale_cache_hx = os.path.join(
                            self.CACHE_DIR, f"{hash_base}_hx_clean.png"
                        )
                        cale_cache_ht = os.path.join(
                            self.CACHE_DIR, f"{hash_base}_ht_clean.png"
                        )

                        if (
                            os.path.exists(cale_cache_n)
                            and os.path.exists(cale_cache_hx)
                            and os.path.exists(cale_cache_ht)
                        ):
                            pil_n = Image.open(cale_cache_n)
                            pil_hx = Image.open(cale_cache_hx)
                            pil_ht = Image.open(cale_cache_ht)
                        else:
                            orig = Image.open(cale_c)
                            orig.thumbnail(
                                (100, 100), Image.Resampling.LANCZOS
                            )

                            bg = Image.new("RGB", (100, 100), "#2b2b2b")
                            x_off = (100 - orig.width) // 2
                            y_off = (100 - orig.height) // 2

                            if orig.mode in ("RGBA", "LA") or (
                                orig.mode == "P" and "transparency" in orig.info
                            ):
                                rgba_orig = orig.convert("RGBA")
                                bg.paste(
                                    rgba_orig,
                                    (x_off, y_off),
                                    mask=rgba_orig.split()[3],
                                )
                            else:
                                bg.paste(orig, (x_off, y_off))

                            pil_i = bg

                            pil_n = pil_i.copy()
                            pil_hx = pil_i.copy()
                            pil_ht = pil_i.copy()

                            if deseneaza_triunghi:
                                d_n = ImageDraw.Draw(pil_n)
                                d_n.polygon(
                                    [(99, 0), (79, 0), (99, 20)], fill="#5865F2"
                                )
                                d_hx = ImageDraw.Draw(pil_hx)
                                d_hx.polygon(
                                    [(99, 0), (79, 0), (99, 20)], fill="#5865F2"
                                )
                                d_ht = ImageDraw.Draw(pil_ht)
                                d_ht.polygon(
                                    [(99, 0), (79, 0), (99, 20)], fill="#4752C4",
                                )

                            def draw_x(im, c_x):
                                dr = ImageDraw.Draw(im)
                                dr.line(
                                    [(5, 5), (13, 13)], fill="#000000", width=3
                                )
                                dr.line(
                                    [(5, 13), (13, 5)], fill="#000000", width=3
                                )
                                dr.line([(5, 5), (13, 13)], fill=c_x, width=1)
                                dr.line([(5, 13), (13, 5)], fill=c_x, width=1)
                            draw_x(pil_n, "#888888")
                            draw_x(pil_hx, "#e0e0e0")
                            draw_x(pil_ht, "#888888")

                            pil_n.save(cale_cache_n, "PNG")
                            pil_hx.save(cale_cache_hx, "PNG")
                            pil_ht.save(cale_cache_ht, "PNG")

                        img_n = customtkinter.CTkImage(
                            light_image=pil_n, dark_image=pil_n, size=(100, 100)
                        )
                        img_hx = customtkinter.CTkImage(
                            light_image=pil_hx,
                            dark_image=pil_hx,
                            size=(100, 100),
                        )
                        img_ht = customtkinter.CTkImage(
                            light_image=pil_ht,
                            dark_image=pil_ht,
                            size=(100, 100),
                        )

                        n_scurt = (
                            f_name
                            if len(f_name) < 15
                            else f_name[:12] + "..."
                        )
                        t_et = (
                            f"[{n_clasa_ef.upper()}]\n{n_scurt}"
                            if self.clasa_selectata is None
                            else n_scurt
                        )

                        rezultate_ui.append(
                            (
                                c,
                                r,
                                img_n,
                                img_hx,
                                img_ht,
                                cale_c,
                                n_fara,
                                n_clasa_ef,
                                t_et,
                                deseneaza_triunghi,
                                False,
                                None,
                            )
                        )

                    c += 1
                    if c >= max_c:
                        c = 0
                        r += 1
                except Exception as e:
                    print(f"Eroare procesare thumbnail {cale_c}: {e}")

            if token == self.load_token:
                self.after(
                    0, self._randeaza_UI_final, rezultate_ui, lbl_loading, token
                )

        threading.Thread(target=proceseaza_chunk, daemon=True).start()

    def _randeaza_UI_final(self, rezultate_ui, lbl_loading, token):
        if token != self.load_token:
            return

        lbl_loading.destroy()
        self.imagini_salvate.clear()

        numar_necesar = len(rezultate_ui)
        numar_existent = len(self.pool_widgeturi)

        for i, item in enumerate(rezultate_ui):
            (
                cc,
                rc,
                imn,
                imhx,
                imht,
                pac,
                nfe,
                nce,
                tet,
                deseneaza_triunghi,
                is_md,
                im_md_sel,
            ) = item

            if imn:
                self.imagini_salvate.append(imn)
            if imhx:
                self.imagini_salvate.append(imhx)
            if imht:
                self.imagini_salvate.append(imht)
            if im_md_sel:
                self.imagini_salvate.append(im_md_sel)

            try:
                if i < numar_existent:
                    w_dict = self.pool_widgeturi[i]
                    cadru = w_dict["cadru"]
                    ico = w_dict["ico"]
                    lbl = w_dict["lbl"]

                    lbl.configure(text=tet)
                    cadru.grid(row=rc, column=cc, padx=15, pady=15)
                else:
                    cadru = customtkinter.CTkFrame(
                        self.galerie_imagini,
                        fg_color="transparent",
                        corner_radius=0,
                    )
                    cadru.grid(row=rc, column=cc, padx=15, pady=15)
                    cont = customtkinter.CTkFrame(cadru, fg_color="transparent")
                    cont.pack()
                    ico = customtkinter.CTkLabel(cont, text="", cursor="hand2")
                    ico.pack()
                    lbl = customtkinter.CTkLabel(
                        cont, text=tet, font=("Roboto", 12), cursor="hand2"
                    )
                    lbl.pack(pady=(4, 0))

                    self._bind_scroll_la_widget(cadru, self._pe_rotire_galerie)
                    self._bind_scroll_la_widget(cont, self._pe_rotire_galerie)
                    self._bind_scroll_la_widget(ico, self._pe_rotire_galerie)
                    self._bind_scroll_la_widget(lbl, self._pe_rotire_galerie)

                    self.pool_widgeturi.append(
                        {"cadru": cadru, "cont": cont, "ico": ico, "lbl": lbl}
                    )

                ico.unbind("<Motion>")
                ico.unbind("<Leave>")
                ico.unbind("<Button-1>")
                ico.unbind("<Enter>")

                if is_md:
                    is_selected_now = pac in self.imagini_selectate_stergere
                    ico.configure(image=im_md_sel if is_selected_now else imn)

                    def toggle_select(
                        e,
                        pac_val=pac,
                        ic=ico,
                        im_unsel=imn,
                        im_sel=im_md_sel,
                    ):
                        if pac_val in self.imagini_selectate_stergere:
                            self.imagini_selectate_stergere.remove(pac_val)
                            ic.configure(image=im_unsel)
                        else:
                            self.imagini_selectate_stergere.add(pac_val)
                            ic.configure(image=im_sel)

                    ico.bind("<Button-1>", toggle_select)

                else:
                    ico.configure(image=imn)
                    stare_hover = [0]

                    def m(
                        e,
                        im_n=imn,
                        im_hx=imhx,
                        im_ht=imht,
                        dt_live=deseneaza_triunghi,
                        ic=ico,
                    ):
                        in_x = e.x <= 20 and e.y <= 20
                        in_triangle = dt_live and (
                            e.x >= 79 and e.y >= 0 and e.y <= (e.x - 79)
                        )

                        if in_x:
                            if stare_hover[0] != 1:
                                stare_hover[0] = 1
                                ic.configure(image=im_hx)
                        elif in_triangle:
                            if stare_hover[0] != 2:
                                stare_hover[0] = 2
                                ic.configure(image=im_ht)
                        else:
                            if stare_hover[0] != 0:
                                stare_hover[0] = 0
                                ic.configure(image=im_n)

                    def l(e, im_n=imn, ic=ico):
                        stare_hover[0] = 0
                        ic.configure(image=im_n)
                        self._pe_leave_imagine(e)

                    def cl(
                        e,
                        pac_val=pac,
                        nfe_val=nfe,
                        nce_val=nce,
                        dt_live=deseneaza_triunghi,
                    ):
                        self._pe_leave_imagine(e)

                        in_x = e.x <= 20 and e.y <= 20
                        in_triangle = dt_live and (
                            e.x >= 79 and e.y >= 0 and e.y <= (e.x - 79)
                        )

                        if in_x:
                            self.sterge_imagine(pac_val, nfe_val, nce_val)
                        elif in_triangle:
                            self.scoate_din_clasa(pac_val, nfe_val, nce_val)
                        else:
                            self.editeaza_adnotare(pac_val, nce_val)

                    ico.bind("<Motion>", m)
                    ico.bind("<Leave>", l)
                    ico.bind("<Button-1>", cl)
                    ico.bind(
                        "<Enter>", lambda e, p=pac: self._pe_enter_imagine(e, p)
                    )
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