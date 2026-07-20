import customtkinter
import tkinter
import os
from PIL import Image, ImageDraw

from snipping_tool import SnippingAnnotator

class TrainWindow(customtkinter.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.geometry("1280x720")
        self.title("Meniu Antrenare și Galerie")
        
        self.transient(parent)
        self.grab_set()

        self.DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))
        self.MAPA_IMAGINI = os.path.join(self.DIRECTOR_CURENT, "images")
        self.MAPA_ALL = os.path.join(self.MAPA_IMAGINI, "all")
        self.MAPA_MODELE = os.path.join(self.DIRECTOR_CURENT, "models") 

        # Ne asigurăm că folderul principal și subfolderul 'all' există
        os.makedirs(self.MAPA_ALL, exist_ok=True)

        self.imagini_salvate = []
        self.clase_existente = []
        
        self.timer_previzualizare = None
        self.fereastra_previzualizare = None
        self.clasa_selectata = None # None înseamnă afișare "All"

        self.snipper = SnippingAnnotator(
            parent_app=self, 
            output_dir=self.DIRECTOR_CURENT, 
            hotkey="ctrl+shift+s"
        )
        self.snipper.start_background_listener()

        self.creeaza_pagini()
        self.arata_pagina_principala()

    def creeaza_pagini(self):
        # ==========================================
        # 1. PAGINA PRINCIPALĂ
        # ==========================================
        self.main_page = customtkinter.CTkFrame(self, fg_color="transparent")
        
        title = customtkinter.CTkLabel(master=self.main_page, text="Sigma", font=("Roboto", 32, "bold"))
        title.pack(pady=(20, 0), padx=10)

        frame_1 = customtkinter.CTkFrame(master=self.main_page)
        frame_1.pack(pady=20, padx=60, fill="both", expand=True)
        frame_1.columnconfigure(0, weight=1)
        frame_1.columnconfigure(1, weight=1)
        frame_1.columnconfigure(2, weight=1)

        column1 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
        column1.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
        button1 = customtkinter.CTkButton(master=column1, text="Galerie", command=self.arata_pagina_galerie)
        button1.pack(pady=10, padx=10)

        column2 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
        column2.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
        button2 = customtkinter.CTkButton(master=column2, text="Învățare", command=self.arata_pagina_invatare)
        button2.pack(pady=10, padx=10)

        column3 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
        column3.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)
        button3 = customtkinter.CTkButton(master=column3, text="Închide", command=lambda: self.master.event_generate("<<CloseTrainWindow>>") if hasattr(self, "master") else self.destroy())
        button3.pack(pady=10, padx=10)

        # ==========================================
        # 2. GALERIA
        # ==========================================
        self.second_page = customtkinter.CTkFrame(self, fg_color="transparent")

        top_bar = customtkinter.CTkFrame(self.second_page, fg_color="transparent", height=40)
        top_bar.pack(fill="x", padx=20, pady=10)

        buton_adauga_clasa = customtkinter.CTkButton(
            top_bar, text="+", width=35, height=35, corner_radius=8,        
            font=("Roboto", 20, "bold"), fg_color="#1f538d", hover_color="#14375e",    
            command=self.deschide_dialog_clasa
        )
        buton_adauga_clasa.pack(side="left", padx=(0, 8), pady=(0, 6))

        self.tab_scroll_frame = customtkinter.CTkScrollableFrame(
            master=top_bar, orientation="horizontal", height=38,                    
            fg_color="transparent", scrollbar_button_color="#242424", scrollbar_button_hover_color="gray"
        )
        self.tab_scroll_frame.pack(side="left", fill="x", expand=True)

        self.tab_scroll_frame.bind("<Enter>", lambda event: self._asculta_rotita(True))
        self.tab_scroll_frame.bind("<Leave>", lambda event: self._asculta_rotita(False))

        self.btn_toate_clasele = customtkinter.CTkButton(
            master=self.tab_scroll_frame, text="All", width=120, height=32, 
            fg_color="#1f538d", hover_color="#14375e", command=lambda: self.filtreaza_dupa_clasa(None)
        )
        self.btn_toate_clasele.pack(side="left", padx=5)

        bottom_bar = customtkinter.CTkFrame(self.second_page, fg_color="transparent")
        bottom_bar.pack(fill="x", side="bottom", padx=20, pady=20)

        buton_refresh_galerie = customtkinter.CTkButton(bottom_bar, text="Actualizează Galeria", command=self.incarca_imagini)
        buton_refresh_galerie.pack(side="left")

        buton_inapoi_galerie = customtkinter.CTkButton(bottom_bar, text="Înapoi", command=self.arata_pagina_principala)
        buton_inapoi_galerie.pack(side="right")

        self.galerie_imagini = customtkinter.CTkScrollableFrame(self.second_page)
        self.galerie_imagini.pack(fill="both", expand=True, padx=20, pady=10)

        # ==========================================
        # 3. ÎNVĂȚARE
        # ==========================================
        self.third_page = customtkinter.CTkFrame(self, fg_color="transparent")

        titlu_invatare = customtkinter.CTkLabel(master=self.third_page, text="Meniul de Învățare", font=("Roboto", 28, "bold"))
        titlu_invatare.pack(pady=(30, 10))

        frame_optiuni_invatare = customtkinter.CTkFrame(master=self.third_page)
        frame_optiuni_invatare.pack(pady=20, padx=60, fill="both", expand=True)
        frame_optiuni_invatare.columnconfigure(0, weight=1)
        frame_optiuni_invatare.columnconfigure(1, weight=1)

        sectiune_neantrenat = customtkinter.CTkFrame(master=frame_optiuni_invatare, fg_color="transparent")
        sectiune_neantrenat.grid(row=0, column=0, sticky="nsew", padx=20, pady=50)
        btn_neantrenat = customtkinter.CTkButton(master=sectiune_neantrenat, text="Model neantrenat", height=40, font=("Roboto", 16))
        btn_neantrenat.pack(expand=True) 

        sectiune_antrenat = customtkinter.CTkFrame(master=frame_optiuni_invatare, fg_color="transparent")
        sectiune_antrenat.grid(row=0, column=1, sticky="nsew", padx=20, pady=50)

        self.meniu_modele = customtkinter.CTkOptionMenu(
            master=sectiune_antrenat, values=["Se încarcă..."], height=35, font=("Roboto", 20), dynamic_resizing=True
        )
        self.meniu_modele.pack(pady=(7,0))

        buton_inapoi_invatare = customtkinter.CTkButton(master=self.third_page, text="Înapoi", command=self.arata_pagina_principala)
        buton_inapoi_invatare.pack(side="bottom", pady=20, padx=20, anchor="e")

    def _pe_rotire_mouse(self, event):
        if event.delta:
            cantitate = -1 * int(event.delta / 120)
        else:
            cantitate = 1 if event.num == 5 else -1
        self.tab_scroll_frame._parent_canvas.xview_scroll(cantitate * 3, "units")

    def _asculta_rotita(self, activa):
        if activa:
            self.bind_all("<MouseWheel>", self._pe_rotire_mouse)
            self.bind_all("<Button-4>", self._pe_rotire_mouse)
            self.bind_all("<Button-5>", self._pe_rotire_mouse)
        else:
            self.unbind_all("<MouseWheel>")
            self.unbind_all("<Button-4>")
            self.unbind_all("<Button-5>")

    def arata_pagina_galerie(self):
        self.main_page.pack_forget()
        self.third_page.pack_forget()
        self.second_page.pack(fill="both", expand=True)
        
        if os.path.exists(self.MAPA_IMAGINI):
            for nume_folder in os.listdir(self.MAPA_IMAGINI):
                cale_folder = os.path.join(self.MAPA_IMAGINI, nume_folder)
                if os.path.isdir(cale_folder) and nume_folder.lower() != "all":
                    if nume_folder not in self.clase_existente:
                        self.clase_existente.append(nume_folder)
                        self.creeaza_fila_clasa(nume_folder)

        self.incarca_imagini()

    def arata_pagina_invatare(self):
        self.main_page.pack_forget()
        self.second_page.pack_forget()
        self.third_page.pack(fill="both", expand=True)
        self.actualizeaza_modele() 

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
        customtkinter.CTkLabel(aviz, text=f"Clasa '{nume}' există deja!\nAlegeți altă denumire.", font=("Roboto", 14)).pack(pady=(25, 15))
        customtkinter.CTkButton(aviz, text="Am înțeles", width=120, command=aviz.destroy).pack(pady=10)

    def deschide_dialog_clasa(self):
        dialog = customtkinter.CTkToplevel(self)
        dialog.geometry("450x220")
        dialog.title("Clasa")
        dialog.transient(self)
        dialog.grab_set()
        
        customtkinter.CTkLabel(dialog, text="Scrieți denumirea clasei:", font=("Roboto", 16)).pack(pady=(25, 10), padx=20, anchor="w")
        entry_clasa = customtkinter.CTkEntry(dialog, width=410, font=("Roboto", 14))
        entry_clasa.pack(pady=10, padx=20)
        
        frame_butoane = customtkinter.CTkFrame(dialog, fg_color="transparent")
        frame_butoane.pack(pady=(15, 0), padx=20, fill="x")
        
        def pe_ok():
            nume_clasa = entry_clasa.get().strip()
            if nume_clasa:
                if nume_clasa.lower() == "all" or nume_clasa.lower() in [c.lower() for c in self.clase_existente]:
                    self.arata_avertisment_duplicat(nume_clasa)
                else:
                    self.clase_existente.append(nume_clasa)
                    self.creeaza_fila_clasa(nume_clasa)
                    dialog.destroy()
            else:
                dialog.destroy()

        customtkinter.CTkButton(frame_butoane, text="Cancel", width=100, fg_color="transparent", border_width=1, command=dialog.destroy).pack(side="right", padx=(10, 0))
        customtkinter.CTkButton(frame_butoane, text="Ok", width=100, command=pe_ok).pack(side="right")

    def filtreaza_dupa_clasa(self, nume_clasa):
        self.clasa_selectata = nume_clasa
        
        if nume_clasa is None:
            self.btn_toate_clasele.configure(fg_color="#1f538d")
        else:
            self.btn_toate_clasele.configure(fg_color="#2b2b2b")

        for widget in self.tab_scroll_frame.winfo_children():
            if isinstance(widget, customtkinter.CTkButton) and widget != self.btn_toate_clasele:
                if widget.cget("text") == nume_clasa:
                    widget.configure(fg_color="#1f538d")
                else:
                    widget.configure(fg_color="#2b2b2b")

        self.incarca_imagini()

    def creeaza_fila_clasa(self, nume):
        if nume.lower() == "all":
            return
        if nume not in self.clase_existente:
            self.clase_existente.append(nume)

        for widget in self.tab_scroll_frame.winfo_children():
            if isinstance(widget, customtkinter.CTkButton) and widget.cget("text") == nume:
                return

        fila = customtkinter.CTkButton(
            master=self.tab_scroll_frame, text=nume, width=120, height=32, 
            fg_color="#2b2b2b", hover_color="#3b3b3b", command=lambda: self.filtreaza_dupa_clasa(nume)
        )
        fila.pack(side="left", padx=5)

        if nume not in self.snipper.classes:
            self.snipper.classes.append(nume)
            self.snipper.save_classes()

    def _pe_enter_imagine(self, event, cale_imagine):
        if self.timer_previzualizare is not None:
            self.after_cancel(self.timer_previzualizare)
        self.timer_previzualizare = self.after(500, lambda: self._arata_previzualizare_hover(cale_imagine, event.x_root, event.y_root))

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
        self.fereastra_previzualizare.geometry(f"+{x_root + 15}+{y_root + 15}")
        
        try:
            imagine_originala = Image.open(cale_imagine).convert("RGBA")
            imagine_originala.thumbnail((450, 450), Image.Resampling.LANCZOS)
            imagine_mare_ctk = customtkinter.CTkImage(light_image=imagine_originala, dark_image=imagine_originala, size=imagine_originala.size)
            label_imagine = customtkinter.CTkLabel(self.fereastra_previzualizare, text="", image=imagine_mare_ctk)
            label_imagine.pack(padx=5, pady=5)
        except Exception as e:
            print(f"Eroare hover: {e}")
            if self.fereastra_previzualizare:
                self.fereastra_previzualizare.destroy()
                self.fereastra_previzualizare = None

    def sterge_imagine(self, calea_imagine, nume_fara_extensie, nume_clasa_efectiva):
        """Șterge imaginea curentă și fișierul de adnotare asociat, apoi reîncarcă galeria."""
        try:
            if os.path.exists(calea_imagine):
                os.remove(calea_imagine)
            
            if nume_clasa_efectiva and nume_clasa_efectiva.lower() != "all":
                cale_txt = os.path.join(self.DIRECTOR_CURENT, "labels", nume_clasa_efectiva, nume_fara_extensie + ".txt")
                if os.path.exists(cale_txt):
                    os.remove(cale_txt)
            else:
                for c_ex in self.clase_existente:
                    cale_txt = os.path.join(self.DIRECTOR_CURENT, "labels", c_ex, nume_fara_extensie + ".txt")
                    if os.path.exists(cale_txt):
                        os.remove(cale_txt)
                        break
            
            self.incarca_imagini()
        except Exception as e:
            print(f"Eroare la ștergerea imaginii: {e}")

    def incarca_imagini(self):
        if not os.path.exists(self.MAPA_IMAGINI):
            os.makedirs(self.MAPA_IMAGINI)
            return

        for widget in self.galerie_imagini.winfo_children():
            widget.destroy()
        self.imagini_salvate.clear()

        extensii_valide = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        coloana_curenta, randul_curent, maxim_coloane = 0, 0, 6

        foldere_de_verificat = []
        if self.clasa_selectata is None:
            foldere_de_verificat.append(self.MAPA_ALL)
            if os.path.exists(self.MAPA_IMAGINI):
                for nume_folder in os.listdir(self.MAPA_IMAGINI):
                    cale_f = os.path.join(self.MAPA_IMAGINI, nume_folder)
                    if os.path.isdir(cale_f) and nume_folder.lower() != "all":
                        foldere_de_verificat.append(cale_f)
        else:
            cale_clasa_selectata = os.path.join(self.MAPA_IMAGINI, self.clasa_selectata)
            if os.path.exists(cale_clasa_selectata):
                foldere_de_verificat.append(cale_clasa_selectata)

        for cale_folder in foldere_de_verificat:
            if not os.path.exists(cale_folder):
                continue
            
            nume_clasa_folder = os.path.basename(cale_folder)
            is_in_all_folder = (nume_clasa_folder.lower() == "all")

            for nume_fisier in os.listdir(cale_folder):
                if nume_fisier.lower().endswith(extensii_valide):
                    cale_completa = os.path.join(cale_folder, nume_fisier)
                    nume_fara_extensie, _ = os.path.splitext(nume_fisier)
                    
                    nume_clasa_efectiva = "all"
                    este_adnotata = False

                    if not is_in_all_folder:
                        nume_clasa_efectiva = nume_clasa_folder
                        cale_adnotare = os.path.join(self.DIRECTOR_CURENT, "labels", nume_clasa_folder, nume_fara_extensie + ".txt")
                        este_adnotata = os.path.exists(cale_adnotare)
                    else:
                        for c_ex in self.clase_existente:
                            cale_txt_test = os.path.join(self.DIRECTOR_CURENT, "labels", c_ex, nume_fara_extensie + ".txt")
                            if os.path.exists(cale_txt_test):
                                este_adnotata = True
                                nume_clasa_efectiva = c_ex
                                break

                    try:
                        imagine_originala = Image.open(cale_completa)
                        
                        if imagine_originala.mode in ('RGBA', 'LA') or (imagine_originala.mode == 'P' and 'transparency' in imagine_originala.info):
                            fundal = Image.new("RGBA", imagine_originala.size, "#2b2b2b")
                            if imagine_originala.mode != 'RGBA':
                                imagine_originala = imagine_originala.convert('RGBA')
                            imagine_pil = Image.alpha_composite(fundal, imagine_originala).convert("RGB")
                        else:
                            imagine_pil = imagine_originala.convert("RGB")

                        imagine_pil = imagine_pil.resize((150, 150), Image.Resampling.LANCZOS)

                        # Triunghi albastru în dreapta-sus pentru imagini adnotate
                        if self.clasa_selectata is None and este_adnotata:
                            draw_b = ImageDraw.Draw(imagine_pil)
                            puncte_triunghi = [(149, 0), (119, 0), (149, 30)]
                            draw_b.polygon(puncte_triunghi, fill="#1f538d")

                        # Creăm variantele de imagini pentru stare normală și hover
                        img_normal = imagine_pil.copy()
                        img_hover = imagine_pil.copy()

                        def deseneaza_x(img, culoare_x):
                            draw = ImageDraw.Draw(img)
                            # Umbră neagră subțire
                            draw.line([(5, 5), (13, 13)], fill="#000000", width=3)
                            draw.line([(5, 13), (13, 5)], fill="#000000", width=3)
                            # Linia X-ului mai mică (8x8px) și mai subțire
                            draw.line([(5, 5), (13, 13)], fill=culoare_x, width=1)
                            draw.line([(5, 13), (13, 5)], fill=culoare_x, width=1)

                        deseneaza_x(img_normal, "#888888")  # Gri normal
                        deseneaza_x(img_hover, "#e0e0e0")   # Gri deschis la hover

                        ctk_img_normal = customtkinter.CTkImage(light_image=img_normal, dark_image=img_normal, size=(150, 150))
                        ctk_img_hover = customtkinter.CTkImage(light_image=img_hover, dark_image=img_hover, size=(150, 150))

                        self.imagini_salvate.append(ctk_img_normal)
                        self.imagini_salvate.append(ctk_img_hover)

                        nume_scurt = nume_fisier if len(nume_fisier) < 15 else nume_fisier[:12] + "..."
                        
                        if self.clasa_selectata is None:
                            text_eticheta = f"[{nume_clasa_efectiva.upper()}]\n{nume_scurt}"
                        else:
                            text_eticheta = nume_scurt
                        
                        cadru_imagine = customtkinter.CTkFrame(
                            master=self.galerie_imagini, fg_color="transparent", corner_radius=0
                        )
                        cadru_imagine.grid(row=randul_curent, column=coloana_curenta, padx=15, pady=15)

                        container_interior = customtkinter.CTkFrame(master=cadru_imagine, fg_color="transparent")
                        container_interior.pack(padx=0, pady=0)

                        iconita = customtkinter.CTkLabel(
                            master=container_interior, text="", image=ctk_img_normal, 
                            cursor="hand2", fg_color="transparent", bg_color="transparent"
                        )
                        iconita.pack()

                        label_text = customtkinter.CTkLabel(
                            master=container_interior, text=text_eticheta,
                            font=("Roboto", 12), fg_color="transparent", cursor="hand2"
                        )
                        label_text.pack(pady=(4, 0))

                        # Funcție izolată pentru gestionarea evenimentelor specifice FIECĂREI poze în parte (rezolvă bug-ul de late binding)
                        def creeaza_evenimente_imagine(lbl_ico, img_norm, img_hov, path_c, name_fe, name_ce):
                            is_hovered_x = [False]

                            def pe_motion_iconita(e):
                                in_zona_x = (e.x <= 20 and e.y <= 20)  # Zona activă redusă
                                if in_zona_x and not is_hovered_x[0]:
                                    is_hovered_x[0] = True
                                    lbl_ico.configure(image=img_hov)
                                elif not in_zona_x and is_hovered_x[0]:
                                    is_hovered_x[0] = False
                                    lbl_ico.configure(image=img_norm)

                            def pe_leave_iconita(e):
                                is_hovered_x[0] = False
                                lbl_ico.configure(image=img_norm)
                                self._pe_leave_imagine(e)

                            def pe_click_imagine(e):
                                if e.x <= 20 and e.y <= 20:
                                    self.sterge_imagine(path_c, name_fe, name_ce)
                                else:
                                    nc = None if name_ce == "all" else name_ce
                                    self.snipper.open_annotation_ui(image_path=path_c, default_class=nc)

                            return pe_motion_iconita, pe_leave_iconita, pe_click_imagine

                        motion_h, leave_h, click_h = creeaza_evenimente_imagine(
                            iconita, ctk_img_normal, ctk_img_hover, cale_completa, nume_fara_extensie, nume_clasa_efectiva
                        )

                        def deschide_adnotare_text(e, c=cale_completa, cl=nume_clasa_efectiva):
                            nc = None if cl == "all" else cl
                            self.snipper.open_annotation_ui(image_path=c, default_class=nc)

                        iconita.bind("<Motion>", motion_h)
                        iconita.bind("<Button-1>", click_h)
                        iconita.bind("<Leave>", leave_h)

                        label_text.bind("<Button-1>", deschide_adnotare_text)
                        container_interior.bind("<Button-1>", deschide_adnotare_text)
                        cadru_imagine.bind("<Button-1>", deschide_adnotare_text)

                        def enter_widget(e, cale=cale_completa):
                            self._pe_enter_imagine(e, cale)

                        iconita.bind("<Enter>", enter_widget)
                        label_text.bind("<Enter>", enter_widget)
                        label_text.bind("<Leave>", self._pe_leave_imagine)
                        cadru_imagine.bind("<Enter>", enter_widget)
                        cadru_imagine.bind("<Leave>", self._pe_leave_imagine)

                        coloana_curenta += 1
                        if coloana_curenta >= maxim_coloane:
                            coloana_curenta = 0  
                            randul_curent += 1   
                    except Exception as e:
                        print(f"Eroare la încărcarea {nume_fisier}: {e}")
    def actualizeaza_modele(self):
        if not os.path.exists(self.MAPA_MODELE):
            os.makedirs(self.MAPA_MODELE)
        fisiere_modele = [f for f in os.listdir(self.MAPA_MODELE) if os.path.isfile(os.path.join(self.MAPA_MODELE, f))]
        if not fisiere_modele:
            fisiere_modele = ["Niciun model găsit în mapă"]
        self.meniu_modele.configure(values=fisiere_modele)
        self.meniu_modele.set("Model antrenat")