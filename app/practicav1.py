import customtkinter
import os
from PIL import Image

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("1280x720")
app.title("Muia lungă mult aduce")

# --- Setăm căile către mapele noastre ---
DIRECTOR_CURENT = os.path.dirname(os.path.abspath(__file__))
MAPA_IMAGINI = os.path.join(DIRECTOR_CURENT, "images")
MAPA_MODELE = os.path.join(DIRECTOR_CURENT, "models") 

# --- Funcții pentru navigare ---
def arata_pagina_galerie():
    main_page.pack_forget()
    third_page.pack_forget()
    second_page.pack(fill="both", expand=True)
    incarca_imagini()

def arata_pagina_invatare():
    main_page.pack_forget()
    second_page.pack_forget()
    third_page.pack(fill="both", expand=True)
    actualizeaza_modele() 

def arata_pagina_principala():
    second_page.pack_forget()
    third_page.pack_forget()
    main_page.pack(fill="both", expand=True)

# --- Funcții pentru gestionarea claselor (Tab-uri) ---
def deschide_dialog_clasa():
    # Creăm fereastra mică (pop-up)
    dialog = customtkinter.CTkToplevel(app)
    dialog.geometry("450x220")
    dialog.title("Clasa")
    dialog.transient(app) # O face "slave" față de fereastra principală
    dialog.grab_set()     # Blochează interacțiunea cu fereastra principală până la închidere
    
    # Textul de deasupra
    label_dialog = customtkinter.CTkLabel(dialog, text="Scrieți denumirea clasei:", font=("Roboto", 16))
    label_dialog.pack(pady=(25, 10), padx=20, anchor="w")
    
    # Boxa de text (Entry)
    entry_clasa = customtkinter.CTkEntry(dialog, width=410, font=("Roboto", 14))
    entry_clasa.pack(pady=10, padx=20)
    
    # Un cadru la baza ferestrei pentru a așeza butoanele OK și Cancel pe dreapta
    frame_butoane = customtkinter.CTkFrame(dialog, fg_color="transparent")
    frame_butoane.pack(pady=(15, 0), padx=20, fill="x")
    
    def pe_ok():
        nume_clasa = entry_clasa.get()
        if nume_clasa.strip(): # Verificăm să nu fie gol
            creeaza_fila_clasa(nume_clasa)
        dialog.destroy() # Închidem fereastra după

    def pe_cancel():
        dialog.destroy()

    # Butoanele
    btn_cancel = customtkinter.CTkButton(frame_butoane, text="Cancel", width=100, fg_color="transparent", border_width=1, command=pe_cancel)
    btn_cancel.pack(side="right", padx=(10, 0))
    
    btn_ok = customtkinter.CTkButton(frame_butoane, text="Ok", width=100, command=pe_ok)
    btn_ok.pack(side="right")

def creeaza_fila_clasa(nume):
    # Creăm butonul ce acționează ca o filă, interiorul zonei de scroll orizontal
    fila = customtkinter.CTkButton(
        master=tab_scroll_frame, 
        text=nume, 
        width=120, 
        height=32,
        fg_color="#2b2b2b",  # Culoare ușor diferită pentru file
        hover_color="#3b3b3b"
    )
    fila.pack(side="left", padx=5)

# --- Funcție pentru încărcarea imaginilor ---
imagini_salvate = []

def incarca_imagini():
    if not os.path.exists(MAPA_IMAGINI):
        os.makedirs(MAPA_IMAGINI)
        return

    for widget in galerie_imagini.winfo_children():
        widget.destroy()
    imagini_salvate.clear()

    extensii_valide = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    fisiere_imagini = [f for f in os.listdir(MAPA_IMAGINI) if f.lower().endswith(extensii_valide)]

    coloana_curenta = 0
    randul_curent = 0
    maxim_coloane = 6  

    for nume_fisier in fisiere_imagini:
        cale_completa = os.path.join(MAPA_IMAGINI, nume_fisier)
        try:
            imagine_pil = Image.open(cale_completa)
            imagine_ctk = customtkinter.CTkImage(light_image=imagine_pil, dark_image=imagine_pil, size=(150, 150))
            imagini_salvate.append(imagine_ctk) 

            nume_afisat = nume_fisier if len(nume_fisier) < 15 else nume_fisier[:12] + "..."

            iconita = customtkinter.CTkLabel(
                master=galerie_imagini,
                text=nume_afisat,
                image=imagine_ctk,
                compound="top", 
                font=("Roboto", 12)
            )
            iconita.grid(row=randul_curent, column=coloana_curenta, padx=20, pady=20)

            coloana_curenta += 1
            if coloana_curenta >= maxim_coloane:
                coloana_curenta = 0  
                randul_curent += 1   
        except Exception as e:
            print(f"Eroare la încărcarea {nume_fisier}: {e}")

# --- Funcție pentru actualizarea listei de modele ---
def actualizeaza_modele():
    if not os.path.exists(MAPA_MODELE):
        os.makedirs(MAPA_MODELE)
    
    fisiere_modele = [f for f in os.listdir(MAPA_MODELE) if os.path.isfile(os.path.join(MAPA_MODELE, f))]
    
    if not fisiere_modele:
        fisiere_modele = ["Niciun model găsit în mapă"]
    
    meniu_modele.configure(values=fisiere_modele)
    meniu_modele.set("Model antrenat")  


# ==========================================
# 1. PAGINA PRINCIPALĂ
# ==========================================
main_page = customtkinter.CTkFrame(app, fg_color="transparent")
main_page.pack(fill="both", expand=True)

title = customtkinter.CTkLabel(master=main_page, text="Sigma", font=("Roboto", 32, "bold"))
title.pack(pady=(20, 0), padx=10)

frame_1 = customtkinter.CTkFrame(master=main_page)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)
frame_1.columnconfigure(0, weight=1)
frame_1.columnconfigure(1, weight=1)
frame_1.columnconfigure(2, weight=1)

column1 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
column1.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
button1 = customtkinter.CTkButton(master=column1, text="Galerie", command=arata_pagina_galerie)
button1.pack(pady=10, padx=10)

column2 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
column2.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=20)
button2 = customtkinter.CTkButton(master=column2, text="Învățare", command=arata_pagina_invatare)
button2.pack(pady=10, padx=10)

column3 = customtkinter.CTkFrame(master=frame_1, width=100, corner_radius=0)
column3.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)
button3 = customtkinter.CTkButton(master=column3, text="Detecție în timp real")
button3.pack(pady=10, padx=10)


# ==========================================
# 2. PAGINA NOUĂ (GALERIA)
# ==========================================
second_page = customtkinter.CTkFrame(app, fg_color="transparent")

# --- Bara de sus (Butonul + și filele scrollabile) ---
top_bar = customtkinter.CTkFrame(second_page, fg_color="transparent")
top_bar.pack(fill="x", padx=20, pady=10)

buton_adauga_clasa = customtkinter.CTkButton(top_bar, text="+", width=40, height=40, font=("Roboto", 18, "bold"), command=deschide_dialog_clasa)
buton_adauga_clasa.pack(side="left", padx=(0, 10))

# Cadru scrollabil orizontal pentru a ține filele (clasele) generate
tab_scroll_frame = customtkinter.CTkScrollableFrame(
    master=top_bar, 
    orientation="horizontal", 
    height=45, 
    fg_color="transparent",
    scrollbar_button_color="#242424", # Face scrollbar-ul aproape invizibil, apare la hover
    scrollbar_button_hover_color="gray"
)
tab_scroll_frame.pack(side="left", fill="x", expand=True)

# --- Bara de jos ---
bottom_bar = customtkinter.CTkFrame(second_page, fg_color="transparent")
bottom_bar.pack(fill="x", side="bottom", padx=20, pady=20)

buton_refresh_galerie = customtkinter.CTkButton(bottom_bar, text="Actualizează Galeria", command=incarca_imagini)
buton_refresh_galerie.pack(side="left")

buton_inapoi_galerie = customtkinter.CTkButton(bottom_bar, text="Înapoi", command=arata_pagina_principala)
buton_inapoi_galerie.pack(side="right")

# --- Zona de imagini ---
galerie_imagini = customtkinter.CTkScrollableFrame(second_page)
galerie_imagini.pack(fill="both", expand=True, padx=20, pady=10)


# ==========================================
# 3. PAGINA PENTRU ÎNVĂȚARE
# ==========================================
third_page = customtkinter.CTkFrame(app, fg_color="transparent")

titlu_invatare = customtkinter.CTkLabel(master=third_page, text="Meniul de Învățare", font=("Roboto", 28, "bold"))
titlu_invatare.pack(pady=(30, 10))

frame_optiuni_invatare = customtkinter.CTkFrame(master=third_page)
frame_optiuni_invatare.pack(pady=20, padx=60, fill="both", expand=True)
frame_optiuni_invatare.columnconfigure(0, weight=1)
frame_optiuni_invatare.columnconfigure(1, weight=1)

sectiune_neantrenat = customtkinter.CTkFrame(master=frame_optiuni_invatare, fg_color="transparent")
sectiune_neantrenat.grid(row=0, column=0, sticky="nsew", padx=20, pady=50)

btn_neantrenat = customtkinter.CTkButton(master=sectiune_neantrenat, text="Model neantrenat", height=40, font=("Roboto", 16))
btn_neantrenat.pack(expand=True) 

sectiune_antrenat = customtkinter.CTkFrame(master=frame_optiuni_invatare, fg_color="transparent")
sectiune_antrenat.grid(row=0, column=1, sticky="nsew", padx=20, pady=50)

meniu_modele = customtkinter.CTkOptionMenu(
    master=sectiune_antrenat, 
    values=["Se încarcă..."], 
    height=40, 
    font=("Roboto", 14),
    dynamic_resizing=True
)
meniu_modele.pack()

buton_inapoi_invatare = customtkinter.CTkButton(master=third_page, text="Înapoi", command=arata_pagina_principala)
buton_inapoi_invatare.pack(side="bottom", pady=20, padx=20, anchor="e")

app.mainloop()