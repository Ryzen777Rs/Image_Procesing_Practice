# Image_Procesing_Practice


<br />
<div align="center">
  <!-- Înlocuiți calea de mai jos cu logo-ul real al proiectului vostru -->
    <img src="app/logos/night.gif" alt="Logo" width="670" height="670">
  </a>

  <h3 align="center">Detectarea Obiectelor și Antrenarea Modelelor</h3>
</div>


## 🚀 Despre Proiect

Acest proiect reprezintă o soluție software integrată, dezvoltată în Python, concepută pentru recunoașterea vizuală a obiectelor și antrenarea personalizată a modelelor de inteligență artificială. Aplicația oferă o interfață grafică intuitivă, împărțită în module distincte pentru detecție în timp real și gestionarea seturilor de date.

### 🛠️ Tehnologii Utilizate

* [![Python][Python.org]][Python-url]
* [![YOLO][Yolo-badge]][Yolo-url]

---

## 💻 Instrucțiuni de Instalare și Rulare

Pentru a asigura o experiență de utilizare fluidă, procesul de instalare a fost automatizat.

1. **Pregătirea mediului:** Nu este necesară instalarea manuală a limbajului Python. Pachetul include un script de configurare care va instala automat versiunea Python 3.12.17 și dependențele necesare.
2. **Lansarea aplicației:** Navigați în folderul principal al proiectului și rulați fișierul `Start.bat`. Procesul de inițializare poate dura câteva momente, după care se va deschide fereastra principală a programului.
---

## 🎯 Arhitectura și Funcționalitățile Programului

Aplicația este structurată în două componente majore, accesibile prin intermediul interfeței grafice: Modulul de Detecție și Modulul de Antrenare.

### 🔍 1. Modulul de Detecție
Aceasta este fereastra principală a aplicației, dedicată identificării obiectelor pe baza modelelor preexistente.

* **🎥 Sursa de intrare:** Utilizatorul poate selecta fluxul video de la o cameră web conectată sau poate opta pentru capturarea ecranului (Screen Capture).
* **⚙️ Pragul de încredere (Confidence):** Permite ajustarea nivelului de precizie. O valoare mai mare va forța modelul să afișeze doar detecțiile de care este foarte sigur.
* **🧠 Gestionarea Modelelor YOLO:** 
  * Interfața include butonul **Install YOLO Models**.
  * Toodată, butonul din colț stânga este responsabil de alegerea modelului curent
* **🎨 Personalizare:** Buton dedicat în colțul din stânga-sus pentru schimbarea temei aplicației.
* **🚀 Tranziția:** Accesarea butonului **Antrenează** va deschide o fereastră nouă pentru modelare avansată.

### 📈 2. Modulul de Antrenare
Acest modul avansat este divizat în trei secțiuni principale: Galerie, Învățare și Detecție LIVE.

#### 🖼️ Galerie (Gestionarea Seturilor de Date)
* **Import:** Imaginile pot fi adăugate din fișierele locale sau importate direct de pe internet.
* **Sistem de Adnotare:** Utilizatorii pot adnota manual imaginile salvate, atribuindu-le clase specifice.
* **Snipping Tool:** Prin scurtătura `Ctrl+Shift+S`, utilizatorul poate decupa o porțiune de pe ecran care va fi salvată automat în Galerie pentru adnotare.


#### ⚙️ Învățare (Configurarea Antrenamentului)
Interfața este organizată în trei coloane logice:
* **Coloana 1 (Clase):** Afișează lista tuturor claselor definite anterior.
* **Coloana 2 (Parametri):** Permite configurarea detaliată a procesului (epoci, dimensiune, etc.), folosind inițial date auto-detectate optime.
* **Coloana 3 (Execuție):** Se selectează modelul de bază, se atribuie o denumire și se apasă butonul **ANTRENARE**.

#### 🔴 Detecție LIVE (Urmărire Dinamică)
* Utilizatorul plasează un obiect în fața camerei și îi conturează manual perimetrul pe ecran. Algoritmul AI va prelua acest contur și va încerca să urmărească (track) obiectul în timp real.

---

## 👥 Contribuitori

Acest proiect a fost realizat cu efortul comun al următoarei echipe:

| Rol | Nume |
| :--- | :--- |
| **👨‍💻 Dezvoltator** | Gudîma Liviu |
| **👨‍💻 Dezvoltator** | Șilo Alexandr |


<!-- MARKDOWN LINKS & IMAGES -->
<!-- Acestea sunt variabilele folosite pentru insigne. Înlocuiți "nume-utilizator/nume-repo" cu datele voastre reale de GitHub. -->
[contributors-shield]: https://img.shields.io/github/contributors/Ryzen777Rs/Ryzen777Rs.svg?style=for-the-badge
[contributors-url]: https://github.com/Ryzen777Rs
[forks-shield]: https://img.shields.io/github/forks/nume-utilizator/nume-repo.svg?style=for-the-badge
[forks-url]: https://github.com/nume-utilizator/nume-repo/network/members
[stars-shield]: https://img.shields.io/github/stars/nume-utilizator/nume-repo.svg?style=for-the-badge
[stars-url]: https://github.com/nume-utilizator/nume-repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/nume-utilizator/nume-repo.svg?style=for-the-badge
[issues-url]: https://github.com/nume-utilizator/nume-repo/issues
[Python.org]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Yolo-badge]: https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black
[Yolo-url]: https://ultralytics.com/