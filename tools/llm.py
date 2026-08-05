import os
import random
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ====== KONFIGURATION ======
LLM_HOST = "http://localhost:1234/v1"
OUTPUT_DIR = "./fakes"
PROMPTS_FILE = "./prompts.json"
TOPICS_FILE = "./topics.json"
# ===========================

# Standard-Prompts
DEFAULT_PROMPTS = {
    "Politikerrede (konservativ)": """Du schreibst eine politische Rede im konservativen Stil.
- Formell, würdevoll, pathetisch
- Betont Tradition, Ordnung, Sicherheit, Leistung
- Verwendet Floskeln wie 'meine sehr geehrten Damen und Herren', 'Verantwortung', 'Fundament', 'Standhaftigkeit'
- Keine Umgangssprache, keine emotionalen Ausbrüche
- Struktur: Einleitung (Werte), Hauptteil (Themen), Schluss (Appell)""",

    "Politikerrede (progressiv)": """Du schreibst eine politische Rede im progressiven Stil.
- Sachlich, zukunftsorientiert, optimistisch
- Betont Gerechtigkeit, Nachhaltigkeit, Zusammenhalt, Wandel
- Verwendet Floskeln wie 'liebe Mitbürgerinnen und Mitbürger', 'gestalten', 'erneuern', 'fair'
- Keine Umgangssprache, aber emotionale Wärme
- Struktur: Einleitung (Vision), Hauptteil (Forderungen), Schluss (Aufbruch)""",

    "Politikerrede (populistisch)": """Du schreibst eine politische Rede im populistischen Stil.
- Emotional, polarisierend, vereinfachend
- Betont 'das Volk', 'Eliten', 'gesunder Menschenverstand', 'Ende des Stillstands'
- Verwendet Floskeln wie 'die da oben', 'wir Bürger', 'aufwachen', 'nicht länger hinnehmen'
- Nah an Umgangssprache, aber keine vulgären Ausdrücke
- Struktur: Einleitung (Problem), Hauptteil (Feindbild), Schluss (Lösung)""",

    "Wirtschaftsbericht": """Du schreibst einen sachlichen Wirtschaftsbericht.
- Nüchtern, datenbasiert, analytisch
- Betont Wachstum, Effizienz, Wettbewerb, Stabilität
- Verwendet Floskeln wie 'vor diesem Hintergrund', 'zusammenfassend', 'führt zu'
- Keine Emotionen, keine Umgangssprache
- Struktur: Einleitung (Lage), Hauptteil (Datenanalyse), Schluss (Ausblick)""",

    "Technologie-Erklärung": """Du erklärst ein technisches Thema für Laien.
- Verständlich, strukturiert, aber nicht zu vereinfacht
- Verwendet Analogien, Beispiele, Schritt-für-Schritt-Erklärungen
- Floskeln wie 'das bedeutet', 'zum Beispiel', 'vereinfacht gesagt'
- Keine Umgangssprache, aber nah am Alltag
- Struktur: Einleitung (Problem), Hauptteil (Funktionsweise), Schluss (Zusammenfassung)""",

    "Social Media Post (LinkedIn)": """Du schreibst einen professionellen LinkedIn-Post.
- Kurz, prägnant, inspirierend
- Betont persönliche Erfahrung, Erfolg, Lektionen
- Verwendet Floskeln wie 'ich habe gelernt', 'meine wichtigste Erkenntnis', 'teile ich mit euch'
- Keine Umgangssprache, aber persönlicher Ton
- Struktur: Haken (Hook), Hauptteil (Story), Schluss (Aufruf)""",

    "Social Media Post (Reddit)": """Du schreibst einen Reddit-Post (r/de oder r/FragReddit).
- Umgangssprachlich, direkt, persönlich
- Stellt eine Frage oder teilt eine Meinung
- Verwendet Floskeln wie 'ich frage mich', 'was haltet ihr von', 'meine Erfahrung zeigt'
- Nah an Alltagssprache, keine Beleidigungen
- Struktur: Einleitung (Kontext), Hauptteil (Frage/Meinung), Schluss (Aufruf)""",

    "Wissenschaftliche Zusammenfassung": """Du schreibst eine Zusammenfassung einer wissenschaftlichen Studie.
- Präzise, neutral, faktenbasiert
- Betont Methode, Ergebnisse, Schlussfolgerung
- Verwendet Floskeln wie 'die Studie zeigt', 'zusammenfassend', 'weiterer Forschungsbedarf'
- Keine Emotionen, keine Umgangssprache
- Struktur: Einleitung (Fragestellung), Hauptteil (Methodik/Ergebnisse), Schluss (Fazit)""",

    "Kochrezept": """Du schreibst ein Kochrezept.
- Klar, Schritt-für-Schritt, präzise
- Betont Zutaten, Zubereitung, Tipps
- Verwendet Floskeln wie 'zunächst', 'dann', 'zum Schluss'
- Keine Umgangssprache, aber einladender Ton
- Struktur: Zutatenliste, Zubereitung (Schritte), Tipps""",

    "Freie Unterhaltung": """Du schreibst einen unterhaltsamen, lockeren Text über ein Alltagsthema.
- Umgangssprachlich, humorvoll, persönlich
- Keine strengen Regeln – einfach schreiben, wie man mit Freunden redet
- Verwendet Floskeln wie 'ich meine', 'echt jetzt', 'also'
- Nah an gesprochener Sprache
- Keine feste Struktur""",

    "Nachrichtenartikel": """Du schreibst einen sachlichen Nachrichtenartikel.
- Neutral, faktenbasiert, journalistisch
- Betont Ereignisse, Hintergründe, Reaktionen
- Verwendet Floskeln wie 'wie bekannt wurde', 'nach Angaben von', 'derweil'
- Keine Emotionen, keine Umgangssprache
- Struktur: Einleitung (Ereignis), Hauptteil (Hintergründe), Schluss (Ausblick)""",

    "Essay (philosophisch)": """Du schreibst einen philosophischen Essay.
- Tiefgründig, reflektierend, komplex
- Betont Gedanken, Fragen, Perspektiven
- Verwendet Floskeln wie 'man könnte argumentieren', 'vor diesem Hintergrund', 'letztlich'
- Keine Umgangssprache, aber persönlicher Ton
- Struktur: Einleitung (These), Hauptteil (Argumentation), Schluss (Fazit)"""
}

DEFAULT_TOPICS = [
    "Klimawandel und seine Auswirkungen",
    "Die Zukunft der künstlichen Intelligenz",
    "Bildungspolitik in Deutschland",
    "Nachhaltige Wirtschaft",
    "Digitale Transformation im öffentlichen Sektor",
    "Soziale Ungleichheit",
    "Gesundheitssystem Reformen",
    "Mobilität der Zukunft",
    "Europäische Integration",
    "Demographischer Wandel",
    "Chancen und Risiken der Digitalisierung",
    "Arbeitsmarkt der Zukunft",
    "Umweltschutz und Wirtschaftswachstum",
    "Bürgerrechte im digitalen Zeitalter",
    "Energiewende und Versorgungssicherheit",
    "Integration und Migration",
    "Steuerpolitik und soziale Gerechtigkeit",
    "Wohnungsmarkt und bezahlbarer Wohnraum",
    "Landwirtschaft und Ernährungssicherheit",
    "Kulturelle Identität in der globalisierten Welt"
]

class LLMSpammerPro:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM-Spammer Pro - Multi-Prompt Generator")
        self.root.geometry("1200x800")
        
        # Verbindung
        self.client = None
        self.connected = False
        
        # Prompts & Topics laden
        self.prompts = self._load_prompts()
        self.topics = self._load_topics()
        self.current_prompt_name = tk.StringVar(value=list(self.prompts.keys())[0] if self.prompts else "")
        
        # GUI bauen
        self._build_ui()
        self._check_connection()
        self._update_prompt_display()
        
    def _load_prompts(self):
        if os.path.exists(PROMPTS_FILE):
            try:
                with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_PROMPTS, f, indent=2, ensure_ascii=False)
        return DEFAULT_PROMPTS.copy()
    
    def _load_topics(self):
        if os.path.exists(TOPICS_FILE):
            try:
                with open(TOPICS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        with open(TOPICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_TOPICS, f, indent=2, ensure_ascii=False)
        return DEFAULT_TOPICS.copy()
    
    def _save_prompts(self):
        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.prompts, f, indent=2, ensure_ascii=False)
    
    def _save_topics(self):
        # Topics aus Textfeld holen
        topics_raw = self.topics_text.get("1.0", tk.END).strip()
        topics = [t.strip() for t in topics_raw.split("\n") if t.strip()]
        self.topics = topics
        with open(TOPICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)
    
    def _build_ui(self):
        # Header
        tk.Label(self.root, text="LLM-Spammer Pro", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Status
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5, fill="x", padx=20)
        self.status_label = tk.Label(status_frame, text="🔴 Nicht verbunden", fg="red", font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        tk.Button(status_frame, text="🔄 Verbindung prüfen", command=self._check_connection).pack(side=tk.LEFT, padx=10)
        
        # Hauptcontainer
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ===== LINKER BEREICH: Prompt-Manager =====
        left_frame = ttk.LabelFrame(main_frame, text="Prompt-Manager", padding=10)
        left_frame.pack(side=tk.LEFT, fill="both", expand=False, padx=(0, 10), ipadx=10)
        
        # Prompt-Liste
        tk.Label(left_frame, text="Verfügbare Prompts:").pack(anchor="w")
        self.prompt_listbox = tk.Listbox(left_frame, height=10, width=35)
        self.prompt_listbox.pack(fill="x", pady=5)
        self.prompt_listbox.bind('<<ListboxSelect>>', self._on_prompt_select)
        
        # Buttons für Prompt-Management
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill="x", pady=5)
        tk.Button(btn_frame, text="➕ Neu", command=self._new_prompt, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="💾 Speichern", command=self._save_prompt, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="🗑 Löschen", command=self._delete_prompt, width=8).pack(side=tk.LEFT, padx=2)
        
        # Prompt-Name
        tk.Label(left_frame, text="Name:").pack(anchor="w", pady=(10,0))
        self.prompt_name_entry = tk.Entry(left_frame, width=35, textvariable=self.current_prompt_name)
        self.prompt_name_entry.pack(fill="x", pady=5)
        
        # Prompt-Text
        tk.Label(left_frame, text="System-Prompt:").pack(anchor="w")
        self.prompt_text = scrolledtext.ScrolledText(left_frame, height=15, width=40)
        self.prompt_text.pack(fill="both", expand=True, pady=5)
        
        # ===== RECHTER BEREICH: Generierung =====
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill="both", expand=True)
        
        # Konfiguration
        config_frame = ttk.LabelFrame(right_frame, text="Generierungs-Konfiguration", padding=10)
        config_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(config_frame, text="Anzahl Texte:").grid(row=0, column=0, sticky="w")
        self.num_texts = tk.IntVar(value=10)
        tk.Spinbox(config_frame, from_=1, to=1000, textvariable=self.num_texts, width=10).grid(row=0, column=1, padx=10)
        
        tk.Label(config_frame, text="Zielordner:").grid(row=1, column=0, sticky="w", pady=5)
        self.output_dir = tk.StringVar(value=OUTPUT_DIR)
        tk.Entry(config_frame, textvariable=self.output_dir, width=40).grid(row=1, column=1, padx=10)
        tk.Button(config_frame, text="📁", command=self._select_folder).grid(row=1, column=2)
        
        # Themen-Editor
        topic_frame = ttk.LabelFrame(right_frame, text="Themen (eine Zeile = ein Thema) - EDITIERBAR", padding=10)
        topic_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Toolbar für Topics
        topic_toolbar = tk.Frame(topic_frame)
        topic_toolbar.pack(fill="x", pady=(0, 5))
        tk.Button(topic_toolbar, text="💾 Themen speichern", command=self._save_topics, bg="lightblue").pack(side=tk.LEFT, padx=2)
        tk.Button(topic_toolbar, text="📂 Themen laden", command=self._load_topics_from_file, bg="lightgray").pack(side=tk.LEFT, padx=2)
        tk.Button(topic_toolbar, text="➕ Standard zurücksetzen", command=self._reset_topics, bg="lightyellow").pack(side=tk.LEFT, padx=2)
        
        self.topics_text = scrolledtext.ScrolledText(topic_frame, height=8)
        self.topics_text.pack(fill="both", expand=True)
        # Standard-Themen einfügen
        self.topics_text.insert("1.0", "\n".join(self.topics))
        
        # Generierungs-Button
        self.generate_btn = tk.Button(
            right_frame,
            text="🚀 TEXTE GENERIEREN",
            command=self._generate,
            bg="green",
            fg="white",
            height=2,
            width=30,
            font=("Arial", 12, "bold")
        )
        self.generate_btn.pack(pady=10)
        
        # Progress
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, length=500)
        self.progress_bar.pack(pady=5)
        self.progress_label = tk.Label(right_frame, text="Bereit")
        self.progress_label.pack()
        
        # Log
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(10,0))
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True)
        
        # Initiale Prompt-Liste füllen
        self._refresh_prompt_list()
        
    def _select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
    
    def _refresh_prompt_list(self):
        self.prompt_listbox.delete(0, tk.END)
        for name in self.prompts.keys():
            self.prompt_listbox.insert(tk.END, name)
    
    def _on_prompt_select(self, event):
        selection = self.prompt_listbox.curselection()
        if selection:
            name = self.prompt_listbox.get(selection[0])
            self.current_prompt_name.set(name)
            self._update_prompt_display()
    
    def _update_prompt_display(self):
        name = self.current_prompt_name.get()
        if name in self.prompts:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", self.prompts[name])
    
    def _new_prompt(self):
        name = f"Neuer Prompt {len(self.prompts)+1}"
        self.prompts[name] = "Schreibe einen Text im Stil von..."
        self.current_prompt_name.set(name)
        self._refresh_prompt_list()
        self._update_prompt_display()
        self.log(f"➕ Neuer Prompt erstellt: {name}")
    
    def _save_prompt(self):
        name = self.current_prompt_name.get()
        if not name.strip():
            messagebox.showerror("Fehler", "Name darf nicht leer sein")
            return
        text = self.prompt_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Fehler", "Prompt-Text darf nicht leer sein")
            return
        self.prompts[name] = text
        self._save_prompts()
        self._refresh_prompt_list()
        self.log(f"💾 Prompt gespeichert: {name}")
    
    def _delete_prompt(self):
        name = self.current_prompt_name.get()
        if name in self.prompts:
            if messagebox.askyesno("Löschen", f"Prompt '{name}' wirklich löschen?"):
                del self.prompts[name]
                self._save_prompts()
                self._refresh_prompt_list()
                if self.prompts:
                    self.current_prompt_name.set(list(self.prompts.keys())[0])
                    self._update_prompt_display()
                self.log(f"🗑 Prompt gelöscht: {name}")
    
    def _load_topics_from_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("Text", "*.txt")])
        if path:
            try:
                if path.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        topics = json.load(f)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        topics = [t.strip() for t in f.readlines() if t.strip()]
                self.topics_text.delete("1.0", tk.END)
                self.topics_text.insert("1.0", "\n".join(topics))
                self.log(f"📂 Themen geladen: {len(topics)} Themen")
            except Exception as e:
                self.log(f"❌ Fehler beim Laden: {e}")
    
    def _reset_topics(self):
        if messagebox.askyesno("Reset", "Standard-Themen wiederherstellen?"):
            self.topics_text.delete("1.0", tk.END)
            self.topics_text.insert("1.0", "\n".join(DEFAULT_TOPICS))
            self.log("↩️ Standard-Themen wiederhergestellt")
    
    def _check_connection(self):
        try:
            test_client = OpenAI(base_url=LLM_HOST, api_key="not-needed")
            models = test_client.models.list()
            if models.data:
                self.client = test_client
                self.connected = True
                self.status_label.config(text="✅ Verbunden (Modell geladen)", fg="green")
                self.log("✅ LLM Studio verbunden - Modell bereit")
                return True
            else:
                self.connected = False
                self.status_label.config(text="⚠️ Verbunden, aber kein Modell geladen", fg="orange")
                self.log("⚠️ LLM Studio verbunden, aber kein Modell geladen")
                return False
        except Exception as e:
            self.connected = False
            self.status_label.config(text="🔴 Nicht verbunden", fg="red")
            self.log(f"❌ Keine Verbindung zu LLM Studio: {str(e)}")
            return False
    
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def _generate(self):
        if not self.connected:
            messagebox.showerror("Fehler", "LLM Studio nicht verbunden!")
            return
        
        # Daten holen
        num = self.num_texts.get()
        output_dir = self.output_dir.get()
        prompt_name = self.current_prompt_name.get()
        system_prompt = self.prompts.get(prompt_name)
        
        if not system_prompt:
            messagebox.showerror("Fehler", "System-Prompt nicht gefunden")
            return
        
        # Themen aus Textfeld holen
        topics_raw = self.topics_text.get("1.0", tk.END).strip()
        topics = [t.strip() for t in topics_raw.split("\n") if t.strip()]
        if not topics:
            messagebox.showerror("Fehler", "Mindestens ein Thema erforderlich")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Nächste ID finden
        next_id = self._get_next_id(output_dir)
        
        self.generate_btn.config(state="disabled", text="⏳ Generiere...")
        self.progress_var.set(0)
        self.progress_label.config(text=f"Starte mit ID {next_id}")
        self.log(f"🚀 Generiere {num} Texte mit Prompt: {prompt_name}")
        self.log(f"   Themen: {len(topics)} verfügbar")
        
        success_count = 0
        
        for i in range(num):
            topic = random.choice(topics)
            
            try:
                response = self.client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Thema: {topic}\nSchreibe einen zusammenhängenden Text (ca. 200-400 Wörter)."}
                    ],
                    temperature=0.8,
                    max_tokens=1024,
                    top_p=0.95
                )
                text = response.choices[0].message.content.strip()
                
                if text and len(text) > 100:
                    filename = f"fake_{next_id:04d}.txt"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    success_count += 1
                    self.log(f"✅ {filename} gespeichert (Thema: {topic[:30]}...)")
                    next_id += 1
                else:
                    self.log(f"⚠️ Text {i+1} zu kurz oder leer")
                    
            except Exception as e:
                self.log(f"❌ Fehler bei Text {i+1}: {str(e)}")
            
            progress = int((i + 1) / num * 100)
            self.progress_var.set(progress)
            self.progress_label.config(text=f"Fortschritt: {i+1}/{num} ({success_count} erfolgreich)")
            self.root.update()
        
        self.progress_label.config(text=f"Fertig! {success_count} Texte generiert in {output_dir}")
        self.generate_btn.config(state="normal", text="🚀 TEXTE GENERIEREN")
        self.log(f"\n✅ {success_count} Texte erfolgreich generiert (IDs {next_id-success_count} bis {next_id-1})")
        self.progress_var.set(0)
    
    def _get_next_id(self, output_dir):
        existing = list(Path(output_dir).glob("fake_*.txt"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.split("_")[1])
                numbers.append(num)
            except:
                continue
        return max(numbers) + 1 if numbers else 1

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMSpammerPro(root)
    root.mainloop()