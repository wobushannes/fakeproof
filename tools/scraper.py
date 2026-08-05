import csv
import os
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import hashlib

class CSVSQLiteImporter:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV -> SQLite Importer (Keine Dubletten)")
        self.root.geometry("700x400")
        
        # DB Pfad (fest im Skript-Ordner)
        self.db_path = os.path.join(os.path.dirname(__file__), "text_corpus.db")
        self._init_db()
        
        # GUI Vars
        self.input_dir = tk.StringVar()
        self.status = tk.StringVar(value="Bereit")
        self.progress = tk.IntVar()
        self.table_name = tk.StringVar(value="speeches")
        
        self._build_ui()
        self._count_db_entries()
        
    def _init_db(self):
        """Erstellt Tabelle mit UNIQUE-Constraint (Text-Hash)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speeches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE NOT NULL,   -- MD5 für Dublettencheck
                text_content TEXT NOT NULL,
                source_file TEXT,
                char_count INTEGER,
                word_count INTEGER,
                import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON speeches (text_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON speeches (source_file);")
        conn.commit()
        conn.close()

    def _build_ui(self):
        # Header
        tk.Label(self.root, text="CSV -> SQLite Bulk Importer", font=("Arial", 14, "bold")).pack(pady=10)
        
        # DB Status
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5, fill="x", padx=20)
        tk.Label(status_frame, text="Datenbank:").pack(side=tk.LEFT)
        self.db_label = tk.Label(status_frame, text=self.db_path, fg="blue", font=("Arial", 8))
        self.db_label.pack(side=tk.LEFT, padx=10)
        self.entry_count_label = tk.Label(status_frame, text="📄 0 Einträge", fg="green")
        self.entry_count_label.pack(side=tk.RIGHT)
        
        # Input Ordner
        tk.Label(self.root, text="CSV-Ordner:").pack(anchor="w", padx=20, pady=(10,0))
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=20)
        tk.Entry(input_frame, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(input_frame, text="Durchsuchen", command=self._select_input).pack(side=tk.RIGHT, padx=5)
        
        # Tabelle
        tk.Label(self.root, text="Ziel-Tabelle:").pack(anchor="w", padx=20, pady=(10,0))
        table_frame = tk.Frame(self.root)
        table_frame.pack(fill="x", padx=20)
        tk.Entry(table_frame, textvariable=self.table_name, width=30).pack(side=tk.LEFT)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="🔍 Count DB", command=self._count_db_entries, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🧹 Leeren (Reset)", command=self._reset_db, width=15, bg="orange").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🚀 IMPORT STARTEN", command=self._import, width=20, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Progress
        tk.Label(self.root, text="Fortschritt:").pack(anchor="w", padx=20)
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress, length=600)
        self.progress_bar.pack(padx=20, pady=5)
        
        # Status / Log
        log_frame = tk.LabelFrame(self.root, text="Log", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = tk.Text(log_frame, height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        
        self.status_label = tk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w")
        self.status_label.pack(fill="x", padx=20, pady=5)

    def _select_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()

    def _count_db_entries(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM speeches")
        count = cursor.fetchone()[0]
        conn.close()
        self.entry_count_label.config(text=f"📄 {count} Einträge")
        self.log(f"📊 Datenbank enthält {count} Einträge (keine Dubletten)")

    def _reset_db(self):
        if messagebox.askyesno("Reset", "Wirklich alle Daten aus der DB löschen?"):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM speeches")
            conn.commit()
            conn.close()
            self._count_db_entries()
            self.log("🧹 Datenbank geleert.")

    def _import(self):
        input_path = self.input_dir.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Fehler", "Bitte gültigen CSV-Ordner wählen.")
            return

        csv_files = list(Path(input_path).glob("*.csv"))
        if not csv_files:
            messagebox.showerror("Fehler", f"Keine CSV-Dateien in {input_path}")
            return

        self.status.set("Import läuft...")
        self.progress.set(0)
        self.root.update()

        total_files = len(csv_files)
        inserted = 0
        skipped = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for idx, csv_file in enumerate(csv_files):
            try:
                with open(csv_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    if 'Speech' not in reader.fieldnames:
                        self.log(f"⚠️ Überspringe {csv_file.name}: Spalte 'Speech' fehlt.")
                        continue

                    for row_num, row in enumerate(reader):
                        text = row.get('Speech', '').strip()
                        if not text or len(text) < 20:
                            continue

                        # 1. Text bereinigen
                        clean_text = ' '.join(text.split())
                        
                        # 2. Hash für Dublettencheck (MD5)
                        text_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()
                        
                        # 3. Metriken
                        char_count = len(clean_text)
                        word_count = len(clean_text.split())

                        try:
                            # 4. INSERT (ignoriert automatisch, falls Hash existiert -> UNIQUE)
                            cursor.execute("""
                                INSERT OR IGNORE INTO speeches 
                                (text_hash, text_content, source_file, char_count, word_count)
                                VALUES (?, ?, ?, ?, ?)
                            """, (text_hash, clean_text, csv_file.name, char_count, word_count))
                            
                            if cursor.rowcount > 0:
                                inserted += 1
                            else:
                                skipped += 1
                                
                        except sqlite3.Error as e:
                            self.log(f"❌ DB-Fehler in {csv_file.name} Zeile {row_num}: {e}")

                    conn.commit()

            except Exception as e:
                self.log(f"❌ Fehler in {csv_file.name}: {str(e)}")

            # Progress
            self.progress.set(int((idx + 1) / total_files * 100))
            self.status.set(f"Verarbeitet: {idx+1}/{total_files} | Neu: {inserted} | Dubletten: {skipped}")
            self.root.update()

        conn.close()
        
        self.status.set(f"✅ Fertig! {inserted} neue Reden importiert, {skipped} Dubletten übersprungen.")
        self.log(f"🎉 Import abgeschlossen: +{inserted} neu, {skipped} Dubletten ignoriert.")
        self._count_db_entries()
        self.progress.set(100)

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVSQLiteImporter(root)
    root.mainloop()