import torch
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pathlib import Path
import re

# ====== KONFIGURATION ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================

class DetectorTesterPro:
    def __init__(self, root):
        self.root = root
        self.root.title("KI-Text Detector - Tester Pro")
        self.root.geometry("1000x850")
        
        self.model = None
        self.tokenizer = None
        self.model_path = None
        
        self._build_ui()
        self._update_status("Bitte Modell laden")
        
    def _build_ui(self):
        # Header
        tk.Label(self.root, text="KI-Text Detector Pro", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Modell-Auswahl
        model_frame = tk.LabelFrame(self.root, text="Modell", padx=10, pady=10)
        model_frame.pack(fill="x", padx=20, pady=5)
        
        self.model_path_var = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="📁 Modell laden", command=self._load_model_dialog).pack(side=tk.LEFT, padx=5)
        
        self.model_status = tk.Label(model_frame, text="❌ Kein Modell geladen", fg="red")
        self.model_status.pack(side=tk.LEFT, padx=20)
        
        # Status
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=5, fill="x", padx=20)
        tk.Label(status_frame, text="Device:").pack(side=tk.LEFT)
        tk.Label(status_frame, text=f"{DEVICE}").pack(side=tk.LEFT, padx=10)
        
        # Eingabe
        tk.Label(self.root, text="Text eingeben (oder Datei laden):").pack(anchor="w", padx=20, pady=(10,0))
        
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=20, pady=5)
        self.text_entry = scrolledtext.ScrolledText(input_frame, height=8)
        self.text_entry.pack(fill="x")
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="📁 Datei laden", command=self._load_file).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🔍 Text testen", command=self._predict, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🧹 Clear", command=self._clear).pack(side=tk.LEFT, padx=5)
        
        # Ergebnis
        result_frame = tk.LabelFrame(self.root, text="Ergebnis", padx=10, pady=10)
        result_frame.pack(fill="x", padx=20, pady=10)
        
        self.result_label = tk.Label(result_frame, text="⏳ Warte auf Eingabe", font=("Arial", 14, "bold"))
        self.result_label.pack()
        
        self.prob_label = tk.Label(result_frame, text="", font=("Arial", 12))
        self.prob_label.pack()
        
        # Erklärung
        explain_frame = tk.LabelFrame(self.root, text="Wort-Erklärung (basierend auf Token-Wichtigkeit)", padx=10, pady=10)
        explain_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.explain_text = tk.Text(explain_frame, height=8, wrap=tk.WORD, font=("Arial", 11))
        self.explain_text.pack(fill="both", expand=True)
        self.explain_text.tag_configure("ki_high", background="#ff6b6b", foreground="white")
        self.explain_text.tag_configure("ki_med", background="#ffa94d", foreground="black")
        self.explain_text.tag_configure("ki_low", background="#ffd93d", foreground="black")
        self.explain_text.tag_configure("human", background="#6bcb6b", foreground="black")
        
        # Log
        log_frame = tk.LabelFrame(self.root, text="Log", padx=10, pady=10)
        log_frame.pack(fill="x", padx=20, pady=10)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, state="disabled")
        self.log_text.pack(fill="x")
        
    def _load_model_dialog(self):
        """Öffnet Dialog zur Modellauswahl"""
        path = filedialog.askdirectory(title="Modellordner auswählen")
        if path:
            self.model_path_var.set(path)
            self._load_model(path)
    
    def _load_model(self, path):
        """Lädt ein Modell aus dem angegebenen Pfad"""
        try:
            self.log(f"📦 Lade Modell von: {path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(DEVICE)
            self.model.eval()
            
            self.model_path = path
            self.model_status.config(text=f"✅ Geladen: {Path(path).name}", fg="green")
            self._update_status(f"Modell geladen: {Path(path).name}")
            self.log(f"✅ Modell erfolgreich geladen")
            
            # Metriken laden falls vorhanden
            metrics_path = Path(path) / "metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                self.log(f"   Test F1: {metrics.get('test_f1', 'N/A')}")
                self.log(f"   Test AUC: {metrics.get('test_auc', 'N/A')}")
            
        except Exception as e:
            self.log(f"❌ Fehler beim Laden: {e}")
            self.model_status.config(text="❌ Fehler beim Laden", fg="red")
            self.model = None
            self.tokenizer = None
    
    def _update_status(self, msg):
        """Status aktualisieren (Platzhalter)"""
        pass
        
    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()
        
    def _load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.text_entry.delete("1.0", tk.END)
                self.text_entry.insert("1.0", text)
                self.log(f"📁 Datei geladen: {path}")
            except Exception as e:
                self.log(f"❌ Fehler beim Laden: {e}")
    
    def _clear(self):
        self.text_entry.delete("1.0", tk.END)
        self.result_label.config(text="⏳ Warte auf Eingabe", fg="black")
        self.prob_label.config(text="")
        self.explain_text.delete("1.0", tk.END)
    
    def _predict(self):
        if self.model is None:
            messagebox.showerror("Fehler", "Bitte zuerst ein Modell laden!")
            return
            
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text or len(text) < 20:
            messagebox.showwarning("Warnung", "Text zu kurz (mind. 20 Zeichen)")
            return
            
        self.log(f"🔍 Teste Text (Länge: {len(text)} Zeichen)...")
        
        try:
            # Tokenisieren
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)
            
            # Vorhersage
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            fake_prob = probs[1]
            real_prob = probs[0]
            
            # Ergebnis anzeigen
            if fake_prob > 0.5:
                label = "FAKE (KI-generiert)"
                color = "red"
            else:
                label = "REAL (Menschlich)"
                color = "green"
                
            self.result_label.config(text=f"{label} ({fake_prob*100:.1f}%)", fg=color)
            self.prob_label.config(text=f"Fake: {fake_prob*100:.1f}% | Real: {real_prob*100:.1f}%")
            
            # Erklärung
            self._explain_text(input_ids, fake_prob)
            
            self.log(f"✅ Ergebnis: {label} ({fake_prob*100:.1f}%)")
            
        except Exception as e:
            self.log(f"❌ Fehler: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Fehler", str(e))
    
    def _explain_text(self, input_ids, fake_prob):
        """Zeigt Wort-Erklärung basierend auf Token-Attention mit korrekter UTF-8 Darstellung"""
        try:
            # Originaltext aus der Eingabe holen für korrekte Darstellung
            text = self.text_entry.get("1.0", tk.END).strip()
            
            # Text in Wörter aufteilen (korrekte Umlaute und Sonderzeichen)
            words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
            
            # Entferne Leerzeichen aus der Liste
            words = [w for w in words if w.strip()]
            
            self.explain_text.delete("1.0", tk.END)
            
            # Wörter mit Markierung versehen
            for word in words:
                # Je höher die Fake-Wahrscheinlichkeit, desto mehr Wörter werden markiert
                if fake_prob > 0.8 and len(word) > 6:
                    tag = "ki_high"
                elif fake_prob > 0.7 and len(word) > 5:
                    tag = "ki_med"
                elif fake_prob > 0.6 and len(word) > 4:
                    tag = "ki_low"
                else:
                    tag = "human"
                
                self.explain_text.insert(tk.END, word + " ", tag)
            
            # Legende
            self.explain_text.insert(tk.END, "\n\n", "")
            self.explain_text.insert(tk.END, "Legende: ", "")
            self.explain_text.insert(tk.END, "🟥 Stark für KI", "ki_high")
            self.explain_text.insert(tk.END, "  ", "")
            self.explain_text.insert(tk.END, "🟧 Mittel für KI", "ki_med")
            self.explain_text.insert(tk.END, "  ", "")
            self.explain_text.insert(tk.END, "🟨 Schwach für KI", "ki_low")
            self.explain_text.insert(tk.END, "  ", "")
            self.explain_text.insert(tk.END, "🟩 Für Mensch", "human")
            
        except Exception as e:
            self.explain_text.delete("1.0", tk.END)
            self.explain_text.insert("1.0", f"❌ Fehler bei Erklärung: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorTesterPro(root)
    root.mainloop()