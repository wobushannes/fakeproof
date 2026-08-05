import torch
import sqlite3
import os
import random
import json
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# ====== KONFIGURATION ======
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints"
# ===========================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class ResumeTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KI-Text Detector Trainer (Resume-fähig)")
        self.root.geometry("1200x800")
        
        # Pfade
        self.real_db_path = tk.StringVar(value="./text_corpus.db")
        self.fake_txt_dir = tk.StringVar(value="./fakes")
        self.output_dir = tk.StringVar(value="./model_output")
        self.checkpoint_dir = tk.StringVar(value=CHECKPOINT_DIR)
        self.selected_checkpoint = tk.StringVar(value="")
        
        # Training State
        self.is_training = False
        self.stop_training = False
        self.is_resuming = False
        self.train_history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_auc": [],
            "epochs_completed": 0
        }
        
        # Modell-Referenzen
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        
        self._build_ui()
        self._update_status("Bereit")
        self._check_checkpoints()
        
    def _build_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        left_frame = ttk.LabelFrame(main_frame, text="Konfiguration", padding=10)
        left_frame.pack(side=tk.LEFT, fill="both", expand=False, padx=(0, 10))
        
        ttk.Label(left_frame, text="Real-DB:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(left_frame, textvariable=self.real_db_path, width=40).grid(row=0, column=1, pady=5)
        ttk.Button(left_frame, text="📁", command=lambda: self._select_file(self.real_db_path, "*.db")).grid(row=0, column=2, padx=5)
        
        ttk.Label(left_frame, text="Fake-Ordner:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(left_frame, textvariable=self.fake_txt_dir, width=40).grid(row=1, column=1, pady=5)
        ttk.Button(left_frame, text="📁", command=lambda: self._select_folder(self.fake_txt_dir)).grid(row=1, column=2, padx=5)
        
        ttk.Label(left_frame, text="Output-Ordner:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(left_frame, textvariable=self.output_dir, width=40).grid(row=2, column=1, pady=5)
        ttk.Button(left_frame, text="📁", command=lambda: self._select_folder(self.output_dir)).grid(row=2, column=2, padx=5)
        
        ttk.Label(left_frame, text="Checkpoint-Ordner:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(left_frame, textvariable=self.checkpoint_dir, width=40).grid(row=3, column=1, pady=5)
        ttk.Button(left_frame, text="📁", command=lambda: self._select_folder(self.checkpoint_dir)).grid(row=3, column=2, padx=5)
        
        ttk.Separator(left_frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        
        checkpoint_frame = ttk.LabelFrame(left_frame, text="Checkpoint-Auswahl", padding=5)
        checkpoint_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)
        
        self.checkpoint_combo = ttk.Combobox(checkpoint_frame, textvariable=self.selected_checkpoint, state="readonly", width=40)
        self.checkpoint_combo.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(checkpoint_frame, text="🔄", command=self._refresh_checkpoints, width=3).pack(side=tk.LEFT)
        
        self.checkpoint_info = ttk.Label(checkpoint_frame, text="Kein Checkpoint ausgewählt", foreground="gray")
        self.checkpoint_info.pack(pady=5)
        
        ttk.Label(left_frame, text="Hyperparameter", font=("Arial", 10, "bold")).grid(row=6, column=0, columnspan=3, pady=5)
        
        ttk.Label(left_frame, text="Modell:").grid(row=7, column=0, sticky="w", pady=2)
        self.model_var = tk.StringVar(value="answerdotai/ModernBERT-base")
        ttk.Combobox(left_frame, textvariable=self.model_var, 
                     values=["answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"],
                     width=38).grid(row=7, column=1, columnspan=2, pady=2)
        
        ttk.Label(left_frame, text="Batch Size:").grid(row=8, column=0, sticky="w", pady=2)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(left_frame, from_=4, to=128, textvariable=self.batch_size_var, width=10).grid(row=8, column=1, sticky="w", pady=2)
        
        ttk.Label(left_frame, text="Epochen:").grid(row=9, column=0, sticky="w", pady=2)
        self.epochs_var = tk.IntVar(value=5)
        ttk.Spinbox(left_frame, from_=1, to=20, textvariable=self.epochs_var, width=10).grid(row=9, column=1, sticky="w", pady=2)
        
        ttk.Label(left_frame, text="Learning Rate:").grid(row=10, column=0, sticky="w", pady=2)
        self.lr_var = tk.DoubleVar(value=2e-5)
        ttk.Entry(left_frame, textvariable=self.lr_var, width=15).grid(row=10, column=1, sticky="w", pady=2)
        
        ttk.Separator(left_frame, orient="horizontal").grid(row=11, column=0, columnspan=3, sticky="ew", pady=10)
        
        self.train_btn = ttk.Button(left_frame, text="🚀 NEUES TRAINING", command=self._start_training)
        self.train_btn.grid(row=12, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.resume_btn = ttk.Button(left_frame, text="⏩ TRAINING FORTSETZEN", command=self._resume_training, state="disabled")
        self.resume_btn.grid(row=13, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.stop_btn = ttk.Button(left_frame, text="⏹ STOP", command=self._stop_training, state="disabled")
        self.stop_btn.grid(row=14, column=0, columnspan=3, pady=5, sticky="ew")
        
        self.status_label = ttk.Label(left_frame, text="Status: Bereit", font=("Arial", 10, "bold"))
        self.status_label.grid(row=15, column=0, columnspan=3, pady=10)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill="both", expand=True)
        
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding=5)
        log_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        
        plot_frame = ttk.LabelFrame(right_frame, text="Training Progress", padding=5)
        plot_frame.pack(fill="both", expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        batch_frame = ttk.LabelFrame(right_frame, text="Batch Loss (aktuell)", padding=5)
        batch_frame.pack(fill="x", pady=(5, 0))
        
        self.batch_loss_label = ttk.Label(batch_frame, text="⏳ Warte auf Batch...")
        self.batch_loss_label.pack(pady=5)
        
    def _select_folder(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)
            self._refresh_checkpoints()
            
    def _select_file(self, var, pattern):
        path = filedialog.askopenfilename(filetypes=[("Database", pattern)])
        if path:
            var.set(path)
    
    def _update_status(self, msg):
        self.status_label.config(text=f"Status: {msg}")
        self.root.update()
        
    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()
    
    def _refresh_checkpoints(self):
        checkpoint_dir = self.checkpoint_dir.get()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_combo['values'] = []
            self.checkpoint_combo.set('')
            self.checkpoint_info.config(text="Kein Checkpoint gefunden", foreground="gray")
            self.resume_btn.config(state="disabled")
            return
        
        checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint_*"), 
                           key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not checkpoints:
            self.checkpoint_combo['values'] = []
            self.checkpoint_combo.set('')
            self.checkpoint_info.config(text="Kein Checkpoint gefunden", foreground="gray")
            self.resume_btn.config(state="disabled")
            return
        
        checkpoint_names = [cp.name for cp in checkpoints]
        self.checkpoint_combo['values'] = checkpoint_names
        
        if checkpoint_names:
            self.checkpoint_combo.set(checkpoint_names[0])
            self._update_checkpoint_info(checkpoints[0])
            self.resume_btn.config(state="normal")
    
    def _update_checkpoint_info(self, checkpoint_path):
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    meta = json.load(f)
                self.checkpoint_info.config(
                    text=f"Epoch {meta.get('epoch', '?')} | F1: {meta.get('val_f1', 0):.4f} | AUC: {meta.get('val_auc', 0):.4f}",
                    foreground="green"
                )
            except:
                self.checkpoint_info.config(text="Checkpoint beschädigt", foreground="red")
        else:
            self.checkpoint_info.config(text="Keine Metadaten", foreground="orange")
    
    def _check_checkpoints(self):
        self._refresh_checkpoints()
    
    def _save_checkpoint(self, model, optimizer, scheduler, epoch, history, best_f1, tokenizer):
        checkpoint_dir = self.checkpoint_dir.get()
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_f1,
            'history': history
        }, os.path.join(checkpoint_path, "training_state.pt"))
        
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump({
                'epoch': epoch + 1,
                'val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
                'val_auc': history['val_auc'][-1] if history['val_auc'] else 0,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        self.log(f"💾 Checkpoint gespeichert: {checkpoint_path}")
        self._refresh_checkpoints()
    
    def _load_checkpoint(self, checkpoint_name):
        checkpoint_dir = self.checkpoint_dir.get()
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
        
        if not checkpoint_path.exists():
            self.log(f"❌ Checkpoint nicht gefunden: {checkpoint_path}")
            return None, None, None, None, None
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            model.to(DEVICE)
            
            state_path = checkpoint_path / "training_state.pt"
            if not state_path.exists():
                self.log(f"❌ training_state.pt nicht gefunden in {checkpoint_path}")
                return None, None, None, None, None
            
            state = torch.load(state_path, map_location=DEVICE)
            
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                self.log(f"❌ metadata.json nicht gefunden in {checkpoint_path}")
                return None, None, None, None, None
            
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            
            self.log(f"📦 Checkpoint geladen: {checkpoint_name} (Epoch {meta.get('epoch', '?')})")
            return model, tokenizer, state, meta, checkpoint_path
            
        except Exception as e:
            self.log(f"❌ Fehler beim Laden des Checkpoints: {e}")
            return None, None, None, None, None
    
    def _load_balanced_data(self):
        """Lädt ALLE Fakes + genauso viele Reale - NUR LANGE TEXTE"""
        self.log("📂 Lade Daten...")
        
        fake_dir = self.fake_txt_dir.get()
        fake_files = list(Path(fake_dir).glob("*.txt"))
        
        if not fake_files:
            self.log(f"❌ Keine TXT-Dateien in {fake_dir}")
            return None, None
        
        self.log(f"   Gefundene Fake-Dateien: {len(fake_files)}")
        fake_texts = []
        
        for f in fake_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    if len(text) > 50:
                        fake_texts.append(text)
            except Exception as e:
                self.log(f"   ⚠️ Fehler bei {f.name}: {e}")
        
        self.log(f"   ✅ {len(fake_texts)} Fake-Texte geladen")
        
        db_path = self.real_db_path.get()
        if not os.path.exists(db_path):
            self.log(f"❌ Datenbank nicht gefunden: {db_path}")
            return None, None
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # ===== NUR LANGE TEXTE =====
        cursor.execute("""
            SELECT text_content FROM speeches 
            WHERE text_content IS NOT NULL AND LENGTH(text_content) > 500
            ORDER BY LENGTH(text_content) DESC
            LIMIT ?
        """, (len(fake_texts),))
        # ===========================
        
        real_texts = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        self.log(f"   ✅ {len(real_texts)} Real-Texte geladen (nur > 500 Zeichen)")
        self.log(f"   📏 Durchschnittliche Länge Real: {sum(len(t) for t in real_texts)/len(real_texts):.0f} Zeichen" if real_texts else "   📏 Keine Real-Texte")
        self.log(f"   📏 Durchschnittliche Länge Fake: {sum(len(t) for t in fake_texts)/len(fake_texts):.0f} Zeichen" if fake_texts else "   📏 Keine Fake-Texte")
        
        texts = fake_texts + real_texts
        labels = [1] * len(fake_texts) + [0] * len(real_texts)
        
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined)
        
        self.log(f"✅ Balanciert: {len(fake_texts)} Fake vs. {len(real_texts)} Real")
        return list(texts), list(labels)
    
    def _train_epoch(self, model, dataloader, optimizer, scheduler, criterion, device, epoch):
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        batch_losses = []
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress):
            if self.stop_training:
                return None, None
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            batch_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
            
            progress.set_postfix({"loss": loss.item()})
            
            if batch_idx % 10 == 0:
                avg_loss = sum(batch_losses[-10:]) / len(batch_losses[-10:]) if batch_losses else 0
                self.batch_loss_label.config(text=f"Batch {batch_idx+1}: Loss = {loss.item():.4f} (Avg = {avg_loss:.4f})")
                self.root.update()
        
        f1 = f1_score(true_labels, predictions, average='macro') if predictions else 0
        return total_loss / len(dataloader), f1
    
    def _evaluate(self, model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.extend(probs[:, 1])
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(true_labels, predictions, average='macro')
        auc = roc_auc_score(true_labels, probabilities)
        return total_loss / len(dataloader), f1, auc
    
    def _save_model(self, model, tokenizer, metrics):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir.get(), f"model_{timestamp}")
        os.makedirs(model_path, exist_ok=True)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        with open(os.path.join(model_path, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.log(f"✅ Modell gespeichert: {model_path}")
        return model_path
    
    def _training_thread(self, resume=False, checkpoint_data=None):
        try:
            texts, labels = self._load_balanced_data()
            if texts is None:
                self._update_status("Fehler beim Laden")
                self.train_btn.config(state="normal")
                self.resume_btn.config(state="normal" if self._has_checkpoints() else "disabled")
                self.stop_btn.config(state="disabled")
                self.is_training = False
                return
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                texts, labels, test_size=0.3, random_state=42, stratify=labels
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            self.log(f"\n📊 Split:")
            self.log(f"   Train: {len(X_train)} ({sum(y_train)} Fake)")
            self.log(f"   Val:   {len(X_val)} ({sum(y_val)} Fake)")
            self.log(f"   Test:  {len(X_test)} ({sum(y_test)} Fake)")
            
            if resume and checkpoint_data:
                model, tokenizer, state, meta, checkpoint_path = checkpoint_data
                if model is None:
                    self.log("❌ Fehler: Kein gültiger Checkpoint zum Fortsetzen")
                    self._update_status("Fehler beim Fortsetzen")
                    return
                    
                self.current_epoch = state['epoch'] if 'epoch' in state else 0
                best_f1 = state['best_f1'] if 'best_f1' in state else 0.0
                history = state['history'] if 'history' in state else {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": [], "epochs_completed": 0}
                self.train_history = history
                self.log(f"⏩ Training fortsetzen bei Epoch {self.current_epoch}")
            else:
                model_name = self.model_var.get()
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                model.to(DEVICE)
                self.current_epoch = 0
                best_f1 = 0.0
                history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": [], "epochs_completed": 0}
                self.train_history = history
            
            train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
            val_dataset = TextDataset(X_val, y_val, tokenizer, MAX_LEN)
            test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)
            
            batch_size = self.batch_size_var.get()
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            if resume and checkpoint_data and model is not None:
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_var.get(), weight_decay=WEIGHT_DECAY)
                if 'optimizer_state_dict' in state:
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                
                total_steps = len(train_loader) * self.epochs_var.get()
                warmup_steps = int(total_steps * WARMUP_RATIO)
                scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
                if 'scheduler_state_dict' in state:
                    try:
                        scheduler.load_state_dict(state['scheduler_state_dict'])
                    except Exception as e:
                        self.log(f"⚠️ Scheduler konnte nicht geladen werden: {e}")
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_var.get(), weight_decay=WEIGHT_DECAY)
                total_steps = len(train_loader) * self.epochs_var.get()
                warmup_steps = int(total_steps * WARMUP_RATIO)
                scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
            
            criterion = torch.nn.CrossEntropyLoss()
            
            self.log(f"\n🚀 Starte Training (Batch Size {batch_size})")
            self._update_status(f"Training läuft...")
            
            for epoch in range(self.current_epoch, self.epochs_var.get()):
                if self.stop_training:
                    self.log("\n⏹ Training gestoppt")
                    break
                
                self.log(f"\n📈 Epoch {epoch+1}/{self.epochs_var.get()}")
                
                train_loss, train_f1 = self._train_epoch(
                    model, train_loader, optimizer, scheduler, criterion, DEVICE, epoch
                )
                
                if train_loss is None:
                    break
                
                val_loss, val_f1, val_auc = self._evaluate(model, val_loader, criterion, DEVICE)
                
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_f1"].append(val_f1)
                history["val_auc"].append(val_auc)
                history["epochs_completed"] = epoch + 1
                self.train_history = history
                
                self.log(f"   Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
                self.log(f"   Val Loss:   {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
                
                self._update_plot()
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self.log(f"   ✅ Neues Bestes Modell (F1: {best_f1:.4f})")
                
                self._save_checkpoint(model, optimizer, scheduler, epoch, history, best_f1, tokenizer)
            
            if not self.stop_training:
                self.log(f"\n📊 Teste bestes Modell...")
                checkpoint_dir = self.checkpoint_dir.get()
                checkpoints = list(Path(checkpoint_dir).glob("checkpoint_*"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(latest)
                        model.to(DEVICE)
                    except Exception as e:
                        self.log(f"⚠️ Konnte bestes Modell nicht laden: {e}")
                
                test_loss, test_f1, test_auc = self._evaluate(model, test_loader, criterion, DEVICE)
                
                self.log(f"\n🎯 TEST RESULTS:")
                self.log(f"   Loss: {test_loss:.4f}")
                self.log(f"   F1:   {test_f1:.4f}")
                self.log(f"   AUC:  {test_auc:.4f}")
                
                self._save_model(model, tokenizer, {
                    "test_f1": test_f1,
                    "test_auc": test_auc,
                    "best_val_f1": best_f1,
                    "epochs": self.epochs_var.get(),
                    "model_name": self.model_var.get(),
                    "fake_count": len([l for l in labels if l == 1]),
                    "real_count": len([l for l in labels if l == 0]),
                    "timestamp": datetime.now().isoformat()
                })
            
            self._update_status("Training abgeschlossen")
            
        except Exception as e:
            self.log(f"❌ Fehler: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            self._update_status("Fehler")
        
        finally:
            self.is_training = False
            self.train_btn.config(state="normal")
            self.resume_btn.config(state="normal" if self._has_checkpoints() else "disabled")
            self.stop_btn.config(state="disabled")
    
    def _update_plot(self):
        self.ax.clear()
        
        epochs = range(1, len(self.train_history["val_f1"]) + 1)
        
        if len(self.train_history["val_f1"]) > 0:
            self.ax.plot(epochs, self.train_history["val_f1"], 'b-', label='Val F1')
            self.ax.plot(epochs, self.train_history["val_auc"], 'g-', label='Val AUC')
            self.ax.plot(epochs, self.train_history["val_loss"], 'r-', label='Val Loss')
        
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Score')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def _has_checkpoints(self):
        checkpoint_dir = self.checkpoint_dir.get()
        if not os.path.exists(checkpoint_dir):
            return False
        checkpoints = list(Path(checkpoint_dir).glob("checkpoint_*"))
        return len(checkpoints) > 0
    
    def _start_training(self):
        if self.is_training:
            return
        
        if not os.path.exists(self.real_db_path.get()):
            messagebox.showerror("Fehler", "Real-DB existiert nicht!")
            return
        if not os.path.exists(self.fake_txt_dir.get()):
            messagebox.showerror("Fehler", "Fake-Ordner existiert nicht!")
            return
        
        self.is_training = True
        self.stop_training = False
        self.train_history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": [], "epochs_completed": 0}
        self.train_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log("\n" + "="*60)
        self.log("🚀 NEUES TRAINING GESTARTET")
        self.log("="*60)
        
        thread = threading.Thread(target=self._training_thread, args=(False, None))
        thread.daemon = True
        thread.start()
    
    def _resume_training(self):
        if self.is_training:
            return
        
        checkpoint_name = self.selected_checkpoint.get()
        if not checkpoint_name:
            messagebox.showerror("Fehler", "Bitte wähle einen Checkpoint aus der Liste!")
            return
        
        model, tokenizer, state, meta, checkpoint_path = self._load_checkpoint(checkpoint_name)
        if model is None:
            messagebox.showerror("Fehler", f"Checkpoint '{checkpoint_name}' konnte nicht geladen werden!")
            return
        
        self.is_training = True
        self.stop_training = False
        self.train_btn.config(state="disabled")
        self.resume_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log("\n" + "="*60)
        self.log(f"⏩ TRAINING FORTGESETZT von {checkpoint_name}")
        self.log("="*60)
        
        thread = threading.Thread(target=self._training_thread, args=(True, (model, tokenizer, state, meta, checkpoint_path)))
        thread.daemon = True
        thread.start()
    
    def _stop_training(self):
        self.stop_training = True
        self.log("\n⏹ Stoppe Training... (warte auf aktuellen Batch)")
        self.stop_btn.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeTrainerGUI(root)
    root.mainloop()