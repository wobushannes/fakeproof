import os
import random
import nltk
from pathlib import Path
from textaugment import EDA
import time

# ===== KONFIGURATION =====
FAKE_DIR = "./data_roh/fake"  # Dein Fake-Ordner
OUTPUT_DIR = "./data_roh/fake_augmented"  # Augmentierte Ausgabe
AUGMENTATIONS_PER_TEXT = 3  # Wie viele Variationen pro Text
# ==========================

def setup_nltk():
    """NLTK-Daten einmalig herunterladen"""
    try:
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("✅ NLTK-Daten geladen")
    except Exception as e:
        print(f"⚠️ NLTK-Fehler: {e}")

def augment_texts():
    """Augmentiert alle Texte im Fake-Ordner"""
    
    # 1. Setup
    setup_nltk()
    
    # Ordner erstellen
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 2. Alle TXT-Dateien finden
    fake_files = list(Path(FAKE_DIR).glob("*.txt"))
    
    if not fake_files:
        print(f"❌ Keine TXT-Dateien in {FAKE_DIR}")
        return
    
    print(f"📂 Gefundene Dateien: {len(fake_files)}")
    
    # 3. EDA initialisieren
    try:
        aug = EDA()
        print("✅ EDA initialisiert")
    except Exception as e:
        print(f"❌ EDA-Fehler: {e}")
        return
    
    # 4. Jede Datei augmentieren
    total_generated = 0
    
    for file_path in fake_files:
        try:
            # Original lesen
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if len(text) < 50:
                print(f"   ⚠️ Überspringe {file_path.name} (zu kurz)")
                continue
            
            # Original speichern (1x)
            base_name = file_path.stem
            orig_out = Path(OUTPUT_DIR) / f"{base_name}_orig.txt"
            with open(orig_out, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Augmentierte Versionen (AUGMENTATIONS_PER_TEXT x)
            for i in range(AUGMENTATIONS_PER_TEXT):
                try:
                    # Zufällige Augmentation
                    method = random.choice(['synonym', 'insert', 'swap', 'delete'])
                    
                    if method == 'synonym':
                        new_text = aug.synonym_replacement(text)
                    elif method == 'insert':
                        new_text = aug.random_insertion(text)
                    elif method == 'swap':
                        new_text = aug.random_swap(text)
                    else:  # delete
                        new_text = aug.random_deletion(text)
                    
                    # Speichern
                    out_path = Path(OUTPUT_DIR) / f"aug_{base_name}_{i+1}.txt"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(new_text)
                    
                    total_generated += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Fehler bei {file_path.name} Variante {i+1}: {e}")
            
            print(f"   ✅ {file_path.name} → {AUGMENTATIONS_PER_TEXT} Variationen")
            
        except Exception as e:
            print(f"   ❌ Fehler bei {file_path.name}: {e}")
    
    # 5. Zusammenfassung
    print(f"\n{'='*50}")
    print(f"✅ Fertig!")
    print(f"   Originale: {len(fake_files)}")
    print(f"   Generiert: {total_generated}")
    print(f"   Gesamt:    {len(fake_files) + total_generated}")
    print(f"   Ordner:    {OUTPUT_DIR}")
    print(f"{'='*50}")

def check_files():
    """Prüft ob neue Dateien da sind"""
    output_files = list(Path(OUTPUT_DIR).glob("*.txt"))
    print(f"\n📊 Dateien im Ausgabeordner: {len(output_files)}")
    
    if output_files:
        print(f"   Beispiel: {output_files[0].name}")
        with open(output_files[0], 'r', encoding='utf-8') as f:
            preview = f.read()[:200]
            print(f"   Preview: {preview}...")

if __name__ == "__main__":
    print("🚀 STARTE TEXT-AUGMENTATION")
    print("="*50)
    
    # 1. Startzeit
    start = time.time()
    
    # 2. Augmentieren
    augment_texts()
    
    # 3. Prüfen
    check_files()
    
    # 4. Dauer
    duration = time.time() - start
    print(f"\n⏱️ Dauer: {duration:.2f} Sekunden")
    
    print("\n💡 Nächster Schritt: Kopiere die augmentierten Texte in deinen Fake-Ordner:")
    print(f"   cp {OUTPUT_DIR}/*.txt {FAKE_DIR}/")