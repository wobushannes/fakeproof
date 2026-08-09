import os
import random
import string
import tkinter as tk
from tkinter import filedialog, messagebox

def random_name(length=12):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def rename_files(folder):
    files = os.listdir(folder)
    
    # Nur Dateien (keine Ordner)
    files = [f for f in files if os.path.isfile(os.path.join(folder, f))]
    
    used_names = set()
    
    for file in files:
        old_path = os.path.join(folder, file)
        name, ext = os.path.splitext(file)
        
        # neuen Namen generieren (ohne Kollision)
        while True:
            new_name = random_name()
            if new_name not in used_names:
                used_names.add(new_name)
                break
        
        new_filename = new_name + ext
        new_path = os.path.join(folder, new_filename)
        
        os.rename(old_path, new_path)

def select_folder():
    folder = filedialog.askdirectory()
    if folder:
        try:
            rename_files(folder)
            messagebox.showinfo("Fertig", "Alle Dateien wurden zufällig umbenannt.")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

# GUI
root = tk.Tk()
root.title("Random File Renamer")
root.geometry("300x150")

label = tk.Label(root, text="Ordner auswählen und Dateien randomisieren")
label.pack(pady=10)

btn = tk.Button(root, text="Ordner auswählen", command=select_folder)
btn.pack(pady=10)

root.mainloop()