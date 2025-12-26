"""
Script pour télécharger les modèles pré-entraînés nécessaires
"""
import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Télécharge un fichier avec barre de progression"""
    print(f"Téléchargement de {os.path.basename(destination)}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rProgression: {percent}%", end='')
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\n Téléchargement terminé!")

def setup_models():
    """Configure et télécharge les modèles nécessaires"""
    
    # Créer le dossier models s'il n'existe pas
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    
    # Créer les dossiers nécessaires
    folders = [
        "models",
        "face_dataset",
        "input_videos",
        "output_videos"
    ]
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        print(f" Dossier '{folder}' créé/vérifié")
    
    print("\n" + "=" * 60)
    print(" Configuration terminée!")
    print("=" * 60)

if __name__ == "__main__":
    setup_models()
