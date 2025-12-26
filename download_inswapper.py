"""
Script pour télécharger manuellement le modèle inswapper
"""
import os
from pathlib import Path

def download_inswapper():
    """Télécharge le modèle inswapper_128.onnx"""
    
    print("=" * 60)
    print("TÉLÉCHARGEMENT DU MODÈLE INSWAPPER")
    print("=" * 60)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "inswapper_128.onnx"
    
    if model_path.exists():
        print(f"Le modèle existe déjà: {model_path}")
        file_size = model_path.stat().st_size / (1024 * 1024)
        print(f"   Taille: {file_size:.2f} MB")
        return
    
    print("\nLe modèle inswapper_128.onnx est nécessaire pour le face swap.")
    print("Taille: ~554 MB\n")
    
    # Essayer avec gdown
    try:
        print("Tentative de téléchargement avec gdown...\n")
        
        try:
            import gdown
        except ImportError:
            print("Installation de gdown...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'gdown', '-q'])
            import gdown
        
        # URL Google Drive du modèle
        url = "https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
        
        print("Téléchargement en cours (peut prendre 5-10 minutes)...")
        gdown.download(url, str(model_path), quiet=False)
        
        if model_path.exists():
            file_size = model_path.stat().st_size / (1024 * 1024)
            print(f"\n Téléchargement réussi!")
            print(f"   Fichier: {model_path}")
            print(f"   Taille: {file_size:.2f} MB")
        else:
            raise Exception("Le fichier n'a pas été téléchargé")
            
    except Exception as e:
        print(f"\n Erreur: {e}\n")
        print("=" * 60)
        print(" INSTRUCTIONS DE TÉLÉCHARGEMENT MANUEL")
        print("=" * 60)
        
        print("\n Option REQUIS Google Drive, huggingface ou github")

if __name__ == "__main__":
    download_inswapper()
