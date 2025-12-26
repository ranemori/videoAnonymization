"""
Module d'analyse et de visualisation vidéo
Permet de comparer avant/après, analyser les modifications, et générer des métriques
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique
import matplotlib.pyplot as plt
from pathlib import Path

class VideoAnalyzer:
    """
    Analyse et compare des vidéos avant/après traitement
    """
    
    def __init__(self):
        """Initialise l'analyseur vidéo"""
        self.metrics = {
            'mse': [],  # Mean Squared Error
            'psnr': [],  # Peak Signal-to-Noise Ratio
            'ssim': []  # Structural Similarity (nécessite skimage)
        }
    
    def create_side_by_side(self, original_frame, processed_frame, labels=None):
        """
        Crée une comparaison côte à côte de deux frames
        
        Args:
            original_frame: Frame originale
            processed_frame: Frame traitée
            labels: Tuple de labels (label_original, label_processed)
            
        Returns:
            Frame combinée avec les deux images côte à côte
        """
        # S'assurer que les deux frames ont la même taille
        if original_frame.shape != processed_frame.shape:
            processed_frame = cv2.resize(processed_frame, 
                                        (original_frame.shape[1], original_frame.shape[0]))
        
        # Créer une image côte à côte
        side_by_side = np.hstack([original_frame, processed_frame])
        
        # Ajouter des labels si fournis
        if labels:
            label_original, label_processed = labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            font_color = (255, 255, 255)
            
            # Position du texte
            h, w = original_frame.shape[:2]
            y_pos = 40
            
            # Label pour l'original
            cv2.putText(side_by_side, label_original, (20, y_pos), 
                       font, font_scale, font_color, font_thickness)
            
            # Label pour le traité
            cv2.putText(side_by_side, label_processed, (w + 20, y_pos), 
                       font, font_scale, font_color, font_thickness)
        
        return side_by_side
    
    def create_difference_map(self, original_frame, processed_frame):
        """
        Crée une carte de différence d'intensité entre deux frames
        
        Args:
            original_frame: Frame originale
            processed_frame: Frame traitée
            
        Returns:
            Carte de différence (heatmap)
        """
        # S'assurer que les deux frames ont la même taille
        if original_frame.shape != processed_frame.shape:
            processed_frame = cv2.resize(processed_frame, 
                                        (original_frame.shape[1], original_frame.shape[0]))
        
        # Calculer la différence absolue
        diff = cv2.absdiff(original_frame, processed_frame)
        
        # Convertir en niveaux de gris pour visualisation
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Appliquer une colormap pour visualisation (heatmap)
        diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        
        return diff_colored
    
    def calculate_histogram(self, frame, title="Histogramme"):
        """
        Calcule et dessine l'histogramme RGB d'une frame
        
        Args:
            frame: Frame à analyser
            title: Titre du graphique
            
        Returns:
            Image du graphique d'histogramme
        """
        # Calculer les histogrammes pour chaque canal
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(10, 4))
        plt.title(title)
        plt.xlabel("Intensité de pixel")
        plt.ylabel("Nombre de pixels")
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f'Canal {color.upper()}')
        
        plt.legend()
        plt.xlim([0, 256])
        
        # Convertir le plot en image
        plt.tight_layout()
        
        # Sauvegarder dans un buffer
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Convertir en image OpenCV
        img_array = np.frombuffer(buf.read(), dtype=np.uint8)
        buf.close()
        plt.close()
        
        hist_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return hist_img
    
    def create_comparison_grid(self, original_frame, processed_frame):
        """
        Crée une grille de comparaison 2x2:
        - Original | Traité
        - Différence | Histogrammes
        
        Args:
            original_frame: Frame originale
            processed_frame: Frame traitée
            
        Returns:
            Grille de comparaison complète
        """
        # S'assurer que les frames ont la même taille
        if original_frame.shape != processed_frame.shape:
            processed_frame = cv2.resize(processed_frame, 
                                        (original_frame.shape[1], original_frame.shape[0]))
        
        h, w = original_frame.shape[:2]
        
        # 1. Côte à côte
        side_by_side = self.create_side_by_side(
            original_frame, processed_frame, 
            labels=("Original", "Traité")
        )
        
        # 2. Carte de différence
        diff_map = self.create_difference_map(original_frame, processed_frame)
        diff_resized = cv2.resize(diff_map, (w, h))
        
        # 3. Histogrammes (créer une image vide temporairement)
        hist_original = self.calculate_histogram(original_frame, "Histogramme Original")
        hist_processed = self.calculate_histogram(processed_frame, "Histogramme Traité")
        
        # Redimensionner les histogrammes
        hist_original = cv2.resize(hist_original, (w, h))
        hist_processed = cv2.resize(hist_processed, (w, h))
        
        # Créer la ligne du bas (différence + histogrammes)
        bottom_left = diff_resized
        bottom_right = cv2.addWeighted(hist_original, 0.5, hist_processed, 0.5, 0)
        bottom_row = np.hstack([bottom_left, bottom_right])
        
        # Combiner en grille 2x2
        grid = np.vstack([side_by_side, bottom_row])
        
        return grid
    
    def calculate_mse(self, original_frame, processed_frame):
        """
        Calcule le Mean Squared Error entre deux frames
        
        Args:
            original_frame: Frame originale
            processed_frame: Frame traitée
            
        Returns:
            Valeur MSE (plus bas = plus similaire)
        """
        if original_frame.shape != processed_frame.shape:
            processed_frame = cv2.resize(processed_frame, 
                                        (original_frame.shape[1], original_frame.shape[0]))
        
        mse = np.mean((original_frame.astype(float) - processed_frame.astype(float)) ** 2)
        return mse
    
    def calculate_psnr(self, original_frame, processed_frame):
        """
        Calcule le Peak Signal-to-Noise Ratio
        
        Args:
            original_frame: Frame originale
            processed_frame: Frame traitée
            
        Returns:
            Valeur PSNR en dB (plus haut = meilleure qualité)
        """
        mse = self.calculate_mse(original_frame, processed_frame)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def add_metrics_overlay(self, frame, original_frame, processed_frame):
        """
        Ajoute des métriques de qualité sur la frame
        
        Args:
            frame: Frame sur laquelle ajouter les métriques
            original_frame: Frame originale pour calcul
            processed_frame: Frame traitée pour calcul
            
        Returns:
            Frame avec métriques affichées
        """
        mse = self.calculate_mse(original_frame, processed_frame)
        psnr = self.calculate_psnr(original_frame, processed_frame)
        
        # Ajouter le texte
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        cv2.putText(frame, f"MSE: {mse:.2f}", (10, y_offset), 
                   font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PSNR: {psnr:.2f} dB", (10, y_offset + 30), 
                   font, 0.7, (0, 255, 0), 2)
        
        return frame


def create_analysis_video(input_path, processed_path, output_path, analysis_type='side_by_side'):
    """
    Crée une vidéo d'analyse complète
    
    Args:
        input_path: Chemin de la vidéo originale
        processed_path: Chemin de la vidéo traitée
        output_path: Chemin de la vidéo d'analyse
        analysis_type: Type d'analyse ('side_by_side', 'difference', 'grid')
    """
    analyzer = VideoAnalyzer()
    
    cap_original = cv2.VideoCapture(input_path)
    cap_processed = cv2.VideoCapture(processed_path)
    
    # Propriétés vidéo
    fps = int(cap_original.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Lire la première frame pour déterminer la taille de sortie
    ret1, frame1 = cap_original.read()
    ret2, frame2 = cap_processed.read()
    
    if not ret1 or not ret2:
        print("Erreur: Impossible de lire les vidéos")
        return
    
    # Déterminer la taille de sortie selon le type d'analyse
    if analysis_type == 'side_by_side':
        sample = analyzer.create_side_by_side(frame1, frame2)
    elif analysis_type == 'difference':
        sample = analyzer.create_difference_map(frame1, frame2)
    elif analysis_type == 'grid':
        sample = analyzer.create_comparison_grid(frame1, frame2)
    else:
        sample = frame1
    
    out_height, out_width = sample.shape[:2]
    
    # Reset les vidéos
    cap_original.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_processed.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Créer le writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print(f"Création de la vidéo d'analyse ({analysis_type})...")
    frame_count = 0
    
    while True:
        ret1, original = cap_original.read()
        ret2, processed = cap_processed.read()
        
        if not ret1 or not ret2:
            break
        
        # Créer la frame d'analyse
        if analysis_type == 'side_by_side':
            analysis_frame = analyzer.create_side_by_side(
                original, processed, 
                labels=("Original", "Anonymisé")
            )
        elif analysis_type == 'difference':
            analysis_frame = analyzer.create_difference_map(original, processed)
        elif analysis_type == 'grid':
            analysis_frame = analyzer.create_comparison_grid(original, processed)
        
        out.write(analysis_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Traité {frame_count}/{total_frames} frames...")
    
    cap_original.release()
    cap_processed.release()
    out.release()
    
    print(f" Vidéo d'analyse créée: {output_path}")
