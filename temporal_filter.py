"""
Filtres temporels pour lisser les vidéos et réduire le scintillement entre frames
"""
import numpy as np
import cv2
from collections import deque

class TemporalFilter:
    """
    Applique des filtres temporels pour lisser les transitions entre frames
    Réduit le scintillement et améliore la cohérence visuelle
    """
    
    def __init__(self, method='moving_average', window_size=5):
        """
        Initialise le filtre temporel
        
        Args:
            method: Type de filtre ('moving_average', 'exponential', 'kalman')
            window_size: Taille de la fenêtre pour moyenne glissante
        """
        self.method = method
        self.window_size = window_size
        self.frame_buffer = deque(maxlen=window_size)
        
        # Pour filtre exponentiel
        self.alpha = 0.3  # Coefficient de lissage (0-1)
        self.previous_frame = None
        
        # Pour Kalman Filter
        self.kalman_state = None
        self.kalman_initialized = False
    
    def apply(self, frame):
        """
        Applique le filtre temporel sur une frame
        
        Args:
            frame: Frame actuelle
            
        Returns:
            Frame filtrée
        """
        if self.method == 'moving_average':
            return self.moving_average_filter(frame)
        elif self.method == 'exponential':
            return self.exponential_filter(frame)
        elif self.method == 'kalman':
            return self.kalman_filter(frame)
        else:
            return frame
    
    def moving_average_filter(self, frame):
        """
        Filtre à moyenne glissante
        Calcule la moyenne des N dernières frames
        
        Args:
            frame: Frame actuelle
            
        Returns:
            Frame lissée (moyenne des frames récentes)
        """
        # Ajouter la frame au buffer
        self.frame_buffer.append(frame.astype(np.float32))
        
        # Si pas assez de frames, retourner la frame actuelle
        if len(self.frame_buffer) < 2:
            return frame
        
        # Calculer la moyenne des frames dans le buffer
        averaged_frame = np.mean(self.frame_buffer, axis=0)
        
        return averaged_frame.astype(np.uint8)
    
    def exponential_filter(self, frame):
        """
        Filtre exponentiel (lissage exponentiel)
        Formule: output = alpha * current + (1-alpha) * previous
        
        Args:
            frame: Frame actuelle
            
        Returns:
            Frame lissée exponentiellement
        """
        if self.previous_frame is None:
            self.previous_frame = frame.astype(np.float32)
            return frame
        
        # Appliquer le lissage exponentiel
        current = frame.astype(np.float32)
        smoothed = self.alpha * current + (1 - self.alpha) * self.previous_frame
        
        # Mettre à jour la frame précédente
        self.previous_frame = smoothed
        
        return smoothed.astype(np.uint8)
    
    def kalman_filter(self, frame):
        """
        Filtre de Kalman pour prédiction et correction
        Utilise un modèle de mouvement simple pour prédire et corriger
        
        Args:
            frame: Frame actuelle
            
        Returns:
            Frame filtrée par Kalman
        """
        if not self.kalman_initialized:
            # Initialiser l'état Kalman avec la première frame
            self.kalman_state = frame.astype(np.float32)
            self.kalman_initialized = True
            return frame
        
        # Paramètres du filtre de Kalman
        process_variance = 1e-5  # Variance du processus (modèle)
        measurement_variance = 1e-4  # Variance de la mesure (observation)
        
        # Initialiser la covariance d'erreur
        if not hasattr(self, 'error_covariance'):
            self.error_covariance = np.ones_like(frame, dtype=np.float32)
        
        # Prédiction (on suppose que l'état ne change pas beaucoup)
        predicted_state = self.kalman_state
        predicted_covariance = self.error_covariance + process_variance
        
        # Calcul du gain de Kalman
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance)
        
        # Correction avec la mesure actuelle
        measurement = frame.astype(np.float32)
        self.kalman_state = predicted_state + kalman_gain * (measurement - predicted_state)
        
        # Mise à jour de la covariance d'erreur
        self.error_covariance = (1 - kalman_gain) * predicted_covariance
        
        return self.kalman_state.astype(np.uint8)
    
    def reset(self):
        """Réinitialise le filtre (utile entre différentes vidéos)"""
        self.frame_buffer.clear()
        self.previous_frame = None
        self.kalman_state = None
        self.kalman_initialized = False
        if hasattr(self, 'error_covariance'):
            delattr(self, 'error_covariance')


class AdaptiveTemporalFilter:
    """
    Filtre temporel adaptatif qui détecte le mouvement
    et ajuste le lissage en conséquence
    """
    
    def __init__(self, low_motion_alpha=0.7, high_motion_alpha=0.2):
        """
        Args:
            low_motion_alpha: Lissage pour zones statiques (plus élevé = plus de lissage)
            high_motion_alpha: Lissage pour zones en mouvement (plus bas = moins de lissage)
        """
        self.low_motion_alpha = low_motion_alpha
        self.high_motion_alpha = high_motion_alpha
        self.previous_frame = None
    
    def apply(self, frame):
        """
        Applique un lissage adaptatif basé sur la détection de mouvement
        
        Args:
            frame: Frame actuelle
            
        Returns:
            Frame avec lissage adaptatif
        """
        if self.previous_frame is None:
            self.previous_frame = frame.astype(np.float32)
            return frame
        
        current = frame.astype(np.float32)
        previous = self.previous_frame
        
        # Calculer la différence entre frames (mouvement)
        diff = cv2.absdiff(current, previous)
        motion_magnitude = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Normaliser le mouvement (0-1)
        motion_norm = motion_magnitude / 255.0
        
        # Calculer l'alpha adaptatif
        # Plus de mouvement = alpha plus petit = moins de lissage
        adaptive_alpha = self.high_motion_alpha + \
                        (self.low_motion_alpha - self.high_motion_alpha) * (1 - motion_norm)
        
        # Étendre alpha à 3 canaux
        adaptive_alpha = np.stack([adaptive_alpha] * 3, axis=-1)
        
        # Appliquer le lissage adaptatif
        smoothed = adaptive_alpha * current + (1 - adaptive_alpha) * previous
        
        # Mettre à jour
        self.previous_frame = smoothed
        
        return smoothed.astype(np.uint8)
    
    def reset(self):
        """Réinitialise le filtre"""
        self.previous_frame = None
