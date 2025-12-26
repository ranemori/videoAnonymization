"""
Module de swap de visages utilisant InsightFace (modèle pré-entraîné)
"""
import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
import random

class FaceSwapper:
    def __init__(self, face_dataset_path=None):
        """
        Initialise le swapper de visages
        
        Args:
            face_dataset_path: Chemin vers le dossier contenant les visages de remplacement
        """
        print(" Initialisation du Face Swapper...")
        
        # Initialiser l'analyseur de visages avec le modèle buffalo_l
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Charger le modèle de swap (inswapper)
        model_path = self.download_swapper_model()
        self.swapper = insightface.model_zoo.get_model(model_path)
        
        # Charger le dataset de visages de remplacement
        self.replacement_faces = []
        if face_dataset_path:
            self.load_face_dataset(face_dataset_path)
        
        # Dictionnaire pour mapper chaque personne à un visage spécifique
        self.person_to_face = {}
        self.next_face_index = 0
        
        print(" Face Swapper initialisé!")
    
    def download_swapper_model(self):
        """Télécharge le modèle inswapper s'il n'existe pas"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / "inswapper_128.onnx"
        
        if not model_path.exists():
            print(" Téléchargement du modèle inswapper_128.onnx...")
            
            try:
                # Utiliser insightface pour télécharger le modèle
                from insightface.model_zoo import get_model
                import gdown
                
                # URL alternative depuis Google Drive
                url = "https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
                
                print("   Téléchargement depuis Google Drive...")
                gdown.download(url, str(model_path), quiet=False)
                print("\n Modèle téléchargé avec succès!")
                
            except ImportError:
                # Si gdown n'est pas installé, essayer avec requests
                print("   Installation de gdown pour le téléchargement...")
                import subprocess
                subprocess.check_call(['pip', 'install', 'gdown', '-q'])
                
                import gdown
                url = "https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu"
                gdown.download(url, str(model_path), quiet=False)
                print("\n Modèle téléchargé avec succès!")
                
            except Exception as e:
                print(f"\n Erreur lors du téléchargement automatique: {e}")
                print("\n" + "="*60)
                print(" TÉLÉCHARGEMENT MANUEL REQUIS")
                print("="*60)
                print("\nOption REQUIS Depuis Hugging Face ou drive")
                print("="*60)
                raise
        
        return str(model_path)
    
    def load_face_dataset(self, dataset_path):
        """
        Charge les visages de remplacement depuis un dossier
        
        Args:
            dataset_path: Chemin vers le dossier contenant les images
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f" Le dossier {dataset_path} n'existe pas")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'*{ext}'))
            image_files.extend(dataset_path.glob(f'*{ext.upper()}'))
        
        print(f" Chargement de {len(image_files)} images du dataset...")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Détecter le visage dans l'image
                faces = self.app.get(img)
                if len(faces) > 0:
                    self.replacement_faces.append(faces[0])
                    print(f" Visage chargé: {img_path.name}")
        
        print(f" {len(self.replacement_faces)} visages de remplacement chargés")
    
    def get_face_for_person(self, face_embedding):
        """
        Retourne un visage de remplacement cohérent pour une personne
        Utilise l'embedding du visage pour identifier la personne
        
        Args:
            face_embedding: L'embedding du visage détecté
            
        Returns:
            Visage de remplacement assigné à cette personne
        """
        if len(self.replacement_faces) == 0:
            return None
        
        # Trouver la personne la plus proche dans notre dictionnaire
        best_match_id = None
        best_similarity = 0.4  # Seuil de similarité (ajustable)
        
        for person_id, stored_embedding in self.person_to_face.items():
            # Calculer la similarité cosinus
            similarity = np.dot(face_embedding, stored_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        # Si on a trouvé une correspondance, utiliser le même visage
        if best_match_id is not None:
            return self.replacement_faces[best_match_id % len(self.replacement_faces)]
        
        # Sinon, c'est une nouvelle personne - assigner un nouveau visage
        new_person_id = len(self.person_to_face)
        self.person_to_face[new_person_id] = face_embedding
        
        # Utiliser le prochain visage disponible (rotation circulaire)
        face_index = self.next_face_index % len(self.replacement_faces)
        self.next_face_index += 1
        
        return self.replacement_faces[face_index]
    
    def swap_faces(self, frame, blur_mode=False):
        """
        Remplace les visages dans une frame avec préservation des expressions
        Chaque personne garde le même visage de remplacement
        
        Args:
            frame: Image/frame à traiter
            blur_mode: Si True, floute les visages au lieu de les remplacer
            
        Returns:
            Frame avec les visages remplacés ou floutés
        """
        # Détecter tous les visages dans la frame
        faces = self.app.get(frame)
        
        if len(faces) == 0:
            return frame
        
        result = frame.copy()
        original = frame.copy()  # Garder l'original pour blending expressions
        
        for face in faces:
            if blur_mode:
                # Mode flou: flouter le visage
                result = self.blur_face(result, face)
            else:
                # Mode swap: remplacer le visage avec un visage cohérent
                # Utiliser l'embedding du visage pour identifier la personne
                replacement_face = self.get_face_for_person(face.embedding)
                if replacement_face is not None:
                    # Swap de base
                    result = self.swapper.get(result, face, replacement_face, paste_back=True)
                    
                    # AMÉLIORATION: Préserver zones d'expression (bouche, yeux)
                    result = self.preserve_expression_zones(original, result, face)
                else:
                    # Si pas de visage de remplacement, flouter
                    result = self.blur_face(result, face)
        
        return result
    
    def preserve_expression_zones(self, original, swapped, face, blend_ratio=0.25):
        """
        Préserve les zones d'expression (bouche, yeux) de l'original
        pour rendre le swap plus fluide et naturel
        
        Args:
            original: Frame originale avec vraies expressions
            swapped: Frame après swap
            face: Objet face avec landmarks
            blend_ratio: Intensité du blending (0-1), 0.25 = 25% original
            
        Returns:
            Frame avec expressions préservées
        """
        # Vérifier si landmarks disponibles
        if not hasattr(face, 'kps') or face.kps is None:
            return swapped
        
        landmarks = face.kps.astype(int)
        
        # Créer masques pour zones d'expression
        mouth_mask = self._create_mouth_mask(swapped.shape[:2], landmarks)
        eyes_mask = self._create_eyes_mask(swapped.shape[:2], landmarks)
        
        # Combiner masques
        combined_mask = np.maximum(mouth_mask, eyes_mask)
        combined_mask = np.stack([combined_mask] * 3, axis=-1)  # 3 canaux RGB
        
        # Blending: mélanger original et swapped dans zones d'expression
        result = swapped.astype(np.float32)
        original_float = original.astype(np.float32)
        
        result = (1 - combined_mask * blend_ratio) * result + \
                 (combined_mask * blend_ratio) * original_float
        
        return result.astype(np.uint8)
    
    def _create_mouth_mask(self, shape, landmarks):
        """
        Crée un masque pour la zone de la bouche
        
        Args:
            shape: Dimensions de l'image (height, width)
            landmarks: Points faciaux (5 points InsightFace)
                      [left_eye, right_eye, nose, left_mouth, right_mouth]
        
        Returns:
            Masque float [0-1] pour zone bouche
        """
        # Points bouche: indices 3 et 4
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]
        mouth_center = ((mouth_left + mouth_right) / 2).astype(int)
        
        # Rayon dynamique basé sur largeur bouche
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        radius = int(mouth_width * 1.2)  # Zone élargie pour inclure lèvres
        
        # Créer masque circulaire
        mask = np.zeros(shape, dtype=np.float32)
        cv2.circle(mask, tuple(mouth_center), radius, 1.0, -1)
        
        # Flou gaussien pour transition douce
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius*0.3)
        
        return mask
    
    def _create_eyes_mask(self, shape, landmarks):
        """
        Crée un masque pour les zones des yeux
        
        Args:
            shape: Dimensions de l'image (height, width)
            landmarks: Points faciaux (5 points InsightFace)
        
        Returns:
            Masque float [0-1] pour zones yeux
        """
        # Points yeux: indices 0 et 1
        left_eye = landmarks[0].astype(int)
        right_eye = landmarks[1].astype(int)
        
        # Rayon basé sur distance entre yeux
        eye_distance = np.linalg.norm(right_eye - left_eye)
        radius = int(eye_distance * 0.35)
        
        # Créer masque avec deux cercles
        mask = np.zeros(shape, dtype=np.float32)
        cv2.circle(mask, tuple(left_eye), radius, 1.0, -1)
        cv2.circle(mask, tuple(right_eye), radius, 1.0, -1)
        
        # Flou gaussien pour transition douce
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius*0.4)
        
        return mask
    
    def blur_face(self, frame, face):
        """Floute un visage dans la frame"""
        # Obtenir la bounding box du visage
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # S'assurer que les coordonnées sont dans les limites de l'image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Extraire la région du visage
        face_region = frame[y1:y2, x1:x2]
        
        # Appliquer un flou gaussien
        blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
        
        # Remplacer la région dans l'image
        frame[y1:y2, x1:x2] = blurred
        
        return frame
    
    def pixelate_face(self, frame, face):
        """Pixelise un visage dans la frame"""
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Extraire et réduire la région
        face_region = frame[y1:y2, x1:x2]
        
        # Pixeliser en réduisant puis agrandissant
        small = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        
        frame[y1:y2, x1:x2] = pixelated
        
        return frame
