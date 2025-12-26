"""
Script principal pour anonymiser des vidéos en remplaçant les visages
Inclut des filtres avancés et des analyses vidéo
"""
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from face_swapper import FaceSwapper
from temporal_filter import TemporalFilter, AdaptiveTemporalFilter
from video_analyzer import VideoAnalyzer
from audio_anonymizer_v2 import AudioAnonymizerV2 as AudioAnonymizer

def anonymize_video(input_path, output_path, face_dataset_path=None, mode='swap', 
                    temporal_filter=None, enable_analysis=False, audio_mode=None):
    """
    Anonymise une vidéo en remplaçant/floutant les visages et en modifiant l'audio
    
    Args:
        input_path: Chemin vers la vidéo d'entrée
        output_path: Chemin vers la vidéo de sortie
        face_dataset_path: Chemin vers le dossier de visages de remplacement
        mode: Mode d'anonymisation
        temporal_filter: Type de filtre temporel ('moving_average', 'exponential', 'kalman', 'adaptive')
        enable_analysis: Activer l'analyse et comparaison vidéo
        audio_mode: Mode d'anonymisation audio
    """
    
    # Vérifier que le fichier d'entrée existe
    if not Path(input_path).exists():
        print(f" Erreur: Le fichier {input_path} n'existe pas")
        return
    
    print("=" * 60)
    print(" ANONYMISATION VIDÉO")
    print("=" * 60)
    print(f" Vidéo d'entrée: {input_path}")
    print(f" Vidéo de sortie: {output_path}")
    print(f" Mode: {mode}")
    print("=" * 60)
    
    # Initialiser le swapper de visages
    swapper = FaceSwapper(face_dataset_path)
    
    # Initialiser le filtre temporel si demandé
    temp_filter = None
    if temporal_filter:
        if temporal_filter == 'adaptive':
            temp_filter = AdaptiveTemporalFilter()
        else:
            temp_filter = TemporalFilter(method=temporal_filter)
    
    # Initialiser l'analyseur si demandé
    analyzer = None
    original_frames = []
    if enable_analysis:
        analyzer = VideoAnalyzer()
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(input_path)
    
    # Obtenir les propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n Propriétés de la vidéo:")
    print(f"   - Résolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Frames totales: {total_frames}")
    print(f"   - Durée: {total_frames/fps:.2f} secondes\n")
    
    # Créer le dossier de sortie s'il n'existe pas
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Créer un fichier temporaire AVI (plus fiable qu'MP4 avec OpenCV)
    import tempfile
    import os
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, f"temp_anonymized_{Path(output_path).stem}.avi")
    
    # Configurer l'encodeur vidéo (XVID pour AVI, plus compatible)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(" Erreur: Impossible de créer le fichier vidéo de sortie")
        print(f"   Chemin temporaire: {temp_video_path}")
        return
    
    # Traiter chaque frame
    frame_count = 0
    blur_mode = (mode == 'blur')
    
    # Créer fenêtre pour affichage en temps réel
    window_name = "Anonymisation en cours (Appuyez sur 'q' pour arrêter)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)  # Taille fenêtre confortable
    
    with tqdm(total=total_frames, desc="Traitement des frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Appliquer l'anonymisation selon le mode
            if mode == 'pixelate':
                # Mode pixelate personnalisé
                faces = swapper.app.get(frame)
                for face in faces:
                    frame = swapper.pixelate_face(frame, face)
                processed_frame = frame
                
            else:
                # Mode swap, blur ou pixelate
                processed_frame = swapper.swap_faces(frame, blur_mode=blur_mode)
            
            # Appliquer le filtre temporel si activé
            if temp_filter:
                processed_frame = temp_filter.apply(processed_frame)
            
            # Sauvegarder la frame originale pour analyse si nécessaire
            if enable_analysis:
                original_frames.append(frame.copy())
            
            # Afficher la frame traitée en temps réel
            cv2.imshow(window_name, processed_frame)
            
            # Permettre d'arrêter avec 'q' (attente de 1ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n Traitement interrompu par l'utilisateur")
                break
            
            # Écrire la frame traitée
            out.write(processed_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Fermer la fenêtre d'aperçu
    cv2.destroyAllWindows()
    
    # Libérer les ressources
    cap.release()
    out.release()
    
    print("\n Conversion AVI → MP4 avec FFmpeg...")
    # Convertir le fichier AVI temporaire en MP4 final avec FFmpeg
    import subprocess
    
    # Toujours copier l'audio de la vidéo originale si disponible
    # (sera anonymisé plus tard si audio_mode est activé)
    cmd = [
        'ffmpeg', '-i', temp_video_path,
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-map', '0:v:0',  # Vidéo du fichier temp
        '-map', '1:a:0?',  # Audio de l'original (si existe)
        '-c:a', 'aac',
        '-y',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f" Erreur FFmpeg conversion:")
        print(f"   {result.stderr}")
    else:
        print(f" Vidéo MP4 créée: {output_path}")
        # Supprimer le fichier AVI temporaire
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    print("\n" + "=" * 60)
    print(" TRAITEMENT TERMINÉ!")
    print("=" * 60)
    print(f" Vidéo sauvegardée: {output_path}")
    print(f"  Frames traitées: {frame_count}")
    print("=" * 60)
    
    # Créer une vidéo d'analyse si demandé
    analysis_path = None
    if enable_analysis and len(original_frames) > 0:
        print("\n Création de la vidéo d'analyse...")
        analysis_path = str(Path(output_path).parent / (Path(output_path).stem + '_analysis.mp4'))
        
        # Créer une vidéo côte à côte
        cap2 = cv2.VideoCapture(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        ret, first_processed = cap2.read()
        if ret:
            side_by_side = analyzer.create_side_by_side(original_frames[0], first_processed)
            h, w = side_by_side.shape[:2]
            out_analysis = cv2.VideoWriter(analysis_path, fourcc, fps, (w, h))
            
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for idx in range(min(len(original_frames), frame_count)):
                ret, processed = cap2.read()
                if ret and idx < len(original_frames):
                    comparison = analyzer.create_side_by_side(
                        original_frames[idx], processed,
                        labels=("Original", "Anonymisé")
                    )
                    out_analysis.write(comparison)
            
            out_analysis.release()
            print(f" Vidéo d'analyse créée: {analysis_path}")
        
        cap2.release()
    
    # Anonymiser l'audio si demandé
    if audio_mode:
        print("\n" + "=" * 60)
        print(" ANONYMISATION AUDIO")
        print("=" * 60)
        
        audio_anonymizer = AudioAnonymizer()
        
        # Créer un fichier temporaire pour la vidéo sans audio anonymisé
        temp_video = str(Path(output_path).parent / (Path(output_path).stem + '_temp.mp4'))
        
        # Renommer la vidéo actuelle en temp
        import shutil
        shutil.move(output_path, temp_video)
        
        # Anonymiser l'audio
        success = audio_anonymizer.anonymize_audio(temp_video, output_path, audio_mode)
        
        if success:
            # Supprimer le fichier temporaire
            import os
            os.remove(temp_video)
            print(" Audio anonymisé avec succès!")
        else:
            # Restaurer le fichier original si échec
            shutil.move(temp_video, output_path)
            print("  Échec de l'anonymisation audio, vidéo sauvegardée sans modification audio")
    
    # Ouvrir automatiquement les vidéos après traitement
    print("\n" + "=" * 60)
    print(" OUVERTURE DES VIDÉOS")
    print("=" * 60)
    
    try:
        import os
        
        # Ouvrir la vidéo anonymisée
        print(f" Ouverture de la vidéo: {output_path}")
        os.startfile(output_path)
        
        # Ouvrir la vidéo d'analyse si elle existe
        if analysis_path and Path(analysis_path).exists():
            import time
            time.sleep(1)  # Petit délai pour éviter d'ouvrir tout en même temps
            print(f" Ouverture de l'analyse: {analysis_path}")
            os.startfile(analysis_path)
    
    except Exception as e:
        print(f" Impossible d'ouvrir automatiquement les vidéos: {e}")
        print(f" Vous pouvez ouvrir manuellement:")
        print(f"   - Vidéo: {output_path}")
        if analysis_path:
            print(f"   - Analyse: {analysis_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Anonymiser une vidéo en remplaçant/floutant les visages'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Chemin vers la vidéo d\'entrée'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Chemin vers la vidéo de sortie'
    )
    
    parser.add_argument(
        '--face-dataset',
        type=str,
        default='face_dataset',
        help='Dossier contenant les visages de remplacement (défaut: face_dataset)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['swap', 'blur', 'pixelate'],
        default='swap',
        help='Mode d\'anonymisation: '
             'swap (remplacement de visage), '
             'blur (flou sur visage), '
             'pixelate (pixelisation de visage)'
    )
    
    parser.add_argument(
        '--temporal-filter',
        type=str,
        choices=['moving_average', 'exponential', 'kalman', 'adaptive'],
        default=None,
        help='Filtre temporel pour lisser la vidéo: '
             'moving_average (moyenne glissante), '
             'exponential (lissage exponentiel), '
             'kalman (filtre de Kalman), '
             'adaptive (adaptatif selon mouvement)'
    )
    
    parser.add_argument(
        '--enable-analysis',
        action='store_true',
        help='Créer une vidéo d\'analyse côte à côte (original vs anonymisé)'
    )
    
    parser.add_argument(
        '--audio-mode',
        type=str,
        choices=['pitch', 'pitch-down', 'formant', 'telephonic', 'radio', 'robot', 
                 'distort', 'speed', 'slow', 'max-anon', 'remove'],
        default=None,
        help='Mode d\'anonymisation audio AVANCÉ: '
             'pitch (voix +5 semitones, homme→femme), '
             'pitch-down (voix -5 semitones, femme→homme), '
             'formant (changement identité vocale), '
             'telephonic (filtre téléphone 300-3400Hz), '
             'radio (filtre radio + compression), '
             'robot (voix robotique/vocoder), '
             'distort (distorsion forte), '
             'speed (accélération ×1.3), '
             'slow (ralentissement ×0.8), '
             'max-anon (MAXIMUM: pitch+formant+filtre), '
             'remove (supprimer audio complètement)'
    )
    
    args = parser.parse_args()
    
    anonymize_video(
        args.input,
        args.output,
        args.face_dataset,
        args.mode,
        args.temporal_filter,
        args.enable_analysis,
        args.audio_mode
    )

if __name__ == "__main__":
    main()
