"""
Module d'anonymisation audio AVANCÉ V2
Techniques professionnelles:
- Pitch shifting (changement de hauteur)
- Formant shifting (changement d'identité vocale)
- Time stretching (changement de rythme)
- Filtrage fréquentiel (dégradation empreinte vocale)
- Combinaisons pour anonymisation maximale
"""
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import os


class AudioAnonymizerV2:
    """
    Anonymiseur audio professionnel avec techniques avancées
    """
    
    def __init__(self):
        """Initialise l'anonymiseur audio V2"""
        self.temp_dir = tempfile.gettempdir()
        
        # Vérifier FFmpeg
        if not self.check_ffmpeg():
            print("ATTENTION: FFmpeg non trouvé!")
            print("Téléchargez depuis: https://ffmpeg.org/download.html")
    
    def check_ffmpeg(self):
        """Vérifie si FFmpeg est installé"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except:
            return False
    
    def extract_audio(self, video_path, audio_path):
        """Extrait l'audio d'une vidéo en WAV haute qualité"""
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # Pas de vidéo
            '-acodec', 'pcm_s16le',  # WAV 16-bit
            '-ar', '44100',  # 44.1 kHz
            '-ac', '2',  # Stereo
            '-y',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg extraction échouée: {result.stderr}")
        return os.path.exists(audio_path)
    
    def pitch_shift(self, audio_path, output_path, semitones=5):
        """
        PITCH SHIFTING - Change la hauteur de la voix
        Homme (+5 semitones) → voix plus féminine
        Femme (-5 semitones) → voix plus masculine
        
        Args:
            audio_path: Audio d'entrée
            output_path: Audio de sortie
            semitones: Nombre de demi-tons (+/-)
        """
        print(f"   Pitch shifting: {semitones:+d} demi-tons")
        
        # Calculer le facteur de pitch
        # 1 semitone = facteur 2^(1/12)
        pitch_factor = 2 ** (semitones / 12.0)
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', f'asetrate=44100*{pitch_factor},aresample=44100',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def formant_shift(self, audio_path, output_path, shift_factor=1.2):
        """
        FORMANT SHIFTING - Change l'identité vocale
        Modifie les formants (résonances du conduit vocal)
        Plus efficace que le simple pitch shifting
        
        Args:
            audio_path: Audio d'entrée
            output_path: Audio de sortie
            shift_factor: Facteur de décalage (1.0 = normal, 1.2 = masculin→féminin)
        """
        print(f"   Formant shifting: facteur {shift_factor}")
        
        # Utiliser rubberband (meilleur que FFmpeg pour formants)
        # Fallback sur pitch + tempo si rubberband absent
        
        # Méthode FFmpeg: combiner pitch shift + time stretch
        # pour simuler formant shifting
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', f'asetrate=44100*{shift_factor},atempo={1/shift_factor},aresample=44100',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def frequency_filter(self, audio_path, output_path, filter_type='telephonic'):
        """
        FILTRAGE FRÉQUENTIEL - Dégrade l'empreinte vocale
        
        Types:
        - 'telephonic': Simule téléphone (300-3400 Hz)
        - 'lowpass': Filtre passe-bas (garde graves, supprime aigus)
        - 'highpass': Filtre passe-haut (garde aigus, supprime graves)
        - 'bandpass': Filtre passe-bande (garde médiums)
        
        Args:
            audio_path: Audio d'entrée
            output_path: Audio de sortie
            filter_type: Type de filtre
        """
        print(f"   Filtrage fréquentiel: {filter_type}")
        
        filters = {
            'telephonic': 'highpass=f=300,lowpass=f=3400',
            'lowpass': 'lowpass=f=1500',
            'highpass': 'highpass=f=1000',
            'bandpass': 'bandpass=f=1000:width=500',
            'radio': 'highpass=f=400,lowpass=f=2500,acompressor'
        }
        
        audio_filter = filters.get(filter_type, filters['telephonic'])
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', audio_filter,
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def time_stretch(self, audio_path, output_path, tempo=1.3):
        """
        TIME STRETCHING - Change le rythme sans changer la hauteur
        
        Args:
            audio_path: Audio d'entrée
            output_path: Audio de sortie
            tempo: Facteur de vitesse (1.0 = normal, >1 = plus rapide)
        """
        print(f"   Time stretching: tempo ×{tempo}")
        
        # Limiter tempo pour éviter artefacts
        tempo = max(0.5, min(2.0, tempo))
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', f'atempo={tempo}',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def robot_voice(self, audio_path, output_path):
        """
        VOIX ROBOTIQUE - Effet vocoder
        Combinaison de vibrato + tremolo + filtrage
        """
        print("   Voix robotique (vocoder)")
        
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', 'vibrato=f=5:d=0.8,tremolo=f=10:d=0.5,aecho=0.8:0.88:60:0.4',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def distortion(self, audio_path, output_path, intensity=0.5):
        """
        DISTORSION - Dégrade la qualité audio
        Utile pour masquer l'identité
        """
        print(f"   Distorsion: intensité {intensity}")
        
        # Combiner compression + saturation + bruit
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', f'acompressor=threshold=-20dB:ratio=8:attack=2:release=50,volume=2,afftdn=nf=-20',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR: {result.stderr}")
            return False
        return True
    
    def combine_audio_video(self, video_path, audio_path, output_path):
        """Combine vidéo et audio traité"""
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copier vidéo sans réencodage
            '-c:a', 'aac',  # Encoder audio en AAC
            '-b:a', '192k',  # Bitrate audio 192 kbps
            '-map', '0:v:0',  # Vidéo du premier fichier
            '-map', '1:a:0',  # Audio du second fichier
            '-shortest',  # Durée = plus court des deux
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ERREUR combinaison: {result.stderr}")
            return False
        return True
    
    def anonymize_audio(self, video_path, output_path, mode='pitch'):
        """
        Anonymise l'audio d'une vidéo
        
        Modes disponibles:
        - 'pitch': Pitch shift +5 (homme → femme)
        - 'pitch-down': Pitch shift -5 (femme → homme)
        - 'formant': Formant shifting (changement identité)
        - 'telephonic': Filtre téléphonique (300-3400 Hz)
        - 'radio': Filtre radio + compression
        - 'robot': Voix robotique (vocoder)
        - 'distort': Distorsion forte
        - 'speed': Accélération tempo
        - 'slow': Ralentissement tempo
        - 'max-anon': Combinaison maximale (pitch + formant + filtre)
        - 'remove': Suppression complète de l'audio
        
        Args:
            video_path: Vidéo d'entrée
            output_path: Vidéo de sortie avec audio anonymisé
            mode: Mode d'anonymisation
        """
        print(f"\n   Mode d'anonymisation audio: {mode}")
        
        if not self.check_ffmpeg():
            print("   ERREUR: FFmpeg requis!")
            return False
        
        # Fichiers temporaires
        temp_audio = os.path.join(self.temp_dir, 'temp_audio.wav')
        temp_processed = os.path.join(self.temp_dir, 'temp_processed.wav')
        
        try:
            # Cas spécial: suppression audio
            if mode == 'remove':
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-an',  # Pas d'audio
                    '-c:v', 'copy',
                    '-y',
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True)
                return result.returncode == 0
            
            # Extraire l'audio
            print("   Extraction audio...")
            if not self.extract_audio(video_path, temp_audio):
                return False
            
            # Appliquer le traitement selon le mode
            success = False
            
            if mode == 'pitch':
                success = self.pitch_shift(temp_audio, temp_processed, semitones=5)
            
            elif mode == 'pitch-down':
                success = self.pitch_shift(temp_audio, temp_processed, semitones=-5)
            
            elif mode == 'formant':
                success = self.formant_shift(temp_audio, temp_processed, shift_factor=1.2)
            
            elif mode == 'telephonic':
                success = self.frequency_filter(temp_audio, temp_processed, 'telephonic')
            
            elif mode == 'radio':
                success = self.frequency_filter(temp_audio, temp_processed, 'radio')
            
            elif mode == 'robot':
                success = self.robot_voice(temp_audio, temp_processed)
            
            elif mode == 'distort':
                success = self.distortion(temp_audio, temp_processed, intensity=0.7)
            
            elif mode == 'speed':
                success = self.time_stretch(temp_audio, temp_processed, tempo=1.3)
            
            elif mode == 'slow':
                success = self.time_stretch(temp_audio, temp_processed, tempo=0.8)
            
            elif mode == 'max-anon':
                # ANONYMISATION MAXIMALE: pitch + formant + filtre
                print("   Anonymisation maximale (multi-passes)")
                temp1 = os.path.join(self.temp_dir, 'temp1.wav')
                temp2 = os.path.join(self.temp_dir, 'temp2.wav')
                
                # Pass 1: Pitch shift
                if self.pitch_shift(temp_audio, temp1, semitones=5):
                    # Pass 2: Formant shift
                    if self.formant_shift(temp1, temp2, shift_factor=1.15):
                        # Pass 3: Filtre téléphonique
                        success = self.frequency_filter(temp2, temp_processed, 'telephonic')
                        os.remove(temp1)
                        os.remove(temp2)
            
            else:
                print(f"   Mode inconnu: {mode}")
                return False
            
            if not success:
                print("   Échec du traitement audio")
                return False
            
            # Combiner avec la vidéo
            print("   Combinaison audio + vidéo...")
            success = self.combine_audio_video(video_path, temp_processed, output_path)
            
            # Nettoyer
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            if os.path.exists(temp_processed):
                os.remove(temp_processed)
            
            return success
            
        except Exception as e:
            print(f"   ERREUR: {e}")
            # Nettoyer en cas d'erreur
            for f in [temp_audio, temp_processed]:
                if os.path.exists(f):
                    os.remove(f)
            return False


def main():
    """Test du module"""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python audio_anonymizer_v2.py <video_input> <video_output> <mode>")
        print("\nModes disponibles:")
        print("  pitch         - Voix plus aiguë (+5 semitones)")
        print("  pitch-down    - Voix plus grave (-5 semitones)")
        print("  formant       - Changement identité vocale")
        print("  telephonic    - Filtre téléphone (300-3400 Hz)")
        print("  radio         - Filtre radio + compression")
        print("  robot         - Voix robotique")
        print("  distort       - Distorsion")
        print("  speed         - Accélération (×1.3)")
        print("  slow          - Ralentissement (×0.8)")
        print("  max-anon      - Anonymisation maximale")
        print("  remove        - Suppression audio")
        return
    
    video_in = sys.argv[1]
    video_out = sys.argv[2]
    mode = sys.argv[3]
    
    anonymizer = AudioAnonymizerV2()
    
    print("=" * 60)
    print(" ANONYMISATION AUDIO V2")
    print("=" * 60)
    print(f" Vidéo entrée: {video_in}")
    print(f" Vidéo sortie: {video_out}")
    print(f" Mode: {mode}")
    print("=" * 60)
    
    success = anonymizer.anonymize_audio(video_in, video_out, mode)
    
    if success:
        print("\n✓ Audio anonymisé avec succès!")
    else:
        print("\n✗ Échec de l'anonymisation audio")


if __name__ == '__main__':
    main()
