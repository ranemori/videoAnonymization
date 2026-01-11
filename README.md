# Projet d'Anonymisation et Traitement Vidéo - Deep Learning

## Description

Ce projet est une **plateforme complète de traitement vidéo** utilisant le deep learning avec des modèles pré-entraînés. Il permet:
- **Anonymisation**: Remplacement des visages et des voix pour protéger l'identité
- **Filtrage avancé**: Application de filtres d'image sophistiqués (médian, bilatéral, Sobel, Canny)
- **Lissage temporel**: Réduction du scintillement avec filtres temporels (moyenne glissante, exponentiel, Kalman)
- **Analyse vidéo**: Comparaisons avant/après, histogrammes, cartes de différence

Demonstration:

https://github.com/user-attachments/assets/1c1f7d00-2a39-4c06-8652-87ee14c349fd

### Fonctionnalités principales

#### Anonymisation
- **Face Swap**: Remplacement de visages avec des visages générés par GAN
- **Face Blur/Pixelate**: Floutage ou pixelisation des visages
- **Tracking de personnes**: Chaque personne garde le même visage de remplacement tout au long de la vidéo
- **Anonymisation audio**: 7 modes de modification de voix (pitch, robot, distorsion, etc.)

#### Filtres d'image avancés
- **Flou Gaussien**: Lissage classique
- **Filtre Médian**: Réduction du bruit tout en préservant les bords
- **Filtre Bilatéral**: Lissage avancé qui préserve les contours
- **Détection de contours**: Sobel et Canny pour extraction de bords

####  Filtres temporels
- **Moyenne glissante**: Lissage sur N frames consécutives
- **Filtre exponentiel**: Lissage exponentiel entre frames
- **Filtre de Kalman**: Prédiction et correction pour réduction du scintillement
- **Filtre adaptatif**: Ajustement automatique selon le mouvement détecté

####  Analyse et visualisation
- **Comparaison côte à côte**: Original vs anonymisé
- **Cartes de différence**: Heatmap des modifications
- **Histogrammes RGB**: Analyse des distributions de couleur
- **Métriques de qualité**: MSE, PSNR pour quantifier les changements

## Technologies utilisées

- **Python 3.10**: Langage de programmation
- **InsightFace**: Détection et swap de visages (modèles pré-entraînés)
- **OpenCV**: Traitement vidéo et segmentation
- **ONNX Runtime**: Exécution rapide des modèles deep learning
- **FFmpeg**: Traitement et anonymisation audio
- **Anaconda**: Gestion d'environnement et dépendances

### Modèles pré-entraînés utilisés

- **buffalo_l** (InsightFace): Détection de visages, landmarks, embeddings
- **inswapper_128.onnx** (528 MB): Modèle de face swap basé sur GAN

## Installation complète

### Prérequis

- **Windows 10/11**
- **Anaconda** installé ([Télécharger ici](https://www.anaconda.com/download))
- **Microsoft Visual C++ Build Tools** (requis pour InsightFace)
- L'anonymisation audio nécessite **FFmpeg**, un outil puissant de traitement audio/vidéo ([Télécharger ici](https://ffmpeg.org/download.html))


### Étape 1: Cloner le repository

git clone https://github.com/ranemori/videoAnonymization.git

### Étape 2: Installer Visual Studio Build Tools

InsightFace nécessite des outils de compilation C++:

1. Télécharger [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Installer avec l'option "**Desktop development with C++**"
3. Redémarrer l'ordinateur

### Étape 3: Créer l'environnement Conda

```powershell ou Anaconda Prompt
# Naviguer vers le dossier du projet
cd videoAnonymization\videoPrivacy

# Créer l'environnement avec TOUTES les dépendances
conda env create -f environment.yml

# Activer l'environnement
conda activate video-anonymization
```

**Note**: Le fichier `environment.yml` installe automatiquement toutes les dépendances nécessaires (insightface, opencv, onnxruntime, etc.). Les packages PyTorch
**Alternative avec pip uniquement** (si vous n'utilisez pas Conda):
```powershell
pip install -r requirements.txt
```

### Étape 4: Télécharger le modèle de face swap

```powershell ou Anaconda Prompt
python download_inswapper.py
```

### Étape 5: Préparer les datasets 
(Facultative si vous voulez utiliser votre propre image)

#### Dataset de visages (face_dataset/)

Ajoutez 10-20 images de visages dans le dossier `face_dataset/`:
- Format: JPG, PNG
- Visages frontaux de bonne qualité
- Diversité recommandée (âge, genre, ethnicité)

### Étape 6: Ajouter votre vidéo

Placez votre vidéo à anonymiser dans `input_videos/`:
```
input_videos/
└── video.mp4    # Votre vidéo
```

## Utilisation

### Commande de base

```powershell ou Anaconda Prompt
conda activate video-anonymization
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/resultat.mp4 --mode swap
```

### Modes d'anonymisation disponibles

####  Modes de base

**1. swap** - Face Swap uniquement
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/face_swap.mp4 --mode swap
```

**2. blur** - Flou gaussien sur visages
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/blur.mp4 --mode blur
```

**3. pixelate** - Pixelisation des visages
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/pixel.mp4 --mode pixelate
```

**Filtre de Kalman** (prédiction et correction)
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/smooth_kalman.mp4 --mode swap --temporal-filter kalman
```

**Filtre adaptatif** (s'adapte au mouvement)
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/smooth_adaptive.mp4 --mode swap --temporal-filter adaptive
```

### Analyse vidéo

**Générer une vidéo de comparaison automatiquement**
```powershell
python anonymize_video.py --input input_videos/video.mp4 --output output_videos/result.mp4 --mode swap --enable-analysis
```
Crée automatiquement `result_analysis.mp4` avec comparaison côte à côte.

**Analyser deux vidéos manuellement**
```powershell
# Comparaison côte à côte
python analyze_video.py --original input_videos/video.mp4 --processed output_videos/result.mp4 --type side_by_side

# Carte de différence
python analyze_video.py --original input_videos/video.mp4 --processed output_videos/result.mp4 --type difference

# Grille complète (original, traité, différence, histogrammes)
python analyze_video.py --original input_videos/video.mp4 --processed output_videos/result.mp4 --type grid
```
### Anonymisation Audio

#### Modes d'anonymisation audio disponibles

**1. pitch** - Voix plus aiguë (+5 demi-tons)
```powershell
python anonymize_audio.py --input video.mp4 --output video_pitch.mp4 --mode pitch
```
Utilisation: Homme → Voix plus féminine

**2. pitch-down** - Voix plus grave (-5 demi-tons)
```powershell
python anonymize_audio.py --input video.mp4 --output video_pitch_down.mp4 --mode pitch-down
```
Utilisation: Femme → Voix plus masculine

**3. robot** - Effet robotique/vocoder
```powershell
python anonymize_audio.py --input video.mp4 --output video_robot.mp4 --mode robot
```
Utilisation: Anonymisation complète avec effet futuriste

**4. distort** - Distorsion audio
```powershell
python anonymize_audio.py --input video.mp4 --output video_distort.mp4 --mode distort
```
Utilisation: Effet "téléphone" ou "radio"

**5. speed** - Accélération (1.3x)
```powershell
python anonymize_audio.py --input video.mp4 --output video_speed.mp4 --mode speed
```

**6. speed-down** - Ralentissement (0.8x)
```powershell
python anonymize_audio.py --input video.mp4 --output video_slow.mp4 --mode speed-down
```

**7. remove** - Suppression complète
```powershell
python anonymize_audio.py --input video.mp4 --output video_silent.mp4 --mode remove
```

#### Anonymisation complète (Vidéo + Audio)

**Commande combinée en une seule étape:**
```powershell
python anonymize_video.py \
  --input input_videos/video.mp4 \
  --output output_videos/anonyme_complet.mp4 \
  --mode body-swap \
  --audio-mode pitch
```

**Exemples de combinaisons:**

**Anonymisation maximale homme → femme**
```powershell
python anonymize_video.py \
  --input video.mp4 \
  --output anonyme.mp4 \
  --mode virtual-tryon \
  --audio-mode pitch \
  --temporal-filter adaptive
```
- Vêtements changés
- Voix plus aiguë
- Lissage temporel

**Anonymisation maximale femme → homme**
```powershell
python anonymize_video.py \
  --input video.mp4 \
  --output anonyme.mp4 \
  --mode virtual-tryon \
  --audio-mode pitch-down
```
- Vêtements changés
- Voix plus grave

**Témoignage anonyme**
```powershell
python anonymize_video.py \
  --input temoignage.mp4 \
  --output temoignage_anonyme.mp4 \
  --mode blur \
  --audio-mode distort
```

**Interview robotique**
```powershell
python anonymize_video.py \
  --input interview.mp4 \
  --output interview_anonyme.mp4 \
  --mode swap \
  --audio-mode robot
```

### Paramètres disponibles

```powershell
python anonymize_video.py --help
```

**Paramètres:**
- `--input`: Vidéo d'entrée (requis)
- `--output`: Vidéo de sortie (requis)
- `--mode`: Mode d'anonymisation (12 modes disponibles)
- `--face-dataset`: Dossier des visages (défaut: face_dataset)
- `--temporal-filter`: Filtre temporel (moving_average, exponential, kalman, adaptive)
- `--audio-mode`: Mode d'anonymisation audio (pitch, pitch-down, robot, distort, speed, speed-down, remove)
- `--enable-analysis`: Créer une vidéo de comparaison

## Structure du projet

```
videoPrivacy/
├── anonymize_video.py              # Script principal (point d'entrée)
├── face_swapper.py                 # Module de face swap avec tracking
├── temporal_filter.py              # Filtres temporels (moyenne, exponentiel, Kalman)
├── video_analyzer.py               # Analyse et comparaison vidéo
├── audio_anonymizer_v2.py             # Module d'anonymisation audio
├── analyze_video.py                # Script utilitaire d'analyse
├── download_inswapper.py           # Télécharge le modèle inswapper
├── environment.yml                 # Configuration environnement Conda
├── requirements.txt                # Dépendances Python
│
├── models/
│   └── inswapper_128.onnx         # Modèle de face swap (528 MB)
│
├── face_dataset/                   # Visages de remplacement
│   ├── face1.jpg
│   ├── face2.jpg
│   └── ...                         # 10-20 images recomman
├── input_videos/                   # Vos vidéos à traiter
│   └── video.mp4
│
├── output_videos/                  # Vidéos anonymisées (résultats)
│   ├── resultat.mp4
│   └── resultat_analysis.mp4      # Comparaison (si --enable-analysis)
│
└── README.md                       # Ce fichier
```

**Installation CUDA (optionnel):**
1. Installer [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
2. Installer cuDNN
3. Redémarrer

##  Contribution1-2
. Améliorer la détection de personnes (modèles plus récents)
. Ajouter un GUI (interface graphique web ou desktop)
. Implémenter des filtres temporels plus sophistiqués
. Ajouter anonymisation des vetements 

1. Forkez le projet
2. Créez une branche (`git checkout -b feature/NouvelleFonctionnalité`)
3. Commitez (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Poussez (`git push origin feature/NouvelleFonctionnalité`)
5. Ouvrez une Pull Request

## Auteur

**Rania**
- GitHub: [ranemori](https://github.com/ranemori)

## Remerciements

- **InsightFace** pour les modèles de détection et swap de visages
- **OpenCV** pour les outils de traitement d'image et vidéo
- **ONNX Runtime** pour l'exécution efficace des modèles
- Communauté open source pour les algorithmes de filtrage

Pour toute question ou suggestion, n'hésitez pas à nous contacter via :
- Email: raniarina.y@gmail.com

---

Fait avec ❤️ 
