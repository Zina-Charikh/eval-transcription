# 🎬 SubQuality

Analyse automatique de la qualité des sous-titres STT  
**France 24 × Whisper (OpenAI) × CC YouTube × Wildmoka**

---

## Structure du projet

```
subquality/
├── README.md
├── requirements.txt
│── france24.fr.srt      # CC YouTube (téléchargés avec yt-dlp)
│── france24.mp4         # Vidéo source
├── data/
│   ├── reference.txt        # Référence terrain (corrigée manuellement)
│   └── transcription.json  # Généré par transcribe.py
├── scripts/
│   ├── transcribe.py        # Étape 1 : Whisper → transcription.json
│   ├── evaluate.py          # Étape 2 : WER normalisé + CER + BLEU
│   ├── dashboard.py         # Étape 3 : frames vidéo + Excel + PNG
│   └── app.py               # Étape 4 : application Streamlit
└── outputs/
    ├── eval_results.json    # Métriques (généré par evaluate.py)
    ├── dashboard.png        # Dashboard visuel
    ├── SubQuality.xlsx      # Excel 4 onglets
    ├── wildmoka_clips.json  # Export API Wildmoka
    └── frames/
        ├── whisper/         # Frames annotées Whisper
        └── cc/              # Frames annotées CC YouTube
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Télécharger la vidéo et les sous-titres

```bash
# Vidéo
yt-dlp "https://www.youtube.com/shorts/Naks0LCYDms" -o data/france24.mp4

# Sous-titres CC YouTube
yt-dlp --write-auto-subs --sub-lang fr --skip-download \
  -o "data/france24" "https://www.youtube.com/shorts/Naks0LCYDms"
```

---

## Ordre d'exécution

```bash
cd scripts/

# 1. Transcrire avec Whisper (génère transcription.json)
python3 transcribe.py

# 2. Évaluer WER normalisé + CER + BLEU (génère eval_results.json)
python3 evaluate.py

# 3. Générer frames vidéo + Excel + PNG (génère outputs/)
python3 dashboard.py

# 4. Lancer l'application Streamlit
streamlit run app.py
```

---

## Métriques utilisées

| Métrique | Description | Interprétation |
|---|---|---|
| **WER normalisé** | Word Error Rate sans ponctuation ni majuscules | Plus bas = mieux. Seuil Wildmoka : 25% |
| **CER** | Character Error Rate — lettre par lettre | Révèle si l'erreur est mineure (1 lettre) ou grave |
| **BLEU** | Qualité des séquences de mots (0-100) | Plus haut = mieux. > 60 = bonne qualité |

### Pourquoi 3 métriques ?

Le WER seul est trop binaire : "Tabaro" au lieu de "Tabarot" = 1 mot faux = WER 25%,
alors que c'est juste une lettre manquante. Le CER (14%) et le BLEU (> 80) montrent
que c'est une erreur mineure. Les 3 métriques ensemble donnent une image complète.

---

## Principales librairies

| Librairie | Rôle | Installation |
|---|---|---|
| `openai-whisper` | Transcription STT (speech-to-text) | `pip install openai-whisper` |
| `jiwer` | Calcul WER et CER | `pip install jiwer` |
| `sacrebleu` | Calcul BLEU score | `pip install sacrebleu` |
| `opencv-python` | Extraction frames vidéo | `pip install opencv-python` |
| `matplotlib` | Graphiques dashboard PNG | `pip install matplotlib` |
| `pandas` | Manipulation des données | `pip install pandas` |
| `openpyxl` | Génération fichier Excel | `pip install openpyxl` |
| `plotly` | Graphiques interactifs Streamlit | `pip install plotly` |
| `streamlit` | Application web interactive | `pip install streamlit` |
| `yt-dlp` | Téléchargement vidéo + sous-titres | `pip install yt-dlp` |

---

## Résultats obtenus

- **WER normalisé Whisper** : ~19% (modèle `base`)
- **CER Whisper** : ~8%
- **BLEU Whisper** : ~65/100
- **Clips Wildmoka** : 8/19 segments WER ≥ 25%

### Limites identifiées

- Whisper `base` est rapide mais imprécis — `large-v3` réduirait le WER à ~12%
- L'alignement CC YouTube par timestamps est approximatif
- Les frames vidéo montrent le contexte visuel mais pas d'analyse image/parole
- Le JSON Wildmoka est simulé — pas d'intégration à l'API réelle

### Prochaines étapes

- Tester `faster-whisper` avec le modèle `large-v3`
- Intégrer l'API Wildmoka réelle
- Ajouter la normalisation WER spécifique au français (nombres, abréviations)
- Computer vision pour concordance image/parole
