# Moroccan Darija LLM (Qwen-2.5-1.5B-Instruct Fine-Tuning)

Ce projet implémente un pipeline complet d'ingénierie de données et d'entraînement (Fine-Tuning LoRA / SFT) pour adapter le modèle de langage **Qwen-2.5-1.5B-Instruct** à la traduction et la génération en **Darija Marocaine** (arabe marocain), couvrant à la fois l'écriture arabe classique et le format Franco-Arabe (Arabizi).

## 📊 Vue d'ensemble du Dataset (Gold Standard)

Notre dataset unifié et nettoyé rassemble **50 743 paires linguistiques** uniques :
- **DODa** : 48 741 exemples (Arabe + Écriture Franco/Latin).
- **Atlaset** (filtré et validé) : 2 002 exemples de la vie quotidienne.

### Tâches d'entraînement (SFT)
À partir de ce corpus, nous avons généré un pipeline de **198 158 tâches d'instruction** réparties comme suit :
1. **Traductions Darija Arabe ➔ Anglais** : 50 344 exemples.
2. **Traductions Anglais ➔ Darija Arabe** : 50 344 exemples.
3. **Traductions Darija Franco ➔ Anglais** : 48 735 exemples.
4. **Traductions Anglais ➔ Darija Franco** : 48 735 exemples.

Le dataset d'entraînement final comprend **20 000 échantillons équilibrés** (`train.jsonl`) et un ensemble de validation de **2 000 échantillons** (`validation.jsonl`).

## 📁 Structure du Projet

```directory
Moroccan_Darija_LLM/
├── data/
│   ├── bronze/       # Données brutes de départ (Atlaset & DODa)
│   ├── silver/       # Données après nettoyage, normalisation et déduplication
│   └── gold/         # Dataset final unifié et fichiers SFT (train.jsonl, validation.jsonl)
├── models/           # Checkpoints locaux et adaptateurs LoRA sauvegardés
├── src/              # Scripts d'exécution principaux
│   ├── train.py      # Script de fine-tuning QLoRA de production
│   └── push.py       # Sauvegarde automatique sur Hugging Face Hub
├── requirements.txt  # Dépendances requises
└── README.md         # Présentation et documentation du projet
```

## 🚀 Installation & Utilisation

### 1. Cloner le dépôt et installer les dépendances
```bash
git clone https://github.com/votre-username/Moroccan-Darija-LLM.git
cd Moroccan-Darija-LLM
pip install -r requirements.txt
```

### 2. Lancer l'entraînement (Fine-Tuning LoRA)
Pour lancer l'entraînement de l'adaptateur sur GPU (recommandé : T4, A100, etc.) :
```bash
python src/train.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --train_file "data/gold/sft/train.jsonl" --val_file "data/gold/sft/validation.jsonl"
```