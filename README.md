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
│   ├── etl.py        # Rebuild du dataset SFT depuis le corpus gold
│   ├── train.py      # Script de fine-tuning QLoRA de production
│   └── push.py       # Sauvegarde automatique sur Hugging Face Hub
├── requirements.txt  # Dépendances requises
└── README.md         # Présentation et documentation du projet
```

## 🚀 Installation & Utilisation

### 1. Cloner le dépôt et installer les dépendances
```bash
git clone https://github.com/Herchan321/Moroccan_Darija_LLM.git
cd Moroccan-Darija-LLM
pip install -r requirements.txt
```

### 2. Rejouer l'ETL
Les scripts ETL sont conservés dans [`src/etl.py`](src/etl.py). Ils reconstruisent
les exemples SFT, suppriment les doublons et créent une séparation train/validation
sans partager la même phrase anglaise entre les deux ensembles :

```bash
python src/etl.py
```

Les sorties sont écrites dans `data/gold/sft/` et le résumé dans
`reports/etl_summary.json`.

### 3. Lancer l'entraînement (Fine-Tuning LoRA)
Pour lancer l'entraînement de l'adaptateur sur GPU (recommandé : T4, A100, etc.) :
```bash
python src/train.py
```

Le script utilise automatiquement les fichiers `train.jsonl` et `validation.jsonl`.
Il fonctionne aussi sur CPU, mais l'entraînement complet peut prendre plusieurs
heures ou jours. Pour commencer avec un petit test :

```bash
python src/train.py --max_train_samples 20 --max_eval_samples 5 --max_length 128
```

Pour lancer le dataset complet sur CPU :

```bash
python src/train.py --batch_size 1 --max_length 256
```

Dans Google Colab, sélectionnez
**Runtime > Change runtime type > T4 GPU**, installez les dépendances, puis exécutez :

```python
!pip install -r requirements.txt
!python src/train.py
```

Pour reprendre un checkpoint après une interruption :

```bash
python src/train.py --resume_from_checkpoint
```

`--allow_cpu` est réservé à un petit test de fonctionnement et n'est pas
recommandé pour entraîner le modèle complet.