# Hetic_Seq2Seq-Additions

Modèle encodeur-décodeur LSTM pour apprendre l'addition de nombres à 2 chiffres. Le modèle prend une addition sous forme de chaîne de caractères (exemple : "12+34") et génère le résultat (exemple : "0046").

## Architecture

### Modèle d'entraînement

- **Encodeur** : Input(None, 13) → LSTM(256, return_state=True) → états (h, c)
- **Décodeur** : Input(None, 13) + états → LSTM(256, return_sequences=True) → Dense(13, softmax)
- **Compilation** : optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']

### Modèles d'inférence

- **encoder_model** : séquence source → états initiaux (h, c)
- **decoder_model** : [token, h, c] → [prédiction, nouveaux états]

### Classe Calculator

Toute la logique est encapsulée dans la classe `Calculator` dans `calculator.py` :

- `generate_data()` : génération de 50 000 additions aléatoires
- `build_vocabulary()` : création du vocabulaire (13 caractères : 0-9, +, START, END)
- `prepare_training_data()` : encodage one-hot des séquences
- `build_model()` : construction de l'architecture Seq2Seq
- `train()` : entraînement avec EarlyStopping et ModelCheckpoint
- `plot_training_history()` : visualisation des courbes
- `build_inference_models()` : création des modèles pour l'inférence
- `decode_sequence()` : décodage auto-régressif (greedy)
- `evaluate_samples()` : évaluation sur exemples du dataset
- `test_addition()` : test sur addition spécifique

## Installation

Python 3.12 recommandé :

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py
```

Le script exécute toutes les étapes :

1. Génération de 50 000 additions aléatoires
2. Construction du vocabulaire
3. Préparation des données (encodage one-hot)
4. Construction du modèle Seq2Seq
5. Entraînement sur 50 epochs (EarlyStopping après 5 epochs sans amélioration)
6. Visualisation des courbes (sauvegardées dans `training_history.png`)
7. Création des modèles d'inférence
8. Évaluation sur 20 exemples du dataset
9. Tests sur 5 additions spécifiques

## Fichiers générés

- `best_model.keras` : meilleur modèle sauvegardé pendant l'entraînement
- `training_history.png` : courbes de loss et accuracy

## Résultats attendus

Avec 50 000 exemples et 50 epochs, le modèle atteint généralement :

- **Accuracy** : ~99-100% sur le dataset d'entraînement
- **Val Accuracy** : ~99-100% sur la validation

Le modèle apprend rapidement l'addition et converge en quelques epochs.
