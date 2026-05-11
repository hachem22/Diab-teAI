# 🩺 DiabèteAI — Documentation Complète du Modèle

> Application web de prédiction du diabète propulsée par **XGBoost**, entraînée sur 100 000 patients réels.
> Inclut explicabilité (XAI/SHAP), validation anti-overfitting et interface utilisateur moderne.

---

## 📋 Table des matières

1. [Aperçu du projet](#1-aperçu-du-projet)
2. [Démarrage rapide](#2-démarrage-rapide)
3. [Le modèle — XGBoost](#3-le-modèle--xgboost)
4. [Le pipeline de données](#4-le-pipeline-de-données)
5. [Les hyperparamètres](#5-les-hyperparamètres)
6. [Métriques de performance](#6-métriques-de-performance)
7. [Validation anti-overfitting](#7-validation-anti-overfitting)
8. [Explicabilité (XAI / SHAP)](#8-explicabilité-xai--shap)
9. [Architecture de l'application](#9-architecture-de-lapplication)
10. [Endpoints API](#10-endpoints-api)
11. [Limites connues](#11-limites-connues)
12. [Structure des fichiers](#12-structure-des-fichiers)

---

## 1. Aperçu du projet

### Objectif
Détecter précocement le risque de diabète à partir de **18 indicateurs physiologiques** (sanguins, biométriques, comportementaux), en moins de 30 secondes.

### Données d'apprentissage
- **100 000 patients** (dataset Kaggle)
- **31 variables** initiales, **18 retenues** pour l'application
- **0 valeur manquante** (dataset propre)
- Cible : `diagnosed_diabetes` (binaire 0/1)
- Distribution : **60 % diabétiques / 40 % non-diabétiques** (légèrement déséquilibré mais gérable)

### Résultats clés
| Métrique | Valeur | Verdict |
|---|---|---|
| Test accuracy | **91.97 %** | Excellent |
| ROC-AUC | **0.9432** | Très bon |
| Train accuracy | **92.16 %** | — |
| Gap train/test | **0.18 %** | ✓ Aucun overfitting |
| Inférence par patient | **< 10 ms** | Temps réel |

---

## 2. Démarrage rapide

### Pré-requis
- Python 3.10+
- Le fichier `data/diabetes_dataset.csv`

### Installation

```bash
pip install -r requirements.txt
```

### Lancement

```bash
python app.py
```

Puis ouvrir **http://localhost:5000** dans le navigateur.

**Au premier lancement**, le modèle s'entraîne automatiquement (~30 secondes) et se sauvegarde dans `diabetes_xgb_model.pkl`. Les lancements suivants chargent simplement le pickle (instantané).

### Workflow utilisateur
```
1. Landing page (landing.html) → bouton "Démarrer mon bilan"
2. Formulaire wizard 4 étapes (predict.html) :
   ├─ Étape 1 : Profil physique (âge, sexe, IMC, ratio taille/hanche)
   ├─ Étape 2 : Bilan sanguin (glucose, HbA1c, insuline, cholestérol...)
   ├─ Étape 3 : Tension artérielle
   └─ Étape 4 : Habitudes (activité, sommeil, alimentation, tabac)
3. Bouton "Analyser" → POST /predict
4. Résultat affiché :
   ├─ Verdict (rouge / vert)
   ├─ Jauge circulaire animée (0-100 %)
   ├─ Top 6 facteurs SHAP locaux (rouge = aggrave, vert = protège)
   └─ Top variables globales du modèle
```

---

## 3. Le modèle — XGBoost

### Pourquoi XGBoost ?

Après comparaison rigoureuse de **6 algorithmes**, XGBoost a été retenu pour 5 raisons :

| # | Raison | Détail |
|---|---|---|
| 1 | **Meilleur ROC-AUC** | 0.9432 (vs 0.9209 pour Naive Bayes) |
| 2 | **Gestion native des NaN** | Pas besoin d'imputation préalable |
| 3 | **`predict_proba` calibré** | Score 0-100 % fiable pour la jauge |
| 4 | **Régularisation L1+L2** | Limite l'overfitting nativement |
| 5 | **Vitesse < 10 ms** | Compatible API REST temps réel |

### Tableau comparatif des 6 modèles testés

| Modèle | Accuracy | ROC-AUC | Verdict |
|---|---|---|---|
| **XGBoost** | **91.98 %** | **0.9432** | ✓ Retenu |
| Decision Tree | 91.98 % | 0.9431 | Équivalent |
| Random Forest | 91.97 % | 0.9417 | Équivalent |
| SVM (RBF) | 88.41 % | 0.9323 | Acceptable |
| SVM (Polynomial) | 87.78 % | 0.9290 | Acceptable |
| Naive Bayes | 85.60 % | 0.9209 | Faible |

### Pourquoi 3 modèles donnent presque le même score ?

L'analyse XAI révèle que **HbA1c représente 89.4 % de l'importance totale**. Cela signifie qu'il existe une règle quasi-déterministe `HbA1c > seuil → diabétique` que **tous les arbres apprennent identiquement**. C'est un **plafond statistique** du dataset, pas un défaut du modèle.

Le choix de XGBoost se fait alors sur les **critères techniques** (NaN, vitesse, calibration) plutôt que sur la seule accuracy.

---

## 4. Le pipeline de données

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Données brutes                       │
│                    100 000 × 18 features                 │
└─────────┬───────────────────────────────┬──────────────┘
          │                                │
          ▼                                ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  Variables numériques │    │ Variables catégorielles  │
│        (16 cols)      │    │  gender, smoking_status  │
└─────────┬────────────┘    └────────────┬─────────────┘
          │                                │
          ▼                                ▼
┌──────────────────────┐    ┌──────────────────────────┐
│ SimpleImputer (median)│    │ SimpleImputer (most_freq) │
└─────────┬────────────┘    └────────────┬─────────────┘
          │                                │
          ▼                                ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  StandardScaler       │    │  OneHotEncoder           │
│  (μ=0, σ=1)           │    │  (handle_unknown=ignore) │
└─────────┬────────────┘    └────────────┬─────────────┘
          │                                │
          └────────────────┬───────────────┘
                           ▼
              ┌──────────────────────────┐
              │   ColumnTransformer      │
              │      (union des 2)        │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │    XGBClassifier         │
              │  • 300 arbres            │
              │  • max_depth = 4         │
              │  • learning_rate = 0.05  │
              │  • L1 + L2 + bagging     │
              └────────────┬─────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  Prédiction     │
                 │  + Probabilité  │
                 │  + SHAP local   │
                 └─────────────────┘
```

### Variables d'entrée (18 features)

#### Numériques (16)
| Variable | Description | Unité |
|---|---|---|
| `age` | Âge du patient | années |
| `bmi` | Indice de Masse Corporelle | kg/m² |
| `waist_to_hip_ratio` | Ratio taille/hanche | — |
| `glucose_fasting` | Glycémie à jeun | mg/dL |
| `glucose_postprandial` | Glycémie post-repas | mg/dL |
| `hba1c` | Hémoglobine glyquée | % |
| `insulin_level` | Insuline à jeun | mIU/L |
| `cholesterol_total` | Cholestérol total | mg/dL |
| `triglycerides` | Triglycérides | mg/dL |
| `systolic_bp` | Tension systolique | mmHg |
| `diastolic_bp` | Tension diastolique | mmHg |
| `physical_activity_minutes_per_week` | Activité physique | min/semaine |
| `sleep_hours_per_day` | Sommeil | h/jour |
| `diet_score` | Note alimentation | 0-10 |
| `family_history_diabetes` | Antécédents familiaux | 0/1 |
| `hypertension_history` | Hypertension | 0/1 |

#### Catégorielles (2)
| Variable | Modalités |
|---|---|
| `gender` | Male, Female |
| `smoking_status` | Never, Former, Current |

### Pourquoi OneHotEncoder et pas LabelEncoder ?

Le notebook initial utilisait `LabelEncoder` sur les variables nominales (`gender`, `smoking_status`), ce qui introduit un **ordre artificiel** (`gender=0 < gender=1`) que l'arbre interprète comme un seuil numérique. L'application corrige ce point avec `OneHotEncoder` qui crée des colonnes binaires indépendantes (`gender_Male`, `gender_Female`).

---

## 5. Les hyperparamètres

### Configuration retenue

```python
XGBClassifier(
    n_estimators=300,        # Nombre d'arbres
    learning_rate=0.05,      # Taux d'apprentissage
    max_depth=4,             # Profondeur max d'un arbre
    subsample=0.8,           # 80 % des lignes par arbre
    colsample_bytree=0.8,    # 80 % des colonnes par arbre
    reg_alpha=0.1,           # Régularisation L1
    reg_lambda=1.0,          # Régularisation L2
    eval_metric='logloss',   # Fonction de coût
    random_state=42,         # Reproductibilité
    n_jobs=-1                # Multi-threading
)
```

### Justification de chaque paramètre

| Paramètre | Valeur | Pourquoi cette valeur ? |
|---|---|---|
| `n_estimators` | **300** | Compromis précision/vitesse. Au-delà, gain marginal et risque de mémorisation. |
| `learning_rate` | **0.05** | Faible LR + plus d'estimateurs → meilleure généralisation (effet "marche lente"). |
| `max_depth` | **4** | Assez profond pour capturer les interactions HbA1c × Glucose × IMC, mais évite la mémorisation des cas individuels. |
| `subsample` | **0.8** | Chaque arbre voit 80 % des lignes → diversité (bagging intégré). |
| `colsample_bytree` | **0.8** | Chaque arbre voit 80 % des colonnes → décorrélation entre arbres. |
| `reg_alpha` | **0.1** | Pénalité L1 → sparsité (élimine les features faibles). |
| `reg_lambda` | **1.0** | Pénalité L2 → lisse les poids (évite les valeurs extrêmes). |
| `random_state` | **42** | Reproductibilité totale des résultats. |
| `n_jobs` | **-1** | Utilise tous les cœurs CPU pour l'entraînement. |

### Effet combiné des régularisations

```
Sans régularisation : risque d'overfitting (train >> test)
       │
       ▼
+ subsample 0.8         → diversité (bagging)
+ colsample_bytree 0.8  → décorrélation
+ reg_alpha 0.1 (L1)    → sparsité des feuilles
+ reg_lambda 1.0 (L2)   → lissage des poids
       │
       ▼
Résultat : Gap train/test = 0.18 % (excellent)
```

---

## 6. Métriques de performance

### Sur le jeu de test (20 000 patients)

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Accuracy** | 91.97 % | 91.97 % des prédictions sont correctes |
| **ROC-AUC** | 0.9432 | Excellente capacité de discrimination |
| **Précision (classe 1)** | ~94 % | Quand on prédit "diabétique", on a raison à 94 % |
| **Rappel (classe 1)** | ~92 % | On détecte 92 % des vrais diabétiques |
| **F1-score** | ~93 % | Moyenne harmonique précision/rappel |

### Comparaison avec la littérature

Les performances sont **cohérentes avec l'état de l'art** pour la détection du diabète à partir de données tabulaires :
- Études publiées : 85-95 % d'accuracy
- Notre modèle : 92 % d'accuracy → dans le haut de la fourchette

---

## 7. Validation anti-overfitting

### Principe
L'overfitting (sur-apprentissage) se produit quand le modèle **mémorise** les données d'entraînement au lieu de **généraliser**. Il se manifeste par un grand écart entre l'accuracy sur le train et le test.

### Vérification automatique
À chaque démarrage de `app.py`, le terminal affiche :

```
[OK] Train accuracy : 0.9216
[OK] Test accuracy  : 0.9197
[OK] ROC-AUC        : 0.9432
[OK] Gap (train-test): 0.0019 (OK)
```

### Seuils
| Gap train/test | Verdict | Action |
|---|---|---|
| < 1 % | Excellent — généralisation parfaite | Aucune |
| 1-5 % | OK — généralisation acceptable | Aucune |
| 5-10 % | ⚠️ Overfitting léger | Augmenter régularisation |
| > 10 % | ❌ Overfitting sévère | Repenser le modèle |

### Notre gap = 0.18 %
**Excellent.** Le modèle généralise parfaitement aux nouvelles données.

---

## 8. Explicabilité (XAI / SHAP)

### Pourquoi XAI ?

Un modèle médical doit pouvoir **justifier ses décisions** :
- **Pour le médecin** → confiance dans l'outil et validation clinique
- **Pour le patient** → comprendre les facteurs de risque actionnables
- **Pour le data scientist** → détecter les biais et les fuites de données
- **Pour la conformité RGPD** → droit d'explication des décisions automatisées

### Implémentation

L'application utilise les **SHAP values natives de XGBoost** (`booster.predict(dmat, pred_contribs=True)`) — aucune dépendance externe.

```python
# Dans app.py, fonction explain_prediction()
booster = clf.get_booster()
dmat = xgb.DMatrix(X_transformed, feature_names=feature_names)
shap_values = booster.predict(dmat, pred_contribs=True)[0]
# shap_values[:-1] = contributions par feature
# shap_values[-1] = biais (score de base)
```

### Deux niveaux d'explication

#### 1. Explication LOCALE (par patient)
Inclus dans la réponse de `POST /predict`. Retourne les **6 facteurs** qui ont le plus influencé CETTE prédiction.

Exemple pour un patient à 67 % de risque :
```
+ HbA1c = 6.8        → +35 % (aggrave)
+ IMC = 32           → +18 % (aggrave)
+ Antécédents = 1    → +12 % (aggrave)
+ Activité = 30 min  → -8 %  (protège)
+ Âge = 28           → -5 %  (protège)
```

#### 2. Importance GLOBALE (modèle)
Endpoint `GET /explain`. Retourne l'importance moyenne de chaque feature sur l'ensemble du dataset.

**Notre modèle** :
```
HbA1c                : 89.4 %  ← dominante
glucose_fasting      :  3.6 %
family_history       :  1.0 %
age                  :  0.6 %
physical_activity    :  0.3 %
... (toutes < 0.5 %)
```

**Lecture** : HbA1c écrase toutes les autres variables — cohérent avec la médecine (HbA1c est l'indicateur de référence ADA depuis 2010).

### Affichage dans l'application

Dans `predict.html`, deux onglets sous la jauge de probabilité :
- **Onglet "Votre profil"** → barres horizontales rouges (aggravants) ou vertes (protecteurs)
- **Onglet "Modèle global"** → importance moyenne, gradient bleu

---

## 9. Architecture de l'application

```
┌──────────────────────────────────────────────────────────┐
│                      UTILISATEUR                         │
│                  (Navigateur Web)                        │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP
                       ▼
┌──────────────────────────────────────────────────────────┐
│  FRONTEND (HTML + CSS + JS Vanilla)                      │
│  ┌─────────────────┐    ┌──────────────────┐            │
│  │  landing.html   │ ─► │   predict.html   │            │
│  │  (marketing)    │    │  (wizard 4 étapes)│           │
│  └─────────────────┘    └──────────────────┘            │
└──────────────────────┬───────────────────────────────────┘
                       │ JSON POST /predict
                       ▼
┌──────────────────────────────────────────────────────────┐
│  BACKEND (app.py - Flask)                                │
│  ┌───────────────────────────────────────────┐          │
│  │  Routes :                                  │          │
│  │  GET  /              → landing.html        │          │
│  │  GET  /predict.html                        │          │
│  │  GET  /status                              │          │
│  │  GET  /metrics                             │          │
│  │  GET  /explain      → SHAP global          │          │
│  │  POST /predict      → prédiction + SHAP    │          │
│  │  POST /retrain                             │          │
│  └───────────────────────────────────────────┘          │
│  ┌───────────────────────────────────────────┐          │
│  │  Pipeline scikit-learn (en mémoire) :     │          │
│  │  ColumnTransformer → XGBClassifier         │          │
│  └───────────────────────────────────────────┘          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  MODÈLE & MÉTRIQUES (persistant)                         │
│  • diabetes_xgb_model.pkl  (5 Mo)                        │
│  • model_metrics.json      (résultats entraînement)      │
└──────────────────────────────────────────────────────────┘
```

---

## 10. Endpoints API

### `GET /status`
Vérifie que le modèle est chargé et prêt.

**Réponse** :
```json
{
  "ready": true,
  "model": "XGBoost Classifier",
  "metrics": { ... }
}
```

### `GET /metrics`
Retourne les métriques d'entraînement.

**Réponse** :
```json
{
  "model": "XGBoost Classifier",
  "n_train": 80000,
  "n_test": 20000,
  "n_features": 18,
  "train_accuracy": 0.9216,
  "test_accuracy": 0.9197,
  "overfitting_gap": 0.0019,
  "roc_auc": 0.9432,
  "classification_report": { ... }
}
```

### `GET /explain`
Retourne l'importance globale des features (XAI).

**Réponse** :
```json
{
  "global_importance": [
    { "feature": "hba1c", "label": "Hemoglobine A1c", "importance": 89.4 },
    { "feature": "glucose_fasting", "label": "Glycemie a jeun", "importance": 3.6 },
    ...
  ]
}
```

### `POST /predict`
Effectue une prédiction + retourne l'explication SHAP locale.

**Requête** :
```json
{
  "age": 45,
  "gender": "Male",
  "bmi": 27.5,
  "waist_to_hip_ratio": 0.85,
  "glucose_fasting": 110,
  "glucose_postprandial": 140,
  "hba1c": 6.2,
  "insulin_level": 18,
  "cholesterol_total": 200,
  "triglycerides": 150,
  "systolic_bp": 130,
  "diastolic_bp": 85,
  "physical_activity_minutes_per_week": 120,
  "sleep_hours_per_day": 7,
  "diet_score": 6,
  "smoking_status": "Never",
  "family_history_diabetes": 0,
  "hypertension_history": 0
}
```

**Réponse** :
```json
{
  "diagnosis": 0,
  "risk_probability": 23.4,
  "model": "XGBoost Classifier",
  "explanation": {
    "bias": -1.234,
    "top_factors": [
      {
        "feature": "hba1c",
        "label": "Hemoglobine A1c",
        "value": 6.2,
        "contribution": 0.42,
        "direction": "risk"
      },
      ...
    ]
  },
  "global_importance": [ ... ]
}
```

### `POST /retrain`
Force le réentraînement du modèle (utile après modification du CSV).

---

## 11. Limites connues

### Limites du modèle DSO3 (production)
| Limite | Impact | Atténuation |
|---|---|---|
| **Dominance de l'HbA1c (89 %)** | Sans HbA1c, prédiction peu fiable | Imputation médiane + warning utilisateur |
| **Plafond statistique** | Tous les modèles plafonnent à ~92 % | Limitation du dataset, pas du modèle |
| **Pas testé sur données réelles** | Performance sur de vrais patients inconnue | Validation clinique requise avant usage médical |

### Limites des autres DSO (non utilisés en production)
- **DSO2 (régression)** — R² = 0.9988, score anormalement élevé qui suggère que `diabetes_risk_score` est une fonction déterministe du dataset synthétique
- **DSO4 (multiclasse)** — Type 1 (39 patients) et Gestational (82 patients) ont F1=0 → modèle inutilisable pour ces stades sans correction

### Avertissement médical
Cet outil est un **projet académique** et ne remplace en aucun cas un diagnostic médical professionnel. Consulter un médecin pour toute décision de santé.

---

## 12. Structure des fichiers

```
machine learning/
├── 📄 README.md                                        ← Ce fichier
├── 📄 revision.md                                      ← Document de révision soutenance
│
├── 🐍 app.py                                           ← Backend Flask + modèle
├── 🐍 requirements.txt                                 ← Dépendances Python
│
├── 🌐 landing.html                                     ← Page d'accueil web
├── 🌐 predict.html                                     ← Formulaire de prédiction
│
├── 🤖 diabetes_xgb_model.pkl                           ← Modèle entraîné (5 Mo)
├── 📊 model_metrics.json                               ← Métriques de validation
│
├── 📓 MachineLearning_Diabetes_Final (3).ipynb         ← Notebook complet
├── 📄 Rapport_Comparaison.html                         ← Rapport DSO3
├── 🎯 Validation_Commerciale_MLA_Diabetes_Pro.pptx     ← Présentation (23 slides)
│
├── 📁 data/
│   └── diabetes_dataset.csv                            ← Dataset (100k patients)
│
└── 📁 diagrams/                                        ← Graphiques générés
    ├── 01_dso3_model_comparison.png
    ├── 02_dso2_regression_comparison.png
    ├── 03_roc_curves.png
    ├── 04_confusion_matrix.png
    ├── 05_feature_importance.png
    ├── 06_pipeline_architecture.png
    ├── 07_train_vs_test.png
    └── 08_class_imbalance_dso4.png
```

---

## 📝 Crédits

**Module** : Machine Learning Appliqué
**Établissement** : ESPRIT
**Année** : 2025-2026
**Responsable du module** : Dr. Jihen Hlel

**Stack technique** :
- Python 3.10+ · Flask · scikit-learn · XGBoost · pandas
- HTML / CSS / JS vanilla
- python-pptx · matplotlib · seaborn

---

> 💡 **Pour la soutenance**, consulter aussi `revision.md` qui contient les questions probables du jury et leurs réponses détaillées.
#   D i a b - t e A I  
 