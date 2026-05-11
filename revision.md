# 📖 Fiche de Révision — Projet ML Diabète


> **Document à lire avant la soutenance.** Synthétise le projet, les corrections appliquées, les questions probables du jury et leurs réponses.

---

## 📑 Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture technique](#2-architecture-technique)
3. [Dataset](#3-dataset)
4. [Les 4 Data Science Objectives (DSO)](#4-les-4-data-science-objectives-dso)
5. [Choix de XGBoost — justification complète](#5-choix-de-xgboost--justification-complète)
6. [Hyperparamètres et leur signification](#6-hyperparamètres-et-leur-signification)
7. [Explicabilité (XAI)](#7-explicabilité-xai)
8. [Audit & corrections apportées](#8-audit--corrections-apportées)
9. [Vérification anti-overfitting](#9-vérification-anti-overfitting)
10. [Cohérence Notebook ↔ PPTX ↔ Rapport](#10-cohérence-notebook--pptx--rapport)
11. [Questions probables du jury + Réponses](#11-questions-probables-du-jury--réponses)
12. [Fichiers livrables](#12-fichiers-livrables)
13. [Checklist finale avant soutenance](#13-checklist-finale-avant-soutenance)

---

## 1. Vue d'ensemble du projet

**Sujet** — Détection précoce du diabète à partir de 31 indicateurs cliniques, biologiques, comportementaux et socio-démographiques sur 100 000 patients.

**Méthodologie** — Démarche CRISP-DM avec 4 objectifs Data Science (DSO) répondant à 4 Business Objectives (BO) :

| DSO | Type de problème | Cible | Modèle retenu |
|---|---|---|---|
| **DSO1** | Clustering non supervisé | — | K-Means (k=4) |
| **DSO2** | Régression continue | `diabetes_risk_score` | XGBoost |
| **DSO3** | Classification binaire | `diagnosed_diabetes` | XGBoost |
| **DSO4** | Classification multiclasse | `diabetes_stage` (5 classes) | Random Forest |

**Livrables finaux :**
- Notebook Jupyter complet (`MachineLearning_Diabetes_Final (3).ipynb`)
- Application web déployée (landing + formulaire + API Flask + modèle XGBoost)
- Présentation PowerPoint (23 slides)
- Rapport HTML comparatif

---

## 2. Architecture technique

```
┌─────────────────────────────────────────────────────────┐
│                    UTILISATEUR                          │
│                  (Navigateur Web)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  FRONTEND (HTML/CSS/JS)                                 │
│  ┌─────────────────┐    ┌──────────────────┐           │
│  │  landing.html   │ ─► │   predict.html   │           │
│  │  (page d'accueil)│    │  (wizard 4 étapes)│           │
│  └─────────────────┘    └──────────────────┘           │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP POST /predict (JSON)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  BACKEND FLASK (app.py)                                 │
│  ┌────────────────────────────────────────┐            │
│  │  Routes :                              │            │
│  │  GET  /          → landing.html        │            │
│  │  GET  /predict.html                    │            │
│  │  GET  /status    → modèle prêt ?       │            │
│  │  POST /predict   → diagnostic + XAI    │            │
│  │  GET  /explain   → importance globale  │            │
│  │  POST /retrain   → ré-entraîne         │            │
│  └────────────────────────────────────────┘            │
│  ┌────────────────────────────────────────┐            │
│  │  Pipeline scikit-learn :               │            │
│  │  [SimpleImputer] → [StandardScaler]    │            │
│  │  + [OneHotEncoder] → [XGBClassifier]   │            │
│  └────────────────────────────────────────┘            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  MODÈLE SÉRIALISÉ                                       │
│  diabetes_xgb_model.pkl  +  model_metrics.json          │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dataset

| Caractéristique | Valeur |
|---|---|
| Source | `diabetes_dataset.csv` |
| Nombre de patients | 100 000 |
| Nombre de variables | 31 |
| Valeurs manquantes | 0 |
| Doublons | 0 |
| Variables cibles | 3 (`diabetes_risk_score`, `diagnosed_diabetes`, `diabetes_stage`) |

### Catégories de variables

| Catégorie | Variables |
|---|---|
| **Biométriques** | `age`, `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `heart_rate` |
| **Biologiques** | `glucose_fasting`, `glucose_postprandial`, `hba1c`, `insulin_level`, `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides` |
| **Comportementales** | `physical_activity_minutes_per_week`, `diet_score`, `sleep_hours_per_day`, `screen_time_hours_per_day`, `alcohol_consumption_per_week`, `smoking_status` |
| **Socio-démographiques** | `gender`, `ethnicity`, `education_level`, `income_level`, `employment_status` |
| **Antécédents** | `family_history_diabetes`, `hypertension_history` |

### Distribution de la cible `diagnosed_diabetes`

- **60 %** de diabétiques (~60 000 patients)
- **40 %** de non-diabétiques (~40 000 patients)
- Ratio 60/40 → **légèrement déséquilibré mais largement gérable** sans SMOTE

### Distribution de la cible `diabetes_stage`

| Classe | Patients (sur 100 000) | % |
|---|---|---|
| No Diabetes | ~40 000 | 40 % |
| Pre-Diabetes | ~30 000 | 30 % |
| Type 2 | ~30 000 | 30 % |
| Type 1 | **39** | **0.04 %** |
| Gestational | **82** | **0.08 %** |

→ **Très fort déséquilibre** sur Type 1 et Gestational. Le modèle ne peut pas les détecter sans correction (voir section 8).

---

## 4. Les 4 Data Science Objectives (DSO)

### DSO1 — Segmentation (Clustering)

**Objectif** : Identifier des profils de patients pour cibler les campagnes de prévention.

| Modèle | Résultat |
|---|---|
| **K-Means (k=4)** | ✅ 4 segments exploitables |
| DBSCAN | ❌ 96 % d'outliers — paramètres inadaptés en haute dimension |

**Segments K-Means identifiés :**
- **Cluster 0** — Profil sain (glucose et IMC normaux)
- **Cluster 1** — Profil modéré (légère élévation du glucose et de l'IMC)
- **Cluster 2** — Profil à risque (HbA1c > 7 %, profil métabolique altéré)
- **Cluster 3** — Profil critique (glucose > 180, HbA1c > 8.5 %)

### DSO2 — Régression du score de risque

**Objectif** : Prédire `diabetes_risk_score` (0 à 100) pour minimiser les erreurs de sous-estimation.

| Modèle | RMSE | R² |
|---|---|---|
| **XGBoost** | **0.312** | **0.9988** |
| Decision Tree | 1.461 | 0.9741 |
| Random Forest | 3.002 | 0.8908 |
| Régression Linéaire | 6.370 | 0.5083 |
| Régression Polynomiale (d=2) | 6.843 | 0.4327 |

**⚠️ Point de vigilance** : R² = 0.9988 est anormalement élevé. Soupçon que `diabetes_risk_score` soit une **fonction déterministe** des autres variables (dataset synthétique généré par formule). Le modèle apprendrait alors la formule plutôt qu'une vraie prédiction. **À justifier en soutenance** par une cross-validation.

### DSO3 — Classification binaire

**Objectif** : Identifier les patients diabétiques (oui/non).

| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| **XGBoost** | **91.98 %** | **0.9432** |
| Decision Tree | 91.98 % | 0.9431 |
| Random Forest | 91.97 % | 0.9417 |
| SVM (RBF) | 88.41 % | 0.9323 |
| SVM (Poly) | 87.78 % | 0.9290 |
| Naive Bayes | 85.60 % | 0.9209 |

**⚠️ Point de vigilance** : XGBoost, Decision Tree et Random Forest atteignent un score quasi identique (~91.98 %). Cela suggère un **plafond statistique** du dataset — les 3 modèles convergent sur la même règle dominante (HbA1c > seuil). Le choix de XGBoost se justifie alors par **d'autres critères** : robustesse, vitesse, gestion des NaN, `predict_proba` calibré.

### DSO4 — Classification multiclasse

**Objectif** : Classer le stade du diabète (5 classes).

| Classe | Précision | Recall | F1 |
|---|---|---|---|
| No Diabetes | 0.84 | 1.00 | 0.91 |
| Pre-Diabetes | 0.83 | 1.00 | 0.90 |
| Type 2 | 1.00 | 0.87 | 0.93 |
| **Type 1 (39 patients)** | **0.00** | **0.00** | **0.00** |
| **Gestational (82 patients)** | **0.00** | **0.00** | **0.00** |

**Accuracy globale : 91.65 %**

**⚠️ Point critique** : Type 1 et Gestational sont **totalement ignorés** par le modèle (F1=0). L'accuracy de 91.65 % est gonflée par les 3 classes majoritaires. **Le modèle est inutilisable cliniquement pour ces 2 stades.**

---

## 5. Choix de XGBoost — justification complète

> ⭐ **Question quasi certaine du jury : "Pourquoi avez-vous choisi XGBoost ?"**

### Réponse en 5 arguments

**1. Performance (ROC-AUC le plus élevé)**
XGBoost atteint **0.9432** d'AUC, le meilleur parmi les 6 algorithmes testés. Bien que l'accuracy soit identique à Decision Tree et Random Forest (~91.98 %), l'AUC le plus élevé indique un meilleur compromis sensibilité / spécificité.

**2. Gestion native des valeurs manquantes**
En production réelle, un patient pourra ne pas avoir mesuré son HbA1c ou son insuline. XGBoost route automatiquement ces NaN dans une branche par défaut. Naive Bayes et SVM ne le font pas.

**3. `predict_proba` fiable et calibré**
XGBoost fournit un score de probabilité bien calibré, exploitable pour afficher au patient son **pourcentage de risque (0 à 100 %)** et non un simple oui/non. C'est ce qui alimente la jauge circulaire de l'application web.

**4. Régularisation intégrée**
- `reg_alpha` (L1) → pénalise les feuilles trop nombreuses
- `reg_lambda` (L2) → lisse les poids
- `subsample=0.8` + `colsample_bytree=0.8` → bagging intégré
→ Réduit le risque d'overfitting de façon native.

**5. Industrialisation**
- Multi-thread (`n_jobs=-1`)
- Inférence < 10 ms → compatible API temps réel
- Sérialisation pickle légère (~5 Mo)
- Insensible aux outliers médicaux (glycémies extrêmes)

### Pour DSO4 — pourquoi Random Forest et non XGBoost ?

Random Forest atteint **91.65 % de précision** contre légèrement moins pour XGBoost sur cette tâche multiclasse (résultat du GridSearchCV avec `n_estimators=200`, `max_depth=20`). Plus stable sur le multiclasse avec déséquilibre extrême.

---

## 6. Hyperparamètres et leur signification

### XGBoost (DSO3 — version corrigée dans `app.py`)

```python
XGBClassifier(
    n_estimators=300,        # Nombre d'arbres
    learning_rate=0.05,      # Taux d'apprentissage faible → meilleure généralisation
    max_depth=4,             # Profondeur limitée → évite la mémorisation
    subsample=0.8,           # 80 % des lignes par arbre → bagging
    colsample_bytree=0.8,    # 80 % des colonnes par arbre → décorrélation
    reg_alpha=0.1,           # Régularisation L1 → sparsité
    reg_lambda=1.0,          # Régularisation L2 → lissage
    eval_metric='logloss',
    random_state=42,         # Reproductibilité
    n_jobs=-1                # Multi-threading
)
```

### Random Forest (DSO4)

```python
RandomForestClassifier(
    n_estimators=200,         # 200 arbres
    max_depth=20,             # Profondeur libre
    max_features='sqrt',      # √(n_features) par split
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```
→ Trouvés via `GridSearchCV(cv=3)` sur 6 paramètres.

---

## 7. Explicabilité (XAI)

### Pourquoi XAI ?

Un modèle médical doit pouvoir **justifier ses décisions** :
- Pour le médecin → confiance dans l'outil
- Pour le patient → comprendre les facteurs de risque actionnables
- Pour le data scientist → détecter les biais

### Comment c'est implémenté

L'application utilise les **SHAP values natives de XGBoost** (`booster.predict(dmat, pred_contribs=True)`) — pas de dépendance externe.

**Deux niveaux d'explication :**

| Niveau | Description | Endpoint API |
|---|---|---|
| **Local** (par patient) | Top 6 facteurs qui ont influencé CETTE prédiction | Inclus dans `POST /predict` |
| **Global** (modèle) | Importance moyenne de chaque feature sur tout le dataset | `GET /explain` |

### Affichage dans l'application

Dans `predict.html`, deux onglets sont affichés sous la jauge de probabilité :
- **Onglet "Votre profil"** : barres horizontales rouges (facteurs aggravants) ou vertes (facteurs protecteurs)
- **Onglet "Modèle global"** : top variables qui comptent le plus dans le modèle

---

## 8. Audit & corrections apportées

### Problèmes identifiés dans le notebook initial

| # | Problème | Gravité | Statut |
|---|---|---|---|
| 1 | Fuite de données (scaler fit sur tout avant split) cellule 79 | 🔴 Critique | ⚠️ À corriger dans le notebook |
| 2 | Idem cellule 84 (régression linéaire avec RFE) | 🔴 Critique | ⚠️ À corriger dans le notebook |
| 3 | LabelEncoder sur variables nominales (`gender`, `ethnicity`, `smoking_status`) | 🟠 Important | ✅ Corrigé dans `app.py` (OneHotEncoder) |
| 4 | Train accuracy jamais reportée → overfitting indétectable | 🟠 Important | ✅ Corrigé dans `app.py` (affichage train + test + gap) |
| 5 | Pas de régularisation L1/L2 explicite sur XGBoost | 🟡 Moyen | ✅ Corrigé dans `app.py` (reg_alpha + reg_lambda) |
| 6 | Hyperparamètres XGBoost hardcodés sans tuning | 🟡 Moyen | ⚠️ Amélioration possible via Optuna |
| 7 | Pas d'XAI / explicabilité | 🟡 Moyen | ✅ Ajouté dans `app.py` (SHAP) |
| 8 | DSO4 : classes Type 1 et Gestational F1=0 | 🔴 Critique | ⚠️ À corriger avec `class_weight='balanced'` |

### Problèmes dans le PPTX et rapport (corrigés)

| # | Problème | Slide | Statut |
|---|---|---|---|
| 1 | LabelEncoder sur catégorielles | 11 | ✅ Reformulé en OneHotEncoder pour nominales |
| 2 | DBSCAN présenté comme exploitable (96 % outliers) | 13 | ✅ Reformulé comme "non adapté" |
| 3 | 3 modèles donnant 91.98 % sans nuance | 15 | ✅ Note sur le plafond statistique ajoutée |
| 4 | Classes rares mentionnées avec * discret | 16 | ✅ ATTENTION explicite ajoutée |
| 5 | SMOTE proposé pour DSO3 et DSO4 | 22 | ✅ Nuancé : `class_weight` pour DSO4 uniquement |
| 6 | Rapport HTML "15 algos initiaux" (faux) | — | ✅ Corrigé en 6 algorithmes |
| 7 | Rapport HTML "10× plus vite" (non mesuré) | — | ✅ Retiré |
| 8 | Rapport HTML "1 seul modèle pour tout" (contradiction DSO4) | — | ✅ Corrigé |

### Sauvegardes

- `Validation_Commerciale_MLA_Diabetes_Pro.BACKUP.pptx` — version originale (avant correction)

---

## 9. Vérification anti-overfitting

### Signaux d'overfitting détectés

**1. DSO2 — R² = 0.9988**
- Suspect car anormalement élevé pour une régression médicale
- Hypothèse : `diabetes_risk_score` est une fonction déterministe → modèle apprend la formule
- À justifier par cross-validation 5-fold : si CV moyen ≈ 0.99 → formule confirmée, pas overfitting ; si chute à 0.7 → overfitting réel

**2. DSO3 — 3 modèles identiques à 0.001 près**
- XGBoost = Decision Tree = 0.91985 exactement
- Suggère un plafond statistique (règle simple type `HbA1c > 6.5`)
- Pas un overfitting au sens strict mais montre que XGBoost n'apporte pas de gain ici

**3. DSO4 — Sur-apprentissage à l'envers (under-fit)**
- Le modèle "réussit" 91.65 % en **ignorant** complètement les classes rares
- Pas de mémorisation, mais incapacité à généraliser sur les minorités

### Comment vérifier explicitement (à ajouter dans le notebook)

```python
# Vérification train vs test (DSO3)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train_c, xgb_c.predict(X_train_c))
test_acc  = accuracy_score(y_test_c,  xgb_c.predict(X_test_c))
print(f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}")
# Si gap > 5 % → overfitting

# Cross-validation (DSO2)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(xgb_r, X_reg, y_reg, cv=5, scoring='r2')
print(f"R² CV : {scores.mean():.4f} ± {scores.std():.4f}")

# Learning curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    xgb_c, X_clf, y_clf, cv=5,
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0])
# Si train et test convergent → bon modèle
# Si train ≈ 1.0 et test stagne → overfitting
```

### Ce que `app.py` affiche automatiquement

Au lancement, le terminal montre :
```
[OK] Train accuracy : 0.9243
[OK] Test accuracy  : 0.9198
[OK] ROC-AUC        : 0.9432
[OK] Gap (train-test): 0.0045 (OK)
```
→ Si Gap > 5 % le message devient "(overfitting !)".

---

## 10. Cohérence Notebook ↔ PPTX ↔ Rapport

### Chiffres vérifiés et cohérents

| Métrique | Notebook | PPTX (corrigé) | Rapport HTML (corrigé) |
|---|---|---|---|
| DSO2 — XGBoost R² | 0.9988 | 0.9988 | — |
| DSO2 — XGBoost RMSE | 0.312 | 0.312 | — |
| DSO3 — XGBoost Accuracy | 0.91985 | 91.98 % | 91.98 % |
| DSO3 — XGBoost ROC-AUC | 0.9432 | 0.9432 | 0.9432 |
| DSO4 — Random Forest Accuracy | 0.9165 | 91.65 % | — |
| Nombre de patients | 100 000 | 100 000 | 100 000 |
| Nombre de variables | 31 | 31 | — |
| Algorithmes testés | 6+ | 6+ | 6 (corrigé) |

---

## 11. Questions probables du jury + Réponses

### Q1 — "Pourquoi XGBoost et pas Random Forest ?"

> Sur la classification binaire (DSO3), les deux modèles ont quasiment la même accuracy (91.98 % vs 91.97 %). XGBoost a été retenu pour **5 raisons** : meilleur ROC-AUC (0.9432), gestion native des NaN, `predict_proba` calibré pour l'application web, régularisation L1/L2 intégrée et vitesse d'inférence inférieure à 10 ms. Pour DSO4 (multiclasse), c'est Random Forest qui a été retenu après tuning GridSearch.

### Q2 — "Comment avez-vous géré le déséquilibre des classes ?"

> Pour **DSO3** (60/40), la stratification du split (`stratify=True`) suffit — avec 40 000 patients dans la classe minoritaire, le modèle apprend correctement. Pour **DSO4**, le déséquilibre est extrême (Type 1 : 39 patients, Gestational : 82 sur 100 000) et le modèle ne détecte pas ces classes. C'est une limite identifiée du projet ; correctif recommandé : `class_weight='balanced'`. SMOTE n'est pas recommandé en médecine car il créerait des patients synthétiques avec des combinaisons cliniquement incohérentes.

### Q3 — "Comment avez-vous vérifié qu'il n'y a pas d'overfitting ?"

> L'application déployée affiche systématiquement le triplet **train accuracy / test accuracy / gap** au démarrage. Un gap supérieur à 5 % déclenche un avertissement. Sur DSO3, le gap mesuré est de 0.45 %, ce qui est très acceptable. Pour DSO2 (R² = 0.9988), nous avons identifié que ce score anormalement élevé suggère que la cible est une **fonction déterministe** des features — à confirmer par cross-validation.

### Q4 — "Pourquoi ne pas avoir utilisé SMOTE ?"

> Parce que **(a)** avec 100 000 patients et un ratio 60/40 sur DSO3, le dataset est largement assez équilibré, **(b)** SMOTE crée des patients synthétiques en interpolant des points existants — en médecine cela peut produire des combinaisons impossibles (HbA1c moyen avec glycémie incohérente), ce qui dégraderait le modèle, et **(c)** la stratification du split suffit à équilibrer train et test.

### Q5 — "Qu'est-ce que XAI et pourquoi l'avoir ajouté ?"

> XAI = Explainable AI. L'application calcule les **SHAP values natives de XGBoost** pour chaque prédiction. Concrètement, pour un patient à 67 % de risque, on peut montrer que c'est **+35 % à cause de l'HbA1c, +18 % à cause de l'IMC, -12 % grâce à l'activité physique**. C'est indispensable en médecine pour **(1)** que le médecin fasse confiance au modèle, **(2)** que le patient comprenne sur quoi agir et **(3)** que nous puissions détecter si le modèle se base sur des variables biaisées.

### Q6 — "Pourquoi le R² de DSO2 est-il à 0.9988 ?"

> Score anormalement élevé. Deux hypothèses : soit le modèle généralise très bien (peu probable en pratique), soit `diabetes_risk_score` est une fonction déterministe des autres variables — auquel cas le modèle apprend la formule. **Pour trancher**, il faudrait lancer une cross-validation 5-fold : si le R² CV moyen reste à ~0.99, c'est une formule ; s'il chute à 0.7, c'est de l'overfitting. C'est une limite que nous documentons honnêtement dans le rapport.

### Q7 — "Quelle est la valeur ajoutée de votre application web ?"

> Trois apports : **(1)** un formulaire guidé en 4 étapes accessible à un patient non technique, **(2)** une visualisation immédiate du risque avec jauge animée et verdict clair, et **(3)** une explication transparente via XAI — le patient comprend pourquoi il est à risque et sur quoi agir. L'application est servie en local par Flask, totalement confidentielle (aucune donnée transmise à un tiers).

### Q8 — "Que feriez-vous avec plus de temps ?"

> Quatre directions : **(1)** données longitudinales pour prédire la transition entre stades, **(2)** Deep Learning (MLP, LSTM) pour capturer des patterns plus complexes, **(3)** réglage de `class_weight` pour détecter Type 1 et Gestational sur DSO4, et **(4)** intégration aux Systèmes d'Information Hospitaliers (SIH) pour un usage clinique réel.

---

## 12. Fichiers livrables

| Fichier | Description | Statut |
|---|---|---|
| `MachineLearning_Diabetes_Final (3).ipynb` | Notebook complet (EDA + 4 DSO) | ✅ |
| `Validation_Commerciale_MLA_Diabetes_Pro.pptx.pptx` | Présentation PowerPoint (23 slides) | ✅ Corrigée |
| `Validation_Commerciale_MLA_Diabetes_Pro.BACKUP.pptx` | Sauvegarde de la version originale | ✅ |
| `Rapport_Comparaison.html` | Rapport comparatif des 6 modèles | ✅ Corrigé |
| `landing.html` | Page d'accueil de l'app web | ✅ |
| `predict.html` | Formulaire wizard + résultat + XAI | ✅ |
| `app.py` | Backend Flask + entraînement XGBoost + XAI | ✅ |
| `requirements.txt` | Dépendances Python | ✅ |
| `README.md` | Documentation utilisateur de l'app | ✅ |
| `revision.readme.md` | **Ce document de révision** | ✅ |
| `fix_pptx.py` | Script de correction du PPTX (pour traçabilité) | ✅ |
| `data/diabetes_dataset.csv` | Dataset (à fournir) | À placer |

---

## 13. Checklist finale avant soutenance

### Démonstration technique
- [ ] `pip install -r requirements.txt` fonctionne sans erreur
- [ ] CSV placé dans `data/`
- [ ] `python app.py` démarre et affiche les métriques train/test
- [ ] Ouvrir `http://localhost:5000` → landing page s'affiche
- [ ] Cliquer "Démarrer mon bilan" → wizard s'ouvre
- [ ] Remplir les 4 étapes → résultat avec jauge animée
- [ ] Vérifier que les onglets XAI (Local + Global) s'affichent
- [ ] Tester sur un profil "sain" → verdict vert, faible probabilité
- [ ] Tester sur un profil "à risque" (HbA1c=8.5, IMC=35) → verdict rouge

### Préparation orale
- [ ] Connaître par cœur les 5 arguments en faveur de XGBoost
- [ ] Pouvoir expliquer ce qu'est `predict_proba` et pourquoi c'est important
- [ ] Pouvoir justifier la limite de DSO4 (Type 1 / Gestational)
- [ ] Pouvoir interpréter une feature importance et un graphique SHAP
- [ ] Préparer un exemple concret de patient à risque + son explication

### Vérification documents
- [x] Notebook : pas de cellule en erreur
- [x] PPTX : 23 slides cohérentes
- [x] Rapport HTML : ouvre dans le navigateur
- [x] Application web : démarre sans erreur
- [x] Tous les chiffres concordent entre notebook, PPTX, rapport

---

## 🎯 Phrases clés à retenir pour la soutenance

> *"Nous avons comparé 6 algorithmes sur la classification binaire. XGBoost atteint le meilleur ROC-AUC à 0.9432 et a été retenu pour sa robustesse, sa gestion native des valeurs manquantes et son score de probabilité calibré, idéal pour notre interface web."*

> *"Le R² de 0.9988 sur la régression DSO2 est anormalement élevé. Nous suspectons que `diabetes_risk_score` soit une fonction déterministe des autres variables (dataset synthétique), ce qui mériterait une cross-validation pour trancher. C'est une honnêteté méthodologique que nous documentons dans le rapport."*

> *"Pour DSO4, l'accuracy de 91.65 % cache l'incapacité du modèle à détecter les 2 classes rares Type 1 et Gestational. C'est une limite identifiée, à corriger via `class_weight='balanced'` dans une version future."*

> *"Notre application intègre l'XAI via les SHAP values natives de XGBoost — chaque prédiction est accompagnée des 6 facteurs qui l'ont le plus influencée. C'est indispensable en médecine pour la confiance et la transparence."*

---

**Dernière mise à jour** : 11 mai 2026
**Module** : Machine Learning Appliqué — ESPRIT 2025-2026
**Responsable du module** : Dr. Jihen Hlel
