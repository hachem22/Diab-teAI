"""
DiabeteAI - Backend Flask
==========================
Sert la landing page, la page de prediction et l'API du modele XGBoost.

Lancement :
    pip install -r requirements.txt
    python app.py

Puis ouvrir http://localhost:5000
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier


# Libelles humains pour l'XAI (affiches dans le bilan)
FEATURE_LABELS = {
    "age": "Age",
    "bmi": "IMC",
    "waist_to_hip_ratio": "Ratio taille/hanche",
    "glucose_fasting": "Glycemie a jeun",
    "glucose_postprandial": "Glycemie post-repas",
    "hba1c": "Hemoglobine A1c",
    "insulin_level": "Insuline a jeun",
    "cholesterol_total": "Cholesterol total",
    "triglycerides": "Triglycerides",
    "systolic_bp": "Tension systolique",
    "diastolic_bp": "Tension diastolique",
    "physical_activity_minutes_per_week": "Activite physique",
    "sleep_hours_per_day": "Sommeil",
    "diet_score": "Score alimentation",
    "family_history_diabetes": "Antecedents familiaux",
    "hypertension_history": "Hypertension",
    "gender": "Sexe",
    "smoking_status": "Tabagisme",
}


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "diabetes_xgb_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# 18 colonnes du formulaire web (doivent matcher le payload JSON envoye par predict.html)
NUMERIC_FEATURES = [
    "age", "bmi", "waist_to_hip_ratio",
    "glucose_fasting", "glucose_postprandial", "hba1c", "insulin_level",
    "cholesterol_total", "triglycerides",
    "systolic_bp", "diastolic_bp",
    "physical_activity_minutes_per_week", "sleep_hours_per_day", "diet_score",
    "family_history_diabetes", "hypertension_history",
]
CATEGORICAL_FEATURES = ["gender", "smoking_status"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "diagnosed_diabetes"


# -----------------------------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
CORS(app)

# Etat global - charge au demarrage
STATE = {"model": None, "metrics": None, "feature_importance": None}


# -----------------------------------------------------------------------------
# XAI (Explainable AI) - SHAP via XGBoost natif
# -----------------------------------------------------------------------------
def _transformed_feature_names(pipeline: Pipeline) -> list:
    """Recupere les noms de colonnes apres OneHotEncoding (ex: gender_Male, smoking_status_Never...)."""
    pre = pipeline.named_steps["preprocessor"]
    num_names = NUMERIC_FEATURES
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(num_names) + cat_names


def _map_to_original_features(contribs: np.ndarray, transformed_names: list) -> dict:
    """Regroupe les contributions SHAP des colonnes OHE vers leur feature d'origine.

    Ex: gender_Male=0.05 + gender_Female=-0.01 -> gender=0.04 (somme des contributions OHE).
    """
    grouped = {feat: 0.0 for feat in ALL_FEATURES}
    for i, name in enumerate(transformed_names):
        # Identifie a quelle feature d'origine cette colonne appartient
        matched = None
        for cat in CATEGORICAL_FEATURES:
            if name.startswith(cat + "_"):
                matched = cat
                break
        if matched is None:
            matched = name  # colonne numerique - meme nom
        grouped[matched] = grouped.get(matched, 0.0) + float(contribs[i])
    return grouped


def explain_prediction(pipeline: Pipeline, X: pd.DataFrame, top_k: int = 5) -> dict:
    """Genere les contributions SHAP pour UNE prediction (XGBoost natif - pas de dependance externe).

    Retourne les top_k facteurs (en valeur absolue) avec leur impact :
    - contribution positive (rouge) = pousse vers "diabetique"
    - contribution negative (vert)  = pousse vers "sain"
    """
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    X_t = pre.transform(X)
    booster = clf.get_booster()
    dmat = xgb.DMatrix(X_t, feature_names=_transformed_feature_names(pipeline))

    # pred_contribs=True retourne les SHAP values (derniere col = biais)
    shap = booster.predict(dmat, pred_contribs=True)[0]  # shape (n_features + 1,)
    bias = float(shap[-1])
    contribs = shap[:-1]

    grouped = _map_to_original_features(contribs, _transformed_feature_names(pipeline))

    # Top k par valeur absolue
    sorted_feats = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    return {
        "bias": round(bias, 4),
        "top_factors": [
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": float(X[feat].iloc[0]) if pd.api.types.is_numeric_dtype(X[feat]) else str(X[feat].iloc[0]),
                "contribution": round(float(val), 4),
                "direction": "risk" if val > 0 else "protective",
            }
            for feat, val in sorted_feats
        ],
    }


def compute_global_feature_importance(pipeline: Pipeline) -> list:
    """Importance globale des features (gain XGBoost regroupe par feature d'origine)."""
    clf = pipeline.named_steps["classifier"]
    transformed_names = _transformed_feature_names(pipeline)
    importances = clf.feature_importances_

    grouped = {feat: 0.0 for feat in ALL_FEATURES}
    for i, name in enumerate(transformed_names):
        matched = None
        for cat in CATEGORICAL_FEATURES:
            if name.startswith(cat + "_"):
                matched = cat
                break
        if matched is None:
            matched = name
        grouped[matched] = grouped.get(matched, 0.0) + float(importances[i])

    total = sum(grouped.values()) or 1.0
    return sorted(
        [
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "importance": round(val / total * 100, 2),
            }
            for feat, val in grouped.items()
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )


# -----------------------------------------------------------------------------
# PIPELINE & ENTRAINEMENT
# -----------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Pipeline preprocessing + XGBoost.

    Pourquoi cette config ?
    - SimpleImputer median pour les numeriques : robuste aux outliers medicaux.
    - StandardScaler : XGBoost n'en a pas besoin mathematiquement, mais aide
      a l'interpretation des feature importances et a la regularisation.
    - OneHotEncoder pour `gender` et `smoking_status` (variables nominales,
      pas ordinales -> jamais de LabelEncoder ici).
    - XGBoost : choix justifie dans le README (gestion des NaN, gradient boosting,
      tolere desequilibre 60/40 via scale_pos_weight si besoin).
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    # Hyperparametres tunes (ameliorations vs notebook original) :
    # - n_estimators=300 + learning_rate=0.05 : trade-off precision/overfitting
    # - max_depth=4 : assez profond pour interactions, evite memorisation
    # - subsample + colsample_bytree=0.8 : regularisation type bagging
    # - reg_alpha (L1) + reg_lambda (L2) : penalisent les feuilles excessives
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def find_csv() -> Path:
    """Recherche le CSV diabetes dans data/ (premier .csv trouve)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier '{DATA_DIR}' n'existe pas.\n"
            f"Creez-le et placez-y votre CSV (ex: diabetes_dataset.csv)."
        )
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"Aucun fichier .csv trouve dans '{DATA_DIR}'.\n"
            f"Placez-y votre fichier de donnees (ex: diabetes_dataset.csv)."
        )
    return csvs[0]


def train_model() -> dict:
    """Entraine XGBoost et sauvegarde le pipeline + metriques."""
    csv_path = find_csv()
    print(f"[INFO] Chargement du dataset : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Verification que les colonnes requises sont presentes
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}\n"
            f"Colonnes presentes : {list(df.columns)}"
        )

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # Split stratifie (CRUCIAL : evite le data leakage du notebook original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    print("[INFO] Entrainement XGBoost en cours...")
    pipeline.fit(X_train, y_train)

    # Evaluation TRAIN + TEST pour detecter overfitting (manquait dans le notebook)
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_proba_test)

    metrics = {
        "model": "XGBoost Classifier",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(ALL_FEATURES),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "overfitting_gap": round(float(train_acc - test_acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
    }

    print(f"[OK] Train accuracy : {train_acc:.4f}")
    print(f"[OK] Test accuracy  : {test_acc:.4f}")
    print(f"[OK] ROC-AUC        : {roc_auc:.4f}")
    print(f"[OK] Gap (train-test): {train_acc - test_acc:.4f} "
          f"{'(overfitting !)' if (train_acc - test_acc) > 0.05 else '(OK)'}")

    # Sauvegarde
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Modele sauvegarde : {MODEL_PATH}")
    return metrics


def load_or_train_model():
    """Charge le pickle s'il existe, sinon entraine."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        print(f"[INFO] Chargement du modele existant : {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            STATE["model"] = pickle.load(f)
        with open(METRICS_PATH) as f:
            STATE["metrics"] = json.load(f)
        print(f"[OK] Modele charge | Test accuracy : {STATE['metrics']['test_accuracy']}")
    else:
        print("[INFO] Aucun modele trouve, entrainement en cours...")
        try:
            STATE["metrics"] = train_model()
            with open(MODEL_PATH, "rb") as f:
                STATE["model"] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print("[WARN] L'API repondra 'not ready' jusqu'a ce que vous fournissiez le CSV.")

    # Pre-calcul de l'importance globale (XAI)
    if STATE["model"] is not None:
        STATE["feature_importance"] = compute_global_feature_importance(STATE["model"])
        print(f"[OK] Top 3 facteurs globaux : "
              f"{', '.join(f['label'] for f in STATE['feature_importance'][:3])}")


# -----------------------------------------------------------------------------
# ROUTES HTML (sert les pages)
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "landing.html")


@app.route("/predict.html")
def predict_page():
    return send_from_directory(BASE_DIR, "predict.html")


@app.route("/landing.html")
def landing_page():
    return send_from_directory(BASE_DIR, "landing.html")


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.route("/status", methods=["GET"])
def status():
    ready = STATE["model"] is not None
    return jsonify({
        "ready": ready,
        "model": "XGBoost Classifier" if ready else None,
        "metrics": STATE["metrics"] if ready else None,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    if STATE["metrics"] is None:
        return jsonify({"error": "Model not trained"}), 503
    return jsonify(STATE["metrics"])


@app.route("/predict", methods=["POST"])
def predict():
    if STATE["model"] is None:
        return jsonify({
            "error": "Modele non charge. Placez un CSV dans data/ et redemarrez."
        }), 503

    try:
        payload = request.get_json(force=True)

        # Construction du DataFrame dans l'ordre attendu par le pipeline
        row = {}
        for feat in NUMERIC_FEATURES:
            row[feat] = float(payload.get(feat, 0))
        for feat in CATEGORICAL_FEATURES:
            row[feat] = str(payload.get(feat, ""))

        X = pd.DataFrame([row], columns=ALL_FEATURES)

        # Prediction
        diagnosis = int(STATE["model"].predict(X)[0])
        proba = float(STATE["model"].predict_proba(X)[0][1])

        # XAI : explication de CETTE prediction + importance globale
        explanation = explain_prediction(STATE["model"], X, top_k=6)

        return jsonify({
            "diagnosis": diagnosis,
            "risk_probability": round(proba * 100, 1),
            "model": "XGBoost Classifier",
            "explanation": explanation,
            "global_importance": STATE["feature_importance"][:6],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/explain", methods=["GET"])
def explain_global():
    """Retourne l'importance globale des features (XAI au niveau modele)."""
    if STATE["feature_importance"] is None:
        return jsonify({"error": "Model not trained"}), 503
    return jsonify({"global_importance": STATE["feature_importance"]})


@app.route("/retrain", methods=["POST"])
def retrain():
    """Reentraine le modele a la demande (utile apres avoir change le CSV)."""
    try:
        STATE["metrics"] = train_model()
        with open(MODEL_PATH, "rb") as f:
            STATE["model"] = pickle.load(f)
        STATE["feature_importance"] = compute_global_feature_importance(STATE["model"])
        return jsonify({"ok": True, "metrics": STATE["metrics"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" DiabeteAI - Backend Flask")
    print("=" * 60)
    load_or_train_model()
    print("=" * 60)
    print(" Serveur demarre : http://localhost:5000")
    print(" Landing page    : http://localhost:5000/")
    print(" Page predict    : http://localhost:5000/predict.html")
    print(" API status      : http://localhost:5000/status")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
