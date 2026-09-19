"""
Heart disease prediction pipeline (Cleveland dataset)

Reproduces the project notebook as a script and adds:
  * predict_record()  - inference for one new patient
  * export_json()     - exports the trained models for the browser UI
                        (heart_disease_predictor.html) and for parity tests

Usage:
    python heart_disease_pipeline.py --data data.csv --out model.json
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  # optional: only used for the comparison table
    XGBClassifier = None

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
CAP_COLS = ["trestbps", "chol", "thalach"]
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
EXPORTABLE = {"Random Forest": "rf", "Logistic Regression": "lr", "MLP": "mlp"}


# ----------------------------------------------------------------- preprocessing
def iqr_bounds(df, cols=CAP_COLS):
    """IQR capping bounds (Eq. 1-2). As in the notebook they are computed on the
    full dataset before the split."""
    out = {}
    for c in cols:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
        iqr = q3 - q1
        out[c] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return out


def clean(df, bounds):
    """Cap outliers and log-transform oldpeak (Eq. 2-3). Returns a copy."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        df[c] = df[c].clip(lo, hi)
    df["oldpeak"] = np.log1p(df["oldpeak"])
    return df


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        **({"XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)}
           if XGBClassifier else {}),
    }


# ----------------------------------------------------------------- training
def train(data_path):
    """Train all models. Returns an 'artifacts' dict used by everything else."""
    raw = pd.read_csv(data_path)
    bounds = iqr_bounds(raw)
    data = clean(raw, bounds)
    X, y = data[FEATURES], data["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

    models, rows = build_models(), []
    for name, m in models.items():
        m.fit(Xtr_s, y_tr)
        pred, proba = m.predict(Xte_s), m.predict_proba(Xte_s)[:, 1]
        rows.append(dict(model=name,
                         acc=round(accuracy_score(y_te, pred), 3),
                         prec=round(precision_score(y_te, pred), 3),
                         rec=round(recall_score(y_te, pred), 3),
                         f1=round(f1_score(y_te, pred), 3),
                         auc=round(roc_auc_score(y_te, proba), 3),
                         cm=confusion_matrix(y_te, pred).tolist()))
    return dict(raw=raw, bounds=bounds, scaler=scaler, models=models,
                metrics=rows, X_test=X_te, y_test=y_te,
                n_train=len(X_tr), n_test=len(X_te))


# ----------------------------------------------------------------- inference
def transform_record(art, record):
    """record: dict with the 13 raw attributes -> scaled 1x13 array."""
    df = pd.DataFrame([{f: float(record[f]) for f in FEATURES}])
    return art["scaler"].transform(clean(df, art["bounds"])[FEATURES].values)


def predict_record(art, record, model="Random Forest"):
    """Probability of heart disease for one raw patient record."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(art["models"][model].predict_proba(transform_record(art, record))[0, 1])


def typical_patient(raw):
    """Median for measurements, most frequent category otherwise."""
    return {c: (float(raw[c].median()) if c in CONTINUOUS else int(raw[c].mode()[0]))
            for c in FEATURES}


# ----------------------------------------------------------------- export for UI
def _pick_examples(art):
    raw, te = art["raw"], art["X_test"].index
    rows = raw.loc[te].copy()
    rows["p"] = [predict_record(art, r) for _, r in rows.iterrows()]
    low = rows[rows.target == 0].sort_values("p").iloc[2]
    high = rows[rows.target == 1].sort_values("p").iloc[-3]
    mid = rows[(rows.p > .4) & (rows.p < .6)].iloc[0]
    conv = lambda r: {**{c: (float(r[c]) if c == "oldpeak" else int(r[c])) for c in FEATURES},
                      "actual": int(r.target), "p": float(r.p)}
    return {"low": conv(low), "high": conv(high), "borderline": conv(mid)}


def export_json(art, path):
    m = art["models"]
    rf, lr, mlp = m["Random Forest"], m["Logistic Regression"], m["MLP"]
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = v[:, 1] / v.sum(1)
        leaf = t.children_left == -1
        trees.append(dict(
            f=[-1 if l else int(f) for f, l in zip(t.feature, leaf)],
            # full-precision thresholds: sklearn compares float32 inputs to them
            t=[float(p1[i]) if leaf[i] else float(t.threshold[i]) for i in range(t.node_count)],
            l=t.children_left.tolist(), r=t.children_right.tolist()))
    raw = art["raw"]
    out = dict(
        cols=FEATURES,
        bounds={k: list(v) for k, v in art["bounds"].items()},
        mean=art["scaler"].mean_.tolist(), scale=art["scaler"].scale_.tolist(),
        lr=dict(w=lr.coef_[0].tolist(), b=float(lr.intercept_[0])),
        mlp=dict(W=[w.tolist() for w in mlp.coefs_], B=[b.tolist() for b in mlp.intercepts_]),
        rf=trees, ref=typical_patient(raw),
        rng={c: [float(raw[c].min()), float(raw[c].max())] for c in FEATURES},
        metrics=art["metrics"], imp=dict(zip(FEATURES, rf.feature_importances_.tolist())),
        examples=_pick_examples(art), split=dict(train=art["n_train"], test=art["n_test"]))
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--out", default="model.json")
    a = ap.parse_args()
    art = train(a.data)
    print(pd.DataFrame(art["metrics"]).drop(columns="cm").sort_values("acc", ascending=False)
          .to_string(index=False))
    print("\nexported ->", export_json(art, a.out))
    print("typical patient probability (RF): %.3f"
          % predict_record(art, typical_patient(art["raw"])))


if __name__ == "__main__":
    main()
