"""Part 2: Model evaluation on Breast Cancer data: CV, confusion matrix,
precision/recall/F1, ROC-AUC, hyperparameter tuning."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, GridSearchCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)

FIG = "outputs/figures/"
data = load_breast_cancer()
X, y = data.data, data.target  # 1 = benign, 0 = malignant
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
out = {"dataset": dict(n_samples=int(X.shape[0]), n_features=int(X.shape[1]),
                       benign=int((y == 1).sum()), malignant=int((y == 0).sum()),
                       train=int(len(ytr)), test=int(len(yte))),
       "cv_baseline": {}}
for n, m in models.items():
    r = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring)
    out["cv_baseline"][n] = {s: [float(r["test_" + s].mean()), float(r["test_" + s].std())] for s in scoring}

# ---- Hyperparameter tuning ----
grids = {
    "Logistic Regression": {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01, 0.1]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
}
tuned, out["tuning"], out["test"] = {}, {}, {}
for n, m in models.items():
    gs = GridSearchCV(m, grids[n], cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
    tuned[n] = gs.best_estimator_
    out["tuning"][n] = dict(best_params={k: (v if isinstance(v, (str, type(None))) else float(v)) for k, v in gs.best_params_.items()},
                            best_cv_roc_auc=float(gs.best_score_))
    p = gs.best_estimator_.predict(Xte); pr = gs.best_estimator_.predict_proba(Xte)[:, 1]
    out["test"][n] = dict(accuracy=accuracy_score(yte, p), precision=precision_score(yte, p),
                          recall=recall_score(yte, p), f1=f1_score(yte, p),
                          roc_auc=roc_auc_score(yte, pr),
                          confusion_matrix=confusion_matrix(yte, p).tolist())
    out["test"][n] = {k: (float(v) if not isinstance(v, list) else v) for k, v in out["test"][n].items()}

best = max(out["test"], key=lambda k: out["test"][k]["roc_auc"])
out["best_model"] = best
bm = tuned[best]
p = bm.predict(Xte)
out["classification_report_best"] = classification_report(yte, p, target_names=data.target_names)

# ---- Plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (n, m) in zip(ax, tuned.items()):
    ConfusionMatrixDisplay.from_estimator(m, Xte, yte, display_labels=data.target_names, cmap="Blues", ax=a, colorbar=False)
    a.set_title(n)
plt.tight_layout(); plt.savefig(FIG + "confusion_matrices.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for n, m in tuned.items():
    fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1])
    plt.plot(fpr, tpr, label=f"{n} (AUC={out['test'][n]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves (Test Set)")
plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(FIG + "roc_curves.png", dpi=150); plt.close()

names = list(out["cv_baseline"]); w = .25
plt.figure(figsize=(7, 4))
for i, s in enumerate(["accuracy", "f1", "roc_auc"]):
    plt.bar(np.arange(len(names)) + i * w, [out["cv_baseline"][n][s][0] for n in names], w, label=s)
plt.xticks(np.arange(len(names)) + w, names); plt.ylim(.9, 1.0); plt.legend()
plt.title("5-Fold CV Performance (Baseline Models)"); plt.tight_layout()
plt.savefig(FIG + "cv_comparison.png", dpi=150); plt.close()

json.dump(out, open("outputs/evaluation_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))
