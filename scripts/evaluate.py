import pandas as pd, joblib, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart_disease_clean.csv"
MODEL = ROOT / "models" / "final_model.pkl"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(f"Missing data file: {DATA}")
if not MODEL.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL}")

df = pd.read_csv(DATA)
X, y = df.drop(columns=["target"]), df["target"]
pipe = joblib.load(MODEL)

y_prob = pipe.predict_proba(X)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y, y_pred, digits=3))

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.title("Confusion Matrix (threshold=0.5)")
plt.tight_layout(); plt.savefig(RES / "confusion_matrix.png", dpi=150); plt.close()

RocCurveDisplay.from_estimator(pipe, X, y)
plt.title("ROC Curve"); plt.tight_layout(); plt.savefig(RES / "roc_full.png", dpi=150); plt.close()

PrecisionRecallDisplay.from_estimator(pipe, X, y)
plt.title("PrecisionRecall Curve"); plt.tight_layout(); plt.savefig(RES / "pr_curve.png", dpi=150); plt.close()

print(" Saved evaluation artifacts in results/")
