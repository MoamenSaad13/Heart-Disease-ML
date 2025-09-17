# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

This repository implements the **end-to-end ML pipeline** described in your brief:
preprocessing → PCA → feature selection → supervised & unsupervised modeling → hyperparameter tuning → export → Streamlit UI → optional Ngrok deployment → GitHub-ready structure.

## Dataset
Use the UCI Heart Disease dataset (Cleveland subset is standard). Download it and save as:
```
data/heart_disease.csv
```
**Columns expected (Kaggle-style heart.csv):**
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

> If your CSV has different column names or encoding (as on some UCI mirrors), update `notebooks/01_data_preprocessing.ipynb` and `ui/app.py` where indicated.

## Quickstart
```bash
# (Recommended) create a venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Put the dataset at data/heart_disease.csv (see above)
# Then (option A) run notebooks manually in Jupyter/VSCode
# Or (option B) run everything headless via the helper script:
python run_all.py

# Launch the Streamlit app
streamlit run ui/app.py
```

## Project Structure
```
Heart_Disease_Project/
├── data/
│   └── heart_disease.csv  # (you add this)
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── models/
│   └── final_model.pkl
├── ui/
│   └── app.py
├── deployment/
│   └── ngrok_setup.txt
├── results/
│   └── evaluation_metrics.txt
├── run_all.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes
- All preprocessing is wrapped in a `Pipeline` so the saved `.pkl` includes encoders/scalers.
- `ui/app.py` expects `models/final_model.pkl` produced by the tuning notebook or `run_all.py`.
- If you have internet access, `01_data_preprocessing.ipynb` includes an **optional** downloader; otherwise place the CSV manually.
