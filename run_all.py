import subprocess, sys, os, pathlib

root = pathlib.Path(__file__).parent
nb_dir = root / "notebooks"

def run(nb):
    print(f"=== Running {nb} ===")
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=600", "--output", nb, str(nb_dir / nb)]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # Ensure results dir exists
    (root / "results").mkdir(exist_ok=True)
    run("01_data_preprocessing.ipynb")
    run("02_pca_analysis.ipynb")
    run("03_feature_selection.ipynb")
    run("04_supervised_learning.ipynb")
    run("05_unsupervised_learning.ipynb")
    run("06_hyperparameter_tuning.ipynb")
    print("✅ All notebooks executed.")
