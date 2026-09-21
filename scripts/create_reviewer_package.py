"""Script to create a complete, self-contained reviewer release zip package."""
import os
import zipfile

REPO_ROOT = r"d:\PACE-ASD"
OUT_DIR = os.path.join(REPO_ROOT, "release")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_PATH = os.path.join(OUT_DIR, "PACE-ASD-v1.0-reviewer.zip")

# Key individual files
FILES_TO_INCLUDE = [
    "README.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "STATISTICAL_ANALYSIS_PLAN.md",
    "REPRODUCE.md",
    "PREPROCESS_SPEC.md",
    "splits/splits_dryad_v2_dedup.json",
    "processed/labels.csv",
    "results/A1_per_seed.json",
    "results/ablation_results.csv",
    "results/interpretability_metrics_A1.json",
    "results/wilcoxon_results.txt",
    "results/supplement_results.csv",
    "release/README_REVIEWER.md",
    "data/README.md",
    "checkpoints/README.md",
    "outputs/.gitkeep",
    # Reference model checkpoints for seed 42 (all 3 folds)
    "models/A1/fold1_seed42.pt",
    "models/A1/fold2_seed42.pt",
    "models/A1/fold3_seed42.pt",
]

# Entire directories to include recursively
DIRS_TO_INCLUDE = [
    "configs",
    "docs",
    "src",
    "scripts",
    "tests",
    "examples",
    "processed/features",
    "processed/removed_duplicates",
]

print(f"Creating self-contained reviewer package at {ZIP_PATH}...")
added_count = 0

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. Add individual files
    for rel_path in FILES_TO_INCLUDE:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.isfile(full_path):
            arcname = os.path.join("PACE-ASD-v1.0-reviewer", rel_path)
            zf.write(full_path, arcname=arcname)
            added_count += 1
        else:
            print(f"  [WARN] File not found: {rel_path}")

    # 2. Add directories recursively
    for dir_name in DIRS_TO_INCLUDE:
        full_dir = os.path.join(REPO_ROOT, dir_name)
        if os.path.isdir(full_dir):
            for root, _, files in os.walk(full_dir):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    if f.endswith((".pyc", ".jsonl")):
                        continue
                    full_f = os.path.join(root, f)
                    rel_f = os.path.relpath(full_f, REPO_ROOT)
                    arcname = os.path.join("PACE-ASD-v1.0-reviewer", rel_f)
                    zf.write(full_f, arcname=arcname)
                    added_count += 1

file_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"\nReviewer package created successfully!")
print(f"  Path: {ZIP_PATH}")
print(f"  Total files packaged: {added_count}")
print(f"  Archive size: {file_size_mb:.2f} MB")
