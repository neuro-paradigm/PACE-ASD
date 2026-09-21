"""Script to create reviewer release zip package."""
import os
import zipfile

REPO_ROOT = r"d:\PACE-ASD"
OUT_DIR = os.path.join(REPO_ROOT, "release")
os.makedirs(OUT_DIR, exist_ok=True)
ZIP_PATH = os.path.join(OUT_DIR, "PACE-ASD-v1.0-reviewer.zip")

FILES_TO_INCLUDE = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "splits/splits_dryad_v2_dedup.json",
    "processed/labels.csv",
    "processed/features/asd_1.npy",
    "processed/features/td_1.npy",
    "models/A1/fold1_seed42.pt",
    "results/A1_per_seed.json",
    "results/ablation_results.csv",
    "results/interpretability_metrics_A1.json",
    "release/README_REVIEWER.md",
    "data/README.md",
    "checkpoints/README.md",
    "examples/README.md",
    "examples/input/.gitkeep",
    "examples/expected_output/.gitkeep",
    "outputs/.gitkeep",
]

DIRS_TO_INCLUDE = [
    "configs",
    "docs",
    "src",
    "scripts",
    "tests",
]

print(f"Creating reviewer package at {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for rel_path in FILES_TO_INCLUDE:
        full_path = os.path.join(REPO_ROOT, rel_path)
        if os.path.isfile(full_path):
            arcname = os.path.join("PACE-ASD-v1.0-reviewer", rel_path)
            zf.write(full_path, arcname=arcname)
            print(f"  Added file: {rel_path}")
        else:
            print(f"  [WARN] File not found: {rel_path}")

    for dir_name in DIRS_TO_INCLUDE:
        full_dir = os.path.join(REPO_ROOT, dir_name)
        if os.path.isdir(full_dir):
            for root, _, files in os.walk(full_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith((".pyc", ".png", ".jsonl")):
                        continue
                    full_f = os.path.join(root, f)
                    rel_f = os.path.relpath(full_f, REPO_ROOT)
                    arcname = os.path.join("PACE-ASD-v1.0-reviewer", rel_f)
                    zf.write(full_f, arcname=arcname)
                    print(f"  Added file: {rel_f}")

file_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"\nReviewer zip package created successfully: {ZIP_PATH} ({file_size_mb:.2f} MB)")
