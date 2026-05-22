"""
ASDMotion - Reconstruction Training Pipeline
Loads split partitions statically from the splits/ directory and trains
models, saving checkpoints to the 'reconstruct/' directory.
"""

import os
import sys
import yaml
import pandas as pd
import torch
import numpy as np

# Ensure src directory is in path
sys.path.insert(0, 'src')
from train import set_seed, train_fold, evaluate_test_set
from report import generate_fold_report

def main():
    config_path = 'configs/config.yaml'
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at '{config_path}'")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for determinism
    set_seed(config['training']['seed'])

    # Override target output directories to 'reconstruct'
    config['output']['models_dir'] = 'reconstruct'
    config['output']['reports_dir'] = 'reports_reconstruct'
    
    os.makedirs(config['output']['models_dir'], exist_ok=True)
    os.makedirs(config['output']['reports_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  ASDMotion - RECONSTRUCTION TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Splits: Statically loaded from 'splits/' directory.")
    print(f"  Checkpoints Destination: '{config['output']['models_dir']}/'")
    print(f"  Reports Destination:     '{config['output']['reports_dir']}/'")
    print("=" * 70)

    splits_dir = 'splits'
    n_folds = config['training']['n_folds']

    all_fold_val_metrics = []
    all_fold_test_metrics = []

    # Read the common test set clips from fold1_test.txt
    test_split_path = os.path.join(splits_dir, 'fold1_test.txt')
    if not os.path.exists(test_split_path):
        print(f"[ERROR] Held-out test split file not found at '{test_split_path}'")
        sys.exit(1)

    test_df = pd.read_csv(test_split_path)
    test_ids = test_df['video_id'].tolist()
    test_subjects = test_df['subject_id'].tolist()
    test_labels = test_df['label'].tolist()

    # Iterate and train for each fold
    for fold_idx in range(1, n_folds + 1):
        train_file = os.path.join(splits_dir, f'fold{fold_idx}_train.txt')
        val_file = os.path.join(splits_dir, f'fold{fold_idx}_val.txt')

        if not os.path.exists(train_file) or not os.path.exists(val_file):
            print(f"[ERROR] Split files for Fold {fold_idx} not found in '{splits_dir}/'.")
            sys.exit(1)

        print(f"\n  Loading partitions for Fold {fold_idx}...")
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        train_ids = train_df['video_id'].tolist()
        train_subjects = train_df['subject_id'].tolist()
        train_labels = train_df['label'].tolist()

        val_ids = val_df['video_id'].tolist()
        val_subjects = val_df['subject_id'].tolist()
        val_labels = val_df['label'].tolist()

        # Execute training fold
        _, best_metrics, t_metrics = train_fold(
            fold_idx=fold_idx,
            train_ids=train_ids,
            train_labels=train_labels,
            train_subjects=train_subjects,
            val_ids=val_ids,
            val_labels=val_labels,
            val_subjects=val_subjects,
            test_ids=test_ids,
            test_labels=test_labels,
            test_subjects=test_subjects,
            config=config,
            device=device
        )
        
        all_fold_val_metrics.append(best_metrics)
        all_fold_test_metrics.append(t_metrics)

    # ── Cross-Fold Summary ────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  CROSS-VALIDATION SUMMARY (VALIDATION SETS)")
    print(f"{'#'*70}")

    metric_keys = ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity', 'ece']
    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*35}")

    for key in metric_keys:
        values = [m[key] for m in all_fold_val_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {key:<15} {mean_val:>8.4f} {std_val:>8.4f}")

    print(f"\n{'#'*70}")
    print(f"  CROSS-VALIDATION SUMMARY (TEST SET)")
    print(f"{'#'*70}")

    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*35}")

    for key in metric_keys:
        values = [m[key] for m in all_fold_test_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {key:<15} {mean_val:>8.4f} {std_val:>8.4f}")

    # ── Held-out Test Evaluation ──────────────────────────
    print(f"\n{'#'*70}")
    print(f"  HELD-OUT TEST SET EVALUATION")
    print(f"{'#'*70}")
    test_subj_unique = len(set(test_subjects))
    print(f"  Test clips: {len(test_ids)} ({sum(test_labels)} ASD, {len(test_labels) - sum(test_labels)} Non-ASD)")
    print(f"  Unique subjects evaluated: {test_subj_unique}")

    per_fold_test, ensemble_metrics, ensemble_cm, ensemble_roc = \
        evaluate_test_set(test_ids, test_labels, test_subjects, config, device)

    if ensemble_metrics:
        print(f"\n  ENSEMBLE TEST RESULTS:")
        print(f"  {'Metric':<15} {'Value':>8}")
        print(f"  {'-'*25}")
        for key in metric_keys:
            if key in ensemble_metrics:
                print(f"  {key:<15} {ensemble_metrics[key]:>8.4f}")

        # Generate test report PDF
        reports_dir = config['output']['reports_dir']
        generate_fold_report(
            fold_idx='test',
            epoch_history=[],
            best_epoch=0,
            best_metrics=ensemble_metrics,
            cm=ensemble_cm,
            roc_data=ensemble_roc,
            output_dir=reports_dir,
            attention_data=None
        )

    print(f"\n  Reconstructed Models saved in: {config['output']['models_dir']}/")
    print(f"  Reconstructed Reports saved in: {config['output']['reports_dir']}/")
    print(f"{'#'*70}\n")

if __name__ == '__main__':
    main()
