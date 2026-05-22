"""
ASDMotion - Batch Inference Script for .npy Sequences
Recursively scans a target directory for all .npy skeleton files, runs the production
ensemble model on each, and outputs a summarized CSV report.
"""

import os
import sys
import glob
import pandas as pd
import torch
import yaml

# Add src/ to system path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from predict import run_production_inference

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASDMotion - Batch Inference")
    parser.add_argument("--dir", type=str, default=r"C:\Users\saite\OneDrive\Desktop\ASD\SUPADATA", 
                        help="Directory containing subfolders with .npy files")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["screening", "strict"], default="screening",
                        help="Inference mode: 'screening' (0.50 threshold) or 'strict' (0.50 threshold)")
    parser.add_argument("--output_csv", type=str, default="batch_inference_results.csv", 
                        help="Path to save the summary CSV file")
    parser.add_argument("--spatial-processor", type=str, choices=["auto", "enable", "disable"], default="enable",
                        help="Spatial processor mode: 'auto' (detect height axis), 'enable' (force spatial reconstruct), 'disable' (skip)")
    parser.add_argument("--models-dir", type=str, default="models_fullbody", help="Directory containing saved model checkpoints")
    parser.add_argument("--zero-lower-limbs", action="store_true", help="Zero out lower limbs to match upper-body training data distribution")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"[ERROR] Target directory not found: {args.dir}")
        return
        
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Recursively find all .npy files in all subfolders
    search_pattern = os.path.join(args.dir, "**", "*.npy")
    npy_files = glob.glob(search_pattern, recursive=True)
    
    if len(npy_files) == 0:
        print(f"[WARNING] No .npy files found in: {args.dir}")
        return
        
    print(f"Found {len(npy_files)} .npy files to process.")
    
    results = []
    
    force_spatial_reconstruct = None
    if args.spatial_processor == "enable":
        force_spatial_reconstruct = True
    elif args.spatial_processor == "disable":
        force_spatial_reconstruct = False

    for idx, npy_path in enumerate(npy_files, start=1):
        rel_path = os.path.relpath(npy_path, args.dir)
        print(f"\nProcessing [{idx}/{len(npy_files)}]: {rel_path}")
        try:
            res = run_production_inference(
                npy_path, 
                config, 
                device, 
                mode=args.mode, 
                zero_lower_limbs=args.zero_lower_limbs,
                custom_models_dir=args.models_dir,
                force_spatial_reconstruct=force_spatial_reconstruct
            )
            if res is not None:
                results.append({
                    'file_name': os.path.basename(npy_path),
                    'relative_path': rel_path,
                    'prediction': res['prediction'],
                    'consensus_agreement': f"{res['agreement']*100:.1f}%",
                    'avg_prob': f"{res['peak_prob']*100:.2f}%",
                    'window_size': res['window_size']
                })
        except Exception as e:
            print(f"[ERROR] Failed to process {rel_path}: {e}")
            
    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\n{'='*75}")
        print(f"  BATCH INFERENCE COMPLETE")
        print(f"{'='*75}")
        print(f"  Total files scanned:  {len(npy_files)}")
        print(f"  Successfully run:     {len(results)}")
        print(f"  ASD Predictions:      {sum(1 for r in results if r['prediction'] == 'ASD')}")
        print(f"  TD Predictions:       {sum(1 for r in results if r['prediction'] == 'TD')}")
        print(f"  Summary saved to:     {args.output_csv}")
        print(f"{'='*75}\n")
    else:
        print("No successful predictions were made.")

if __name__ == "__main__":
    main()
