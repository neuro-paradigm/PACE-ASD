"""
PACE-ASD — Case Study Generator

Runs the complete inference pipeline on one processed subject and produces
a multi-panel case study visualization demonstrating software functionality.

This demonstrates software capabilities ONLY.
One subject does not imply clinical validity.

Usage:
    python scripts/generate_case_study.py \\
        --clip_id asd_1 \\
        --checkpoint models/A1/fold1_seed42.pt \\
        --config configs/inference.yaml \\
        --output outputs/case_study_asd1

    python scripts/generate_case_study.py \\
        --clip_id td_1 \\
        --checkpoint models/A1/fold1_seed42.pt \\
        --output outputs/case_study_td1

Outputs:
    <output>/
    ├── case_study_summary.json
    ├── kinematics.npz
    ├── selected_events.json
    ├── attribution.json
    ├── result.json
    └── visualizations/
        ├── 01_skeleton_frames.png     # Sample frames with landmark overlay
        ├── 02_position_trajectories.png
        ├── 03_velocity_acceleration.png
        ├── 04_bilateral_asymmetry.png
        ├── 05_block_saliency.png
        ├── 06_selected_events.png
        ├── 07_evidence_summary.png
        └── 08_complete_panel.png       # Full multi-panel figure
"""

import argparse
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPTS_DIR)


# ── Landmark connectivity for skeleton overlay ──────────────────────────────────

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

JOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "left_mouth", "right_mouth",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def plot_skeleton_frames(positions: np.ndarray, output_dir: str,
                         n_frames_to_show: int = 6) -> None:
    """Plot a grid of skeleton frames at evenly spaced intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print("  [WARN] No valid frames for skeleton plot")
        return

    # Choose evenly spaced valid frames
    n_show  = min(n_frames_to_show, len(valid_idx))
    indices = valid_idx[np.linspace(0, len(valid_idx) - 1, n_show, dtype=int)]

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4))
    fig.suptitle("Skeleton Landmark Overlays (Sample Frames)", fontsize=13, fontweight="bold")
    if n_show == 1:
        axes = [axes]

    for ax_i, (ax, t) in enumerate(zip(axes, indices)):
        pts = positions[t]  # (33, 2)
        # Scale from normalised coords to plotting coords
        x = pts[:, 0]; y = -pts[:, 1]  # flip Y for display

        # Draw edges
        for i, j in SKELETON_EDGES:
            ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', linewidth=1.0, alpha=0.6)

        # Draw joints
        ax.scatter(x, y, c='red', s=20, zorder=5)
        ax.set_title(f"Frame {t}", fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "01_skeleton_frames.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_skeleton_frames.png")


def plot_position_trajectories(positions: np.ndarray, output_dir: str,
                                selected_indices=None) -> None:
    """Plot position magnitude trajectories for selected body regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = positions.shape[0]
    t_axis = np.arange(T)
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    region_groups = {
        "Head (0–10)": list(range(0, 11)),
        "Arms (11–22)": list(range(11, 23)),
        "Legs (25–32)": list(range(25, 33)),
    }
    colors = {"Head (0–10)": "#E91E63", "Arms (11–22)": "#FF9800", "Legs (25–32)": "#2196F3"}

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle("Position Trajectories by Body Region", fontsize=13, fontweight="bold")

    for region, joints in region_groups.items():
        mag = np.linalg.norm(positions[:, joints, :], axis=-1).mean(axis=1)  # (T,)
        mag_plot = np.where(valid_mask, mag, np.nan)
        ax.plot(t_axis, mag_plot, label=region, color=colors[region], linewidth=1.2)

    if selected_indices is not None:
        for idx in selected_indices:
            if 0 <= idx < T:
                ax.axvline(idx, color="orange", alpha=0.15, linewidth=0.8)
        ax.axvline(selected_indices[0] if len(selected_indices) > 0 else 0,
                   color="orange", alpha=0.5, linewidth=1, label="Selected events")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean position magnitude (normalised)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, T)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_position_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_position_trajectories.png")


def plot_velocity_acceleration(velocities: np.ndarray, accelerations: np.ndarray,
                                positions: np.ndarray, output_dir: str) -> None:
    """Plot velocity and acceleration magnitude over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = positions.shape[0]
    t_axis = np.arange(T)
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Velocity and Acceleration Trajectories", fontsize=13, fontweight="bold")

    vel_mag = np.linalg.norm(velocities, axis=-1).mean(axis=1)  # (T,)
    vel_mag[~valid_mask] = np.nan
    ax1.plot(t_axis, vel_mag, color="#4CAF50", linewidth=1.2)
    ax1.set_ylabel("Mean velocity magnitude")
    ax1.set_title("Velocity")

    acc_mag = np.linalg.norm(accelerations, axis=-1).mean(axis=1)  # (T,)
    acc_mag[~valid_mask] = np.nan
    ax2.plot(t_axis, acc_mag, color="#E91E63", linewidth=1.2)
    ax2.set_ylabel("Mean acceleration magnitude")
    ax2.set_xlabel("Frame")
    ax2.set_title("Acceleration")

    plt.tight_layout()
    path = os.path.join(output_dir, "03_velocity_acceleration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_velocity_acceleration.png")


def plot_bilateral_asymmetry(positions: np.ndarray, output_dir: str) -> dict:
    """Plot bilateral movement asymmetry time-series for key joint pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from asymmetry import compute_bilateral_asymmetry, asymmetry_timeseries, LR_PAIRS

    asym_result = compute_bilateral_asymmetry(positions)

    # Time-series for key joints
    KEY_PAIRS = ["shoulder", "elbow", "wrist"]
    colors = ["#9C27B0", "#FF9800", "#2196F3"]

    fig, axes = plt.subplots(len(KEY_PAIRS), 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "Descriptive Bilateral Movement Asymmetry (Left − Right)\n"
        "[Descriptive measure — not a validated ASD biomarker]",
        fontsize=12, fontweight="bold"
    )

    for ax, pair, color in zip(axes, KEY_PAIRS, colors):
        ts = asymmetry_timeseries(positions, joint_pair=pair)
        T  = len(ts)
        ax.plot(np.arange(T), ts, color=color, linewidth=1.0, label=f"{pair}")
        ax.set_ylabel("L-R diff")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    path = os.path.join(output_dir, "04_bilateral_asymmetry.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_bilateral_asymmetry.png")
    return asym_result


def plot_block_saliency(block_scores, selected_block_starts: list,
                        block_size: int, output_dir: str) -> None:
    """Plot Block-ESG gate saliency scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if block_scores is None:
        print("  [WARN] No block scores available (model may not use ESG)")
        return

    bs = np.array(block_scores)
    n  = len(bs)
    # Find selected blocks
    selected_block_idxs = set(s // block_size for s in selected_block_starts)
    colors = ["#FF9800" if i in selected_block_idxs else "#90CAF9" for i in range(n)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(n), bs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Block index ({block_size} frames / block)")
    ax.set_ylabel("Gate saliency score")
    ax.set_title(
        f"Block-ESG Saliency Scores — Orange = Selected top-M blocks",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "05_block_saliency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_block_saliency.png")


def plot_selected_events(positions: np.ndarray, selected_block_starts: list,
                          block_size: int, output_dir: str) -> None:
    """Highlight selected temporal event blocks on position trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = positions.shape[0]
    t_axis = np.arange(T)
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    pos_mag = np.linalg.norm(positions, axis=-1).mean(axis=1)  # (T,)
    pos_mag[~valid_mask] = np.nan

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_axis, pos_mag, color="#90CAF9", linewidth=1.0, label="Position trajectory")

    for s in selected_block_starts:
        ax.axvspan(s, min(s + block_size, T), alpha=0.3, color="orange",
                   label="_nolegend_" if s != selected_block_starts[0] else "Selected events")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean position magnitude")
    ax.set_title("Selected Temporal Events (Block-ESG Selection)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, T)
    plt.tight_layout()
    path = os.path.join(output_dir, "06_selected_events.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_selected_events.png")


def plot_complete_panel(result: dict, positions: np.ndarray, velocities: np.ndarray,
                         accelerations: np.ndarray, block_scores,
                         selected_block_starts: list, block_size: int,
                         asym_result: dict, output_dir: str) -> None:
    """Generate one combined multi-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from asymmetry import asymmetry_timeseries

    T = positions.shape[0]
    t_axis = np.arange(T)
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4

    fig = plt.figure(figsize=(20, 14))
    cal_prob = result['calibrated_probability']
    pred = result['prediction']
    fig.suptitle(
        f"PACE-ASD Case Study: {result.get('clip_id', 'unknown')}\n"
        f"Calibrated P(ASD) = {cal_prob:.3f}  |  Prediction = {pred}\n"
        "[Software functionality demonstration — single subject does not imply clinical validity]",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # (A) Position
    ax_a = fig.add_subplot(gs[0, :])
    pos_mag = np.linalg.norm(positions, axis=-1).mean(axis=1)
    pos_mag[~valid_mask] = np.nan
    ax_a.plot(t_axis, pos_mag, color="#2196F3", linewidth=1.0)
    for s in selected_block_starts:
        ax_a.axvspan(s, min(s + block_size, T), alpha=0.2, color="orange")
    ax_a.set_title("(A) Position Trajectory (orange = selected events)", fontsize=10)
    ax_a.set_xlabel("Frame"); ax_a.set_ylabel("Magnitude")

    # (B) Velocity
    ax_b = fig.add_subplot(gs[1, 0])
    vel_mag = np.linalg.norm(velocities, axis=-1).mean(axis=1)
    vel_mag[~valid_mask] = np.nan
    ax_b.plot(t_axis, vel_mag, color="#4CAF50", linewidth=1.0)
    ax_b.set_title("(B) Velocity", fontsize=10)
    ax_b.set_xlabel("Frame")

    # (C) Acceleration
    ax_c = fig.add_subplot(gs[1, 1])
    acc_mag = np.linalg.norm(accelerations, axis=-1).mean(axis=1)
    acc_mag[~valid_mask] = np.nan
    ax_c.plot(t_axis, acc_mag, color="#E91E63", linewidth=1.0)
    ax_c.set_title("(C) Acceleration", fontsize=10)
    ax_c.set_xlabel("Frame")

    # (D) Bilateral asymmetry
    ax_d = fig.add_subplot(gs[1, 2])
    from asymmetry import asymmetry_timeseries
    wrist_ts = asymmetry_timeseries(positions, joint_pair="wrist")
    ax_d.plot(t_axis, wrist_ts, color="#9C27B0", linewidth=1.0)
    ax_d.set_title("(D) Wrist L-R asymmetry\n[Descriptive, not a biomarker]", fontsize=10)
    ax_d.set_xlabel("Frame")

    # (E) Block saliency
    ax_e = fig.add_subplot(gs[2, 0])
    if block_scores is not None:
        bs = np.array(block_scores)
        n = len(bs)
        sel_idxs = set(s // block_size for s in selected_block_starts)
        colors_e = ["#FF9800" if i in sel_idxs else "#90CAF9" for i in range(n)]
        ax_e.bar(np.arange(n), bs, color=colors_e)
        ax_e.set_title("(E) Block-ESG Saliency", fontsize=10)
        ax_e.set_xlabel("Block index")

    # (F) Body-region attribution
    ax_f = fig.add_subplot(gs[2, 1])
    if result.get("body_region_attribution"):
        regions = list(result["body_region_attribution"].keys())
        vals    = list(result["body_region_attribution"].values())
        ax_f.barh(regions, vals, color=["#E91E63", "#FF9800", "#9C27B0", "#2196F3"][:len(regions)])
        ax_f.set_xlim(0, 1)
        ax_f.set_title("(F) Body-region attribution", fontsize=10)

    # (G) Kinematic-stream attribution
    ax_g = fig.add_subplot(gs[2, 2])
    if result.get("kinematic_stream_attribution"):
        streams = list(result["kinematic_stream_attribution"].keys())
        svals   = list(result["kinematic_stream_attribution"].values())
        ax_g.bar(streams, svals, color=["#2196F3", "#4CAF50", "#E91E63"])
        ax_g.set_ylim(0, 1)
        ax_g.set_title("(G) Kinematic-stream attribution", fontsize=10)

    plt.savefig(os.path.join(output_dir, "08_complete_panel.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 08_complete_panel.png")


# ── Main ─────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PACE-ASD: Case Study Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_case_study.py \\\
      --clip_id asd_1 \\\
      --checkpoint models/A1/fold1_seed42.pt \\\
      --output outputs/case_study_asd1

  python scripts/generate_case_study.py \\\
      --clip_id td_1 \\\
      --checkpoint models/A1/fold1_seed42.pt \\\
      --output outputs/case_study_td1

IMPORTANT: This demonstrates software functionality.
Single-subject outputs do not imply clinical validity.
"""
    )
    parser.add_argument("--clip_id",    type=str, required=True,
                        help="Clip ID from processed/features/ (e.g. asd_1, td_1)")
    parser.add_argument("--checkpoint", type=str, default="models/A1/fold1_seed42.pt")
    parser.add_argument("--config",     type=str, default="configs/inference.yaml")
    parser.add_argument("--features_dir", type=str, default="processed/features")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"outputs/case_study_{args.clip_id}"

    npy_path = os.path.join(args.features_dir, f"{args.clip_id}.npy")
    if not os.path.isfile(npy_path):
        print(f"  ERROR: Feature file not found: {npy_path}")
        sys.exit(1)
    if not os.path.isfile(args.checkpoint):
        print(f"  ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"\n  PACE-ASD Case Study Generator")
    print(f"  Clip ID      : {args.clip_id}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Output       : {args.output}")
    print(f"  [NOTE] This demonstrates software functionality.")
    print(f"         Single-subject output does not imply clinical validity.\n")

    viz_dir = os.path.join(args.output, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # ── Run inference via infer.py ─────────────────────────────────────────────────
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "infer", os.path.join(SCRIPTS_DIR, "infer.py")
    )
    infer_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(infer_mod)

    class InferArgs:
        input      = None
        input_npy  = npy_path
        checkpoint = args.checkpoint
        config     = args.config
        output     = args.output

    result = infer_mod.run_inference(InferArgs())

    # Load saved kinematics
    kinem = np.load(os.path.join(args.output, "kinematics.npz"))
    positions     = kinem["positions"]
    velocities    = kinem["velocities"]
    accelerations = kinem["accelerations"]

    # Load saved events
    events_path = os.path.join(args.output, "selected_events.json")
    if os.path.isfile(events_path):
        with open(events_path) as f:
            events = json.load(f)
        selected_block_starts = events.get("selected_block_starts", [])
        block_size = events.get("block_size_frames", 15)
        block_scores = events.get("all_block_saliency_scores", None)
    else:
        selected_block_starts = []
        block_size = 15
        block_scores = None

    # ── Generate visualizations ──────────────────────────────────────────────────────
    print("\n  Generating case study visualizations...")

    plot_skeleton_frames(positions, viz_dir)
    plot_position_trajectories(positions, viz_dir, selected_indices=(
        events.get("selected_frame_indices", []) if os.path.isfile(events_path) else None
    ))
    plot_velocity_acceleration(velocities, accelerations, positions, viz_dir)
    asym_result = plot_bilateral_asymmetry(positions, viz_dir)
    plot_block_saliency(block_scores, selected_block_starts, block_size, viz_dir)
    plot_selected_events(positions, selected_block_starts, block_size, viz_dir)
    plot_complete_panel(
        result, positions, velocities, accelerations,
        block_scores, selected_block_starts, block_size,
        asym_result, viz_dir,
    )

    # ── Case study summary JSON ─────────────────────────────────────────────────────
    summary = {
        "clip_id":               args.clip_id,
        "checkpoint":            args.checkpoint,
        "disclaimer":            (
            "Software functionality demonstration. "
            "Single-subject output does not imply clinical validity."
        ),
        "prediction":            result["prediction"],
        "calibrated_probability": result["calibrated_probability"],
        "raw_probability":        result["raw_probability"],
        "selected_events_summary": result["selected_events_summary"],
        "body_region_attribution": result.get("body_region_attribution", {}),
        "kinematic_stream_attribution": result.get("kinematic_stream_attribution", {}),
        "bilateral_asymmetry_summary": {
            "mean_asymmetry": asym_result.get("mean_asymmetry", 0.0),
            "note":           asym_result.get("note", ""),
        },
        "visualizations_generated": [
            "01_skeleton_frames.png", "02_position_trajectories.png",
            "03_velocity_acceleration.png", "04_bilateral_asymmetry.png",
            "05_block_saliency.png", "06_selected_events.png",
            "07_evidence_summary.png", "08_complete_panel.png",
        ],
    }

    summary_path = os.path.join(args.output, "case_study_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Case study summary: {summary_path}")

    print(f"\n  -- Case Study Complete --")
    print(f"  Calibrated P(ASD): {result['calibrated_probability']:.4f}")
    print(f"  Prediction:        {result['prediction']}")
    print(f"  Output:            {args.output}/")


if __name__ == "__main__":
    main()
