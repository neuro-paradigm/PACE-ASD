"""
Generate Publication-Quality Composite Figure of PACE-ASD User-Facing Software Outputs.
Creates Figure 6 for the BMC Medical Informatics and Decision Making Software Article.

Panels:
(A) Pose/skeleton representation across sample frames (de-identified, wireframes only)
(B) Position, velocity, and acceleration kinematic trajectories
(C) Block-ESG temporal saliency scores across blocks
(D) Selected temporal events (top-M contiguous 15-frame blocks highlighted)
(E) Body-region attribution (head, arms, torso, legs)
(F) Kinematic-stream attribution (position, velocity, acceleration) + Calibrated probability box

Output:
  - bmc submission/Figure_6.pdf
  - bmc submission/Figure_6.png
  - bmc submission/figures/Figure_6.pdf
  - bmc submission/figures/Figure_6.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import infer

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

def generate_figure(clip_id="asd_45", checkpoint="models/A1/fold1_seed42.pt", output_dir="outputs/figure_software_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    from types import SimpleNamespace
    args = SimpleNamespace(
        input=None,
        input_npy=f"processed/features/{clip_id}.npy",
        checkpoint=checkpoint,
        config="configs/inference.yaml",
        output=output_dir,
    )

    result = infer.run_inference(args)

    # Load kinematics and event data
    kinem = np.load(os.path.join(output_dir, "kinematics.npz"))
    positions = kinem["positions"]
    velocities = kinem["velocities"]
    accelerations = kinem["accelerations"]

    with open(os.path.join(output_dir, "selected_events.json")) as f:
        events_data = json.load(f)

    with open(os.path.join(output_dir, "attribution.json")) as f:
        attr_data = json.load(f)

    with open(os.path.join(output_dir, "result.json")) as f:
        res_data = json.load(f)

    T = positions.shape[0]
    valid_mask = np.abs(positions).sum(axis=(1, 2)) > 1e-4
    valid_frames = np.where(valid_mask)[0]
    max_valid = valid_frames[-1] if len(valid_frames) > 0 else T
    t_axis = np.arange(T)

    # Create figure layout
    # 3 rows:
    # Row 0: Panel A (Pose / Skeleton Representation - 5 keyframe subplots across the top)
    # Row 1: Panel B (Kinematic trajectories: Pos, Vel, Acc) & Panel C (Block-ESG saliency) & Panel D (Selected events)
    # Row 2: Panel E (Body-region attribution) & Panel F (Kinematic-stream attribution + Probability)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig = plt.figure(figsize=(13.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.1, 1.2, 1.1], hspace=0.38, wspace=0.28)

    # =========================================================================
    # Panel A: Pose / Skeleton wireframes (5 sample frames across top)
    # =========================================================================
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.12)
    sample_frames = [valid_frames[int(idx)] for idx in np.linspace(0, len(valid_frames)-1, 5)]

    # Title for Panel A
    fig.text(0.08, 0.965, "(A) Pose / Skeleton Wireframe Sequences (De-identified 33 Landmarks)",
             fontsize=10.5, fontweight="bold", ha="left", va="bottom")

    for i, f_idx in enumerate(sample_frames):
        ax_a = fig.add_subplot(gs_a[0, i])
        pts = positions[f_idx]
        x = pts[:, 0]
        y = -pts[:, 1] # upright display: head positive, feet negative

        for u, v in SKELETON_EDGES:
            ax_a.plot([x[u], x[v]], [y[u], y[v]], color="#1E88E5", linewidth=1.3, alpha=0.8, zorder=2)
        ax_a.scatter(x, y, color="#D81B60", s=12, zorder=3, edgecolors="none")

        ax_a.set_aspect("equal")
        ax_a.set_xlim(-1.2, 1.2)
        ax_a.set_ylim(-1.5, 1.1)
        ax_a.axis("off")
        ax_a.set_title(f"t = {f_idx} ({(f_idx/30.0):.2f}s)", fontsize=8.5, y=0.92)

    # =========================================================================
    # Panel B: Position, Velocity, and Acceleration Kinematics
    # =========================================================================
    ax_b = fig.add_subplot(gs[1, 0])
    pos_mag = np.linalg.norm(positions, axis=-1).mean(axis=1)
    pos_mag[~valid_mask] = np.nan
    vel_mag = np.linalg.norm(velocities, axis=-1).mean(axis=1)
    vel_mag[~valid_mask] = np.nan
    acc_mag = np.linalg.norm(accelerations, axis=-1).mean(axis=1)
    acc_mag[~valid_mask] = np.nan

    # Normalize trajectories for unified comparative plot or multi-line
    ax_b.plot(t_axis[:max_valid+5], pos_mag[:max_valid+5], label="Position (norm)", color="#1976D2", linewidth=1.2)
    ax_b.plot(t_axis[:max_valid+5], vel_mag[:max_valid+5] / 10.0, label="Velocity (scaled/10)", color="#388E3C", linewidth=1.2)
    ax_b.plot(t_axis[:max_valid+5], acc_mag[:max_valid+5] / 20.0, label="Acceleration (scaled/20)", color="#E53935", linewidth=1.2)

    ax_b.set_title("(B) Kinematic Trajectories", fontweight="bold", pad=8)
    ax_b.set_xlabel("Frame index (30 fps)")
    ax_b.set_ylabel("Kinematic magnitude (a.u.)")
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.85)
    ax_b.set_xlim(0, max_valid + 5)
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # =========================================================================
    # Panel C: Block-ESG Temporal Saliency
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 1])
    block_scores = np.array(events_data["all_block_saliency_scores"])
    n_blocks = len(block_scores)
    selected_blocks = [s // events_data["block_size_frames"] for s in events_data["selected_block_starts"]]
    sel_set = set(selected_blocks)

    colors_c = ["#FF8F00" if b in sel_set else "#B0BEC5" for b in range(n_blocks)]
    bars_c = ax_c.bar(np.arange(n_blocks), block_scores, color=colors_c, width=0.72, edgecolor="black", linewidth=0.5)

    ax_c.set_title("(C) Block-ESG Saliency Scores", fontweight="bold", pad=8)
    ax_c.set_xlabel("15-frame block index (0.5 s)")
    ax_c.set_ylabel("Saliency score $S_b$")
    ax_c.set_xlim(-0.8, n_blocks - 0.2)
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Custom legend for selection
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#FF8F00", lw=6),
                    Line2D([0], [0], color="#B0BEC5", lw=6)]
    ax_c.legend(custom_lines, ["Selected (Top-8)", "Unselected"], loc="upper right", framealpha=0.85)

    # =========================================================================
    # Panel D: Selected Temporal Events Over Timeline
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 2])
    block_size = events_data["block_size_frames"]

    # Draw timeline with highlighted blocks
    timeline_y = 0.5
    ax_d.plot([0, max_valid], [timeline_y, timeline_y], color="#78909C", linewidth=2.5, zorder=1)

    # Highlight selected blocks
    # Group contiguous blocks for clean labeling
    sorted_blocks = sorted(selected_blocks)
    runs = []
    curr_run = [sorted_blocks[0]]
    for b in sorted_blocks[1:]:
        if b == curr_run[-1] + 1:
            curr_run.append(b)
        else:
            runs.append(curr_run)
            curr_run = [b]
    runs.append(curr_run)

    for run in runs:
        start = run[0] * block_size
        end = min((run[-1] + 1) * block_size, max_valid)
        rect = Rectangle((start, 0.25), end - start, 0.5,
                         facecolor="#FF8F00", edgecolor="#D84315", linewidth=1.0, alpha=0.85, zorder=3)
        ax_d.add_patch(rect)
        label = f"B{run[0]}" if len(run) == 1 else f"B{run[0]}–B{run[-1]}"
        ax_d.text((start + end)/2, 0.5, label, ha="center", va="center",
                  fontsize=8, fontweight="bold", color="black", zorder=4)

    # Add unselected valid regions in gray
    for b_idx in range(n_blocks):
        if b_idx not in sel_set:
            start = b_idx * block_size
            end = min(start + block_size, max_valid)
            if start < max_valid:
                rect = Rectangle((start, 0.35), end - start, 0.3,
                                 facecolor="#ECEFF1", edgecolor="#CFD8DC", linewidth=0.5, alpha=0.7, zorder=2)
                ax_d.add_patch(rect)

    ax_d.set_title("(D) Selected Contiguous Events (Top-8)", fontweight="bold", pad=8)
    ax_d.set_xlabel("Frame index (30 fps)")
    ax_d.set_yticks([])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_xlim(0, max_valid + 5)
    ax_d.grid(True, axis="x", linestyle="--", alpha=0.4)

    # =========================================================================
    # Panel E: Body-Region Attribution
    # =========================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    body_attr = attr_data["body_region_attribution"]
    regions = ["Head", "Arms", "Torso", "Legs"]
    region_keys = ["head", "arms", "torso", "legs"]
    vals_e = [body_attr[k] for k in region_keys]
    colors_e = ["#8E24AA", "#0288D1", "#00897B", "#F4511E"]

    y_pos = np.arange(len(regions))
    bars_e = ax_e.barh(y_pos, vals_e, color=colors_e, height=0.55, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars_e, vals_e):
        ax_e.text(val + 0.015, bar.get_y() + bar.get_height()/2.0, f"{val*100:.1f}%",
                  va="center", ha="left", fontsize=8.5, fontweight="bold")

    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels(regions)
    ax_e.set_xlim(0, max(vals_e) * 1.35)
    ax_e.set_title("(E) Body-Region Attribution", fontweight="bold", pad=8)
    ax_e.set_xlabel("Normalized Gradient × Input attribution")
    ax_e.grid(True, axis="x", linestyle="--", alpha=0.4)

    # =========================================================================
    # Panel F: Kinematic-Stream Attribution & Calibrated Probability
    # =========================================================================
    gs_f = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1:], width_ratios=[1.1, 1.25], wspace=0.28)

    ax_f1 = fig.add_subplot(gs_f[0, 0])
    kinem_attr = attr_data["kinematic_stream_attribution"]
    streams = ["Position", "Velocity", "Acceleration"]
    stream_keys = ["position", "velocity", "acceleration"]
    vals_f = [kinem_attr[k] for k in stream_keys]
    colors_f = ["#1976D2", "#388E3C", "#E53935"]

    x_pos = np.arange(len(streams))
    bars_f = ax_f1.bar(x_pos, vals_f, color=colors_f, width=0.55, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars_f, vals_f):
        ax_f1.text(bar.get_x() + bar.get_width()/2.0, val + 0.02, f"{val*100:.1f}%",
                   ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_f1.set_xticks(x_pos)
    ax_f1.set_xticklabels(streams)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.set_title("(F1) Kinematic-Stream Attribution", fontweight="bold", pad=8)
    ax_f1.set_ylabel("Stream contribution fraction")
    ax_f1.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Right sub-panel: Diagnostic summary card
    ax_f2 = fig.add_subplot(gs_f[0, 1])
    ax_f2.axis("off")

    p_cal = res_data["calibrated_probability"]
    p_raw = res_data["raw_probability"]
    pred = res_data["prediction"]

    summary_text = (
        "PACE-ASD Research Output\n"
        "------------------------------------\n"
        f"Subject Recording : {clip_id} (De-identified)\n"
        f"Raw Logit / Prob  : {p_raw:.4f}\n"
        f"Calibrated P(ASD) : {p_cal:.4f}\n"
        f"Classification    : {pred} (at threshold 0.50)\n"
        f"Selected Events   : 8 contiguous blocks (120 tokens)\n"
        f"Primary Region    : {regions[np.argmax(vals_e)]} ({max(vals_e)*100:.1f}%)\n"
        f"Primary Kinematic : {streams[np.argmax(vals_f)]} ({max(vals_f)*100:.1f}%)\n"
        "------------------------------------\n"
        "* Research software output only.\n"
        "* Not a validated diagnostic device."
    )

    box_color = "#FFF3E0" if pred == "ASD" else "#E8F5E9"
    border_color = "#E65100" if pred == "ASD" else "#2E7D32"

    fancy_box = FancyBboxPatch((0.05, 0.05), 0.90, 0.88, boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=box_color, edgecolor=border_color, linewidth=1.5,
                               transform=ax_f2.transAxes, zorder=1)
    ax_f2.add_patch(fancy_box)
    ax_f2.text(0.10, 0.50, summary_text, transform=ax_f2.transAxes,
               fontsize=8.5, fontfamily="monospace", va="center", ha="left", zorder=2)
    ax_f2.set_title("(F2) Calibrated Prediction & Metadata", fontweight="bold", pad=8)

    # Save to multiple formats and destinations
    destinations = [
        os.path.join(output_dir, "Figure_6.png"),
        os.path.join(output_dir, "Figure_6.pdf"),
        "bmc submission/Figure_6.png",
        "bmc submission/Figure_6.pdf",
        "bmc submission/figures/Figure_6.png",
        "bmc submission/figures/Figure_6.pdf",
    ]

    for dest in destinations:
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if dest.endswith(".pdf"):
            fig.savefig(dest, format="pdf", bbox_inches="tight")
        else:
            fig.savefig(dest, format="png", dpi=300, bbox_inches="tight")
        print(f"Saved: {dest}")

    plt.close(fig)

if __name__ == "__main__":
    generate_figure()
