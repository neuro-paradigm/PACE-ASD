"""
PACE-ASD — Dataset Audit (Section 1.1 of Protocol)

Walks D:/dryad, counts subjects per group, verifies every expected
raw video file is present. Writes audit_report.txt.
Exits with code 1 if any regular subject is missing its video.avi.

Usage:
    python src/audit.py --raw_dir "D:/dryad"
    python src/audit.py --raw_dir "D:/dryad" --out audit_report.txt
"""

import argparse
import os
import sys


# ── Expected structure ────────────────────────────────────────────────────────

ASD_DIR     = "Autism/children with ASD"
TD_DIR      = "Typical"
SEVERE_DIR  = "Autism/Severe level of ASD"

REGULAR_GROUPS = {
    "ASD": ASD_DIR,
    "TD":  TD_DIR,
}

N_REGULAR_EXPECTED = {"ASD": 50, "TD": 50}


def audit(raw_dir: str) -> dict:
    """
    Audit the raw dataset directory.

    Returns a dict:
        {
          "subjects": {group: [subject_id, ...]},
          "videos":   {clip_id: abs_path},
          "missing":  [description_str, ...],
          "warnings": [description_str, ...],
        }
    """
    raw_dir = os.path.abspath(raw_dir)
    result = {
        "subjects": {},
        "videos":   {},
        "missing":  [],
        "warnings": [],
    }

    # ── 1. Regular subjects (ASD + TD) ───────────────────────────────────────
    for group, rel_path in REGULAR_GROUPS.items():
        group_dir = os.path.join(raw_dir, rel_path)
        if not os.path.isdir(group_dir):
            result["missing"].append(f"Group directory not found: {group_dir}")
            result["subjects"][group] = []
            continue

        subject_dirs = sorted(
            [d for d in os.listdir(group_dir)
             if os.path.isdir(os.path.join(group_dir, d))],
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
        )
        result["subjects"][group] = subject_dirs

        expected = N_REGULAR_EXPECTED.get(group, 0)
        if len(subject_dirs) != expected:
            result["warnings"].append(
                f"{group}: expected {expected} subjects, found {len(subject_dirs)}"
            )

        for subj in subject_dirs:
            prefix = "asd" if group == "ASD" else "td"
            clip_id = f"{prefix}_{subj}"
            video_dir = os.path.join(group_dir, subj, "video")

            # Try candidate filenames in priority order
            candidates = ["video.avi", "video1.avi"]
            found = None
            for name in candidates:
                p = os.path.join(video_dir, name)
                if os.path.isfile(p):
                    found = p
                    break

            # Fallback: any .avi in video/ that is not Svideo or Tvideo
            if found is None and os.path.isdir(video_dir):
                avis = sorted([
                    f for f in os.listdir(video_dir)
                    if f.lower().endswith(".avi")
                    and not f.lower().startswith("s")
                    and not f.lower().startswith("t")
                ])
                if avis:
                    found = os.path.join(video_dir, avis[0])
                    result["warnings"].append(
                        f"{clip_id}: using fallback '{avis[0]}' instead of video.avi"
                    )

            if found:
                result["videos"][clip_id] = found
            else:
                result["missing"].append(
                    f"Missing video: {os.path.join(video_dir, 'video.avi')}  (clip_id={clip_id})"
                )

    # ── 2. Severe ASD (supplement) ───────────────────────────────────────────
    severe_dir = os.path.join(raw_dir, SEVERE_DIR)
    severe_subjects = []
    if not os.path.isdir(severe_dir):
        result["missing"].append(f"Severe ASD directory not found: {severe_dir}")
    else:
        case_dirs = sorted(
            [d for d in os.listdir(severe_dir)
             if os.path.isdir(os.path.join(severe_dir, d))]
        )
        severe_subjects = case_dirs
        if len(case_dirs) != 9:
            result["warnings"].append(
                f"Severe ASD: expected 9 cases, found {len(case_dirs)}"
            )

        for case in case_dirs:
            case_path = os.path.join(severe_dir, case)
            avis = sorted([
                f for f in os.listdir(case_path)
                if f.lower().endswith(".avi")
            ])
            if not avis:
                result["missing"].append(
                    f"Severe ASD {case}: no .avi files found in {case_path}"
                )
            else:
                for i, avi in enumerate(avis, start=1):
                    clip_id = f"severe_{case}_v{i}"
                    result["videos"][clip_id] = os.path.join(case_path, avi)

    result["subjects"]["SEVERE"] = severe_subjects
    return result


def format_report(raw_dir: str, result: dict) -> str:
    lines = [
        "=" * 70,
        "PACE-ASD Dataset Audit Report",
        f"Raw directory : {raw_dir}",
        "=" * 70,
        "",
        "── Subject Counts ───────────────────────────────────────────────────",
    ]

    total_regular = 0
    for group in ["ASD", "TD", "SEVERE"]:
        n = len(result["subjects"].get(group, []))
        lines.append(f"  {group:<10}: {n} subjects")
        if group != "SEVERE":
            total_regular += n

    total_subjects = total_regular + len(result["subjects"].get("SEVERE", []))
    lines.append(f"  {'TOTAL':<10}: {total_subjects} subjects")
    lines.append("")

    # Video inventory
    regular_vids   = {k: v for k, v in result["videos"].items()
                      if not k.startswith("severe_")}
    severe_vids    = {k: v for k, v in result["videos"].items()
                      if k.startswith("severe_")}

    lines += [
        "── Video Inventory ──────────────────────────────────────────────────",
        f"  Regular (train/val/test) : {len(regular_vids)} clips",
        f"  Severe ASD (supplement)  : {len(severe_vids)} clips",
        f"  TOTAL                    : {len(result['videos'])} clips",
        "",
    ]

    if result["warnings"]:
        lines.append("── Warnings ─────────────────────────────────────────────────────────")
        for w in result["warnings"]:
            lines.append(f"  [WARN] {w}")
        lines.append("")

    if result["missing"]:
        lines.append("── Missing Files (BLOCKING) ─────────────────────────────────────────")
        for m in result["missing"]:
            lines.append(f"  [MISSING] {m}")
        lines.append("")
        lines.append("  ✗  Audit FAILED — resolve missing files before preprocessing.")
    else:
        lines.append("  ✓  Audit PASSED — all expected files present.")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PACE-ASD dataset audit")
    parser.add_argument("--raw_dir", default="D:/dryad",
                        help="Path to the raw Dryad dataset root")
    parser.add_argument("--out", default="audit_report.txt",
                        help="Where to write the audit report")
    args = parser.parse_args()

    result = audit(args.raw_dir)
    report = format_report(args.raw_dir, result)

    print(report)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to: {args.out}")

    if result["missing"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
