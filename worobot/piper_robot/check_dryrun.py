#!/usr/bin/env python3
"""
Dry-run dataset sanity check.

Usage:
    python3 worobot/piper_robot/check_dryrun.py /home/woan/data/heart666888_test_dryrun

Checks:
  1) Episode count + file presence
  2) Action / state shape == 14, dtype float
  3) action != state (master ≠ slave puppet — most important fix)
  4) Gripper dim 6 / dim 13 reach close-command magnitude (~0.06-0.10)
  5) Three videos exist per episode (head/left/right) at 30fps 480x640
  6) Joint range comparison vs fold_clothes_data167 (sanity-check left/right
     not flipped relative to fold)
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main(root_str: str):
    root = Path(root_str)
    if not root.exists():
        sys.exit(f"dataset root not found: {root}")

    # --- 1) Episode files ---
    parquet_dir = root / "data" / "chunk-000"
    parquets = sorted(parquet_dir.glob("episode_*.parquet"))
    print(f"\n[1] Found {len(parquets)} parquet episode(s) under {parquet_dir}")
    if len(parquets) == 0:
        sys.exit("no parquet files - did recording save anything?")

    # --- meta ---
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.load(open(info_path))
        print(f"    fps={info.get('fps')}, total_episodes={info.get('total_episodes')}, "
              f"total_frames={info.get('total_frames')}, robot_type={info.get('robot_type')}")
        cams_meta = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
        print(f"    cameras in meta: {cams_meta}")
        action_names = info.get("features", {}).get("action", {}).get("names")
        state_names = info.get("features", {}).get("observation.state", {}).get("names")
        print(f"    action names: {action_names}")
        print(f"    state  names: {state_names}")

    # --- 2/3/4) Per-episode action/state checks ---
    print("\n[2-4] Action / state checks:")
    g6_max_overall, g13_max_overall = 0.0, 0.0
    a_minus_s_avg = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        a = np.stack(df["action"].values)
        s = np.stack(df["observation.state"].values)
        assert a.shape[1] == 14, f"{pq.name}: action dim {a.shape[1]} != 14"
        assert s.shape[1] == 14, f"{pq.name}: state dim {s.shape[1]} != 14"
        diff = np.abs(a - s).mean()
        a_minus_s_avg.append(diff)
        g6_max_overall = max(g6_max_overall, a[:, 6].max())
        g13_max_overall = max(g13_max_overall, a[:, 13].max())
        print(f"    {pq.name}: T={len(df):4d}, "
              f"|action-state|.mean={diff:.4f}, "
              f"a[6].max={a[:,6].max():.4f}, a[13].max={a[:,13].max():.4f}")

    avg_diff = np.mean(a_minus_s_avg)
    print(f"\n    >>> avg |action - state| over all eps = {avg_diff:.4f}")
    if avg_diff < 0.001:
        print("    !!! action == state !!! master command not separated from slave qpos")
        print("        (this is the kai0 bug we are trying to avoid)")
    elif avg_diff < 0.005:
        print("    ~ action and state very close; teleop lag is small but present")
    else:
        print("    OK: action != state (master command captured separately)")

    print(f"\n    >>> gripper max  dim6 ={g6_max_overall:.4f}  dim13={g13_max_overall:.4f}")
    if max(g6_max_overall, g13_max_overall) < 0.05:
        print("    !!! gripper never close-commanded; either you didn't grasp during recording")
        print("        or the gripper command is being clipped somewhere")
    elif max(g6_max_overall, g13_max_overall) < 0.07:
        print("    ~ partial close commands seen; ok for soft objects, may not grip slippery tube")
    else:
        print("    OK: full close commands (>=0.07) present in data")

    # --- 5) Videos ---
    print("\n[5] Video files:")
    for cam in ("head", "left", "right"):
        vid_dir = root / "videos" / "chunk-000" / f"observation.images.{cam}"
        if not vid_dir.exists():
            print(f"    !! missing dir: {vid_dir}")
            continue
        mp4s = sorted(vid_dir.glob("*.mp4"))
        print(f"    {cam}: {len(mp4s)} mp4 files")
        if mp4s:
            # quick frame count check via torchcodec (already in deps)
            try:
                from torchcodec.decoders import VideoDecoder
                d = VideoDecoder(str(mp4s[0]))
                print(f"        {mp4s[0].name}: {d.metadata.num_frames} frames @ "
                      f"{d.metadata.average_fps:.1f}fps, "
                      f"{d.metadata.width}x{d.metadata.height}")
            except Exception as e:
                print(f"        torchcodec probe failed: {e}")

    # --- 6) Compare ranges with fold_clothes (if path provided) ---
    fold_root_candidates = [
        Path("/data/user/yzhu765/vlash-real/data/fold_clothes_data167"),
    ]
    fold_root = next((p for p in fold_root_candidates if p.exists()), None)
    if fold_root is None:
        print("\n[6] fold reference dataset not on this machine; skipping range comparison.")
        return

    print(f"\n[6] Joint range vs fold_clothes_data167 ({fold_root}):")
    fold_pq = sorted((fold_root / "data" / "chunk-000").glob("*.parquet"))[:5]
    a_fold = np.concatenate([np.stack(pd.read_parquet(p)["action"].values) for p in fold_pq])
    a_new = np.concatenate([np.stack(pd.read_parquet(p)["action"].values) for p in parquets])
    print(f"    {'dim':>3s} | {'new min..max':>22s} | {'fold min..max':>22s}")
    for d in range(14):
        print(f"    {d:>3d} | {a_new[:,d].min():>+8.3f} ..{a_new[:,d].max():>+8.3f} "
              f"| {a_fold[:,d].min():>+8.3f} ..{a_fold[:,d].max():>+8.3f}")
    print("    >>> dim 0..6 should be LEFT arm, dim 7..13 RIGHT arm.")
    print("        signs and magnitudes per dim should be in the same regime as fold.")
    print("        e.g. dim 0 (left shoulder yaw) often mostly negative for both;")
    print("        if new dim 0 is mostly positive but fold dim 0 is mostly negative,")
    print("        the left/right halves of action may be flipped at recording time.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 check_dryrun.py <dataset_root>")
    main(sys.argv[1])
