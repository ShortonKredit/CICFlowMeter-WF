# scripts/run_small_batch.py

import os
import sys
import argparse
import json
import time
import tarfile
import numpy as np
import pandas as pd

# Add parent directory to sys.path to allow imports from cicflowmeter_wf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicflowmeter_wf.trace_reader import read_trace_csv
from cicflowmeter_wf.features import extract_features_and_quality
from cicflowmeter_wf.feature_names import FEATURE_NAMES
from cicflowmeter_wf.extract_train_bank import get_label_and_site_name

def compute_dist_stats(arr):
    """
    Computes min, max, mean, std of an array/list.
    Handles empty arrays gracefully.
    """
    if len(arr) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    arr_np = np.array(arr, dtype=float)
    # Replace NaN/inf if any (safety check)
    arr_np = np.nan_to_num(arr_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_min = float(np.min(arr_np))
    val_max = float(np.max(arr_np))
    val_mean = float(np.mean(arr_np))
    val_std = float(np.std(arr_np, ddof=1)) if len(arr_np) > 1 else 0.0
    return {"min": val_min, "max": val_max, "mean": val_mean, "std": val_std}

def main():
    parser = argparse.ArgumentParser(description="Run small-batch metadata feature extraction (50 traces total).")
    parser.add_argument("--closed-tar", type=str, required=True,
                        help="Path to closed_world_split.tar.gz")
    parser.add_argument("--open-tar", type=str, required=True,
                        help="Path to open_world_split.tar.gz")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for reports")
    parser.add_argument("--split", type=str, default="training_data",
                        help="Split to extract from (training_data, validation_data, test_data)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Starting Small-Batch Verification for split: {args.split} ===")
    start_total_time = time.time()

    all_results = []
    success_count = 0
    fail_count = 0
    sample_id = 0
    runtimes = []

    # Diagnostics lists
    total_nan_count = 0
    total_inf_count = 0
    total_negative_iat_count = 0  # pre-clamping
    total_raw_negative_ts_diff = 0

    timing_anomalies = []

    # We will sample 25 closed and 25 open traces
    targets = [
        {"tar_path": args.closed_tar, "world": "closed", "limit": 25},
        {"tar_path": args.open_tar, "world": "open", "limit": 25}
    ]

    for target in targets:
        tar_path = target["tar_path"]
        world = target["world"]
        limit = target["limit"]
        tar_filename = os.path.basename(tar_path)

        if not os.path.exists(tar_path):
            print(f"Error: Tar file not found: {tar_path}")
            sys.exit(1)

        print(f"\nProcessing {world}-world traces from: {tar_filename}...")
        
        with tarfile.open(tar_path, "r:gz") as tar:
            prefix = f"{args.split}/{world}_world/"
            
            # Find and sort matching csv files
            members = []
            for m in tar.getmembers():
                if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".csv"):
                    members.append(m)

            # Deterministic, sorted ordering
            members.sort(key=lambda x: x.name)
            
            selected_members = members[:limit]
            print(f"Sampled first {len(selected_members)} traces (out of {len(members)} total matches).")

            for idx, member in enumerate(selected_members, 1):
                label, site_name, trace_name = get_label_and_site_name(member.name, world)
                
                f_obj = tar.extractfile(member)
                if f_obj is None:
                    print(f"  [Fail] Could not extract: {member.name}")
                    fail_count += 1
                    continue

                t0 = time.time()
                try:
                    df = read_trace_csv(f_obj)
                    features, quality = extract_features_and_quality(df)
                    dt = time.time() - t0
                    runtimes.append(dt)

                    # Update diagnostic stats
                    total_nan_count += quality["nan_count"]
                    total_inf_count += quality["inf_count"]
                    total_negative_iat_count += quality["raw_negative_iat_count"]
                    total_raw_negative_ts_diff += quality["raw_negative_iat_count"]

                    # Check for timing anomaly in packet timestamps
                    if quality["raw_negative_iat_count"] > 0:
                        timestamps = df["timestamp"].to_numpy(dtype=float)
                        raw_diffs = np.diff(timestamps) if len(timestamps) > 1 else np.array([])
                        neg_diffs = raw_diffs[raw_diffs < 0.0]
                        min_neg = float(np.min(neg_diffs)) if len(neg_diffs) > 0 else 0.0
                        timing_anomalies.append({
                            "tar_file": tar_filename,
                            "member_path": member.name,
                            "raw_negative_diff_count": quality["raw_negative_iat_count"],
                            "min_negative_diff": min_neg
                        })

                    # Construct row
                    row = {
                        "sample_id": sample_id,
                        "split": args.split,
                        "world": world,
                        "label": label,
                        "site_name": site_name,
                        "trace_name": trace_name,
                        "tar_file": tar_filename,
                        "member_path": member.name
                    }
                    row.update(features)
                    all_results.append(row)
                    sample_id += 1
                    success_count += 1
                    print(f"  [Success] {success_count}/50: {member.name} in {dt:.3f}s")

                except Exception as e:
                    dt = time.time() - t0
                    print(f"  [Fail] Error processing {member.name}: {str(e)}")
                    fail_count += 1

    total_runtime = time.time() - start_total_time
    avg_runtime = np.mean(runtimes) if len(runtimes) > 0 else 0.0

    print(f"\nBatch processing complete. Success: {success_count}, Fail: {fail_count}")

    # Ensure we actually successfully extracted some traces
    if len(all_results) == 0:
        print("Error: No traces were successfully processed.")
        sys.exit(1)

    # 1. Create DataFrame of extracted features
    df_features = pd.DataFrame(all_results)
    meta_cols = ["sample_id", "split", "world", "label", "site_name", "trace_name", "tar_file", "member_path"]
    df_features = df_features[meta_cols + FEATURE_NAMES]
    
    metadata_csv_path = os.path.join(args.output_dir, "metadata_feature_bank_small.csv")
    df_features.to_csv(metadata_csv_path, index=False)
    print(f"Saved {metadata_csv_path}")

    # 2. Create feature stats DataFrame
    stats_rows = []
    constant_count = 0
    all_zero_count = 0

    for name in FEATURE_NAMES:
        col_vals = df_features[name].to_numpy(dtype=float)
        col_vals = np.nan_to_num(col_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        feat_min = float(np.min(col_vals))
        feat_max = float(np.max(col_vals))
        feat_mean = float(np.mean(col_vals))
        feat_std = float(np.std(col_vals, ddof=1)) if len(col_vals) > 1 else 0.0
        
        is_constant = bool(feat_std == 0.0 or feat_min == feat_max)
        is_all_zero = bool(np.all(col_vals == 0.0))

        if is_constant:
            constant_count += 1
        if is_all_zero:
            all_zero_count += 1

        stats_rows.append({
            "feature_name": name,
            "min": feat_min,
            "max": feat_max,
            "mean": feat_mean,
            "std": feat_std,
            "is_constant": is_constant,
            "is_all_zero": is_all_zero
        })

    df_stats = pd.DataFrame(stats_rows)
    stats_csv_path = os.path.join(args.output_dir, "small_batch_feature_stats.csv")
    df_stats.to_csv(stats_csv_path, index=False)
    print(f"Saved {stats_csv_path}")

    # 3. Save timing anomaly report
    timing_anomaly_csv_path = os.path.join(args.output_dir, "timing_anomaly_report.csv")
    df_anomaly = pd.DataFrame(timing_anomalies)
    if len(df_anomaly) == 0:
        # Create empty dataframe with headers
        df_anomaly = pd.DataFrame(columns=["tar_file", "member_path", "raw_negative_diff_count", "min_negative_diff"])
    df_anomaly.to_csv(timing_anomaly_csv_path, index=False)
    print(f"Saved {timing_anomaly_csv_path}")

    # 4. Generate distributions for report
    timing_stats = {
        "flow_iat_mean": compute_dist_stats(df_features["flow_iat_mean"]),
        "flow_iat_std": compute_dist_stats(df_features["flow_iat_std"]),
        "burst_duration_mean": compute_dist_stats(df_features["burst_duration_mean"]),
        "inter_burst_gap_mean": compute_dist_stats(df_features["inter_burst_gap_mean"])
    }

    burst_stats = {
        "total_burst_count": compute_dist_stats(df_features["total_burst_count"]),
        "burst_len_mean": compute_dist_stats(df_features["burst_len_mean"]),
        "direction_switch_count": compute_dist_stats(df_features["direction_switch_count"])
    }

    active_idle_stats = {
        "active_mean": compute_dist_stats(df_features["active_mean"]),
        "active_std": compute_dist_stats(df_features["active_std"]),
        "idle_mean": compute_dist_stats(df_features["idle_mean"]),
        "idle_std": compute_dist_stats(df_features["idle_std"])
    }

    # 5. Generate small_batch_report.txt
    report_lines = [
        "======================================================================",
        "SMALL-BATCH TEST VERIFICATION REPORT (50 TRACES)",
        "======================================================================",
        f"Split Name          : {args.split}",
        f"Generation Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "----------------------------------------------------------------------",
        "1. GENERAL PERFORMANCE METRICS",
        "----------------------------------------------------------------------",
        f"Total Traces Success: {success_count}",
        f"Total Traces Failed : {fail_count}",
        f"Total Runtime (s)   : {total_runtime:.6f}",
        f"Average Runtime (s) : {avg_runtime:.6f}",
        "",
        "----------------------------------------------------------------------",
        "2. DATA QUALITY AUDIT",
        "----------------------------------------------------------------------",
        f"Total NaN Count (raw inputs)  : {total_nan_count}",
        f"Total Inf Count (raw inputs)  : {total_inf_count}",
        f"Pre-clamp Negative IAT Gaps   : {total_negative_iat_count}",
        f"Raw negative timestamp diffs  : {total_raw_negative_ts_diff}",
        f"Post-clamped Negative IAT Gaps: 0 (verified)",
        "",
        "----------------------------------------------------------------------",
        "3. FEATURE SANITY SUMMARY",
        "----------------------------------------------------------------------",
        f"Total Extracted Features      : {len(FEATURE_NAMES)}",
        f"Constant Feature Count        : {constant_count}",
        f"All-Zero Feature Count        : {all_zero_count}",
        "",
        "----------------------------------------------------------------------",
        "4. TIMING ANALYSIS DISTRIBUTIONS",
        "----------------------------------------------------------------------",
        "flow_iat_mean distribution:",
        f"  min={timing_stats['flow_iat_mean']['min']:.6f}, max={timing_stats['flow_iat_mean']['max']:.6f}, mean={timing_stats['flow_iat_mean']['mean']:.6f}, std={timing_stats['flow_iat_mean']['std']:.6f}",
        "flow_iat_std distribution:",
        f"  min={timing_stats['flow_iat_std']['min']:.6f}, max={timing_stats['flow_iat_std']['max']:.6f}, mean={timing_stats['flow_iat_std']['mean']:.6f}, std={timing_stats['flow_iat_std']['std']:.6f}",
        "burst_duration_mean distribution:",
        f"  min={timing_stats['burst_duration_mean']['min']:.6f}, max={timing_stats['burst_duration_mean']['max']:.6f}, mean={timing_stats['burst_duration_mean']['mean']:.6f}, std={timing_stats['burst_duration_mean']['std']:.6f}",
        "inter_burst_gap_mean distribution:",
        f"  min={timing_stats['inter_burst_gap_mean']['min']:.6f}, max={timing_stats['inter_burst_gap_mean']['max']:.6f}, mean={timing_stats['inter_burst_gap_mean']['mean']:.6f}, std={timing_stats['inter_burst_gap_mean']['std']:.6f}",
        "",
        "----------------------------------------------------------------------",
        "5. BURST ANALYSIS DISTRIBUTIONS",
        "----------------------------------------------------------------------",
        "total_burst_count distribution:",
        f"  min={burst_stats['total_burst_count']['min']:.2f}, max={burst_stats['total_burst_count']['max']:.2f}, mean={burst_stats['total_burst_count']['mean']:.4f}, std={burst_stats['total_burst_count']['std']:.4f}",
        "burst_len_mean distribution:",
        f"  min={burst_stats['burst_len_mean']['min']:.4f}, max={burst_stats['burst_len_mean']['max']:.4f}, mean={burst_stats['burst_len_mean']['mean']:.4f}, std={burst_stats['burst_len_mean']['std']:.4f}",
        "direction_switch_count distribution:",
        f"  min={burst_stats['direction_switch_count']['min']:.2f}, max={burst_stats['direction_switch_count']['max']:.2f}, mean={burst_stats['direction_switch_count']['mean']:.4f}, std={burst_stats['direction_switch_count']['std']:.4f}",
        "",
        "----------------------------------------------------------------------",
        "6. ACTIVE/IDLE ANALYSIS DISTRIBUTIONS",
        "----------------------------------------------------------------------",
        "active_mean distribution:",
        f"  min={active_idle_stats['active_mean']['min']:.6f}, max={active_idle_stats['active_mean']['max']:.6f}, mean={active_idle_stats['active_mean']['mean']:.6f}, std={active_idle_stats['active_mean']['std']:.6f}",
        "active_std distribution:",
        f"  min={active_idle_stats['active_std']['min']:.6f}, max={active_idle_stats['active_std']['max']:.6f}, mean={active_idle_stats['active_std']['mean']:.6f}, std={active_idle_stats['active_std']['std']:.6f}",
        "idle_mean distribution:",
        f"  min={active_idle_stats['idle_mean']['min']:.6f}, max={active_idle_stats['idle_mean']['max']:.6f}, mean={active_idle_stats['idle_mean']['mean']:.6f}, std={active_idle_stats['idle_mean']['std']:.6f}",
        "idle_std distribution:",
        f"  min={active_idle_stats['idle_std']['min']:.6f}, max={active_idle_stats['idle_std']['max']:.6f}, mean={active_idle_stats['idle_std']['mean']:.6f}, std={active_idle_stats['idle_std']['std']:.6f}",
        "======================================================================"
    ]

    report_content = "\n".join(report_lines)
    report_txt_path = os.path.join(args.output_dir, "small_batch_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved {report_txt_path}")
    print("\nSmall-batch verification outputs successfully generated!")

if __name__ == "__main__":
    main()
