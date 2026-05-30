# scripts/extract_metadata_train.py

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

def main():
    parser = argparse.ArgumentParser(description="Batch extract metadata features from tar.gz or directories.")
    parser.add_argument("--closed-tar", type=str, default=None,
                        help="Path to closed_world_split.tar.gz")
    parser.add_argument("--open-tar", type=str, default=None,
                        help="Path to open_world_split.tar.gz")
    parser.add_argument("--closed-dir", type=str, default=None,
                        help="Path to extracted closed_world directory")
    parser.add_argument("--open-dir", type=str, default=None,
                        help="Path to extracted open_world directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument("--split", type=str, default="training_data",
                        help="Split name to process (training_data, validation_data, test_data)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files processed per category (for testing)")
    
    args = parser.parse_args()

    # Validate arguments
    use_dir_mode = args.closed_dir is not None or args.open_dir is not None
    if use_dir_mode:
        if args.closed_dir is None or args.open_dir is None:
            print("Error: Both --closed-dir and --open-dir must be provided for Directory Mode.")
            sys.exit(1)
    else:
        if args.closed_tar is None or args.open_tar is None:
            print("Error: Must provide either both directories (--closed-dir/--open-dir) or both tar files (--closed-tar/--open-tar).")
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    mode_str = "Directory Mode" if use_dir_mode else "Tar Mode"
    print(f"=== Starting Full Feature Extraction ({mode_str}) ===")
    print(f"Target Split: {args.split}")
    
    start_total_time = time.time()

    all_results = []
    all_failures = []
    timing_anomalies = []
    success_count = 0
    fail_count = 0
    sample_id = 0
    runtimes = []

    # Quality aggregates
    total_nan_count = 0
    total_inf_count = 0
    total_raw_negative_ts_diff = 0

    if use_dir_mode:
        # Prepare targets
        targets = [
            {"dir_path": args.closed_dir, "world": "closed"},
            {"dir_path": args.open_dir, "world": "open"}
        ]

        for target in targets:
            dir_path = target["dir_path"]
            world = target["world"]

            if not os.path.exists(dir_path):
                print(f"Error: Directory path not found: {dir_path}")
                sys.exit(1)

            print(f"\nScanning local files from {world}-world directory: {dir_path}...")
            csv_paths = []
            for root, _, files in os.walk(dir_path):
                for f in files:
                    if f.endswith(".csv"):
                        csv_paths.append(os.path.join(root, f))
            
            # Deterministic alphabetical ordering
            csv_paths = [p.replace("\\", "/") for p in csv_paths]
            csv_paths.sort()

            if args.limit:
                csv_paths = csv_paths[:args.limit]

            total_files = len(csv_paths)
            print(f"Found {total_files} trace CSV files in directory.")

            t_start = time.time()
            for idx, filepath in enumerate(csv_paths, 1):
                parts = filepath.split('/')
                trace_name = parts[-1]
                
                if world == "closed":
                    site_name = parts[-2] if len(parts) >= 2 else "unknown"
                    try:
                        label = int(site_name.split('_')[0])
                    except (ValueError, IndexError):
                        label = -1
                else:
                    site_name = "open_world"
                    label = 100

                # Print progress every 5000 files
                if idx % 5000 == 0 or idx == total_files:
                    elapsed = time.time() - t_start
                    rate = idx / elapsed if elapsed > 0 else 0
                    eta = (total_files - idx) / rate if rate > 0 else 0
                    print(f"  [{world}] Processed {idx}/{total_files} files. Rate: {rate:.1f} files/s. ETA: {eta/60:.1f}m")

                t0 = time.time()
                try:
                    df = read_trace_csv(filepath)
                    features, quality = extract_features_and_quality(df)
                    dt = time.time() - t0
                    runtimes.append(dt)

                    total_nan_count += quality["nan_count"]
                    total_inf_count += quality["inf_count"]
                    total_raw_negative_ts_diff += quality["raw_negative_iat_count"]

                    if quality["raw_negative_iat_count"] > 0:
                        timestamps = df["timestamp"].to_numpy(dtype=float)
                        raw_diffs = np.diff(timestamps) if len(timestamps) > 1 else np.array([])
                        neg_diffs = raw_diffs[raw_diffs < 0.0]
                        min_neg = float(np.min(neg_diffs)) if len(neg_diffs) > 0 else 0.0
                        timing_anomalies.append({
                            "tar_file": "extracted_directory",
                            "member_path": filepath,
                            "raw_negative_diff_count": quality["raw_negative_iat_count"],
                            "min_negative_diff": min_neg
                        })

                    row = {
                        "sample_id": sample_id,
                        "split": args.split,
                        "world": world,
                        "label": label,
                        "site_name": site_name,
                        "trace_name": trace_name,
                        "tar_file": "extracted_directory",
                        "member_path": filepath
                    }
                    for split_cand in ["training_data", "validation_data", "test_data"]:
                        if f"/{split_cand}/" in filepath or filepath.startswith(f"{split_cand}/"):
                            row["split"] = split_cand
                            break

                    row.update(features)
                    all_results.append(row)
                    sample_id += 1
                    success_count += 1
                except Exception as e:
                    fail_count += 1
                    all_failures.append({
                        "tar_file": "extracted_directory",
                        "member_path": filepath,
                        "error": str(e)
                    })

    else:
        # Fallback Tar Mode
        targets = [
            {"tar_path": args.closed_tar, "world": "closed"},
            {"tar_path": args.open_tar, "world": "open"}
        ]

        for target in targets:
            tar_path = target["tar_path"]
            world = target["world"]
            tar_filename = os.path.basename(tar_path)

            if not os.path.exists(tar_path):
                print(f"Error: Tar file not found: {tar_path}")
                sys.exit(1)

            print(f"\nProcessing {world}-world traces from tar: {tar_filename}...")
            with tarfile.open(tar_path, "r:gz") as tar:
                prefix = f"{args.split}/{world}_world/"
                members = []
                for m in tar.getmembers():
                    if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".csv"):
                        members.append(m)
                
                members.sort(key=lambda x: x.name)
                if args.limit:
                    selected_members = members[:args.limit]
                else:
                    selected_members = members
                
                total_files = len(selected_members)
                print(f"Found {total_files} matching traces in tar file.")

                t_start = time.time()
                for idx, member in enumerate(selected_members, 1):
                    label, site_name, trace_name = get_label_and_site_name(member.name, world)
                    
                    if idx % 1000 == 0 or idx == total_files:
                        elapsed = time.time() - t_start
                        rate = idx / elapsed if elapsed > 0 else 0
                        eta = (total_files - idx) / rate if rate > 0 else 0
                        print(f"  [{world}] Processed {idx}/{total_files} files. Rate: {rate:.1f} files/s. ETA: {eta/60:.1f}m")

                    f_obj = tar.extractfile(member)
                    if f_obj is None:
                        fail_count += 1
                        all_failures.append({
                            "tar_file": tar_filename,
                            "member_path": member.name,
                            "error": "Failed to extract file object"
                        })
                        continue

                    t0 = time.time()
                    try:
                        df = read_trace_csv(f_obj)
                        features, quality = extract_features_and_quality(df)
                        dt = time.time() - t0
                        runtimes.append(dt)

                        total_nan_count += quality["nan_count"]
                        total_inf_count += quality["inf_count"]
                        total_raw_negative_ts_diff += quality["raw_negative_iat_count"]

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
                    except Exception as e:
                        fail_count += 1
                        all_failures.append({
                            "tar_file": tar_filename,
                            "member_path": member.name,
                            "error": str(e)
                        })

    total_runtime = time.time() - start_total_time
    avg_runtime = np.mean(runtimes) if len(runtimes) > 0 else 0.0

    print(f"\nBatch processing complete. Success: {success_count}, Fail: {fail_count}")

    # Ensure we actually successfully extracted some traces
    if len(all_results) == 0:
        print("Error: No traces were successfully processed.")
        sys.exit(1)

    # 1. Save main CSV
    df_features = pd.DataFrame(all_results)
    meta_cols = ["sample_id", "split", "world", "label", "site_name", "trace_name", "tar_file", "member_path"]
    df_features = df_features[meta_cols + FEATURE_NAMES]
    
    metadata_csv_path = os.path.join(args.output_dir, "metadata_feature_bank_train_all74.csv")
    print(f"Saving features to: {metadata_csv_path}")
    df_features.to_csv(metadata_csv_path, index=False)

    # Get file size
    file_size_bytes = os.path.getsize(metadata_csv_path) if os.path.exists(metadata_csv_path) else 0
    file_size_mb = file_size_bytes / (1024 * 1024)

    # 2. Save timing anomaly report
    timing_anomaly_csv_path = os.path.join(args.output_dir, "timing_anomaly_report.csv")
    df_anomaly = pd.DataFrame(timing_anomalies)
    if len(df_anomaly) == 0:
        df_anomaly = pd.DataFrame(columns=["tar_file", "member_path", "raw_negative_diff_count", "min_negative_diff"])
    df_anomaly.to_csv(timing_anomaly_csv_path, index=False)
    print(f"Saved {timing_anomaly_csv_path}")

    # 3. Save failed traces if any
    if len(all_failures) > 0:
        failures_csv = os.path.join(args.output_dir, "failed_traces.csv")
        pd.DataFrame(all_failures).to_csv(failures_csv, index=False)
        print(f"Saved failed traces to: {failures_csv}")

    # 4. Save feature names
    feature_names_json = os.path.join(args.output_dir, "feature_names_74.json")
    with open(feature_names_json, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=4)
    print(f"Saved {feature_names_json}")

    # 5. Generate extraction_quality_report.txt
    expected_rows = 133000
    report_lines = [
        "======================================================================",
        "FULL DATASET FEATURE EXTRACTION QUALITY REPORT",
        "======================================================================",
        f"Execution Mode            : {mode_str}",
        f"Target Split              : {args.split}",
        f"Generation Time           : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "----------------------------------------------------------------------",
        "1. GENERAL RUNTIME METRICS",
        "----------------------------------------------------------------------",
        f"Total Traces Success      : {success_count}",
        f"Total Traces Failed       : {fail_count}",
        f"Expected Rows             : {expected_rows}",
        f"Actual Rows               : {len(df_features)}",
        f"Total Runtime (s)         : {total_runtime:.6f}",
        f"Average Runtime (s/trace) : {avg_runtime:.6f}",
        f"Output CSV File Size      : {file_size_mb:.2f} MB",
        "",
        "----------------------------------------------------------------------",
        "2. DATA QUALITY SUMMARY",
        "----------------------------------------------------------------------",
        f"Feature Count             : {len(FEATURE_NAMES)} (verified)",
        f"Total NaN Count           : {total_nan_count}",
        f"Total Inf Count           : {total_inf_count}",
        f"Total Timing Anomaly Traces: {len(timing_anomalies)}",
        f"Total Raw Negative Diffs  : {total_raw_negative_ts_diff}",
        "======================================================================"
    ]

    report_content = "\n".join(report_lines)
    report_txt_path = os.path.join(args.output_dir, "extraction_quality_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved quality report to: {report_txt_path}")

    print(f"\n=== Extraction complete in {total_time/60:.2f} minutes ===")

if __name__ == "__main__":
    main()
