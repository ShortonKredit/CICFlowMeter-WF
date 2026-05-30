# scripts/extract_metadata_train.py

import os
import sys
import argparse
import json
import time
import pandas as pd

# Add parent directory to sys.path to allow imports from cicflowmeter_wf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicflowmeter_wf import process_tar_archive, process_directory
from cicflowmeter_wf.feature_names import FEATURE_NAMES

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
    print(f"=== Starting Metadata Feature Extraction ({mode_str}) for split: {args.split} ===")
    if args.limit:
        print(f"[Verification Mode] Limiting processing to: {args.limit} files.")
    
    start_total_time = time.time()

    all_results = []
    all_failures = []
    sample_id = 0

    if use_dir_mode:
        # 1. Process closed-world from directory
        print(f"\n[1/2] Processing closed-world from directory: {args.closed_dir}")
        closed_results, closed_failures, sample_id = process_directory(
            args.closed_dir, "closed", sample_id, limit=args.limit
        )
        all_results.extend(closed_results)
        all_failures.extend(closed_failures)
        print(f"Closed-world complete: Extracted {len(closed_results)} rows successfully, failed {len(closed_failures)}")

        # 2. Process open-world from directory
        print(f"\n[2/2] Processing open-world from directory: {args.open_dir}")
        open_results, open_failures, sample_id = process_directory(
            args.open_dir, "open", sample_id, limit=args.limit
        )
        all_results.extend(open_results)
        all_failures.extend(open_failures)
        print(f"Open-world complete: Extracted {len(open_results)} rows successfully, failed {len(open_failures)}")

    else:
        # 1. Process closed-world from tar
        if os.path.exists(args.closed_tar):
            print(f"\n[1/2] Processing closed-world from tar: {args.closed_tar}")
            closed_results, closed_failures, sample_id = process_tar_archive(
                args.closed_tar, args.split, "closed", sample_id, limit=args.limit
            )
            all_results.extend(closed_results)
            all_failures.extend(closed_failures)
            print(f"Closed-world complete: Extracted {len(closed_results)} rows successfully, failed {len(closed_failures)}")
        else:
            print(f"\n[Warning] Closed-world tar not found: {args.closed_tar}. Skipping...")

        # 2. Process open-world from tar
        if os.path.exists(args.open_tar):
            print(f"\n[2/2] Processing open-world from tar: {args.open_tar}")
            open_results, open_failures, sample_id = process_tar_archive(
                args.open_tar, args.split, "open", sample_id, limit=args.limit
            )
            all_results.extend(open_results)
            all_failures.extend(open_failures)
            print(f"Open-world complete: Extracted {len(open_results)} rows successfully, failed {len(open_failures)}")
        else:
            print(f"\n[Warning] Open-world tar not found: {args.open_tar}. Skipping...")

    total_time = time.time() - start_total_time

    # 3. Create DataFrame and export results
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # Ensure column ordering matches specification
        meta_cols = ["sample_id", "split", "world", "label", "site_name", "trace_name", "tar_file", "member_path"]
        col_order = meta_cols + FEATURE_NAMES
        df = df[col_order]

        output_csv = os.path.join(args.output_dir, f"metadata_feature_bank_train.csv")
        print(f"\nSaving {len(df)} rows to: {output_csv}")
        df.to_csv(output_csv, index=False)
    else:
        print("\n[Warning] No successfully extracted features to save.")

    # 4. Save failed traces if any
    if len(all_failures) > 0:
        failures_csv = os.path.join(args.output_dir, "failed_traces.csv")
        pd.DataFrame(all_failures).to_csv(failures_csv, index=False)
        print(f"Saved {len(all_failures)} failures list to: {failures_csv}")

    # 5. Export feature names
    feature_names_json = os.path.join(args.output_dir, "feature_names_74.json")
    with open(feature_names_json, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=4)

    # 6. Save log file
    log_data = {
        "mode": mode_str,
        "split": args.split,
        "total_extracted": len(all_results),
        "total_failed": len(all_failures),
        "total_time_seconds": total_time,
        "closed_world_count": sum(1 for r in all_results if r["world"] == "closed"),
        "open_world_count": sum(1 for r in all_results if r["world"] == "open"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    log_json = os.path.join(args.output_dir, "extraction_log.json")
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n=== Extraction complete in {total_time/60:.2f} minutes ===")
    print(log_data)

if __name__ == "__main__":
    main()
