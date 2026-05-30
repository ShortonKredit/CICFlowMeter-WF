# scripts/test_extract_one.py

import os
import sys
import argparse
import numpy as np

# Add parent directory to sys.path to allow imports from cicflowmeter_wf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicflowmeter_wf.trace_reader import read_trace_csv
from cicflowmeter_wf.features import extract_features_from_df
from cicflowmeter_wf.feature_names import FEATURE_NAMES

def main():
    parser = argparse.ArgumentParser(description="Test metadata feature extraction on a single trace CSV.")
    parser.add_argument("--input", type=str, default="sample_data/0_123movies.is_0.csv",
                        help="Path to the input CSV trace file.")
    parser.add_argument("--output", type=str, default="sample_data/0_123movies.is_0_features.csv",
                        help="Path to save the extracted features CSV.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    print(f"Reading trace file: {args.input}")
    df = read_trace_csv(args.input)
    print(f"Loaded DataFrame: {len(df)} rows, columns: {df.columns.tolist()}")

    # Perform extraction
    print("Extracting 74 metadata features...")
    features = extract_features_from_df(df)

    # 1. Assert feature count
    print(f"Number of features extracted: {len(features)}")
    assert len(features) == 74, f"Expected 74 features, but got {len(features)}"

    # 2. Check for NaN/Inf
    nan_features = []
    inf_features = []
    for name, val in features.items():
        if np.isnan(val):
            nan_features.append(name)
        if np.isinf(val):
            inf_features.append(name)

    print(f"NaN features: {nan_features}")
    print(f"Inf features: {inf_features}")
    assert len(nan_features) == 0, f"Features with NaN values detected: {nan_features}"
    assert len(inf_features) == 0, f"Features with Infinite values detected: {inf_features}"

    # 3. Check basic invariants
    total_packets = len(df)
    total_fwd = features["total_fwd_packets"]
    total_bwd = features["total_bwd_packets"]
    print(f"Invariant Check 1: total_fwd ({total_fwd}) + total_bwd ({total_bwd}) = {total_fwd + total_bwd} (Expected: {total_packets})")
    assert total_fwd + total_bwd == total_packets, "Total fwd and bwd packets must equal total packet count"

    flow_duration = features["flow_duration"]
    expected_duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    print(f"Invariant Check 2: flow_duration = {flow_duration:.6f}s (Expected: {expected_duration:.6f}s)")
    assert abs(flow_duration - expected_duration) < 1e-5, "Flow duration does not match timestamp difference"

    pkt_len_mean = features["pkt_len_mean"]
    expected_mean = df["length"].mean()
    print(f"Invariant Check 3: pkt_len_mean = {pkt_len_mean:.4f} (Expected: {expected_mean:.4f})")
    assert abs(pkt_len_mean - expected_mean) < 1e-5, "Packet length mean does not match expected mean"

    total_burst_count = features["total_burst_count"]
    print(f"Invariant Check 4: total_burst_count = {total_burst_count}")
    assert total_burst_count >= 1, "Total burst count should be at least 1"

    print("\n--- Features Summary ---")
    for idx, (name, val) in enumerate(features.items(), 1):
        print(f"{idx:02d}. {name:<30}: {val}")

    # Save to file
    output_path = args.output
    if os.path.exists(output_path):
        base, ext = os.path.splitext(output_path)
        counter = 2
        while True:
            candidate = f"{base}_{counter}{ext}"
            if not os.path.exists(candidate):
                output_path = candidate
                break
            counter += 1

    try:
        import pandas as pd
        pd.DataFrame([features]).to_csv(output_path, index=False)
        print(f"\nSaved extracted features to: {output_path}")
    except Exception as e:
        print(f"\n[Warning] Could not save features to {output_path} ({str(e)}).")
        print("This is usually because the file is open in another application (like VS Code or Excel).")

    print("\nAll local unit tests PASSED successfully!")

if __name__ == "__main__":
    main()
