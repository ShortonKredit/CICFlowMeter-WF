# cicflowmeter_wf/extract_train_bank.py

import os
import tarfile
import json
import time
import pandas as pd
import numpy as np
from .trace_reader import read_trace_csv
from .features import extract_features_from_df
from .feature_names import FEATURE_NAMES

def get_label_and_site_name(member_name: str, world: str) -> tuple:
    """
    Parses label and site_name from the internal member name.
    
    Closed-world name: training_data/closed_world/000_123movies.is/0.csv
    Open-world name: training_data/open_world/0.csv
    """
    parts = member_name.split('/')
    if world == "closed":
        # parts[0] = split (e.g. training_data)
        # parts[1] = closed_world
        # parts[2] = site_name (e.g. 000_123movies.is)
        # parts[3] = trace file name (e.g. 0.csv)
        if len(parts) >= 4:
            site_name = parts[2]
            trace_name = parts[3]
            try:
                label = int(site_name.split('_')[0])
            except (ValueError, IndexError):
                label = -1
            return label, site_name, trace_name
    else:
        # parts[0] = split
        # parts[1] = open_world
        # parts[2] = trace file name (e.g. 0.csv)
        if len(parts) >= 3:
            site_name = "open_world"
            trace_name = parts[2]
            return 100, site_name, trace_name
            
    return -1, "unknown", os.path.basename(member_name)

def process_tar_archive(tar_path: str, split: str, world: str, start_sample_id: int = 0, limit: int = None) -> tuple:
    """
    Processes a single tar.gz split file and extracts metadata features from matching members.
    
    Returns:
        tuple (list_of_results, list_of_failed_traces, next_sample_id)
    """
    results = []
    failed_traces = []
    sample_id = start_sample_id

    tar_filename = os.path.basename(tar_path)
    print(f"Opening tar archive: {tar_path} (size: {os.path.getsize(tar_path) / (1024*1024):.2f} MB)")
    
    start_time = time.time()
    
    with tarfile.open(tar_path, "r:gz") as tar:
        # Filter for members that are files inside the specified split directory
        prefix = f"{split}/{world}_world/"
        
        print(f"Scanning members with prefix: {prefix}")
        members = []
        for m in tar.getmembers():
            if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".csv"):
                members.append(m)
        
        # Sort members by name to guarantee deterministic ordering
        members.sort(key=lambda x: x.name)
        
        if limit is not None and limit > 0:
            members = members[:limit]
        
        total_members = len(members)
        print(f"Found {total_members} matching trace CSV files to process.")
        
        for idx, member in enumerate(members, 1):
            label, site_name, trace_name = get_label_and_site_name(member.name, world)
            
            # Print progress every 1000 files
            if idx % 1000 == 0 or idx == total_members:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (total_members - idx) / rate if rate > 0 else 0
                print(f"  Processed {idx}/{total_members} files. Rate: {rate:.1f} files/s. ETA: {eta/60:.1f}m")
                
            f_obj = tar.extractfile(member)
            if f_obj is None:
                print(f"Warning: Could not extract file object for {member.name}")
                failed_traces.append({
                    "tar_file": tar_filename,
                    "member_path": member.name,
                    "error": "Failed to extract file object"
                })
                continue
                
            try:
                df = read_trace_csv(f_obj)
                features = extract_features_from_df(df)
                
                # Combine metadata/debug columns with feature columns
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "world": world,
                    "label": label,
                    "site_name": site_name,
                    "trace_name": trace_name,
                    "tar_file": tar_filename,
                    "member_path": member.name,
                }
                row.update(features)
                results.append(row)
                sample_id += 1
                
            except Exception as e:
                print(f"Error processing trace {member.name}: {str(e)}")
                failed_traces.append({
                    "tar_file": tar_filename,
                    "member_path": member.name,
                    "error": str(e)
                })
                
    return results, failed_traces, sample_id
