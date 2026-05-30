# cicflowmeter_wf/features.py

import numpy as np
import pandas as pd
from .feature_names import FEATURE_NAMES

def get_std(arr) -> float:
    """
    Computes the sample standard deviation (ddof=1) of an array/list.
    Returns 0.0 if array has fewer than 2 elements.
    """
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))

def get_var(arr) -> float:
    """
    Computes the sample variance (ddof=1) of an array/list.
    Returns 0.0 if array has fewer than 2 elements.
    """
    if len(arr) < 2:
        return 0.0
    return float(np.var(arr, ddof=1))

def extract_features_from_df(df: pd.DataFrame) -> dict:
    """
    Extracts exactly 74 features from a Website Fingerprinting CSV trace DataFrame.
    
    Columns expected in df:
        - timestamp (in seconds, absolute or relative)
        - length (packet size)
        - direction (+1 for outgoing/forward, -1 for incoming/backward)
        - packet_index (optional, used for sorting if present)
    """
    # 0. Ensure sorted packet order
    if "packet_index" in df.columns:
        df = df.sort_values("packet_index").reset_index(drop=True)

    # If the dataframe is empty, return zero for all features
    if len(df) == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    # Extract columns as numpy arrays
    timestamps = df["timestamp"].to_numpy(dtype=float)
    lengths = df["length"].to_numpy(dtype=float)
    directions = df["direction"].to_numpy(dtype=float)

    # 1. Flow Duration & Rates
    flow_duration = float(timestamps[-1] - timestamps[0])
    if flow_duration < 0.0:
        flow_duration = 0.0

    flow_bytes_per_s = float(np.sum(lengths) / flow_duration) if flow_duration > 0.0 else 0.0
    flow_packets_per_s = float(len(lengths) / flow_duration) if flow_duration > 0.0 else 0.0

    # Direction masks
    fwd_mask = (directions == 1)
    bwd_mask = (directions == -1)

    fwd_lengths = lengths[fwd_mask]
    bwd_lengths = lengths[bwd_mask]

    fwd_timestamps = timestamps[fwd_mask]
    bwd_timestamps = timestamps[bwd_mask]

    # Counts and total lengths
    total_fwd_packets = len(fwd_lengths)
    total_bwd_packets = len(bwd_lengths)
    total_len_fwd_packets = float(np.sum(fwd_lengths))
    total_len_bwd_packets = float(np.sum(bwd_lengths))

    # Min/Max/Mean/Std lengths
    fwd_pkt_len_min = float(np.min(fwd_lengths)) if len(fwd_lengths) > 0 else 0.0
    fwd_pkt_len_max = float(np.max(fwd_lengths)) if len(fwd_lengths) > 0 else 0.0
    fwd_pkt_len_mean = float(np.mean(fwd_lengths)) if len(fwd_lengths) > 0 else 0.0
    fwd_pkt_len_std = get_std(fwd_lengths)

    bwd_pkt_len_min = float(np.min(bwd_lengths)) if len(bwd_lengths) > 0 else 0.0
    bwd_pkt_len_max = float(np.max(bwd_lengths)) if len(bwd_lengths) > 0 else 0.0
    bwd_pkt_len_mean = float(np.mean(bwd_lengths)) if len(bwd_lengths) > 0 else 0.0
    bwd_pkt_len_std = get_std(bwd_lengths)

    # IATs
    flow_iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)
    flow_iat_mean = float(np.mean(flow_iats)) if len(flow_iats) > 0 else 0.0
    flow_iat_std = get_std(flow_iats)
    flow_iat_max = float(np.max(flow_iats)) if len(flow_iats) > 0 else 0.0
    flow_iat_min = float(np.min(flow_iats)) if len(flow_iats) > 0 else 0.0

    fwd_iats = np.diff(fwd_timestamps) if len(fwd_timestamps) > 1 else np.array([], dtype=float)
    fwd_iat_min = float(np.min(fwd_iats)) if len(fwd_iats) > 0 else 0.0
    fwd_iat_max = float(np.max(fwd_iats)) if len(fwd_iats) > 0 else 0.0
    fwd_iat_mean = float(np.mean(fwd_iats)) if len(fwd_iats) > 0 else 0.0
    fwd_iat_std = get_std(fwd_iats)
    fwd_iat_total = float(np.sum(fwd_iats)) if len(fwd_iats) > 0 else 0.0

    bwd_iats = np.diff(bwd_timestamps) if len(bwd_timestamps) > 1 else np.array([], dtype=float)
    bwd_iat_min = float(np.min(bwd_iats)) if len(bwd_iats) > 0 else 0.0
    bwd_iat_max = float(np.max(bwd_iats)) if len(bwd_iats) > 0 else 0.0
    bwd_iat_mean = float(np.mean(bwd_iats)) if len(bwd_iats) > 0 else 0.0
    bwd_iat_std = get_std(bwd_iats)
    bwd_iat_total = float(np.sum(bwd_iats)) if len(bwd_iats) > 0 else 0.0

    # Rate features
    fwd_packets_per_s = float(total_fwd_packets / flow_duration) if flow_duration > 0.0 else 0.0
    bwd_packets_per_s = float(total_bwd_packets / flow_duration) if flow_duration > 0.0 else 0.0

    # Overall length stats
    pkt_len_min = float(np.min(lengths)) if len(lengths) > 0 else 0.0
    pkt_len_max = float(np.max(lengths)) if len(lengths) > 0 else 0.0
    pkt_len_mean = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    pkt_len_std = get_std(lengths)
    pkt_len_var = get_var(lengths)

    # Down/Up ratio & segment sizes
    down_up_ratio = float(total_bwd_packets / total_fwd_packets) if total_fwd_packets > 0 else 0.0
    avg_packet_size = pkt_len_mean
    fwd_segment_size_avg = fwd_pkt_len_mean
    bwd_segment_size_avg = bwd_pkt_len_mean

    # Bulk Feature Extraction
    # Ported exactly from Java BasicFlow.java logic
    fbulk_duration = 0.0
    fbulk_packet_count = 0
    fbulk_size_total = 0.0
    fbulk_state_count = 0
    fbulk_packet_count_helper = 0
    fbulk_start_helper = 0.0
    fbulk_size_helper = 0.0
    flast_bulk_ts = 0.0

    bbulk_duration = 0.0
    bbulk_packet_count = 0
    bbulk_size_total = 0.0
    bbulk_state_count = 0
    bbulk_packet_count_helper = 0
    bbulk_start_helper = 0.0
    bbulk_size_helper = 0.0
    blast_bulk_ts = 0.0

    for i in range(len(timestamps)):
        ts = timestamps[i]
        size = lengths[i]
        direction = directions[i]
        
        if direction == 1:
            # Check if interrupted by backward bulk packet
            if blast_bulk_ts > fbulk_start_helper:
                fbulk_start_helper = 0.0
            
            if size > 0:
                if fbulk_start_helper == 0.0:
                    fbulk_start_helper = ts
                    fbulk_packet_count_helper = 1
                    fbulk_size_helper = size
                    flast_bulk_ts = ts
                else:
                    if ts - flast_bulk_ts > 1.0:
                        fbulk_start_helper = ts
                        flast_bulk_ts = ts
                        fbulk_packet_count_helper = 1
                        fbulk_size_helper = size
                    else:
                        fbulk_packet_count_helper += 1
                        fbulk_size_helper += size
                        
                        if fbulk_packet_count_helper == 4:
                            fbulk_state_count += 1
                            fbulk_packet_count += fbulk_packet_count_helper
                            fbulk_size_total += fbulk_size_helper
                            fbulk_duration += ts - fbulk_start_helper
                        elif fbulk_packet_count_helper > 4:
                            fbulk_packet_count += 1
                            fbulk_size_total += size
                            fbulk_duration += ts - flast_bulk_ts
                        flast_bulk_ts = ts
        else:
            # Check if interrupted by forward bulk packet
            if flast_bulk_ts > bbulk_start_helper:
                bbulk_start_helper = 0.0
            
            if size > 0:
                if bbulk_start_helper == 0.0:
                    bbulk_start_helper = ts
                    bbulk_packet_count_helper = 1
                    bbulk_size_helper = size
                    blast_bulk_ts = ts
                else:
                    if ts - blast_bulk_ts > 1.0:
                        bbulk_start_helper = ts
                        blast_bulk_ts = ts
                        bbulk_packet_count_helper = 1
                        bbulk_size_helper = size
                    else:
                        bbulk_packet_count_helper += 1
                        bbulk_size_helper += size
                        
                        if bbulk_packet_count_helper == 4:
                            bbulk_state_count += 1
                            bbulk_packet_count += bbulk_packet_count_helper
                            bbulk_size_total += bbulk_size_helper
                            bbulk_duration += ts - bbulk_start_helper
                        elif bbulk_packet_count_helper > 4:
                            bbulk_packet_count += 1
                            bbulk_size_total += size
                            bbulk_duration += ts - blast_bulk_ts
                        blast_bulk_ts = ts

    # Compute averages
    fwd_bytes_bulk_avg = float(fbulk_size_total / fbulk_state_count) if fbulk_state_count > 0 else 0.0
    fwd_packet_bulk_avg = float(fbulk_packet_count / fbulk_state_count) if fbulk_state_count > 0 else 0.0
    fwd_bulk_rate_avg = float(fbulk_size_total / fbulk_duration) if fbulk_duration > 0.0 else 0.0

    bwd_bytes_bulk_avg = float(bbulk_size_total / bbulk_state_count) if bbulk_state_count > 0 else 0.0
    bwd_packet_bulk_avg = float(bbulk_packet_count / bbulk_state_count) if bbulk_state_count > 0 else 0.0
    bwd_bulk_rate_avg = float(bbulk_size_total / bbulk_duration) if bbulk_duration > 0.0 else 0.0

    # Subflow Features
    # Bound by 1.0 second idle time gaps
    sf_count = 0
    sf_last_packet_ts = -1.0
    for ts in timestamps:
        if sf_last_packet_ts == -1.0:
            sf_last_packet_ts = ts
        else:
            if ts - sf_last_packet_ts > 1.0:
                sf_count += 1
            sf_last_packet_ts = ts

    if sf_count > 0:
        subflow_fwd_packets = float(total_fwd_packets / sf_count)
        subflow_fwd_bytes = float(total_len_fwd_packets / sf_count)
        subflow_bwd_packets = float(total_bwd_packets / sf_count)
        subflow_bwd_bytes = float(total_len_bwd_packets / sf_count)
    else:
        subflow_fwd_packets = 0.0
        subflow_fwd_bytes = 0.0
        subflow_bwd_packets = 0.0
        subflow_bwd_bytes = 0.0

    # Active/Idle Features
    # Activity Timeout of 5.0 seconds (5,000,000 microseconds)
    active_periods = []
    idle_periods = []
    if len(timestamps) > 0:
        start_active_time = timestamps[0]
        end_active_time = timestamps[0]
        for ts in timestamps[1:]:
            if ts - end_active_time > 5.0:
                if end_active_time - start_active_time > 0.0:
                    active_periods.append(end_active_time - start_active_time)
                idle_periods.append(ts - end_active_time)
                start_active_time = ts
                end_active_time = ts
            else:
                end_active_time = ts
        # Note: Java code comments out final active period logging, so it remains commented here:
        # if end_active_time - start_active_time > 0.0:
        #     active_periods.append(end_active_time - start_active_time)

    active_min = float(np.min(active_periods)) if len(active_periods) > 0 else 0.0
    active_max = float(np.max(active_periods)) if len(active_periods) > 0 else 0.0
    active_mean = float(np.mean(active_periods)) if len(active_periods) > 0 else 0.0
    active_std = get_std(active_periods)

    idle_min = float(np.min(idle_periods)) if len(idle_periods) > 0 else 0.0
    idle_max = float(np.max(idle_periods)) if len(idle_periods) > 0 else 0.0
    idle_mean = float(np.mean(idle_periods)) if len(idle_periods) > 0 else 0.0
    idle_std = get_std(idle_periods)

    # ----------------- Group B: WF-Specific Features -----------------
    
    # 59. direction_switch_count
    direction_switch_count = 0
    if len(directions) > 1:
        direction_switch_count = int(np.sum(directions[:-1] != directions[1:]))

    # 60. direction_switch_rate
    direction_switch_rate = float(direction_switch_count / max(len(directions) - 1, 1))

    # Parse Bursts
    bursts = []
    if len(directions) > 0:
        current_dir = directions[0]
        start_idx = 0
        for i in range(1, len(directions)):
            if directions[i] != current_dir:
                count = i - start_idx
                start_ts = timestamps[start_idx]
                end_ts = timestamps[i - 1]
                bursts.append({
                    'dir': current_dir,
                    'count': count,
                    'start_ts': start_ts,
                    'end_ts': end_ts,
                    'duration': max(end_ts - start_ts, 0.0)
                })
                current_dir = directions[i]
                start_idx = i
        # last burst
        count = len(directions) - start_idx
        start_ts = timestamps[start_idx]
        end_ts = timestamps[-1]
        bursts.append({
            'dir': current_dir,
            'count': count,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'duration': max(end_ts - start_ts, 0.0)
        })

    # 61. total_burst_count
    total_burst_count = len(bursts)

    # 62. fwd_burst_count
    fwd_burst_count = sum(1 for b in bursts if b['dir'] == 1)

    # 63. bwd_burst_count
    bwd_burst_count = sum(1 for b in bursts if b['dir'] == -1)

    # Burst lengths in packets
    burst_lengths = [b['count'] for b in bursts]
    # 64. burst_len_mean
    burst_len_mean = float(np.mean(burst_lengths)) if len(bursts) > 0 else 0.0
    # 65. burst_len_std
    burst_len_std = get_std(burst_lengths)
    # 66. burst_len_max
    burst_len_max = float(np.max(burst_lengths)) if len(bursts) > 0 else 0.0

    fwd_burst_lengths = [b['count'] for b in bursts if b['dir'] == 1]
    # 67. fwd_burst_len_mean
    fwd_burst_len_mean = float(np.mean(fwd_burst_lengths)) if len(fwd_burst_lengths) > 0 else 0.0

    bwd_burst_lengths = [b['count'] for b in bursts if b['dir'] == -1]
    # 68. bwd_burst_len_mean
    bwd_burst_len_mean = float(np.mean(bwd_burst_lengths)) if len(bwd_burst_lengths) > 0 else 0.0

    # Burst durations
    burst_durations = [b['duration'] for b in bursts]
    # 69. burst_duration_mean
    burst_duration_mean = float(np.mean(burst_durations)) if len(bursts) > 0 else 0.0
    # 70. burst_duration_std
    burst_duration_std = get_std(burst_durations)
    # 71. burst_duration_max
    burst_duration_max = float(np.max(burst_durations)) if len(bursts) > 0 else 0.0

    # Gaps between bursts
    gaps = []
    for idx in range(len(bursts) - 1):
        gaps.append(max(bursts[idx+1]['start_ts'] - bursts[idx]['end_ts'], 0.0))

    # 72. inter_burst_gap_mean
    inter_burst_gap_mean = float(np.mean(gaps)) if len(gaps) > 0 else 0.0
    # 73. inter_burst_gap_std
    inter_burst_gap_std = get_std(gaps)

    # 74. first_burst_direction
    first_burst_direction = float(bursts[0]['dir']) if len(bursts) > 0 else 0.0

    feature_dict = {
        "flow_duration": flow_duration,
        "flow_bytes_per_s": flow_bytes_per_s,
        "flow_packets_per_s": flow_packets_per_s,
        "total_fwd_packets": float(total_fwd_packets),
        "total_bwd_packets": float(total_bwd_packets),
        "total_len_fwd_packets": total_len_fwd_packets,
        "total_len_bwd_packets": total_len_bwd_packets,
        "fwd_pkt_len_min": fwd_pkt_len_min,
        "fwd_pkt_len_max": fwd_pkt_len_max,
        "fwd_pkt_len_mean": fwd_pkt_len_mean,
        "fwd_pkt_len_std": fwd_pkt_len_std,
        "bwd_pkt_len_min": bwd_pkt_len_min,
        "bwd_pkt_len_max": bwd_pkt_len_max,
        "bwd_pkt_len_mean": bwd_pkt_len_mean,
        "bwd_pkt_len_std": bwd_pkt_len_std,
        "flow_iat_mean": flow_iat_mean,
        "flow_iat_std": flow_iat_std,
        "flow_iat_max": flow_iat_max,
        "flow_iat_min": flow_iat_min,
        "fwd_iat_min": fwd_iat_min,
        "fwd_iat_max": fwd_iat_max,
        "fwd_iat_mean": fwd_iat_mean,
        "fwd_iat_std": fwd_iat_std,
        "fwd_iat_total": fwd_iat_total,
        "bwd_iat_min": bwd_iat_min,
        "bwd_iat_max": bwd_iat_max,
        "bwd_iat_mean": bwd_iat_mean,
        "bwd_iat_std": bwd_iat_std,
        "bwd_iat_total": bwd_iat_total,
        "fwd_packets_per_s": fwd_packets_per_s,
        "bwd_packets_per_s": bwd_packets_per_s,
        "pkt_len_min": pkt_len_min,
        "pkt_len_max": pkt_len_max,
        "pkt_len_mean": pkt_len_mean,
        "pkt_len_std": pkt_len_std,
        "pkt_len_var": pkt_len_var,
        "down_up_ratio": down_up_ratio,
        "avg_packet_size": avg_packet_size,
        "fwd_segment_size_avg": fwd_segment_size_avg,
        "bwd_segment_size_avg": bwd_segment_size_avg,
        "fwd_bytes_bulk_avg": fwd_bytes_bulk_avg,
        "fwd_packet_bulk_avg": fwd_packet_bulk_avg,
        "fwd_bulk_rate_avg": fwd_bulk_rate_avg,
        "bwd_bytes_bulk_avg": bwd_bytes_bulk_avg,
        "bwd_packet_bulk_avg": bwd_packet_bulk_avg,
        "bwd_bulk_rate_avg": bwd_bulk_rate_avg,
        "subflow_fwd_packets": subflow_fwd_packets,
        "subflow_fwd_bytes": subflow_fwd_bytes,
        "subflow_bwd_packets": subflow_bwd_packets,
        "subflow_bwd_bytes": subflow_bwd_bytes,
        "active_min": active_min,
        "active_mean": active_mean,
        "active_max": active_max,
        "active_std": active_std,
        "idle_min": idle_min,
        "idle_mean": idle_mean,
        "idle_max": idle_max,
        "idle_std": idle_std,
        
        "direction_switch_count": float(direction_switch_count),
        "direction_switch_rate": direction_switch_rate,
        "total_burst_count": float(total_burst_count),
        "fwd_burst_count": float(fwd_burst_count),
        "bwd_burst_count": float(bwd_burst_count),
        "burst_len_mean": burst_len_mean,
        "burst_len_std": burst_len_std,
        "burst_len_max": burst_len_max,
        "fwd_burst_len_mean": fwd_burst_len_mean,
        "bwd_burst_len_mean": bwd_burst_len_mean,
        "burst_duration_mean": burst_duration_mean,
        "burst_duration_std": burst_duration_std,
        "burst_duration_max": burst_duration_max,
        "inter_burst_gap_mean": inter_burst_gap_mean,
        "inter_burst_gap_std": inter_burst_gap_std,
        "first_burst_direction": first_burst_direction
    }

    # NaN / Inf protection over all features
    for name in FEATURE_NAMES:
        val = feature_dict[name]
        if np.isnan(val) or np.isinf(val):
            feature_dict[name] = 0.0

    return feature_dict
