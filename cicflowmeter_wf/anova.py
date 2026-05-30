# cicflowmeter_wf/anova.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from .feature_names import FEATURE_NAMES

def get_feature_group(feature_name: str) -> str:
    """
    Returns 'cic' for original 58 CICFlowMeter-style features,
    or 'wf' for the 16 WF-specific features.
    """
    # First 58 are CIC-style features
    cic_count = 58
    try:
        idx = FEATURE_NAMES.index(feature_name)
        if idx < cic_count:
            return "cic"
        return "wf"
    except ValueError:
        return "unknown"

def get_feature_category(feature_name: str) -> str:
    """
    Categorizes features into descriptive subgroups.
    """
    name = feature_name.lower()
    if "burst" in name:
        return "burst"
    elif "subflow" in name:
        return "subflow"
    elif "bulk" in name:
        return "bulk"
    elif "active" in name or "idle" in name:
        return "active_idle"
    elif "iat" in name:
        return "iat"
    elif "pkt_len" in name or "pkt_size" in name or "segment_size" in name:
        return "packet_length"
    elif "packets_per_s" in name or "bytes_per_s" in name:
        return "rate"
    elif "total_fwd_packets" in name or "total_bwd_packets" in name or "total_len" in name:
        return "packet_count_and_total_size"
    elif "duration" in name:
        return "duration"
    elif "switch" in name:
        return "switch"
    else:
        return "other"

def run_anova_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs ANOVA F-test (f_classif) on the feature columns of df against target label.
    
    Parameters:
        df: pd.DataFrame containing the 74 feature columns and 'label' column.
        
    Returns:
        pd.DataFrame containing ANOVA ranking.
    """
    # 1. Identify feature columns in dataframe
    feat_cols = [col for col in FEATURE_NAMES if col in df.columns]
    
    # 2. Check that we have all 74 features
    if len(feat_cols) != len(FEATURE_NAMES):
        print(f"Warning: Only {len(feat_cols)} / {len(FEATURE_NAMES)} expected features found in DataFrame.")
    
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column for ANOVA analysis.")
        
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()

    # Handle any NaN/inf values in X before running ANOVA (e.g. fill with 0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Compute ANOVA F-scores and p-values
    print("Computing ANOVA F-classif scores...")
    f_scores, p_values = f_classif(X, y)

    # 4. Construct ranking DataFrame
    ranking_data = []
    for i, col_name in enumerate(feat_cols):
        f_score = float(f_scores[i])
        p_val = float(p_values[i])
        # In case f_score is NaN/inf (e.g. constant feature), treat as 0.0
        if np.isnan(f_score) or np.isinf(f_score):
            f_score = 0.0
        if np.isnan(p_val) or np.isinf(p_val):
            p_val = 1.0

        ranking_data.append({
            "feature_name": col_name,
            "feature_group": get_feature_group(col_name),
            "category": get_feature_category(col_name),
            "f_score": f_score,
            "p_value": p_val
        })

    ranking_df = pd.DataFrame(ranking_data)
    
    # Sort by f_score descending
    ranking_df = ranking_df.sort_values(by="f_score", ascending=False).reset_index(drop=True)
    
    # Add rank (1-indexed)
    ranking_df.insert(0, "rank", ranking_df.index + 1)
    
    # Add selection boolean flags
    ranking_df["selected_top5"] = ranking_df["rank"] <= 5
    ranking_df["selected_top10"] = ranking_df["rank"] <= 10
    ranking_df["selected_top15"] = ranking_df["rank"] <= 15
    ranking_df["selected_top20"] = ranking_df["rank"] <= 20

    return ranking_df
