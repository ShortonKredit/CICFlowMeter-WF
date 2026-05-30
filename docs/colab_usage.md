# Google Colab Execution Guide: Website Fingerprinting Feature Extraction & ANOVA Pipeline

This guide outlines the step-by-step workflow to execute the frozen metadata feature extraction pipeline and the ANOVA ranking on Google Colab. It covers setting up the environment, running the deterministic 50-trace small-batch test, verifying performance and quality metrics, and executing the full dataset extraction.

---

## 1. Environment Setup

Copy and run the following code blocks sequentially in your Google Colab notebook cells.

### Step 1.1: Mount Google Drive
Mount the Google Drive containing the raw split datasets.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 1.2: Clone the GitHub Repository
Clone the repository and navigate into the project root directory.
```bash
!git clone https://github.com/ShortonKredit/CICFlowMeter-WF.git
%cd CICFlowMeter-WF
```

### Step 1.3: Install Dependencies
Install all required libraries (e.g. `scikit-learn`, `pandas`, `numpy`).
```bash
!pip install -r requirements.txt
```
*(Note: If `requirements.txt` is not present, you can install the dependencies manually: `pip install pandas numpy scikit-learn`)*

---

## 2. Phase 1: Small-Batch Verification (50 Traces)

Before scaling to the full dataset, run the deterministic 50-trace verification batch. This extracts features from the first 25 closed-world and first 25 open-world traces in the training split.

### Execution Command
```bash
!python scripts/run_small_batch.py \
    --closed-tar /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz \
    --open-tar /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/small_batch_test/ \
    --split training_data
```

### Expected Output Directory
All small-batch outputs are saved directly under Google Drive:
`/content/drive/MyDrive/WF/metadata_feature_bank/small_batch_test/`

### Generated Output Files
1. **`metadata_feature_bank_small.csv`**: Contains the 50 processed traces with metadata columns and all 74 features.
2. **`small_batch_feature_stats.csv`**: Features stats summary (min, max, mean, std, and indicators if a feature is constant or all-zero).
3. **`timing_anomaly_report.csv`**: Records trace filenames containing microsecond timestamp inversions, including the count and maximum inversion depth.
4. **`small_batch_report.txt`**: Text report summarizing general runtime, data quality (NaN/inf counts), feature sanity, and distribution profiles of timing, burst, and active/idle features.

---

## 3. Phase 2: Full Dataset Feature Extraction

Once Phase 1 passes successfully and the small-batch report shows no errors, proceed to extract features for the full dataset.

### Step 3.1: Full Extraction (e.g. `training_data` split)
Run extraction on all traces in the selected split. Do not use a limit.
```bash
!python scripts/extract_metadata_train.py \
    --closed-tar /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz \
    --open-tar /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/ \
    --split training_data
```
* **Expected Output**: `/content/drive/MyDrive/WF/metadata_feature_bank/metadata_feature_bank_train.csv` (133k traces)

---

## 4. Phase 3: ANOVA Feature Ranking (Closed-World Only)

Run ANOVA on the generated metadata feature bank to rank features by their importance in distinguishing closed-world sites.

### Execution Command
```bash
!python scripts/run_anova.py \
    --input /content/drive/MyDrive/WF/metadata_feature_bank/metadata_feature_bank_train.csv \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/
```

### Expected Outputs
The command outputs ANOVA ranking details under:
`/content/drive/MyDrive/WF/metadata_feature_bank/`

* **`anova_ranking_train.csv`**: Full sorted table of 74 features ranked by $F$-score descending.
* **`top5_features.json`**, **`top10_features.json`**, **`top15_features.json`**, **`top20_features.json`**: JSON arrays of the top $K$ features.
* **`anova_log.json`**: JSON logs containing run statistics and top-5 preview.

---

## 5. Verification Checklist (PASS Criteria)

Before starting the full extraction or model training, ensure the small-batch report (`small_batch_report.txt`) meets these conditions:
- [ ] **Success Rate**: 50/50 traces successfully processed.
- [ ] **Feature Bank Length**: Pre-processing checks verify exactly 74 features are present in the output.
- [ ] **Data Quality**: Total NaN Count = 0, Total Inf Count = 0.
- [ ] **Negative IAT**: Post-clamped Negative IAT Gaps = 0 (verified that clamping to $\ge 0.0$ worked).
- [ ] **Reproducibility**: Repeated runs produce identical feature arrays and matching md5/sha256 checksums on `metadata_feature_bank_small.csv`.
