# Google Colab Execution Guide: Website Fingerprinting Feature Extraction & ANOVA Pipeline

This guide outlines the step-by-step workflow to execute the frozen metadata feature extraction pipeline and the ANOVA ranking on Google Colab. 

To overcome the performance bottlenecks of Python `tarfile` random seeking and Google Drive FUSE latency, **Directory-Based Local Processing (Directory Mode)** is the default and recommended method. It extracts the archives to the Colab local scratch space (SSD) first, improving extraction speed by up to 3000x (<0.015s per trace).

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
```bash
!pip install -r requirements.txt
```
*(Note: If `requirements.txt` is not present, you can install the dependencies manually: `pip install pandas numpy scikit-learn`)*

---

## 2. Disk Space Audit & Dataset Extraction

To avoid closed-world and open-world folder naming conflicts and bypass FUSE latency, extract the datasets to separate folders on the local SSD.

### Step 2.1: Check Available Disk Space
Check the local VM storage capacity (Colab usually provides ~100GB of fast SSD space) and verify the size of the compressed archives:
```bash
# Check local storage capacity
!df -h /content

# Check dataset archive sizes on Drive
!du -sh /content/drive/MyDrive/WF/raw_splited/*.tar.gz
```

### Step 2.2: Extract Archives to Local SSD
Create target directories and decompress the archives:
```bash
# Create isolated local target directories
!mkdir -p /content/wf_split/closed /content/wf_split/open

# Decompress archives (typically takes ~1-2 minutes total)
!tar -xzf /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz -C /content/wf_split/closed
!tar -xzf /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz -C /content/wf_split/open
```

**Expected Paths after extraction:**
* Closed-world traces: `/content/wf_split/closed/training_data/closed_world/`
* Open-world traces: `/content/wf_split/open/training_data/open_world/`

---

## 3. Phase 1: Small-Batch Verification (50 Traces)

Before scaling to the full dataset, run the deterministic 50-trace verification batch in **Directory Mode**. This processes the first 25 closed-world and first 25 open-world traces.

### Execution Command
```bash
!python scripts/run_small_batch.py \
    --closed-dir /content/wf_split/closed/training_data/closed_world \
    --open-dir /content/wf_split/open/training_data/open_world \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/small_batch_test_dir/
```

### Expected Output Directory
All small-batch outputs are saved directly under Google Drive:
`/content/drive/MyDrive/WF/metadata_feature_bank/small_batch_test_dir/`

### Generated Output Files
1. **`metadata_feature_bank_small.csv`**: Contains the 50 processed traces with metadata columns and all 74 features.
2. **`small_batch_feature_stats.csv`**: Feature statistics (min, max, mean, std, and indicators if constant/all-zero).
3. **`timing_anomaly_report.csv`**: Logs trace filenames containing microsecond timestamp inversions, including the count and maximum inversion depth.
4. **`small_batch_report.txt`**: Summary report detailing general runtimes, profiling breakdown (scanning, reading, calculating, exporting), data quality (NaN/inf counts), and timing/burst/idle distribution profiles.

> [!TIP]
> Read `small_batch_report.txt` and check the **GENERAL PERFORMANCE METRICS & PROFILING BREAKDOWN** section. It will display the speedup achieved in Directory Mode compared to Tar Mode.

---

## 4. Phase 2: Full Dataset Feature Extraction

Once Phase 1 passes successfully and the small-batch report shows no errors, proceed to extract features for the full dataset using local directories.

### Step 4.1: Full Extraction (training_data split)
Run the extraction script pointing to the extracted directories and outputting to `/content/drive/MyDrive/WF/metadata_feature_bank/train/`.
```bash
!python scripts/extract_metadata_train.py \
    --closed-dir /content/wf_split/closed/training_data/closed_world \
    --open-dir /content/wf_split/open/training_data/open_world \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/train/ \
    --split training_data
```

### Expected Outputs
The script generates exactly these 4 outputs under `/content/drive/MyDrive/WF/metadata_feature_bank/train/`:
1. **`metadata_feature_bank_train_all74.csv`**: Main feature CSV containing all successfully extracted traces with exactly 74 features.
2. **`extraction_quality_report.txt`**: Text report containing general runtime metrics, actual row count, failed trace count, total NaN/Inf counts, and timing anomaly counts.
3. **`timing_anomaly_report.csv`**: Logs filenames where raw timestamp inversions were encountered, along with the inversion count and minimum negative delta.
4. **`feature_names_74.json`**: JSON array containing the ordered list of 74 feature names.

---

## 5. Phase 3: ANOVA Feature Ranking (Closed-World Only)

Run ANOVA on the generated metadata feature bank to rank features by their importance in distinguishing closed-world sites. **Only run this step if the quality report verifies 133,000 actual rows and 0 NaN/Inf counts.**

### Execution Command
```bash
!python scripts/run_anova.py \
    --input /content/drive/MyDrive/WF/metadata_feature_bank/train/metadata_feature_bank_train_all74.csv \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/train/
```

### Expected Outputs
The command outputs ANOVA ranking details under:
`/content/drive/MyDrive/WF/metadata_feature_bank/train/`

* **`anova_ranking_train.csv`**: Full sorted table of 74 features ranked by $F$-score descending.
* **`top5_features.json`**, **`top10_features.json`**, **`top15_features.json`**, **`top20_features.json`**: JSON arrays of the top $K$ features.
* **`anova_log.json`**: JSON logs containing run statistics and top-5 preview.

---

## 6. Fallback Option: Tar-based Processing

If you cannot extract the datasets to the local SSD due to disk space limitations, you can fall back to Tar Mode (processes traces directly from GDrive tars sequentially, but note that random seeks will take significantly longer).

### Small-batch Tar Mode:
```bash
!python scripts/run_small_batch.py \
    --closed-tar /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz \
    --open-tar /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/small_batch_test_tar/ \
    --split training_data
```

### Full-extraction Tar Mode:
```bash
!python scripts/extract_metadata_train.py \
    --closed-tar /content/drive/MyDrive/WF/raw_splited/closed_world_split.tar.gz \
    --open-tar /content/drive/MyDrive/WF/raw_splited/open_world_split.tar.gz \
    --output-dir /content/drive/MyDrive/WF/metadata_feature_bank/train/ \
    --split training_data
```
