# scripts/run_anova.py

import os
import sys
import argparse
import json
import time
import pandas as pd

# Add parent directory to sys.path to allow imports from cicflowmeter_wf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicflowmeter_wf.anova import run_anova_ranking

def main():
    parser = argparse.ArgumentParser(description="Chạy phân tích ANOVA để xếp hạng đặc trưng siêu dữ liệu.")
    parser.add_argument("--input", type=str, required=True,
                        help="Đường dẫn đến tệp tin CSV chứa bảng đặc trưng trích xuất (metadata_feature_bank_train.csv)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Thư mục đầu ra để lưu các kết quả ANOVA")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Không tìm thấy tệp đầu vào '{args.input}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Bắt đầu phân tích ANOVA từ tệp: {args.input} ===")
    start_time = time.time()

    # 1. Đọc dữ liệu
    print("Đang đọc tệp CSV dữ liệu...")
    df = pd.read_csv(args.input)
    print(f"Đã đọc thành công {len(df)} dòng dữ liệu.")

    # 2. Chạy ANOVA
    ranking_df = run_anova_ranking(df)

    # 3. Lưu kết quả ranking
    output_ranking_csv = os.path.join(args.output_dir, "anova_ranking_train.csv")
    ranking_df.to_csv(output_ranking_csv, index=False)
    print(f"Đã lưu bảng xếp hạng đặc trưng vào: {output_ranking_csv}")

    # 4. Trích xuất các danh sách top-k đặc trưng và lưu dưới dạng JSON
    top_k_sizes = [5, 10, 15, 20]
    top_features = {}
    
    for k in top_k_sizes:
        features_list = ranking_df.head(k)["feature_name"].tolist()
        top_features[k] = features_list
        
        json_path = os.path.join(args.output_dir, f"top{k}_features.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(features_list, f, indent=4)
        print(f"Đã lưu Top {k} đặc trưng vào: {json_path}")

    # 5. Lưu log ANOVA
    elapsed_time = time.time() - start_time
    
    # Lấy thông tin top 5 đặc trưng hàng đầu làm ví dụ
    top5_info = ranking_df.head(5)[["feature_name", "f_score"]].to_dict(orient="records")

    anova_log = {
        "input_file": args.input,
        "total_samples": len(df),
        "total_features": len(ranking_df),
        "elapsed_time_seconds": elapsed_time,
        "top5_features": top5_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    log_path = os.path.join(args.output_dir, "anova_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(anova_log, f, indent=4)

    print(f"\n=== Hoàn thành ANOVA trong {elapsed_time:.2f} giây ===")
    print(json.dumps(anova_log, indent=4))

if __name__ == "__main__":
    main()
