# scripts/extract_metadata_train.py

import os
import sys
import argparse
import json
import time
import pandas as pd

# Add parent directory to sys.path to allow imports from cicflowmeter_wf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicflowmeter_wf.extract_train_bank import process_tar_archive
from cicflowmeter_wf.feature_names import FEATURE_NAMES

def main():
    parser = argparse.ArgumentParser(description="Batch trích xuất đặc trưng siêu dữ liệu từ tệp tar.gz.")
    parser.add_argument("--closed-tar", type=str, required=True,
                        help="Đường dẫn đến tệp closed_world_split.tar.gz")
    parser.add_argument("--open-tar", type=str, required=True,
                        help="Đường dẫn đến tệp open_world_split.tar.gz")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Thư mục đầu ra để lưu kết quả")
    parser.add_argument("--split", type=str, default="training_data",
                        help="Tên phân tách dữ liệu cần xử lý (ví dụ: training_data, validation_data, test_data)")
    
    args = parser.parse_args()

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Bắt đầu trích xuất đặc trưng siêu dữ liệu cho phân tách: {args.split} ===")
    start_total_time = time.time()

    all_results = []
    all_failures = []
    sample_id = 0

    # 1. Xử lý closed-world
    if os.path.exists(args.closed_tar):
        print(f"\n[1/2] Đang xử lý closed-world từ: {args.closed_tar}")
        closed_results, closed_failures, sample_id = process_tar_archive(
            args.closed_tar, args.split, "closed", sample_id
        )
        all_results.extend(closed_results)
        all_failures.extend(closed_failures)
        print(f"Hoàn thành closed-world: Trích xuất thành công {len(closed_results)} dòng, thất bại {len(closed_failures)}")
    else:
        print(f"\n[Cảnh báo] Không tìm thấy tệp closed-world: {args.closed_tar}. Bỏ qua...")

    # 2. Xử lý open-world
    if os.path.exists(args.open_tar):
        print(f"\n[2/2] Đang xử lý open-world từ: {args.open_tar}")
        open_results, open_failures, sample_id = process_tar_archive(
            args.open_tar, args.split, "open", sample_id
        )
        all_results.extend(open_results)
        all_failures.extend(open_failures)
        print(f"Hoàn thành open-world: Trích xuất thành công {len(open_results)} dòng, thất bại {len(open_failures)}")
    else:
        print(f"\n[Cảnh báo] Không tìm thấy tệp open-world: {args.open_tar}. Bỏ qua...")

    total_time = time.time() - start_total_time

    # 3. Tạo DataFrame và lưu kết quả
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        
        # Đảm bảo thứ tự cột đúng quy định
        meta_cols = ["sample_id", "split", "world", "label", "site_name", "trace_name", "tar_file", "member_path"]
        col_order = meta_cols + FEATURE_NAMES
        df = df[col_order]

        output_csv = os.path.join(args.output_dir, f"metadata_feature_bank_train.csv")
        print(f"\nĐang lưu {len(df)} dòng dữ liệu vào: {output_csv}")
        df.to_csv(output_csv, index=False)
    else:
        print("\n[Cảnh báo] Không có dữ liệu trích xuất thành công để lưu!")

    # 4. Lưu danh sách thất bại nếu có
    if len(all_failures) > 0:
        failures_csv = os.path.join(args.output_dir, "failed_traces.csv")
        pd.DataFrame(all_failures).to_csv(failures_csv, index=False)
        print(f"Đã lưu danh sách {len(all_failures)} trace thất bại vào: {failures_csv}")

    # 5. Lưu danh sách tên đặc trưng
    feature_names_json = os.path.join(args.output_dir, "feature_names_74.json")
    with open(feature_names_json, "w", encoding="utf-8") as f:
        json.dump(FEATURE_NAMES, f, indent=4)

    # 6. Lưu log quá trình trích xuất
    log_data = {
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

    print(f"\n=== Hoàn thành trích xuất trong {total_time/60:.2f} phút ===")
    print(log_data)

if __name__ == "__main__":
    main()
