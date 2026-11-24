"""
main.py - Điểm chạy tổng hợp toàn dự án (TV5 & TV6).

    python main.py
    python main.py --input data/supermarket_sales.csv --output-dir outputs
    python main.py --skip-charts
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src import preprocessing, revenue_stats, multivariate_analysis, visualizer, insights  # noqa: E402

CHART_FILES = {
    "trend": "01_doanh_thu_xu_huong.png",
    "category": "02_doanh_thu_mat_hang.png",
    "heatmap": "03_heatmap_khung_gio_thu.png",
    "multi": "04_da_chieu_hang_khung_gio.png",
    "boxplot": "05_boxplot_phan_phoi_chi_nhanh.png",
}


def _first(result):
    """Một số hàm có thể trả tuple (vd: doanh thu, số đơn) -> lấy bảng đầu tiên."""
    return result[0] if isinstance(result, tuple) else result


def _step(idx, total, title):
    print(f"\n[{idx}/{total}] {title} ...")
    return time.time()


def _done(t0):
    print(f"      xong ({time.time() - t0:.1f}s)")


def parse_args():
    p = argparse.ArgumentParser(description="Phân tích & trực quan hóa doanh thu cửa hàng tiện lợi (VN)")
    p.add_argument("--input", default=str(BASE_DIR / "data" / "supermarket_sales.csv"),
                   help="Đường dẫn tệp CSV gốc")
    p.add_argument("--output-dir", default=str(BASE_DIR / "outputs"), help="Thư mục xuất kết quả")
    p.add_argument("--skip-charts", action="store_true", help="Bỏ qua bước vẽ biểu đồ")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):          # tránh lỗi tiếng Việt trên terminal Windows
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / "cleaned_sales_vn.csv"
    n = 5

    print("=" * 70)
    print(" PHÂN TÍCH DOANH THU CỬA HÀNG TIỆN LỢI VIỆT NAM")
    print("=" * 70)
    if not input_path.exists():
        sys.exit(f"Không tìm thấy tệp dữ liệu: {input_path}")

    # 1. Tiền xử lý (TV1)
    t = _step(1, n, "Tiền xử lý dữ liệu (TV1)")
    df = preprocessing.run_preprocessing(str(input_path), str(cleaned_path))
    if df is None:                                   # phòng khi hàm chỉ ghi tệp mà không trả về
        df = pd.read_csv(cleaned_path, parse_dates=["Date"])
    print(f"      {len(df)} dòng, {df.shape[1]} cột -> {cleaned_path}")
    _done(t)

    # 2. Thống kê (TV2)
    t = _step(2, n, "Thống kê doanh thu cốt lõi (TV2)")
    stats = revenue_stats.run_revenue_analysis(df)
    revenue_stats.export_revenue_results(stats, output_dir=out_dir)
    stats["kpi"] = insights.compute_kpis(df)
    _done(t)

    # 3. Đa chiều (TV3)
    t = _step(3, n, "Phân tích đa chiều (TV3)")
    multi = {
        "hour_weekday": multivariate_analysis.analyze_hour_by_weekday(df),
        "category_timeslot": multivariate_analysis.analyze_category_by_timeslot(df),
        "three_d": multivariate_analysis.analyze_3d_category_timeslot_weekday(df),
        "branch_category_timeslot": multivariate_analysis.analyze_branch_category_timeslot(df),
    }
    _done(t)

    # 4. Biểu đồ (TV4)
    t = _step(4, n, "Vẽ 5 biểu đồ (TV4)")
    if args.skip_charts:
        print("      bỏ qua (--skip-charts)")
    else:
        visualizer.generate_all_charts(df, stats, multi, output_dir=str(chart_dir))
        print(f"      đã lưu {len(CHART_FILES)} biểu đồ -> {chart_dir}")
    _done(t)

    # 5. Nhận định + báo cáo (TV5)
    t = _step(5, n, "Trích xuất nhận định & xuất báo cáo (TV5)")
    result = insights.generate_business_insights(df, stats, multi)
    report = insights.export_text_report(result, stats, out_dir / "business_report.txt")
    print(f"      {len(result)} nhận định -> {report}")
    _done(t)

    print("\n" + "=" * 70)
    print(" HOÀN TẤT. Kết quả nằm trong:", out_dir)
    print("=" * 70)
    for ins in result:
        print(f"\n* Nhận định {ins['id']}: {ins['finding']}")


if __name__ == "__main__":
    main()
