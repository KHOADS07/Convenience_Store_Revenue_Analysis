"""
tests/test_pipeline.py - TV6: Bộ kiểm thử tự động (Unittest) cho toàn bộ pipeline.

Chạy kiểm thử từ thư mục gốc dự án:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src import (  # noqa: E402
    insights,
    multivariate_analysis,
    preprocessing,
    revenue_stats,
    visualizer,
)


def _formatted_call(func, *args, **kwargs):
    """Chạy hàm, giữ nguyên toàn bộ log in ra nhưng xuống dòng & thụt lề gọn gàng."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args, **kwargs)
    logs = buf.getvalue().strip()
    if logs:
        print("\n" + textwrap.indent(logs, "      | "), end="\n    ")
    return result


class TestConvenienceStorePipeline(unittest.TestCase):
    """Kiểm thử toàn diện 5 bước xử lý dữ liệu, phân tích, vẽ biểu đồ và sinh nhận định."""

    @classmethod
    def setUpClass(cls):
        """Khởi tạo thư mục tạm và chạy tiền xử lý dữ liệu thật một lần cho các test đọc."""
        cls.raw_csv = BASE_DIR / "data" / "supermarket_sales.csv"
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="test_sales_vn_"))
        cls.cleaned_csv = cls.temp_dir / "cleaned_sales_vn.csv"

        if cls.raw_csv.exists():
            print("\n" + "=" * 72)
            print(" [SETUP] NẠP & TIỀN XỬ LÝ DỮ LIỆU THỰC TẾ (TV1)")
            print("=" * 72, end="")
            cls.df_clean = _formatted_call(
                preprocessing.run_preprocessing,
                str(cls.raw_csv),
                str(cls.cleaned_csv),
            )
            print("\n" + "=" * 72)
            if cls.df_clean is None:
                cls.df_clean = pd.read_csv(cls.cleaned_csv, parse_dates=["Date"])
        else:
            cls.df_clean = cls._build_synthetic_clean_df()

    @classmethod
    def tearDownClass(cls):
        """Dọn dẹp thư mục tạm sau khi hoàn tất toàn bộ ca kiểm thử."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @staticmethod
    def _build_synthetic_clean_df() -> pd.DataFrame:
        return pd.DataFrame({
            "Invoice ID": [f"INV-{i:03d}" for i in range(1, 13)],
            "Branch": ["A", "B", "C"] * 4,
            "KhuVuc_VN": ["Quận 1 - Văn phòng", "Thủ Đức - Đại học", "Quận 7 - Dân cư"] * 4,
            "Product line": ["Food and beverages", "Health and beauty"] * 6,
            "NhomHang_VN": ["Đồ ăn nhanh & Thức uống", "Chăm sóc cá nhân & Mỹ phẩm mini"] * 6,
            "Total": [100.0, 200.0, 150.0, 300.0] * 3,
            "Total_VND": [2_500_000.0, 5_000_000.0, 3_750_000.0, 7_500_000.0] * 3,
            "Total_Clean": [100.0, 200.0, 150.0, 300.0] * 3,
            "Total_VND_Clean": [2_500_000.0, 5_000_000.0, 3_750_000.0, 7_500_000.0] * 3,
            "Gross_Income_VND": [119_000.0, 238_000.0, 178_500.0, 357_000.0] * 3,
            "Quantity": [2, 4, 3, 6] * 3,
            "Rating": [8.5, 7.2, 9.0, 6.8] * 3,
            "Date": pd.date_range("2019-01-01", periods=12, freq="D"),
            "Month": [1] * 12,
            "DayOfWeek": ["Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"] * 2,
            "IsWeekend": [False, False, False, False, True, True] * 2,
            "Hour": [10, 12, 15, 19] * 3,
            "TimeSlot": ["10h-12h Sáng", "12h-14h Trưa", "14h-18h Chiều", "18h-21h Tối"] * 3,
            "Payment_VN": ["Ví điện tử / QR", "Tiền mặt", "Thẻ ngân hàng"] * 4,
        })

    def test_01_preprocessing_and_vn_mapping(self):
        """[TV1] Data Contract: đủ cột chuẩn, 0 NaN, tỷ giá VNĐ & ánh xạ chi nhánh VN."""
        required_cols = [
            "Invoice ID", "Branch", "KhuVuc_VN", "Product line", "NhomHang_VN",
            "Total", "Total_VND", "Total_Clean", "Total_VND_Clean",
            "Gross_Income_VND", "Date", "Month", "DayOfWeek", "IsWeekend",
            "Hour", "TimeSlot", "Payment_VN",
        ]
        for col in required_cols:
            self.assertIn(col, self.df_clean.columns, f"Thiếu cột bắt buộc: {col}")

        self.assertEqual(int(self.df_clean[required_cols].isna().sum().sum()), 0)
        diff = np.abs(self.df_clean["Total_VND"] - self.df_clean["Total"] * 25000.0).max()
        self.assertLess(diff, 1e-3, "Sai lệch tỷ giá quy đổi Total_VND")

        expected_branches = {"Quận 1 - Văn phòng", "Thủ Đức - Đại học", "Quận 7 - Dân cư"}
        self.assertTrue(expected_branches.issubset(set(self.df_clean["KhuVuc_VN"].unique())))

    def test_02_missing_value_imputation(self):
        """[TV1] Tiêm giả lập NaN (số & chuỗi), xác minh điền khuyết median/mode đưa NaN về 0."""
        df_dirty = pd.DataFrame({
            "Invoice ID": ["A1", "A2", "A3", "A4", "A5"],
            "Total": [100.0, np.nan, 200.0, 300.0, 400.0],
            "Quantity": [2.0, 4.0, np.nan, 6.0, 8.0],
            "Payment": ["Ewallet", "Ewallet", np.nan, "Cash", "Ewallet"],
        })

        df_fixed = _formatted_call(preprocessing.handle_missing_values, df_dirty.copy())

        self.assertEqual(int(df_fixed.isna().sum().sum()), 0)
        self.assertAlmostEqual(float(df_fixed.loc[1, "Total"]), 250.0)
        self.assertAlmostEqual(float(df_fixed.loc[2, "Quantity"]), 5.0)
        self.assertEqual(df_fixed.loc[2, "Payment"], "Ewallet")

    def test_03_outlier_detection_iqr(self):
        """[TV1] Chèn hóa đơn đột biến 50.000.000 VNĐ (2000 USD), kiểm tra phát hiện IQR & chặn Winsorizing."""
        df_sample = pd.DataFrame({
            "Invoice ID": [f"ID-{i}" for i in range(10)],
            "Total": [100.0, 110.0, 120.0, 115.0, 105.0, 125.0, 130.0, 118.0, 122.0, 2000.0],
        })

        df_out = _formatted_call(
            preprocessing.detect_and_handle_outliers,
            df_sample.copy(),
            col="Total",
            method="iqr",
        )

        self.assertIn("Total_Clean", df_out.columns)
        self.assertLess(float(df_out.loc[9, "Total_Clean"]), 2000.0)
        self.assertAlmostEqual(float(df_out.loc[0, "Total_Clean"]), 100.0)

    def test_04_revenue_consistency(self):
        """[TV2 & TV3] Đối chiếu chéo tổng doanh thu giữa Ngày, Tháng, Mặt hàng và Pivot khớp 100%."""
        expected_total = float(self.df_clean["Total_VND"].sum())

        stats = revenue_stats.run_revenue_analysis(self.df_clean)
        rev_day = float(stats["by_day"]["total_revenue"].sum())
        rev_month = float(stats["by_month"]["total_revenue"].sum())
        rev_product = float(stats["by_product"]["total_revenue"].sum())

        hw_res = multivariate_analysis.analyze_hour_by_weekday(self.df_clean)
        pivot_hw = hw_res[0] if isinstance(hw_res, tuple) else hw_res
        rev_pivot = float(pivot_hw.to_numpy().sum())

        self.assertAlmostEqual(rev_day, expected_total, delta=1.0)
        self.assertAlmostEqual(rev_month, expected_total, delta=1.0)
        self.assertAlmostEqual(rev_product, expected_total, delta=1.0)
        self.assertAlmostEqual(rev_pivot, expected_total, delta=1.0)

    def test_05_multivariate_pivot_and_insights(self):
        """[TV3 & TV5] Kiểm tra ma trận đa chiều không chứa NaN và TV5 sinh đủ 4+ nhận định động."""
        ct_res = multivariate_analysis.analyze_category_by_timeslot(self.df_clean)
        pivot_ct = ct_res[0] if isinstance(ct_res, tuple) else ct_res

        self.assertFalse(pivot_ct.isna().any().any())
        self.assertGreaterEqual(pivot_ct.shape[0], 2)

        stats = revenue_stats.run_revenue_analysis(self.df_clean)
        stats["kpi"] = insights.compute_kpis(self.df_clean)
        multi = {"category_timeslot": ct_res}

        ins_list = insights.generate_business_insights(self.df_clean, stats, multi)
        self.assertGreaterEqual(len(ins_list), 4)
        for item in ins_list:
            for key in ("id", "title", "finding", "evidence", "explanation", "action"):
                self.assertIn(key, item)

        report_path = self.temp_dir / "test_business_report.txt"
        insights.export_text_report(ins_list, stats, report_path)
        self.assertTrue(report_path.exists())
        self.assertGreater(report_path.stat().st_size, 200)

    def test_06_chart_generation(self):
        """[TV4] Kiểm tra xuất đủ 5 biểu đồ PNG hợp lệ (dung lượng > 1 KB) trong thư mục đầu ra."""
        chart_out_dir = self.temp_dir / "charts"
        stats = revenue_stats.run_revenue_analysis(self.df_clean)
        multi = {
            "hour_weekday": multivariate_analysis.analyze_hour_by_weekday(self.df_clean),
            "category_timeslot": multivariate_analysis.analyze_category_by_timeslot(self.df_clean),
        }

        _formatted_call(
            visualizer.generate_all_charts,
            self.df_clean,
            stats,
            multi,
            output_dir=str(chart_out_dir),
        )

        expected_files = [
            "01_doanh_thu_xu_huong.png",
            "02_doanh_thu_mat_hang.png",
            "03_heatmap_khung_gio_thu.png",
            "04_da_chieu_hang_khung_gio.png",
            "05_boxplot_phan_phoi_chi_nhanh.png",
        ]
        for fname in expected_files:
            fpath = chart_out_dir / fname
            self.assertTrue(fpath.exists(), f"Không tìm thấy biểu đồ: {fname}")
            self.assertGreater(fpath.stat().st_size, 1024, f"Biểu đồ {fname} bị rỗng hoặc lỗi")


if __name__ == "__main__":
    unittest.main(verbosity=2)