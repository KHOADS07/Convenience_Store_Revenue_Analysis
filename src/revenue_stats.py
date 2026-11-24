"""
TV2 - Thống kê doanh thu cốt lõi
File: src/revenue_stats.py

Chức năng:
- Thống kê mô tả bằng Pandas và NumPy.
- Tổng hợp doanh thu và lợi nhuận theo ngày.
- Tổng hợp doanh thu và lợi nhuận theo tháng.
- Tổng hợp doanh thu và lợi nhuận theo mặt hàng.

Cách dùng:
    import pandas as pd
    from revenue_stats import run_revenue_analysis

    df = pd.read_csv("outputs/cleaned_sales_vn.csv")
    results = run_revenue_analysis(df)

    print(results["descriptive"])
    print(results["by_day"])
    print(results["by_month"])
    print(results["by_product"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Các tên cột được chấp nhận. Ưu tiên tên chuẩn hóa của nhóm (từ TV1).
COLUMN_ALIASES = {
    "revenue": [
        "Total_VND_Clean",
        "Total_Clean",
        "Total / Total_VND",
        "Total",
        "Total_VND",
        "doanh_thu",
    ],
    "profit": [
        "Gross_Income_VND",
        "Gross Income",
        "gross_income",
        "loi_nhuan",
    ],
    "date": [
        "Date",
        "date",
        "Ngày",
        "Ngay",
    ],
    "product": [
        "NhomHang_VN",
        "Product line",
        "Product line / NhomHang_VN",
        "Product_Line",
        "product",
        "Mặt hàng",
    ],
}


def _find_column(df: pd.DataFrame, field: str) -> str:
    """Tìm tên cột tương ứng; báo lỗi rõ ràng nếu không tìm thấy."""
    candidates = COLUMN_ALIASES[field]
    for name in candidates:
        if name in df.columns:
            return name

    raise KeyError(
        f"Không tìm thấy cột '{field}'. "
        f"Các cột hiện có: {list(df.columns)}. "
        f"Tên cột được hỗ trợ: {candidates}"
    )


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột cần dùng và loại bỏ dòng thiếu dữ liệu quan trọng."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào phải là pandas.DataFrame.")

    revenue_col = _find_column(df, "revenue")
    profit_col = _find_column(df, "profit")
    date_col = _find_column(df, "date")
    product_col = _find_column(df, "product")

    data = df[[revenue_col, profit_col, date_col, product_col]].copy()
    data.columns = ["revenue", "profit", "date", "product"]

    # Chuyển doanh thu và lợi nhuận về kiểu số.
    data["revenue"] = pd.to_numeric(data["revenue"], errors="coerce")
    data["profit"] = pd.to_numeric(data["profit"], errors="coerce")

    # Dữ liệu từ TV1 đã ở chuẩn ISO (YYYY-MM-DD), không dùng dayfirst=True để tránh parse sai.
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Loại bỏ bản ghi thiếu các trường cần thiết cho phân tích.
    data = data.dropna(subset=["revenue", "profit", "date", "product"]).copy()
    data["product"] = data["product"].astype(str).str.strip()
    data = data[data["product"] != ""]

    return data


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thống kê mô tả cho doanh thu và lợi nhuận:
    count, mean, std, min, 25%, 50%, 75%, max, median, variance.
    """
    numeric = data[["revenue", "profit"]]
    stats = numeric.describe().T
    stats["median"] = numeric.median()
    stats["variance"] = numeric.var()
    stats["total"] = numeric.sum()
    stats["skewness"] = numeric.skew()
    stats["kurtosis"] = numeric.kurtosis()

    # Dùng NumPy để bổ sung các chỉ số tổng hợp.
    stats["numpy_mean"] = [
        np.mean(numeric[col].to_numpy()) for col in numeric.columns
    ]
    stats["numpy_std"] = [
        np.std(numeric[col].to_numpy(), ddof=1) for col in numeric.columns
    ]
    return stats.round(2)


def revenue_profit_by_day(data: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp số hóa đơn, doanh thu và lợi nhuận theo ngày."""
    result = (
        data.assign(day=data["date"].dt.date)
        .groupby("day", as_index=False)
        .agg(
            invoice_count=("revenue", "size"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            average_revenue=("revenue", "mean"),
            average_profit=("profit", "mean"),
        )
        .sort_values("day")
        .reset_index(drop=True)
    )
    return result.round(2)


def revenue_profit_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp số hóa đơn, doanh thu và lợi nhuận theo tháng."""
    result = (
        data.assign(month=data["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            invoice_count=("revenue", "size"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            average_revenue=("revenue", "mean"),
            average_profit=("profit", "mean"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    return result.round(2)


def revenue_profit_by_product(data: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp số hóa đơn, doanh thu và lợi nhuận theo mặt hàng."""
    result = (
        data.groupby("product", as_index=False)
        .agg(
            invoice_count=("revenue", "size"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            average_revenue=("revenue", "mean"),
            average_profit=("profit", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )
    return result.round(2)


def run_revenue_analysis(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Hàm chạy toàn bộ phần TV2.

    Trả về dictionary gồm:
    descriptive, by_day, by_month, by_product.
    """
    data = _prepare_data(df)

    if data.empty:
        raise ValueError(
            "Không còn dữ liệu hợp lệ sau khi làm sạch. "
            "Hãy kiểm tra các cột doanh thu, lợi nhuận, ngày và mặt hàng."
        )

    return {
        "descriptive": descriptive_statistics(data),
        "by_day": revenue_profit_by_day(data),
        "by_month": revenue_profit_by_month(data),
        "by_product": revenue_profit_by_product(data),
    }


def export_revenue_results(
    results: Dict[str, pd.DataFrame],
    output_dir: Path | str = ROOT / "outputs",
) -> None:
    """Xuất 4 bảng kết quả thống kê của TV2 vào thư mục outputs/."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results["descriptive"].to_csv(
        out_path / "descriptive_statistics.csv", encoding="utf-8-sig"
    )
    results["by_day"].to_csv(
        out_path / "revenue_profit_by_day.csv", index=False, encoding="utf-8-sig"
    )
    results["by_month"].to_csv(
        out_path / "revenue_profit_by_month.csv", index=False, encoding="utf-8-sig"
    )
    results["by_product"].to_csv(
        out_path / "revenue_profit_by_product.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    input_path = ROOT / "outputs" / "cleaned_sales_vn.csv"
    output_dir = ROOT / "outputs"

    try:
        sales_df = pd.read_csv(input_path)
        analysis_results = run_revenue_analysis(sales_df)

        for name, result_df in analysis_results.items():
            print(f"\n{'=' * 15} {name.upper()} {'=' * 15}")
            print(result_df.to_string(index=False))

        export_revenue_results(analysis_results, output_dir=output_dir)
        print(f"\nĐã hoàn thành phân tích TV2 và xuất 4 file CSV vào {output_dir}.")

    except FileNotFoundError:
        print(
            f"Không tìm thấy file dữ liệu: {input_path}\n"
            "Hãy chạy src/preprocessing.py trước để tạo file CSV đã làm sạch."
        )
    except (KeyError, ValueError, TypeError) as error:
        print(f"Lỗi dữ liệu: {error}")