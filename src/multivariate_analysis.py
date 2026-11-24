"""Module Phân tích Kết hợp Đa chiều (Multivariate Analysis - OLAP) cho chuỗi Cửa hàng Tiện lợi.

Phụ trách: Thành viên 3 (TV3) - Chuyên viên Phân tích Kết hợp Đa chiều.
Nhiệm vụ:
    - Xây dựng bảng chéo (Pivot Table) Khung giờ × Thứ trong tuần (tìm khung giờ vàng).
    - Phân tích kết hợp Nhóm mặt hàng × Khung giờ (doanh thu & số lượng đơn).
    - Phân tích đa chiều 3D (Mặt hàng × Khung giờ × Thứ/Ngày) xếp hạng Top 5 và Bottom 5.
    - Mở rộng phân tích đa chiều theo 3 chi nhánh đặc trưng tại TP.HCM:
        + Branch A: Quận 1 - Khu Văn phòng Trung tâm
        + Branch B: Thủ Đức - Khu Đại học & Ký túc xá
        + Branch C: Quận 7 - Khu Dân cư & Chung cư cao tầng
    - Cung cấp dữ liệu chuẩn hóa (multi_dict) cho TV4 (Visualizer), TV5 (Insights/Main), TV6 (QA).
"""

from collections import namedtuple
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Định nghĩa các hằng số chuẩn hóa bối cảnh bán lẻ Việt Nam
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_sales_vn.csv"

WEEKDAYS_ORDER = [
    "Thứ 2",
    "Thứ 3",
    "Thứ 4",
    "Thứ 5",
    "Thứ 6",
    "Thứ 7",
    "Chủ nhật",
]

TIME_SLOTS_ORDER = [
    "10h-12h Sáng",
    "12h-14h Trưa",
    "14h-18h Chiều",
    "18h-21h Tối",
]

BRANCH_ORDER = [
    "Quận 1 - Văn phòng",
    "Thủ Đức - Đại học",
    "Quận 7 - Dân cư",
]

# Cấu trúc namedtuple để trả về kết quả phân tích 3 chiều và chi nhánh linh hoạt
ThreeDimResult = namedtuple("ThreeDimResult", ["full", "top_5", "bottom_5"])
BranchResult = namedtuple("BranchResult", ["full", "top_by_slot", "pivot"])


def _validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
    """Kiểm tra tính hợp lệ của DataFrame đầu vào theo Data Contract."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dữ liệu đầu vào bắt buộc phải là một pandas.DataFrame.")
    if df.empty:
        raise ValueError("DataFrame rỗng, không có dữ liệu để phân tích.")
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame thiếu các cột bắt buộc: {', '.join(sorted(missing))}")


def analyze_hour_by_weekday(
    df: pd.DataFrame,
    value_col: str = "Total_VND",
    aggfunc: str = "sum"
) -> pd.DataFrame:
    """Tạo bảng chéo 2 chiều (pivot_table) Giờ giao dịch (Hour) × Thứ trong tuần (DayOfWeek).

    Mục tiêu:
        - Nhận diện các 'điểm nóng doanh thu' và xác định khung giờ vàng theo từng ngày.
        - Xử lý các ô khuyết bằng fill_value=0.
        - Sắp xếp cột thứ theo thứ tự logic (Thứ 2 -> Chủ nhật) thay vì bảng chữ cái.
        - Sắp xếp giờ từ 10 đến 20.

    Tham số:
        df: DataFrame sạch từ preprocessing.
        value_col: Cột giá trị tính toán (mặc định 'Total_VND').
        aggfunc: Hàm tổng hợp ('sum', 'count', 'mean').

    Trả về:
        pivot_hw: DataFrame với index là Hour, columns là DayOfWeek.
                  Đồng thời lưu khung giờ vàng vào thuộc tính `.attrs`.
    """
    _validate_dataframe(df, ["Hour", "DayOfWeek", value_col])

    pivot = df.pivot_table(
        index="Hour",
        columns="DayOfWeek",
        values=value_col,
        aggfunc=aggfunc,
        fill_value=0
    )

    # Đảm bảo thứ tự logic Thứ 2 -> Chủ nhật
    ordered_cols = [col for col in WEEKDAYS_ORDER if col in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols, fill_value=0)

    # Đảm bảo thứ tự tăng dần theo giờ
    pivot = pivot.sort_index()

    # Tính toán khung giờ vàng (Golden Hour)
    total_by_hour = pivot.sum(axis=1)
    golden_hour_overall = int(total_by_hour.idxmax())
    golden_hour_by_day = {
        col: int(pivot[col].idxmax()) for col in pivot.columns
    }

    # Lưu metadata để TV5 và người dùng dễ dàng truy xuất
    pivot.attrs["golden_hour_overall"] = golden_hour_overall
    pivot.attrs["golden_hour_by_day"] = golden_hour_by_day
    pivot.attrs["hourly_total"] = total_by_hour

    return pivot


def analyze_category_by_timeslot(
    df: pd.DataFrame,
    value_col: str = "Total_VND"
) -> pd.DataFrame:
    """Phân tích kết hợp Nhóm mặt hàng (NhomHang_VN) × Khung giờ (TimeSlot).

    Mục tiêu:
        - So sánh doanh thu và số lượng đơn hàng của từng nhóm mặt hàng qua 4 khung giờ.
        - Phát hiện sự chuyển dịch giỏ hàng: F&B bùng nổ giờ trưa vs Đồ gia dụng/Mỹ phẩm buổi tối.
        - Đảm bảo các khung giờ được sắp xếp theo nhịp sinh hoạt đô thị: Sáng -> Trưa -> Chiều -> Tối.
        - Tương thích 100% với hàm `plot_multivariate_category_timeslot` của TV4 (visualizer.py).

    Tham số:
        df: DataFrame sạch từ preprocessing.
        value_col: Cột giá trị doanh thu (mặc định 'Total_VND').

    Trả về:
        pivot_rev: DataFrame pivot doanh thu (Index: NhomHang_VN, Columns: TimeSlot).
                   Thuộc tính `.attrs['orders_pivot']` chứa pivot số lượng đơn hàng,
                   `.attrs['summary']` chứa bảng tổng hợp chi tiết kèm tỷ trọng %.
    """
    _validate_dataframe(df, ["NhomHang_VN", "TimeSlot", value_col, "Invoice ID"])

    # 1. Bảng Pivot Doanh thu (phục vụ vẽ biểu đồ TV4 và tính toán tài chính)
    pivot_rev = df.pivot_table(
        index="NhomHang_VN",
        columns="TimeSlot",
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )
    ordered_slots = [s for s in TIME_SLOTS_ORDER if s in pivot_rev.columns]
    pivot_rev = pivot_rev.reindex(columns=ordered_slots, fill_value=0)

    # 2. Bảng Pivot Số lượng đơn hàng (phục vụ đối soát tần suất mua)
    pivot_orders = df.pivot_table(
        index="NhomHang_VN",
        columns="TimeSlot",
        values="Invoice ID",
        aggfunc="count",
        fill_value=0
    )
    pivot_orders = pivot_orders.reindex(columns=ordered_slots, fill_value=0)

    # 3. Tạo bảng tóm tắt chi tiết 2 chỉ số kết hợp
    summary_records = []
    total_revenue = df[value_col].sum()
    for cat in pivot_rev.index:
        for slot in ordered_slots:
            rev = float(pivot_rev.loc[cat, slot])
            orders = int(pivot_orders.loc[cat, slot])
            aov = rev / orders if orders > 0 else 0.0
            share_total = (rev / total_revenue) * 100 if total_revenue > 0 else 0.0
            slot_total_rev = pivot_rev[slot].sum()
            share_slot = (rev / slot_total_rev) * 100 if slot_total_rev > 0 else 0.0

            summary_records.append({
                "NhomHang_VN": cat,
                "TimeSlot": slot,
                "DoanhThu_VND": rev,
                "SoLuongDon": orders,
                "AOV_VND": aov,
                "TyTrong_ToanChuoi_Pct": round(share_total, 2),
                "TyTrong_TrongKhungGio_Pct": round(share_slot, 2),
            })

    summary_df = pd.DataFrame(summary_records)

    # Gắn các kết quả bổ trợ vào .attrs để không làm thay đổi định dạng DataFrame chính
    pivot_rev.attrs["orders_pivot"] = pivot_orders
    pivot_rev.attrs["summary"] = summary_df

    return pivot_rev


def analyze_3d_category_timeslot_weekday(
    df: pd.DataFrame,
    value_col: str = "Total_VND",
    top_n: int = 5
) -> ThreeDimResult:
    """Phân tích tổ hợp 3 chiều đầy đủ: Mặt hàng × Khung giờ × Thứ trong tuần.

    Mục tiêu:
        - Tìm kiếm các tổ hợp doanh thu cao nhất (Top 5 - Tổ hợp vàng) và thấp nhất (Bottom 5).
        - Tính giá trị đơn hàng trung bình (AOV) và tỷ trọng % doanh thu đóng góp.
        - Làm minh chứng số liệu định lượng cho Nhận định 2 của TV5.

    Tham số:
        df: DataFrame sạch từ preprocessing.
        value_col: Cột doanh thu (mặc định 'Total_VND').
        top_n: Số lượng tổ hợp đầu/cuối bảng cần trích xuất (mặc định 5).

    Trả về:
        ThreeDimResult namedtuple gồm:
            - full: DataFrame tổng hợp toàn bộ các tổ hợp 3 chiều (đã sắp xếp giảm dần).
            - top_5: DataFrame Top 5 tổ hợp doanh thu lớn nhất.
            - bottom_5: DataFrame Bottom 5 tổ hợp doanh thu thấp nhất.
    """
    _validate_dataframe(df, ["NhomHang_VN", "TimeSlot", "DayOfWeek", value_col, "Invoice ID"])

    total_revenue = df[value_col].sum()

    grouped = df.groupby(["NhomHang_VN", "TimeSlot", "DayOfWeek"], as_index=False, observed=False).agg(
        DoanhThu_VND=(value_col, "sum"),
        SoLuongDon=("Invoice ID", "count"),
    )

    grouped["AOV_VND"] = grouped["DoanhThu_VND"] / grouped["SoLuongDon"]
    grouped["TyTrong_Pct"] = (grouped["DoanhThu_VND"] / total_revenue) * 100

    # Sắp xếp giảm dần theo doanh thu
    grouped_sorted = grouped.sort_values(by="DoanhThu_VND", ascending=False).reset_index(drop=True)

    top_df = grouped_sorted.head(top_n).copy().reset_index(drop=True)
    bottom_df = grouped_sorted.tail(top_n).copy().reset_index(drop=True)

    # Làm tròn hiển thị
    for target in (grouped_sorted, top_df, bottom_df):
        target["TyTrong_Pct"] = target["TyTrong_Pct"].round(2)
        target["AOV_VND"] = target["AOV_VND"].round(0)

    return ThreeDimResult(full=grouped_sorted, top_5=top_df, bottom_5=bottom_df)


def analyze_branch_category_timeslot(
    df: pd.DataFrame,
    value_col: str = "Total_VND"
) -> BranchResult:
    """Mở rộng phân tích đa chiều theo Chi nhánh bán lẻ Việt Nam (Branch/KhuVuc_VN).

    Bối cảnh thực tiễn Việt Nam:
        - Quận 1 (Khu Văn phòng): Sức mua bữa trưa 12h-14h (F&B) cực cao.
        - Thủ Đức (Khu Đại học): Sinh viên tập trung mua sắm buổi tối 18h-21h.
        - Quận 7 (Khu Chung cư Dân cư): Tiêu thụ Đồ gia dụng & Chăm sóc cá nhân vào chiều/tối.

    Tham số:
        df: DataFrame sạch từ preprocessing.
        value_col: Cột doanh thu (mặc định 'Total_VND').

    Trả về:
        BranchResult namedtuple gồm:
            - full: DataFrame chi tiết kết hợp Chi nhánh × Khung giờ × Nhóm hàng.
            - top_by_slot: Nhóm mặt hàng bán chạy nhất (Top 1) tại từng Chi nhánh theo từng Khung giờ.
            - pivot: Bảng Pivot 2 tầng (MultiIndex: [KhuVuc_VN, NhomHang_VN] × Columns: TimeSlot).
    """
    _validate_dataframe(df, ["KhuVuc_VN", "TimeSlot", "NhomHang_VN", value_col, "Invoice ID"])

    grouped = df.groupby(["KhuVuc_VN", "TimeSlot", "NhomHang_VN"], as_index=False, observed=False).agg(
        DoanhThu_VND=(value_col, "sum"),
        SoLuongDon=("Invoice ID", "count"),
    )

    grouped["AOV_VND"] = (grouped["DoanhThu_VND"] / grouped["SoLuongDon"]).round(0)

    # Tính tỷ trọng doanh thu trong từng chi nhánh
    branch_totals = df.groupby("KhuVuc_VN")[value_col].sum().to_dict()
    grouped["TyTrong_ChiNhanh_Pct"] = grouped.apply(
        lambda row: round((row["DoanhThu_VND"] / branch_totals.get(row["KhuVuc_VN"], 1.0)) * 100, 2),
        axis=1,
    )

    # Xác định Top 1 mặt hàng theo từng Chi nhánh và từng Khung giờ
    idx_max = grouped.groupby(["KhuVuc_VN", "TimeSlot"])["DoanhThu_VND"].idxmax()
    top_by_slot = grouped.loc[idx_max].sort_values(by=["KhuVuc_VN", "TimeSlot"]).reset_index(drop=True)

    # Pivot table đa tầng
    pivot = df.pivot_table(
        index=["KhuVuc_VN", "NhomHang_VN"],
        columns="TimeSlot",
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )
    ordered_slots = [s for s in TIME_SLOTS_ORDER if s in pivot.columns]
    pivot = pivot.reindex(columns=ordered_slots, fill_value=0)

    return BranchResult(full=grouped, top_by_slot=top_by_slot, pivot=pivot)


def run_multivariate_analysis(
    df_or_path: Optional[Union[pd.DataFrame, str, Path]] = None
) -> Dict[str, object]:
    """Hàm điều phối toàn bộ pipeline phân tích kết hợp đa chiều của TV3.

    Quy trình:
        1. Nạp và kiểm tra dữ liệu sạch.
        2. Chạy 4 phép phân tích đa chiều bắt buộc.
        3. Đóng gói kết quả thành `multi_dict` đúng chuẩn Data Contract cho TV4, TV5, TV6.
        4. In báo cáo tóm tắt các phát hiện kinh doanh nổi bật.

    Trả về:
        multi_dict: Dictionary chứa các Pivot DataFrames, Top/Bottom combinations và ma trận chi nhánh.
    """
    # 1. Đọc dữ liệu
    if df_or_path is None:
        df_or_path = DEFAULT_DATA_PATH

    if isinstance(df_or_path, (str, Path)):
        csv_path = Path(df_or_path)
        if not csv_path.is_file():
            # Nếu chưa có file cleaned, thử gọi preprocessing để tạo tự động
            from preprocessing import run_preprocessing
            print(f"[TV3] Chưa tìm thấy {csv_path}, đang kích hoạt pipeline tiền xử lý (TV1)...")
            df = run_preprocessing(output_path=csv_path)
        else:
            df = pd.read_csv(csv_path)
    elif isinstance(df_or_path, pd.DataFrame):
        df = df_or_path.copy()
    else:
        raise TypeError("df_or_path phải là pandas.DataFrame hoặc đường dẫn tệp CSV hợp lệ.")

    print("=" * 80)
    print("TV3: BẮT ĐẦU PHÂN TÍCH KẾT HỢP ĐA CHIỀU (MULTIVARIATE ANALYSIS - OLAP)")
    print("=" * 80)

    # 2. Thực hiện các phân tích đa chiều
    pivot_hw = analyze_hour_by_weekday(df, value_col="Total_VND")
    pivot_ct = analyze_category_by_timeslot(df, value_col="Total_VND")
    pivot_ct_orders = pivot_ct.attrs["orders_pivot"]
    ct_summary = pivot_ct.attrs["summary"]
    res_3d = analyze_3d_category_timeslot_weekday(df, value_col="Total_VND", top_n=5)
    res_branch = analyze_branch_category_timeslot(df, value_col="Total_VND")

    golden_hour_overall = pivot_hw.attrs["golden_hour_overall"]
    golden_hour_by_day = pivot_hw.attrs["golden_hour_by_day"]

    # 3. Đóng gói từ điển đa chiều (Data Contract cho toàn bộ nhóm)
    multi_dict = {
        "hour_weekday": pivot_hw,
        "category_timeslot": pivot_ct,
        "category_timeslot_orders": pivot_ct_orders,
        "category_timeslot_summary": ct_summary,
        "3d_full": res_3d.full,
        "top_5_combinations": res_3d.top_5,
        "bottom_5_combinations": res_3d.bottom_5,
        "branch_category_timeslot": res_branch.full,
        "branch_pivot": res_branch.pivot,
        "top_by_branch_slot": res_branch.top_by_slot,
        "golden_hour_overall": golden_hour_overall,
        "golden_hour_by_day": golden_hour_by_day,
    }

    # 4. In báo cáo tóm tắt ra màn hình chuẩn mực
    print(f"\n[1] MA TRẬN KHUNG GIỜ × THỨ TRONG TUẦN (Hour × DayOfWeek):")
    print(f"    - Kích thước ma trận: {pivot_hw.shape} (11 khung giờ từ 10h-20h × 7 ngày trong tuần)")
    print(f"    - Khung giờ vàng toàn chuỗi (Golden Hour): {golden_hour_overall}h:00 - {golden_hour_overall}:59")
    print(f"      (Đạt doanh thu: {pivot_hw.loc[golden_hour_overall].sum():,.0f} VNĐ)")
    print("    - Khung giờ đỉnh theo từng ngày:")
    for day, hr in golden_hour_by_day.items():
        rev_hr = pivot_hw.loc[hr, day]
        print(f"      + {day:8s}: {hr}h:00 ({rev_hr:,.0f} VNĐ)")

    print(f"\n[2] PHÂN TÍCH NHÓM MẶT HÀNG × KHUNG GIỜ (Category × TimeSlot):")
    print(f"    - Nhóm hàng dẫn đầu ca Trưa (12h-14h): {pivot_ct['12h-14h Trưa'].idxmax()} "
          f"({pivot_ct.loc[pivot_ct['12h-14h Trưa'].idxmax(), '12h-14h Trưa']:,.0f} VNĐ)")
    print(f"    - Nhóm hàng dẫn đầu ca Tối  (18h-21h): {pivot_ct['18h-21h Tối'].idxmax()} "
          f"({pivot_ct.loc[pivot_ct['18h-21h Tối'].idxmax(), '18h-21h Tối']:,.0f} VNĐ)")

    print(f"\n[3] TOP 5 TỔ HỢP ĐA CHIỀU VÀNG (Mặt hàng × Khung giờ × Thứ):")
    print("-" * 88)
    print(f"{'Mặt hàng':<35} | {'Khung giờ':<14} | {'Thứ':<9} | {'Doanh thu (VNĐ)':<15} | {'Tỷ trọng':<8} | {'Đơn'}")
    print("-" * 88)
    for _, row in res_3d.top_5.iterrows():
        print(f"{row['NhomHang_VN']:<35} | {row['TimeSlot']:<14} | {row['DayOfWeek']:<9} | "
              f"{row['DoanhThu_VND']:>15,.0f} | {row['TyTrong_Pct']:>6.2f}% | {row['SoLuongDon']:>3}")
    print("-" * 88)

    print(f"\n[4] BOTTOM 5 TỔ HỢP ĐA CHIỀU THẤP ĐIỂM:")
    print("-" * 88)
    for _, row in res_3d.bottom_5.iterrows():
        print(f"{row['NhomHang_VN']:<35} | {row['TimeSlot']:<14} | {row['DayOfWeek']:<9} | "
              f"{row['DoanhThu_VND']:>15,.0f} | {row['TyTrong_Pct']:>6.2f}% | {row['SoLuongDon']:>3}")
    print("-" * 88)

    print(f"\n[5] MẶT HÀNG DẪN ĐẦU THEO CHI NHÁNH & KHUNG GIỜ (TP.HCM):")
    for _, row in res_branch.top_by_slot.iterrows():
        print(f"    * {row['KhuVuc_VN']:<20} | {row['TimeSlot']:<14} -> {row['NhomHang_VN']:<35} "
              f"({row['DoanhThu_VND']:>12,.0f} đ | {row['SoLuongDon']} đơn)")

    print("\n" + "=" * 80)
    print("TV3: HOÀN THÀNH PHÂN TÍCH ĐA CHIỀU XUẤT SẮC - SẴN SÀNG CHUYỂN TIẾP CHO TV4, TV5, TV6!")
    print("=" * 80)

    return multi_dict


if __name__ == "__main__":
    run_multivariate_analysis()
