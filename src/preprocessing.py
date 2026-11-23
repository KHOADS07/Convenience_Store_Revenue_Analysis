"""Chuẩn hóa dữ liệu Supermarket Sales cho pipeline phân tích doanh thu."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
USD_TO_VND = 25_000
BRANCH_CODES = {"Alex": "A", "Cairo": "B", "Giza": "C"}
REGIONS = {
    "A": "Quận 1 - Văn phòng",
    "B": "Thủ Đức - Đại học",
    "C": "Quận 7 - Dân cư",
}
PRODUCTS = {
    "Food and beverages": "Đồ ăn nhanh & Thức uống",
    "Health and beauty": "Chăm sóc cá nhân & Mỹ phẩm mini",
    "Electronic accessories": "Phụ kiện công nghệ & Văn phòng phẩm",
    "Home and lifestyle": "Đồ gia dụng tiện ích",
    "Sports and travel": "Thể thao & Du lịch",
    "Fashion accessories": "Phụ kiện thời trang",
}
PAYMENTS = {
    "Ewallet": "Ví điện tử / QR",
    "Cash": "Tiền mặt",
    "Credit card": "Thẻ ngân hàng",
}
WEEKDAYS = dict(enumerate(["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]))
TIME_SLOTS = ["10h-12h Sáng", "12h-14h Trưa", "14h-18h Chiều", "18h-21h Tối"]
NUMERIC_COLUMNS = {
    "Unit price", "Quantity", "Tax 5%", "Total", "cogs",
    "gross margin percentage", "gross income", "Rating",
}
REQUIRED_COLUMNS = {
    "Invoice ID", "Branch", "Product line", "Date", "Time",
    "Payment", "Total", "gross income", "Quantity", "Rating",
}


def load_raw_data(filepath):
    """Đọc CSV, chuẩn hóa tên cột và giữ hóa đơn đầu tiên theo mã."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {filepath}")
    df = pd.read_csv(filepath, dtype={"Invoice ID": "string"})
    print(f"Input: {len(df)} rows, {len(df.columns)} columns.")
    df.columns = df.columns.str.strip()
    if "Total" not in df.columns:
        df = df.rename(columns={"Sales": "Total"})
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing_columns))}")
    invoice_ids = df["Invoice ID"].str.strip().replace("", pd.NA)
    df["Invoice ID"] = invoice_ids
    # Mã thiếu được giữ đến bước báo cáo chất lượng dữ liệu.
    duplicate_ids = invoice_ids.notna() & invoice_ids.duplicated()
    print(f"Duplicate invoices removed: {duplicate_ids.sum()}.")
    return df.loc[~duplicate_ids].reset_index(drop=True)


def handle_missing_values(df):
    """Báo cáo dữ liệu thiếu rồi điền median/mode theo kiểu dữ liệu."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    for col in NUMERIC_COLUMNS.intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    counts = df.isna().sum()
    print(f"Rows with missing values: {df.isna().any(axis=1).sum()}.")
    for col, count in counts.items():
        if count:
            print(f"  {col}: {count} ({count / len(df):.2%})")
    # Mã hóa đơn là khóa. Điền theo mode sẽ tạo mã trùng.
    df = df.dropna(subset=["Invoice ID"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Không còn hóa đơn có mã hợp lệ.")
    for col in df.columns[df.isna().any()]:
        values = df[col].dropna()
        if values.empty:
            raise ValueError(f"Cột {col} thiếu toàn bộ, không thể tính median hoặc mode.")
        fill = values.median() if pd.api.types.is_numeric_dtype(values) else values.mode().iloc[0]
        df[col] = df[col].fillna(fill)
    return df


def detect_and_handle_outliers(df, col="Total", method="iqr"):
    """Báo cả IQR và Z-score, rồi cắt ngưỡng theo phương pháp được chọn."""
    method = method.lower()
    if method not in {"iqr", "zscore"}:
        raise ValueError("method phải là 'iqr' hoặc 'zscore'.")
    df = df.copy()
    values = df[col].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Cột {col} phải có dữ liệu số hữu hạn.")
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    mean, std = values.mean(), values.std()
    bounds = {
        "iqr": (q1 - 1.5 * iqr, q3 + 1.5 * iqr),
        "zscore": (mean - 3 * std, mean + 3 * std),
    }
    flags = {}
    for name, (lower, upper) in bounds.items():
        flags[name] = (values < lower) | (values > upper)
        print(f"{col} {name}: {flags[name].sum()} outliers, bounds [{lower:.4f}, {upper:.4f}].")
    lower, upper = bounds[method]
    df[f"{col}_Clean"] = np.clip(values, lower, upper)
    flagged = flags["iqr"] | flags["zscore"]
    report = df.loc[flagged, ["Invoice ID", col, f"{col}_Clean"]].copy()
    report["IsIQR"] = flags["iqr"][flagged]
    report["IsZScore"] = flags["zscore"][flagged]
    df.attrs["outlier_report"] = report
    df["Is_Outlier"] = flags[method]
    if not report.empty:
        print(report.to_string(index=False))
    return df


def engineer_vn_features(df):
    """Tạo đặc trưng thời gian và các cột ánh xạ sang bối cảnh Việt Nam."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="raise")
    raw_time = df["Time"].astype("string")
    times = pd.to_datetime(raw_time, format="%I:%M:%S %p", errors="coerce")
    for fmt in ("%I:%M %p", "%H:%M:%S", "%H:%M"):
        unparsed = times.isna()
        times.loc[unparsed] = pd.to_datetime(raw_time.loc[unparsed], format=fmt, errors="coerce")
    if times.isna().any():
        raise ValueError(f"Giờ giao dịch không hợp lệ: {raw_time[times.isna()].iloc[0]}")
    df["Time"] = df["Date"] + (times - times.dt.normalize())
    df["Month"] = df["Date"].dt.month.astype(int)
    weekday = df["Date"].dt.dayofweek
    df["DayOfWeek"] = weekday.map(WEEKDAYS)
    df["IsWeekend"] = weekday >= 5
    df["Hour"] = times.dt.hour.astype(int)
    df["TimeSlot"] = pd.cut(df["Hour"], [10, 12, 14, 18, 21], labels=TIME_SLOTS, right=False)
    df["Branch"] = df["Branch"].replace(BRANCH_CODES)
    for source, target, mapping in (
        ("Branch", "KhuVuc_VN", REGIONS),
        ("Product line", "NhomHang_VN", PRODUCTS),
        ("Payment", "Payment_VN", PAYMENTS),
    ):
        df[target] = df[source].map(mapping)
        if df[target].isna().any():
            unknown = df.loc[df[target].isna(), source].unique()
            raise ValueError(f"{source} có giá trị chưa ánh xạ: {unknown.tolist()}")
    if df["TimeSlot"].isna().any():
        raise ValueError("Có hóa đơn ngoài các khung giờ 10:00 đến trước 21:00.")
    df["Total"] = df["Total"].astype(float)
    df["Total_VND"] = df["Total"] * USD_TO_VND
    df["Total_VND_Clean"] = df["Total_Clean"] * USD_TO_VND
    df["Gross_Income_VND"] = df["gross income"].astype(float) * USD_TO_VND
    df["Unit_Price_VND"] = df["Unit price"].astype(float) * USD_TO_VND
    df["COGS_VND"] = df["cogs"].astype(float) * USD_TO_VND
    return df


def run_preprocessing(input_path=ROOT / "data" / "supermarket_sales.csv",
                      output_path=ROOT / "outputs" / "cleaned_sales_vn.csv",):
                          
    """Chạy pipeline, xuất CSV và trả về DataFrame cho các bước sau."""
    df = load_raw_data(input_path)
    df = handle_missing_values(df)
    df = detect_and_handle_outliers(df)
    df = engineer_vn_features(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Exported {len(df)} invoices to {output_path}.")
    return df


if __name__ == "__main__":
   run_preprocessing()
