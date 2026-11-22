"""
insights.py - TV5: Trích xuất nhận định kinh doanh + xuất báo cáo.

Mọi con số trong nhận định đều được tính động từ DataFrame (không gõ cứng),
nên đổi tệp dữ liệu đầu vào thì nhận định tự cập nhật.

Cột cần có (theo Data Contract của TV1):
    Invoice ID, Total_VND, Hour, TimeSlot, DayOfWeek, NhomHang_VN,
    KhuVuc_VN, Payment_VN   (tuỳ chọn: Gross_Income_VND, Rating, Date)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REV = "Total_VND"
USD_TO_VND = 25000
REQUIRED_COLS = ["Total_VND", "Hour", "TimeSlot", "DayOfWeek",
                 "NhomHang_VN", "KhuVuc_VN", "Payment_VN"]


# ----------------------------------------------------------------- helpers
def _vn_num(text: str) -> str:
    """Đổi 1,234.5 -> 1.234,5 (kiểu số Việt Nam)."""
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_vnd(x: float) -> str:
    return _vn_num(f"{x:,.0f}") + " VNĐ"


def fmt_trieu(x: float) -> str:
    return _vn_num(f"{x / 1e6:,.1f}") + " triệu VNĐ"


def fmt_pct(x: float) -> str:
    return _vn_num(f"{x:.1f}") + "%"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if REV not in df.columns and "Total" in df.columns:
        df[REV] = df["Total"] * USD_TO_VND
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"insights.py: DataFrame thiếu cột theo Data Contract: {missing}")
    if "Invoice ID" not in df.columns:
        df["Invoice ID"] = df.index.astype(str)
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """KPI tổng quan - tính trực tiếp từ DataFrame sạch."""
    df = _prepare(df)
    kpi = {
        "tong_doanh_thu": float(df[REV].sum()),
        "so_don": int(len(df)),
        "aov": float(df[REV].mean()),
        "so_khu_vuc": int(df["KhuVuc_VN"].nunique()),
    }
    if "Gross_Income_VND" in df.columns:
        kpi["loi_nhuan_gop"] = float(df["Gross_Income_VND"].sum())
        kpi["bien_loi_nhuan"] = kpi["loi_nhuan_gop"] / kpi["tong_doanh_thu"] * 100
    if "Rating" in df.columns:
        kpi["rating_tb"] = float(df["Rating"].mean())
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], errors="coerce")
        kpi["tu_ngay"], kpi["den_ngay"] = d.min(), d.max()
    return kpi


def _slot_context(slot: str) -> str:
    s = str(slot)
    if s.startswith("10"):
        return "nhịp mua sắm buổi sáng và chuẩn bị bữa trưa"
    if s.startswith("12"):
        return "giờ nghỉ trưa của dân văn phòng và sinh viên"
    if s.startswith("14"):
        return "giờ xế chiều và tan tầm"
    if s.startswith("18"):
        return "giờ tan làm/tan học, ăn tối nhanh và mua đồ dùng về nhà"
    return "nhịp sinh hoạt đô thị"


# -------------------------------------------------------------- Nhận định 1
def _insight_time_slots(df: pd.DataFrame) -> dict:
    total = df[REV].sum()
    g = df.groupby("TimeSlot").agg(doanh_thu=(REV, "sum"), so_don=(REV, "size"))
    g["so_gio"] = df.groupby("TimeSlot")["Hour"].nunique()
    g["ty_trong"] = g["doanh_thu"] / total * 100
    g["dt_moi_gio"] = g["doanh_thu"] / g["so_gio"]       # công bằng giữa khung 2h và 4h

    top, low = g["dt_moi_gio"].idxmax(), g["dt_moi_gio"].idxmin()
    gap_slot = (g.loc[top, "dt_moi_gio"] - g.loc[low, "dt_moi_gio"]) / g.loc[low, "dt_moi_gio"] * 100

    hourly = df.groupby("Hour")[REV].sum()
    peak_h, low_h = int(hourly.idxmax()), int(hourly.idxmin())
    gap_hour = (hourly.max() - hourly.min()) / hourly.min() * 100

    evidence = [
        f"{slot}: {fmt_trieu(r.doanh_thu)} | {int(r.so_don)} đơn | {fmt_pct(r.ty_trong)} doanh thu "
        f"| {fmt_trieu(r.dt_moi_gio)}/giờ"
        for slot, r in g.sort_values("dt_moi_gio", ascending=False).iterrows()
    ]
    evidence.append(
        f"Giờ cao nhất: {peak_h}h ({fmt_trieu(hourly.max())}); giờ thấp nhất: {low_h}h "
        f"({fmt_trieu(hourly.min())}) - chênh {fmt_pct(gap_hour)}."
    )
    return {
        "title": "Khung giờ vàng và nhịp sinh hoạt đô thị",
        "finding": (f"Khung {top} có doanh thu bình quân/giờ cao nhất, hơn khung thấp nhất ({low}) "
                    f"{fmt_pct(gap_slot)}; giờ bán tốt nhất là {peak_h}h."),
        "evidence": evidence,
        "explanation": (f"Khung {top} trùng với {_slot_context(top)}, trong khi khung {low} "
                        f"là {_slot_context(low)} nên lượng khách thấp hơn."),
        "action": (f"Dồn nhân viên part-time (sinh viên) vào khung {top} và quanh {peak_h}h; "
                   f"giảm định biên khung {low}; chuẩn bị hàng trước {peak_h}h."),
    }


# -------------------------------------------------------------- Nhận định 2
def _insight_multivariate(df: pd.DataFrame) -> dict:
    total = df[REV].sum()
    combo = (df.groupby(["NhomHang_VN", "TimeSlot", "DayOfWeek"])[REV]
               .agg(doanh_thu="sum", so_don="size", aov="mean")
               .sort_values("doanh_thu", ascending=False))
    top3 = combo.head(3)
    evidence = [
        f"Top {i}: {cat} | {slot} | {day} -> {fmt_trieu(r.doanh_thu)} "
        f"({fmt_pct(r.doanh_thu / total * 100)} tổng), {int(r.so_don)} đơn, AOV {fmt_vnd(r.aov)}"
        for i, ((cat, slot, day), r) in enumerate(top3.iterrows(), 1)
    ]
    cat_rev = df.groupby("NhomHang_VN")[REV].sum().sort_values(ascending=False)
    lead = cat_rev.index[0]
    evidence.append(f"Nhóm hàng dẫn đầu toàn chuỗi: {lead} - {fmt_trieu(cat_rev.iloc[0])} "
                    f"({fmt_pct(cat_rev.iloc[0] / total * 100)}).")
    (c1, s1, d1) = top3.index[0]
    lead_slot = df[df["NhomHang_VN"] == lead].groupby("TimeSlot")[REV].sum().idxmax()
    return {
        "title": "Tổ hợp đa chiều: Mặt hàng x Khung giờ x Thứ",
        "finding": (f"Tổ hợp doanh thu cao nhất là {c1} - {s1} - {d1}; "
                    f"nhóm {lead} bán mạnh nhất ở khung {lead_slot}."),
        "evidence": evidence,
        "explanation": (f"Cơ cấu giỏ hàng dịch chuyển theo giờ và ngày: nhu cầu của {c1} "
                        f"tập trung vào {s1} ({d1}), phù hợp với thói quen mua sắm đô thị."),
        "action": (f"Thiết kế combo/khuyến mãi cho {c1} đúng khung {s1}; tăng tồn kho {c1} "
                   f"trước {d1}; xem xét kích hoạt thêm cho tổ hợp thấp điểm."),
    }


# -------------------------------------------------------------- Nhận định 3
def _insight_payment(df: pd.DataFrame) -> dict:
    n, total = len(df), df[REV].sum()
    g = df.groupby("Payment_VN").agg(so_don=(REV, "size"), doanh_thu=(REV, "sum"), aov=(REV, "mean"))
    g["ty_le_don"] = g["so_don"] / n * 100
    g["ty_trong_dt"] = g["doanh_thu"] / total * 100
    g = g.sort_values("so_don", ascending=False)

    evidence = [
        f"{m}: {int(r.so_don)} đơn ({fmt_pct(r.ty_le_don)}) | {fmt_trieu(r.doanh_thu)} "
        f"({fmt_pct(r.ty_trong_dt)}) | AOV {fmt_vnd(r.aov)}"
        for m, r in g.iterrows()
    ]
    top_method = g.index[0]
    wallet = next((m for m in g.index if "Ví" in str(m) or "QR" in str(m)), None)
    finding = f"{top_method} là kênh thanh toán nhiều giao dịch nhất ({fmt_pct(g.iloc[0].ty_le_don)} số đơn)."
    explanation = "Thanh toán không tiền mặt ngày càng phổ biến ở đô thị Việt Nam (MoMo/ZaloPay/VNPay-QR)."
    action = f"Duy trì kênh {top_method}; theo dõi AOV từng kênh để thiết kế ưu đãi phù hợp."

    if wallet is not None:
        share_by_branch = (df.assign(_w=(df["Payment_VN"] == wallet))
                             .groupby("KhuVuc_VN")["_w"].mean() * 100).sort_values(ascending=False)
        evidence += [f"Tỷ lệ đơn {wallet} tại {kv}: {fmt_pct(v)}" for kv, v in share_by_branch.items()]
        best, worst = share_by_branch.index[0], share_by_branch.index[-1]
        finding = (f"{wallet} chiếm {fmt_pct(g.loc[wallet, 'ty_le_don'])} số đơn, AOV {fmt_vnd(g.loc[wallet, 'aov'])}; "
                   f"dùng nhiều nhất ở {best} ({fmt_pct(share_by_branch.iloc[0])}).")
        if "Tiền mặt" in g.index:
            diff = (g.loc[wallet, "aov"] - g.loc["Tiền mặt", "aov"]) / g.loc["Tiền mặt", "aov"] * 100
            evidence.append(f"AOV {wallet} so với Tiền mặt: {'+' if diff >= 0 else ''}{fmt_pct(diff)}")
        explanation = (f"Khách ở {best} quen thanh toán số; {worst} còn dư địa chuyển đổi "
                       f"({fmt_pct(share_by_branch.iloc[-1])}).")
        action = (f"Hợp tác ví điện tử (MoMo/ZaloPay/VNPay) hoàn tiền/giảm giá theo ngưỡng đơn "
                  f"tại {worst} để nâng tỷ lệ thanh toán số; giữ ưu đãi tại {best}.")
    return {"title": "Hành vi thanh toán số tại Việt Nam", "finding": finding,
            "evidence": evidence, "explanation": explanation, "action": action}


# -------------------------------------------------------------- Nhận định 4
def _insight_outliers(df: pd.DataFrame) -> dict:
    vals = df[REV].to_numpy(dtype=float)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[REV] < lo) | (df[REV] > hi)
    out = df[mask]
    n, total = len(df), df[REV].sum()
    evidence = [f"Ngưỡng IQR: [{fmt_vnd(lo)} ; {fmt_vnd(hi)}] (Q1={fmt_vnd(q1)}, Q3={fmt_vnd(q3)})",
                f"Số đơn outlier: {len(out)}/{n} ({fmt_pct(len(out) / n * 100)})"]
    if out.empty:
        return {"title": "Tác động của đơn ngoại lai (mua sỉ / sự kiện)",
                "finding": "Không phát hiện hóa đơn ngoại lai theo ngưỡng IQR trong dữ liệu này.",
                "evidence": evidence,
                "explanation": "Giá trị hóa đơn phân bố khá đồng đều, không có đơn đột biến.",
                "action": "Tiếp tục giám sát; chưa cần chương trình riêng cho đơn giá trị lớn."}
    share = out[REV].sum() / total * 100
    by_cat = out["NhomHang_VN"].value_counts()
    by_kv = out["KhuVuc_VN"].value_counts()
    top_cat, top_kv = by_cat.index[0], by_kv.index[0]
    evidence += [f"Doanh thu từ đơn outlier: {fmt_trieu(out[REV].sum())} ({fmt_pct(share)} tổng doanh thu)",
                 f"Nhóm hàng nhiều outlier nhất: {top_cat} ({int(by_cat.iloc[0])} đơn)",
                 f"Khu vực nhiều outlier nhất: {top_kv} ({int(by_kv.iloc[0])} đơn)",
                 f"AOV đơn outlier: {fmt_vnd(out[REV].mean())} so với AOV chung {fmt_vnd(df[REV].mean())}"]
    return {
        "title": "Tác động của đơn ngoại lai (mua sỉ / sự kiện)",
        "finding": (f"{len(out)} đơn outlier ({fmt_pct(len(out) / n * 100)} số đơn) đóng góp "
                    f"{fmt_pct(share)} doanh thu, tập trung ở {top_cat} và {top_kv}."),
        "evidence": evidence,
        "explanation": "Các đơn giá trị lớn thường là mua sắm số lượng nhiều (sự kiện, gia đình, mua dự trữ).",
        "action": (f"Xây dựng chương trình thành viên và gói mua số lượng lớn (giá sỉ mini) cho {top_kv}, "
                   f"ưu tiên nhóm {top_cat}."),
    }


# --------------------------------------------------------------- Hàm chính
def generate_business_insights(df: pd.DataFrame, stats_dict: dict | None = None,
                               multi_dict: dict | None = None) -> list[dict]:
    """Sinh 4 nhận định (Phát hiện + Số liệu + Giải thích + Đề xuất) từ dữ liệu.
    stats_dict / multi_dict giữ trong chữ ký theo đề bài; số liệu được tính lại
    trực tiếp từ df (cùng nguồn Total_VND) nên khớp với TV2/TV3."""
    df = _prepare(df)
    insights = [_insight_time_slots(df), _insight_multivariate(df),
                _insight_payment(df), _insight_outliers(df)]
    for i, ins in enumerate(insights, 1):
        ins["id"] = i
    return insights


def export_text_report(insights: list[dict], stats_dict: dict | None, filepath) -> Path:
    """Xuất KPI + nhận định ra outputs/business_report.txt."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    kpi = (stats_dict or {}).get("kpi", {})
    line = "=" * 78
    out = [line, "BÁO CÁO PHÂN TÍCH DOANH THU CỬA HÀNG TIỆN LỢI (VN)",
           f"Thời điểm xuất: {datetime.now():%d/%m/%Y %H:%M}", line, "", "I. CHỈ SỐ KPI TỔNG QUAN", "-" * 78]
    if kpi:
        out.append(f"  Tổng doanh thu     : {fmt_vnd(kpi['tong_doanh_thu'])}")
        out.append(f"  Số hóa đơn         : {kpi['so_don']}")
        out.append(f"  AOV (đơn trung bình): {fmt_vnd(kpi['aov'])}")
        if "loi_nhuan_gop" in kpi:
            out.append(f"  Lợi nhuận gộp      : {fmt_vnd(kpi['loi_nhuan_gop'])} ({fmt_pct(kpi['bien_loi_nhuan'])})")
        if "rating_tb" in kpi:
            rating = _vn_num("{:.2f}".format(kpi["rating_tb"]))
            out.append(f"  Rating trung bình  : {rating}")
        if "tu_ngay" in kpi and pd.notna(kpi["tu_ngay"]):
            out.append(f"  Giai đoạn          : {kpi['tu_ngay']:%d/%m/%Y} - {kpi['den_ngay']:%d/%m/%Y}")
    else:
        out.append("  (Không có KPI)")
    out += ["", f"II. {len(insights)} NHẬN ĐỊNH KINH DOANH", "-" * 78]
    for ins in insights:
        out += ["", f"NHẬN ĐỊNH {ins['id']}: {ins['title'].upper()}",
                f"  * Phát hiện chính : {ins['finding']}", "  * Số liệu chứng minh:"]
        out += [f"      - {e}" for e in ins["evidence"]]
        out += [f"  * Bối cảnh VN     : {ins['explanation']}", f"  * Đề xuất hành động: {ins['action']}"]
    out += ["", line, "Số liệu được tính động từ dữ liệu sạch (outputs/cleaned_sales_vn.csv).", line]
    path.write_text("\n".join(out), encoding="utf-8")
    return path
