# **ĐỒ ÁN MÔN HỌC: PHÂN TÍCH & TRỰC QUAN HÓA DOANH THU CHUỖI CỬA HÀNG TIỆN LỢI TẠI VIỆT NAM**
## **Học phần: Ngôn ngữ lập trình Python — Mã học phần: `012012410005`**
## **NHÓM 5
**Công nghệ sử dụng:** Python 3.10+, Pandas (OLAP Pivot), NumPy (Vectorization, IQR/Z-score), Matplotlib (300 DPI) & Unittest  
**Kho mã nguồn:** `https://github.com/KHOADS07/Convenience_Store_Revenue_Analysis`  
**Dữ liệu đầu vào:** Kaggle *Supermarket Sales* (1.000 giao dịch chuẩn hóa sang bối cảnh bán lẻ đô thị Việt Nam)

---

## 1. Giới thiệu & Phương pháp Bản địa hóa Dữ liệu

Dự án xây dựng một pipeline phân tích dữ liệu khép kín (End-to-End Data Pipeline) nhằm khai phá hành vi mua sắm, khung giờ cao điểm và cơ cấu doanh thu của chuỗi cửa hàng tiện lợi/mini-mart đô thị tại TP.HCM.

* **Ánh xạ bối cảnh Việt Nam:** Chuẩn hóa 3 chi nhánh sang mô hình cửa hàng đặc trưng tại TP.HCM (`Quận 1 - Văn phòng`, `Thủ Đức - Đại học`, `Quận 7 - Dân cư`), 6 nhóm ngành hàng tiêu dùng thiết yếu và 4 khung giờ sinh hoạt đô thị (`10h-12h Sáng`, `12h-14h Trưa`, `14h-18h Chiều`, `18h-21h Tối`).
* **Lưu ý thang đo tiền tệ:** Hệ thống giữ nguyên tỷ giá chuẩn **`1 USD = 25.000 VNĐ`** để bảo toàn tuyệt đối cấu trúc phân phối, độ lệch chuẩn và tỷ trọng gốc của 1.000 giao dịch thực từ Kaggle. Quy mô giỏ hàng trung bình **~8,07 triệu VNĐ/đơn** (tổng doanh thu **~8,07 tỷ VNĐ**) phản ánh mô hình siêu thị tiện lợi quy mô lớn / Superette phục vụ nhu cầu mua sắm hộ gia đình và văn phòng.

---

## 2. Kiến trúc Dự án & Phân công Thành viên

```text
Convenience_Store_Revenue_Analysis/
├── data/
│   └── supermarket_sales.csv          # Dữ liệu thô 1.000 hóa đơn, 17 cột gốc từ Kaggle
├── src/
│   ├── preprocessing.py               # (TV1) Tiền xử lý, ánh xạ ngữ nghĩa VN, xử lý Missing & Outlier (IQR)
│   ├── revenue_stats.py               # (TV2) Thống kê mô tả, xu hướng Ngày (7-day MA), Tháng (MoM%) & Mặt hàng
│   ├── multivariate_analysis.py       # (TV3) Phân tích đa chiều OLAP (Mặt hàng × Khung giờ × Thứ × Chi nhánh)
│   ├── visualizer.py                  # (TV4) Xuất tự động 5 biểu đồ trực quan hóa chuẩn báo cáo (300 DPI)
│   └── insights.py                    # (TV5) Trích xuất động 4 nhận định kinh doanh & khuyến nghị hành động
├── tests/
│   └── test_pipeline.py               # (TV6) Bộ 6 ca kiểm thử tự động (unittest) & tiêm lỗi giả lập
├── outputs/
│   ├── cleaned_sales_vn.csv           # Dữ liệu sạch sau chuẩn hóa & xử lý ngoại lai
│   ├── business_report.txt            # Báo cáo KPI & 4 nhận định kinh doanh tự động
│   └── charts/                        # Thư mục chứa 5 biểu đồ phân tích (.png)
├── main.py                            # (TV5) Điểm điều phối chạy toàn bộ pipeline từ Bước 1 đến Bước 5
├── requirements.txt                   # Danh sách thư viện phụ thuộc (pandas, numpy, matplotlib)
└── README.md                          # Tài liệu hướng dẫn và tổng quan kiến trúc dự án
```

| Thành viên | Tầng chức năng | Tệp tin phụ trách | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Nguyễn Đức Tài Nhân - 075207000402(TV1)** | Tiền xử lý & Làm sạch (ETL) | `src/preprocessing.py` | Đọc CSV, tách đặc trưng ngày/giờ, ánh xạ bối cảnh VN, xử lý Missing (Median/Mode) & Outlier (IQR Winsorizing) |
| **Nguyễn Á Thành - 077207004059(TV2)** | Thống kê cốt lõi (Core Stats) | `src/revenue_stats.py` | Tính thống kê mô tả (NumPy/Pandas), doanh thu theo Ngày (7-day MA), Tháng (MoM%) & Mặt hàng |
| **Nguyễn Đức Tuấn Anh - 066207017711(TV3)** | Phân tích đa chiều (OLAP) | `src/multivariate_analysis.py` | Xây dựng ma trận Pivot đa chiều (`Mặt hàng × Khung giờ × Thứ × Chi nhánh`) & xếp hạng Top/Bottom 5 |
| **Huỳnh Quang Bảo - 079207006037(TV4)** | Trực quan hóa (Visualization) | `src/visualizer.py` | Thiết kế và xuất tự động 5 biểu đồ Matplotlib độ phân giải 300 DPI (đơn vị Triệu VNĐ) |
| **Trần Lê Đăng Khôi - 049207015682(TV5)** | Nhận định & Điều phối | `src/insights.py`, `main.py` | Tính toán động 4 nhận định kinh doanh kèm đề xuất hành động & điều phối luồng chạy `main.py` |
| **Phan Đăng Khoa - 056207003633(TV6)** | Kiểm thử & Đảm bảo chất lượng | `tests/test_pipeline.py` | Bộ 6 ca kiểm thử tự động (`unittest`) xác minh toàn bộ pipeline & tiêm lỗi giả lập |

---

## 3. Hướng dẫn Cài đặt & Thực thi

**Cách 1: Chạy trực tiếp trên Google Colab**

    !git clone https://github.com/KHOADS07/Convenience_Store_Revenue_Analysis.git
    %cd Convenience_Store_Revenue_Analysis
    !python main.py
    !python -m unittest discover -s tests -v

**Cách 2: Chạy trên máy cục bộ (Local)**

    git clone https://github.com/KHOADS07/Convenience_Store_Revenue_Analysis.git
    cd Convenience_Store_Revenue_Analysis
    pip install -r requirements.txt
    python main.py
    python -m unittest discover -s tests -v

---

## 4. Kết quả Đầu ra Chính

Sau khi chạy `python main.py`, toàn bộ kết quả được lưu tự động vào thư mục `outputs/`:

1. **`outputs/cleaned_sales_vn.csv`**: 1.000 bản ghi sạch kèm cột doanh thu gốc (`Total_VND`) và doanh thu đã cắt ngưỡng ngoại lai (`Total_VND_Clean` theo biên IQR `[-9,90 triệu ; 24,79 triệu VNĐ]`).
2. **`outputs/business_report.txt`**: Tổng hợp các chỉ số phân phối (Tổng doanh thu `8.074.168.725 VNĐ`, Trung bình `8.074.168,72 VNĐ/đơn`, Trung vị `6.346.200 VNĐ/đơn`, Lợi nhuận gộp `384.484.225 VNĐ`) và 4 nhận định chiến lược.
3. **`outputs/charts/`**:
   * `01_doanh_thu_xu_huong.png`: Xu hướng doanh thu ngày (7-day MA) & tăng trưởng theo tháng.
   * `02_doanh_thu_mat_hang.png`: Cơ cấu doanh thu & tỷ trọng % theo 6 nhóm mặt hàng.
   * `03_heatmap_khung_gio_thu.png`: Bản đồ nhiệt khung giờ vàng theo thứ trong tuần.
   * `04_da_chieu_hang_khung_gio.png`: Tổ hợp đa chiều Mặt hàng × Khung giờ VN.
   * `05_boxplot_phan_phoi_chi_nhanh.png`: Phân phối giá trị hóa đơn & 9 điểm ngoại lai theo 3 chi nhánh TP.HCM.
