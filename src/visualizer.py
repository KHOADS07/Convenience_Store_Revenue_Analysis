import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def format_trieu_vnd(x, pos):
    """Hàm định dạng trục tiền tệ sang đơn vị Triệu VNĐ"""
    return f"{x / 1_000_000:.1f}"

def plot_daily_monthly_trend(df_day, df_month, save_path):
    """
    Biểu đồ 1: Biểu đồ đường doanh thu ngày (kèm MA 7 ngày) & Biểu đồ cột doanh thu tháng.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(df_day['Date'], df_day['Total_VND'], marker='o', linestyle='-', color='#1f77b4', alpha=0.6, label='Doanh thu ngày')
    if '7-day MA' in df_day.columns:
        ax1.plot(df_day['Date'], df_day['7-day MA'], color='#d62728', linewidth=2, label='Trung bình trượt 7 ngày')
    ax1.set_title("Biến động doanh thu theo ngày và hiệu ứng mua sắm cuối tuần", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Doanh thu (Triệu VNĐ)")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_trieu_vnd))
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    bars = ax2.bar(df_month['Month'].astype(str), df_month['Total_VND'], color='#2ca02c')
    ax2.set_title("Tổng doanh thu các tháng và tốc độ tăng trưởng (MoM)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Doanh thu (Triệu VNĐ)")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_trieu_vnd))

    if 'MoM_Growth' in df_month.columns:
        for bar, growth in zip(bars, df_month['MoM_Growth']):
            yval = bar.get_height()
            label = f"{growth*100:.1f}%" if not np.isnan(growth) else "-"
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), label, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_category_revenue(df_cat, save_path):
    """
    Biểu đồ 2: Biểu đồ cột ngang doanh thu & tỷ trọng % theo mặt hàng.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_cat = df_cat.sort_values(by='Total_VND', ascending=True)
    
    categories = df_cat['NhomHang_VN']
    revenues = df_cat['Total_VND']
    shares = df_cat['Market_Share'] if 'Market_Share' in df_cat.columns else (revenues / revenues.sum() * 100)

    colors = ['#ff7f0e' if i == len(categories)-1 else '#1f77b4' for i in range(len(categories))]
    
    bars = ax.barh(categories, revenues, color=colors)
    ax.set_title("Nhóm Đồ ăn nhanh & Thức uống (F&B) dẫn đầu cơ cấu doanh thu", fontsize=14, fontweight='bold')
    ax.set_xlabel("Doanh thu (Triệu VNĐ)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_trieu_vnd))
    ax.set_xlim(0, revenue.max() * 1.12)

    mean_rev = revenues.mean()
    ax.axvline(mean_rev, color='red', linestyle='--', linewidth=1.5, label='Trung bình ngành')
    ax.legend(loc='lower right')

    for bar, share in zip(bars, shares):
        xval = bar.get_width()
        ax.text(xval + (xval*0.01), bar.get_y() + bar.get_height()/2, f"{share:.1f}%", ha='left', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_hour_weekday_heatmap(pivot_hw, save_path):
    """
    Biểu đồ 3: Heatmap ma trận Khung giờ × Thứ trong tuần.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cax = ax.imshow(pivot_hw.values, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot_hw.columns)))
    ax.set_yticks(np.arange(len(pivot_hw.index)))
    ax.set_xticklabels(pivot_hw.columns)
    ax.set_yticklabels(pivot_hw.index)
    
    ax.set_title("Ma trận doanh thu theo Khung giờ và Thứ (Đỉnh điểm 19h & 13h-15h)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Thứ trong tuần")
    ax.set_ylabel("Khung giờ giao dịch")

    for i in range(len(pivot_hw.index)):
        for j in range(len(pivot_hw.columns)):
            val = pivot_hw.iloc[i, j]
            text_color = "white" if val > pivot_hw.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val/1_000_000:.1f}M", ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(cax, ax=ax, label='Doanh thu (Triệu VNĐ)', format=ticker.FuncFormatter(format_trieu_vnd))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_multivariate_category_timeslot(pivot_ct, save_path):
    """
    Biểu đồ 4: Biểu đồ cột nhóm (Grouped Bar chart) Mặt hàng phân tách theo 4 Khung giờ.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    categories = pivot_ct.index
    timeslots = pivot_ct.columns
    x = np.arange(len(categories))
    width = 0.2  

    for i, ts in enumerate(timeslots):
        offset = (i - len(timeslots)/2) * width + width/2
        ax.bar(x + offset, pivot_ct[ts], width, label=ts)

    ax.set_title("Sự dịch chuyển cơ cấu giỏ hàng theo nhịp sống đô thị (Sáng - Tối)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Doanh thu (Triệu VNĐ)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_trieu_vnd))
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(title="Khung giờ")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_outlier_and_branch_boxplot(df, save_path):
    """
    Biểu đồ 5: Boxplot phân phối giá trị hóa đơn theo 3 chi nhánh Việt Nam.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    branches = ["Quận 1 - Văn phòng", "Thủ Đức - Đại học", "Quận 7 - Dân cư"]
    data_to_plot = [df[df['KhuVuc_VN'] == branch]['Total_VND'].dropna() for branch in branches]

    flierprops = dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.6)
    box = ax.boxplot(data_to_plot, patch_artist=True, flierprops=flierprops)

    colors = ['#aec7e8', '#ffbb78', '#98df8a']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_xticklabels(branches)
    ax.set_title("Phân phối giá trị hóa đơn và điểm ngoại lai (Outliers) theo 3 chi nhánh", fontsize=14, fontweight='bold')
    ax.set_ylabel("Giá trị hóa đơn (VNĐ)")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_all_charts(df_clean, stats_dict, multi_dict, output_dir="outputs/charts/"):
    """Hàm tổng phối chạy toàn bộ 5 biểu đồ, được gọi từ main.py"""
    create_dir_if_not_exists(output_dir)

    # 1. Chuẩn hóa bảng theo ngày (hỗ trợ cả key của TV2 lẫn key test)
    df_day = (stats_dict.get('by_day') if 'by_day' in stats_dict else stats_dict['day']).copy()
    df_day = df_day.rename(columns={'day': 'Date', 'total_revenue': 'Total_VND'})
    df_day['Date'] = pd.to_datetime(df_day['Date'])
    df_day = df_day.sort_values('Date')
    if '7-day MA' not in df_day.columns:
        df_day['7-day MA'] = df_day['Total_VND'].rolling(7).mean()

    # 2. Chuẩn hóa bảng theo tháng
    df_month = (stats_dict.get('by_month') if 'by_month' in stats_dict else stats_dict['month']).copy()
    df_month = df_month.rename(columns={'month': 'Month', 'total_revenue': 'Total_VND'})
    if 'MoM_Growth' not in df_month.columns:
        df_month['MoM_Growth'] = df_month['Total_VND'].pct_change()

    # 3. Chuẩn hóa bảng theo mặt hàng
    df_cat = (stats_dict.get('by_product') if 'by_product' in stats_dict else stats_dict['category']).copy()
    df_cat = df_cat.rename(columns={'product': 'NhomHang_VN', 'total_revenue': 'Total_VND'})

    pivot_hw = multi_dict['hour_weekday']
    pivot_ct = multi_dict['category_timeslot']

    plot_daily_monthly_trend(df_day, df_month, os.path.join(output_dir, '01_doanh_thu_xu_huong.png'))
    plot_category_revenue(df_cat, os.path.join(output_dir, '02_doanh_thu_mat_hang.png'))
    plot_hour_weekday_heatmap(pivot_hw, os.path.join(output_dir, '03_heatmap_khung_gio_thu.png'))
    plot_multivariate_category_timeslot(pivot_ct, os.path.join(output_dir, '04_da_chieu_hang_khung_gio.png'))
    plot_outlier_and_branch_boxplot(df_clean, os.path.join(output_dir, '05_boxplot_phan_phoi_chi_nhanh.png'))

    print(f"TV4: Da xuat thanh cong 5 bieu do tai '{output_dir}'.")
    
if __name__ == "__main__":
    df = pd.read_csv("outputs/cleaned_sales_vn.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    df_day = df.groupby('Date', as_index=False)['Total_VND'].sum().sort_values('Date')
    df_day['7-day MA'] = df_day['Total_VND'].rolling(7).mean()

    df_month = df.groupby('Month', as_index=False)['Total_VND'].sum()
    df_month['MoM_Growth'] = df_month['Total_VND'].pct_change()

    df_cat = df.groupby('NhomHang_VN', as_index=False)['Total_VND'].sum()
    df_cat['Market_Share'] = df_cat['Total_VND'] / df_cat['Total_VND'].sum() * 100

    days_order = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
    pivot_hw = df.pivot_table(index='Hour', columns='DayOfWeek', values='Total_VND', aggfunc='sum', fill_value=0)
    pivot_hw = pivot_hw.reindex(columns=[d for d in days_order if d in pivot_hw.columns])

    slots_order = ["10h-12h Sáng", "12h-14h Trưa", "14h-18h Chiều", "18h-21h Tối"]
    pivot_ct = df.pivot_table(index='NhomHang_VN', columns='TimeSlot', values='Total_VND', aggfunc='sum', fill_value=0, observed=False)
    pivot_ct = pivot_ct.reindex(columns=[s for s in slots_order if s in pivot_ct.columns])

    stats_dict = {'day': df_day, 'month': df_month, 'category': df_cat}
    multi_dict = {'hour_weekday': pivot_hw, 'category_timeslot': pivot_ct}

    generate_all_charts(df, stats_dict, multi_dict)
