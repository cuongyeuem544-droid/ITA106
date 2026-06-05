import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "learnx.csv.csv"
CHART_DIR = "charts"
REPORT_FILE = "bao_cao_giai_doan_1.txt"
OUTLIER_FILE = "outliers_learnx.csv"

os.makedirs(CHART_DIR, exist_ok=True)


def read_dataset(path):
    df = pd.read_csv(path)
    print("===== 1. TONG QUAN DU LIEU =====")
    print("So dong, so cot:", df.shape)
    print("Danh sach cot:")
    print(list(df.columns))
    print("\n5 dong dau tien:")
    print(df.head())
    return df


def clean_dataset(df):
    print("\n===== 2. LAM SACH DU LIEU =====")
    rows_before = len(df)

    print("\nGia tri thieu ban dau:")
    print(df.isna().sum())

    print("\nSo dong trung lap:", df.duplicated().sum())
    df = df.drop_duplicates().copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col] = df[col].fillna(mode_value.iloc[0])
            else:
                df[col] = df[col].fillna("Khong ro")

    invalid_count = 0
    rules = {
        "avg_session_minutes": (0, None),
        "sessions_per_week": (0, None),
        "completion_rate": (0, 100),
        "courses_enrolled": (0, None),
        "total_spent_usd": (0, None)
    }

    for col, (min_value, max_value) in rules.items():
        if col in df.columns:
            before = len(df)
            if min_value is not None:
                df = df[df[col] >= min_value]
            if max_value is not None:
                df = df[df[col] <= max_value]
            invalid_count += before - len(df)

    rows_after = len(df)

    print("\nGia tri thieu sau khi xu ly:")
    print(df.isna().sum())
    print("\nSo dong ban dau:", rows_before)
    print("So dong sau khi lam sach:", rows_after)
    print("So dong du lieu sai/bat thuong da loai bo:", invalid_count)

    return df, rows_before, rows_after, invalid_count


def save_histogram(df, col, title, file_name):
    if col not in df.columns:
        print("Khong co cot:", col)
        return

    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=20, edgecolor="black")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("So luong nguoi dung")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, file_name))
    plt.show()
    plt.close()

def save_boxplot(df, columns):
    columns = [col for col in columns if col in df.columns]
    if len(columns) == 0:
        return

    plt.figure(figsize=(10, 5))
    df[columns].boxplot(rot=30)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "boxplot_outliers.png"))
    plt.show()
    plt.close()

def find_outliers(df, columns):
    result = []

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        min_value = q1 - 1.5 * iqr
        max_value = q3 + 1.5 * iqr

        outliers = df[(df[col] < min_value) | (df[col] > max_value)].copy()
        outliers["cot_bat_thuong"] = col
        outliers["gia_tri_bat_thuong"] = outliers[col]
        result.append(outliers)

    if len(result) == 0:
        return pd.DataFrame()

    return pd.concat(result, ignore_index=True)


def analyze_special_users(df):
    avg_time = "avg_session_minutes"
    total_courses = "courses_enrolled"
    total_money = "total_spent_usd"

    high_study = pd.DataFrame()
    many_course_low_study = pd.DataFrame()
    high_spending = pd.DataFrame()

    if avg_time in df.columns:
        high_study = df[df[avg_time] > df[avg_time].quantile(0.95)]

    if total_courses in df.columns and avg_time in df.columns:
        many_course_low_study = df[
            (df[total_courses] > df[total_courses].quantile(0.75)) &
            (df[avg_time] <= df[avg_time].quantile(0.25))
        ]

    if total_money in df.columns:
        high_spending = df[df[total_money] > df[total_money].quantile(0.95)]

    return {
        "high_study": len(high_study),
        "many_course_low_study": len(many_course_low_study),
        "high_spending": len(high_spending)
    }


def write_report(df, rows_before, rows_after, invalid_count, outliers, special_groups):
    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write("BAO CAO GIAI DOAN 1 - PHAN TICH DU LIEU LEARNX\n\n")

        file.write("1. Gioi thieu va mo ta du lieu\n")
        file.write("- Nguon du lieu: bo du lieu hanh vi nguoi dung LearnX.\n")
        file.write(f"- So ban ghi ban dau: {rows_before}\n")
        file.write(f"- So ban ghi sau khi lam sach: {rows_after}\n")
        file.write(f"- So thuoc tinh: {df.shape[1]}\n")
        file.write(f"- Cac thuoc tinh trong du lieu: {list(df.columns)}\n\n")

        file.write("2. Lam sach du lieu\n")
        file.write("- Da kiem tra gia tri thieu trong tung cot.\n")
        file.write("- Da dien gia tri thieu o cot so bang trung vi.\n")
        file.write("- Da dien gia tri thieu o cot chu bang gia tri xuat hien nhieu nhat.\n")
        file.write("- Da kiem tra va xoa cac dong du lieu trung lap.\n")
        file.write("- Da kiem tra du lieu sai hoac bat thuong.\n")
        file.write("- Cac gia tri am da duoc loai bo.\n")
        file.write("- Completion rate ngoai khoang 0 den 100 da duoc loai bo.\n")
        file.write(f"- Tong so dong du lieu sai/bat thuong da loai bo: {invalid_count}\n\n")

        file.write("3. Truc quan hoa du lieu\n")
        file.write("- Da ve bieu do phan phoi thoi gian hoc trung binh.\n")
        file.write("- Da ve bieu do phan phoi so lan truy cap moi tuan.\n")
        file.write("- Da ve bieu do phan phoi muc do hoan thanh khoa hoc.\n")
        file.write("- Da ve boxplot de phat hien outlier.\n\n")

        file.write("4. Phat hien hanh vi bat thuong\n")
        file.write(f"- Tong so outlier phat hien bang IQR: {len(outliers)}\n")
        file.write(f"- So nguoi dung hoc cuc ky nhieu: {special_groups['high_study']}\n")
        file.write(f"- So nguoi dung dang ky nhieu khoa nhung hoc it: {special_groups['many_course_low_study']}\n")
        file.write(f"- So nguoi dung chi tieu cao bat thuong: {special_groups['high_spending']}\n\n")

        file.write("5. Insight cho doi san pham\n")
        file.write("- Nen chu y nhom nguoi dung dang ky nhieu khoa nhung thoi gian hoc thap.\n")
        file.write("- Nen gui thong bao nhac nho hoc tap de tang ty le hoan thanh khoa hoc.\n")
        file.write("- Nhom hoc nhieu co the duoc goi y cac khoa hoc nang cao.\n")
        file.write("- Nhom chi tieu cao nen duoc cham soc rieng de toi uu doanh thu.\n")


def main():
    df = read_dataset(DATA_FILE)
    df, rows_before, rows_after, invalid_count = clean_dataset(df)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    print("\n===== THONG KE MO TA CAC COT SO =====")
    print(df[numeric_columns].describe())

    avg_time = "avg_session_minutes"
    weekly_sessions = "sessions_per_week"
    completion_rate = "completion_rate"
    total_courses = "courses_enrolled"
    total_money = "total_spent_usd"

    analysis_columns = [avg_time, weekly_sessions, completion_rate, total_courses, total_money]

    save_histogram(df, avg_time, "Phan phoi thoi gian hoc trung binh", "phan_phoi_thoi_gian_hoc.png")
    save_histogram(df, weekly_sessions, "Phan phoi so lan truy cap moi tuan", "so_lan_truy_cap_moi_tuan.png")
    save_histogram(df, completion_rate, "Phan phoi muc do hoan thanh khoa hoc", "muc_do_hoan_thanh_khoa_hoc.png")
    save_boxplot(df, analysis_columns)

    outliers = find_outliers(df, analysis_columns)
    print("\nTong so outlier phat hien:", len(outliers))

    if len(outliers) > 0:
        outliers.to_csv(OUTLIER_FILE, index=False)
        print("Da luu file outlier:", OUTLIER_FILE)

    special_groups = analyze_special_users(df)
    print("Nguoi dung hoc cuc ky nhieu:", special_groups["high_study"])
    print("Dang ky nhieu khoa nhung hoc it:", special_groups["many_course_low_study"])
    print("Chi tieu cao bat thuong:", special_groups["high_spending"])

    write_report(df, rows_before, rows_after, invalid_count, outliers, special_groups)

    print("\n===== HOAN THANH GIAI DOAN 1 =====")
    print("Cac bieu do nam trong thu muc:", CHART_DIR)
    print("Bao cao nam trong file:", REPORT_FILE)
    print("File outlier neu co:", OUTLIER_FILE)


if __name__ == "__main__":
    main()