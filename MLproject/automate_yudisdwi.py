import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import os

# ── Konfigurasi path ──────────────────────────────────────────────────────────
INPUT_PATH  = "dataset/data.csv"
OUTPUT_PATH = "dataset/clean_data.csv"

# ── Definisi fitur ────────────────────────────────────────────────────────────
numeric_features = [
    'Application_order', 'Previous_qualification_grade', 'Admission_grade',
    'Age_at_enrollment', 'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited', 'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations', 'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

ordinal_features  = ['Marital_status', 'Mothers_qualification', 'Fathers_qualification']

nominal_features  = [
    'Application_mode', 'Course', 'Daytime_evening_attendance',
    'Previous_qualification', 'Nacionality',
    'Mothers_occupation', 'Fathers_occupation',
    'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender',
    'Scholarship_holder', 'International'
]


def load_data(path: str) -> pd.DataFrame:
    print(f"[1/6] Memuat dataset dari {path} ...")
    df = pd.read_csv(path, sep=';')
    print(f"      Shape awal: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("[1/6] Menangani missing values ...")
    missing = df.isnull().sum().sum()
    print(f"      Total missing values: {missing}")

    for col in numeric_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in ordinal_features + nominal_features:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"      Missing values setelah penanganan: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Menghapus duplikat ...")
    n_dup = df.duplicated().sum()
    print(f"      Duplikat ditemukan: {n_dup}")
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
    print(f"      Shape setelah drop duplikat: {df.shape}")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] Mendeteksi dan menghapus outlier (IQR) ...")
    valid_num = [c for c in numeric_features if c in df.columns]
    Q1  = df[valid_num].quantile(0.25)
    Q3  = df[valid_num].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((df[valid_num] < (Q1 - 1.5 * IQR)) |
                  (df[valid_num] > (Q3 + 1.5 * IQR))).any(axis=1)

    df_clean = df[condition].reset_index(drop=True)
    print(f"      Baris dihapus: {df.shape[0] - df_clean.shape[0]}")
    print(f"      Shape setelah remove outlier: {df_clean.shape}")
    return df_clean


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Encoding fitur kategorikal ...")
    df_enc = df.copy()

    # Ordinal Encoding
    valid_ord = [c for c in ordinal_features if c in df_enc.columns]
    if valid_ord:
        ordinal_mappings = [
            sorted(df_enc[c].unique().tolist()) for c in valid_ord
        ]
        oe = OrdinalEncoder(
            categories=ordinal_mappings,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df_enc[valid_ord] = oe.fit_transform(df_enc[valid_ord])

    # One-Hot Encoding
    valid_nom = [c for c in nominal_features if c in df_enc.columns]
    if valid_nom:
        df_enc = pd.get_dummies(df_enc, columns=valid_nom, drop_first=False)

    # Label Encoding target
    if 'Status' in df_enc.columns:
        le = LabelEncoder()
        df_enc['Status'] = le.fit_transform(df_enc['Status'])
        print(f"      Kelas target: {list(le.classes_)}")

    print(f"      Shape setelah encoding: {df_enc.shape}")
    return df_enc


def binning(df: pd.DataFrame, df_before_encode: pd.DataFrame) -> pd.DataFrame:
    print("[6/6] Binning fitur kontinu ...")

    if 'Admission_grade' in df_before_encode.columns:
        df['Admission_grade_bin'] = pd.cut(
            df_before_encode['Admission_grade'],
            bins=[0, 120, 140, 160, 200],
            labels=[0, 1, 2, 3]   # Low=0, Medium=1, High=2, Very High=3
        ).astype(float)

    if 'Age_at_enrollment' in df_before_encode.columns:
        df['Age_group_bin'] = pd.cut(
            df_before_encode['Age_at_enrollment'],
            bins=[17, 20, 25, 35, 70],
            labels=[0, 1, 2, 3]   # 18-20=0, 21-25=1, 26-35=2, 35+=3
        ).astype(float)

    print(f"      Shape setelah binning: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n✅ Dataset bersih disimpan ke: {path}")
    print(f"   Shape akhir: {df.shape}")


def main():
    print("=" * 50)
    print("  PREPROCESSING PIPELINE - Eksperimen SML")
    print("=" * 50)

    df = load_data(INPUT_PATH)
    df = handle_missing_values(df)       # Step 1
    df = remove_duplicates(df)           # Step 2
    # Step 3: Standarisasi dilakukan di pipeline modeling, bukan disimpan ke CSV
    df = remove_outliers(df)             # Step 4
    df_before_encode = df.copy()         # simpan sebelum encode untuk binning
    df = encode_features(df)             # Step 5
    df = binning(df, df_before_encode)   # Step 6
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()