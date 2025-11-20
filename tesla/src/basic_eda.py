# src/basic_eda.py

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# -------------------------------------------------------
# 1) 기본 정보 요약
# -------------------------------------------------------
def summarize_dataframe(df: pd.DataFrame) -> None:
    print("=== DataFrame Shape ===")
    print(df.shape, "\n")

    print("=== Columns & dtypes ===")
    print(df.dtypes, "\n")

    print("=== Head ===")
    print(df.head(), "\n")

    print("=== Tail ===")
    print(df.tail(), "\n")

    print("=== Missing Values ===")
    print(df.isna().sum(), "\n")

    print("=== Missing Ratio (%) ===")
    print((df.isna().mean() * 100).round(2), "\n")


# -------------------------------------------------------
# 2) 기술통계 (numeric 기준)
# -------------------------------------------------------
def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Numeric Summary Statistics ===")
    stats = df.describe().T
    print(stats)
    return stats


# -------------------------------------------------------
# 3) 결측치 시각화
# -------------------------------------------------------
def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print(">> No Missing Values Found.")
        return

    plt.figure(figsize=(10, 4))
    missing.sort_values().plot(kind="barh")
    plt.title("Missing Values per Column")
    plt.xlabel("Count")
    plt.grid(True, axis="x")
    plt.show()


# -------------------------------------------------------
# 4) 카테고리컬 컬럼 value_counts
# -------------------------------------------------------
def count_categorical(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not in DataFrame.")
            continue

        print(f"\n=== Value Counts: {col} ===")
        print(df[col].value_counts())
        print()


# -------------------------------------------------------
# 5) 상관관계 Heatmap
# -------------------------------------------------------
def plot_corr_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Only)")
    plt.show()


# -------------------------------------------------------
# 6) 날짜(Year, Month) 분포
# -------------------------------------------------------
def plot_year_month_dist(df: pd.DataFrame) -> None:
    if "Year" in df.columns:
        plt.figure(figsize=(8, 3))
        df["Year"].value_counts().sort_index().plot(kind="bar")
        plt.title("Distribution by Year")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.show()

    if "Month" in df.columns:
        plt.figure(figsize=(8, 3))
        df["Month"].value_counts().sort_index().plot(kind="bar")
        plt.title("Distribution by Month")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.show()


# -------------------------------------------------------
# 7) 기본적인 분포 그래프
# -------------------------------------------------------
def plot_basic_distributions(df: pd.DataFrame) -> None:
    numeric_cols = [
        "Estimated_Deliveries",
        "Production_Units",
        "Avg_Price_USD",
        "Battery_Capacity_kWh",
        "Range_km",
        "Charging_Stations",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.grid(True, axis="y")
        plt.show()
