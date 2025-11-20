from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 정렬
def load_tesla_data(
        csv_path: str | Path,
        sort: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if sort:
        df = df.sort_values(["Year", "Month", "Region", "Model"]).reset_index(drop=True)
    return df

# Region 목록 추출
def get_region_names(
        df: pd.DataFrame,
        region_col: str = "Region",
        sort: bool = True,
) -> List[str]:
    regions = df[region_col].dropna().unique().tolist()
    if sort:
        regions = sorted(regions)
    return regions

# 지역별 df
def split_by_region(
        df: pd.DataFrame,
        region_col: str = "Region",
        drop_region_col: bool=True,
) -> Dict[str, pd.DataFrame]:
    region_dfs: Dict[str, pd.DataFrame] = {}
    region_names = get_region_names(df, region_col=region_col, sort=True)
    for region in region_names:
        temp_df = df[df[region_col] == region].copy()
        temp_df.sort_values(by=["Year", "Month"], inplace=True)
        temp_df.reset_index(drop=True, inplace=True)

        if drop_region_col:
            temp_df.drop(columns=region_col, inplace=True)
        
        region_dfs[region] = temp_df

    return region_dfs

# date 지정 
def add_date_column(
        df: pd.DataFrame,
        year_col: str = "Year",
        month_col: str = "Month",
        date_col: str = "Date"
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(
        df[year_col].astype(int).astype(str)
        +"-"
        +df[month_col].astype(int).astype(str)
        +"-01"
        )
    return df

# region별 csv 저장
def save_region_dfs(
        region_dfs: Dict[str, pd.DataFrame],
        out_dir: str|Path,
        prefix: str = "tesla_",
        index: bool=False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for region, rdf in region_dfs.items():
        safe_region = (
            str(region)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )
    out_path = out_dir / f"{prefix}{safe_region}.csv"
    rdf.to_csv(out_path, index=index)