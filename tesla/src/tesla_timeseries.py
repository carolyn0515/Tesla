from __future__ import annotations
from typing import Optional, Tuple, Union, List
import pandas as pd
import matplotlib.pyplot as plt
from .tesla_data import add_date_column


# 공통: Date 보장 + datetime 변환
def _prepare_ts_df(
    df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    if date_col not in df.columns:
        df = add_date_column(df, date_col=date_col)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


# 공통: 설명 출력용 헬퍼 --------------------------------------------
def _print_description(
    title_kor: str,
    region_name: Optional[str],
    purpose: str,
    key_points: List[str],
) -> None:
    region_str = f" ({region_name})" if region_name else ""
    print("\n" + "=" * 72)
    print(f"[그래프 설명] {title_kor}{region_str}")
    print("- 목적:", purpose)
    print("- 핵심 포인트:")
    for p in key_points:
        print(f"  • {p}")
    print("=" * 72 + "\n")


# 1) 월별 판매량 트렌드 --------------------------------------------
def plot_monthly_deliveries(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.Series]]:
    df = _prepare_ts_df(df, date_col=date_col)
    monthly = df.groupby(date_col)["Estimated_Deliveries"].sum()

    if explain:
        _print_description(
            title_kor="월별 판매량 트렌드",
            region_name=region_name,
            purpose="시간이 지남에 따라 월별 추정 판매량(Estimated_Deliveries)의 증가/감소 패턴을 확인하기 위함.",
            key_points=[
                "어느 시기에 판매가 급증/급감했는지 확인",
                "장기적으로 우상향/우하향 추세인지 관찰",
                "특정 이벤트(출시, 정책, 경기변동 등)와 시점이 맞물리는지 해석에 활용",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(monthly.index, monthly.values, marker="o")
    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Monthly Estimated Deliveries{title_region}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated Deliveries")
    ax.grid(True)

    if print_stats and not monthly.empty:
        print("=== [Monthly Deliveries Stats]", title_region, "===")
        print(f"기간: {monthly.index.min().date()} ~ {monthly.index.max().date()}")
        print(f"총합: {monthly.sum():,.0f}")
        print(f"평균: {monthly.mean():,.1f}")
        print(f"최소: {monthly.min():,.0f}, 최대: {monthly.max():,.0f}")
        print("최근 5개(monthly):")
        print(monthly.tail(), "\n")

    if return_data:
        return ax, monthly
    return ax


# 2) 생산량 vs 판매량 비교 시계열 -----------------------------------
def plot_production_vs_deliveries(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.DataFrame]]:
    df = _prepare_ts_df(df, date_col=date_col)
    monthly = (
        df.groupby(date_col)[["Estimated_Deliveries", "Production_Units"]]
        .sum()
    )

    if explain:
        _print_description(
            title_kor="생산량 vs 판매량 시계열 비교",
            region_name=region_name,
            purpose="같은 기간 동안 생산량과 판매량이 어떻게 움직이는지, 생산이 수요를 따라가고 있는지 확인하기 위함.",
            key_points=[
                "생산량이 판매량보다 지속적으로 높은지/낮은지 확인",
                "공급 부족(판매>생산) 또는 재고 증가(생산>판매) 가능성 탐색",
                "두 시계열 간 상관관계를 통해 수급 균형 여부를 해석",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        monthly.index,
        monthly["Estimated_Deliveries"],
        marker="o",
        label="Estimated Deliveries",
    )
    ax.plot(
        monthly.index,
        monthly["Production_Units"],
        marker="s",
        linestyle="--",
        label="Production Units",
    )

    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Production vs Deliveries{title_region}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(True)

    if print_stats and not monthly.empty:
        ratio = monthly["Estimated_Deliveries"] / monthly["Production_Units"]
        corr = monthly["Estimated_Deliveries"].corr(monthly["Production_Units"])
        print("=== [Production vs Deliveries Stats]", title_region, "===")
        print(f"기간: {monthly.index.min().date()} ~ {monthly.index.max().date()}")
        print("평균 값:")
        print(monthly.mean().round(2))
        print("판매/생산 비율(평균):", ratio.mean().round(3))
        print("판매량-생산량 상관계수:", round(corr, 3))
        print("최근 5개(monthly):")
        print(monthly.tail(), "\n")

    if return_data:
        return ax, monthly
    return ax


# 3) 평균 가격 시계열 ----------------------------------------------
def plot_avg_price_ts(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.Series]]:
    df = _prepare_ts_df(df, date_col=date_col)
    monthly_price = df.groupby(date_col)["Avg_Price_USD"].mean()

    if explain:
        _print_description(
            title_kor="평균 판매 가격 시계열",
            region_name=region_name,
            purpose="시간 경과에 따라 평균 판매 가격(Avg_Price_USD)이 어떻게 변하는지 확인하기 위함.",
            key_points=[
                "신모델 출시, 옵션 변화, 환율 등의 영향으로 가격 레벨이 바뀌는지 확인",
                "장기적으로 고가화(업셀링) 또는 저가화(보급형 확대) 추세인지 파악",
                "가격 변동과 판매량 변화가 같이 움직이는지 다른 분석과 연결할 수 있음",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(monthly_price.index, monthly_price.values, marker="o")
    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Average Price Over Time{title_region}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Price (USD)")
    ax.grid(True)

    if print_stats and not monthly_price.empty:
        print("=== [Average Price Stats]", title_region, "===")
        print(f"기간: {monthly_price.index.min().date()} ~ {monthly_price.index.max().date()}")
        print(f"평균 가격: {monthly_price.mean():,.2f} USD")
        print(f"최소: {monthly_price.min():,.2f}, 최대: {monthly_price.max():,.2f}")
        print("최근 5개(monthly):")
        print(monthly_price.tail(), "\n")

    if return_data:
        return ax, monthly_price
    return ax


# 4) Model별 시장 점유율 변화 -------------------------------------
def plot_model_share_ts(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.DataFrame]]:
    df = _prepare_ts_df(df, date_col=date_col)
    pivot = (
        df.groupby([date_col, "Model"])["Estimated_Deliveries"]
        .sum()
        .unstack("Model")
        .fillna(0.0)
    )
    share = pivot.div(pivot.sum(axis=1), axis=0)

    if explain:
        _print_description(
            title_kor="모델별 시장 점유율(stackplot)",
            region_name=region_name,
            purpose="각 시점에서 모델별 판매 비중(점유율)이 어떻게 변화하는지 확인하기 위함.",
            key_points=[
                "어느 시점부터 어떤 모델이 '주력 모델'로 떠오르는지 확인",
                "기존 모델의 비중이 줄고 신모델이 치고 올라오는 교체 패턴 관찰",
                "특정 모델에 지나치게 의존하는 구조인지, 포트폴리오가 분산되어 있는지 평가",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.stackplot(share.index, share.T.values, labels=share.columns)

    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Model Market Share Over Time{title_region}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of Deliveries")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")

    if print_stats and not share.empty:
        last_date = share.index.max()
        print("=== [Model Share Stats]", title_region, "===")
        print(f"기간: {share.index.min().date()} ~ {share.index.max().date()}")
        print("전체 기간 평균 점유율:")
        print(share.mean().round(3).sort_values(ascending=False))
        print(f"\n가장 최근 날짜({last_date.date()}) 점유율:")
        print(share.loc[last_date].round(3).sort_values(ascending=False), "\n")

    if return_data:
        return ax, share
    return ax


# 5) Battery Capacity / Range 변화 추세 ---------------------------
def plot_battery_and_range_ts(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.DataFrame]]:
    df = _prepare_ts_df(df, date_col=date_col)

    monthly = (
        df.groupby(date_col)[["Battery_Capacity_kWh", "Range_km"]]
        .mean()
    )

    if explain:
        _print_description(
            title_kor="배터리 용량 & 주행거리 시계열",
            region_name=region_name,
            purpose="배터리 용량과 1회 충전 주행거리(Range)가 함께 어떻게 개선되어 왔는지 살펴보기 위함.",
            key_points=[
                "배터리 용량 증가가 곧바로 주행거리 증가로 이어지는지 확인",
                "어느 시점부터 효율 개선(동일 용량 대비 더 긴 주행거리)이 나타나는지 관찰",
                "기술 발전의 속도와 판매 전략(고성능/롱레인지 모델 강조 등) 해석에 활용",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # 1번째 y축: Battery
    ax.plot(
        monthly.index,
        monthly["Battery_Capacity_kWh"],
        marker="o",
        label="Battery Capacity (kWh)",
    )
    ax.set_ylabel("Battery Capacity (kWh)")
    ax.grid(True, axis="y")

    # 2번째 y축: Range
    ax2 = ax.twinx()
    ax2.plot(
        monthly.index,
        monthly["Range_km"],
        marker="s",
        linestyle="--",
        label="Range (km)",
    )
    ax2.set_ylabel("Range (km)")

    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Battery Capacity & Range Over Time{title_region}")
    ax.set_xlabel("Date")

    # 범례 합치기
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    if print_stats and not monthly.empty:
        corr = monthly["Battery_Capacity_kWh"].corr(monthly["Range_km"])
        print("=== [Battery & Range Stats]", title_region, "===")
        print(f"기간: {monthly.index.min().date()} ~ {monthly.index.max().date()}")
        print("평균 값:")
        print(monthly.mean().round(2))
        print("Battery-Range 상관계수:", round(corr, 3))
        print("최근 5개(monthly):")
        print(monthly.tail(), "\n")

    if return_data:
        return ax, monthly
    return ax


# 6) Charging Stations vs 판매량 -------------------------------
def plot_infra_vs_sales_ts(
    df: pd.DataFrame,
    region_name: Optional[str] = None,
    date_col: str = "Date",
    ax: Optional[plt.Axes] = None,
    explain: bool = True,
    print_stats: bool = True,
    return_data: bool = False,
) -> Union[plt.Axes, Tuple[plt.Axes, pd.DataFrame]]:
    df = _prepare_ts_df(df, date_col=date_col)

    monthly = (
        df.groupby(date_col)[["Estimated_Deliveries", "Charging_Stations"]]
        .sum()
    )

    if explain:
        _print_description(
            title_kor="충전 인프라 vs 판매량 시계열",
            region_name=region_name,
            purpose="충전소 수(인프라 확충)가 판매량 증가와 얼마나 연결되어 있는지 확인하기 위함.",
            key_points=[
                "충전 인프라가 빠르게 늘어난 구간에서 판매량도 같이 증가하는지 확인",
                "인프라는 늘었는데 판매는 정체된 구간이 있는지(비효율 가능성) 탐색",
                "충전 인프라 전략이 실제 수요 창출에 어느 정도 기여하는지 정성적으로 해석",
            ],
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # 판매량: 왼쪽 y축
    ax.plot(
        monthly.index,
        monthly["Estimated_Deliveries"],
        marker="o",
        label="Estimated Deliveries",
    )
    ax.set_ylabel("Estimated Deliveries")
    ax.grid(True, axis="y")

    # 충전소 수: 오른쪽 y축
    ax2 = ax.twinx()
    ax2.plot(
        monthly.index,
        monthly["Charging_Stations"],
        marker="s",
        linestyle="--",
        label="Charging Stations",
    )
    ax2.set_ylabel("Charging Stations")

    title_region = f" - {region_name}" if region_name else ""
    ax.set_title(f"Infrastructure vs Sales Over Time{title_region}")
    ax.set_xlabel("Date")

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    if print_stats and not monthly.empty:
        corr = monthly["Estimated_Deliveries"].corr(monthly["Charging_Stations"])
        print("=== [Infrastructure vs Sales Stats]", title_region, "===")
        print(f"기간: {monthly.index.min().date()} ~ {monthly.index.max().date()}")
        print("평균 값:")
        print(monthly.mean().round(2))
        print("판매량-충전소 상관계수:", round(corr, 3))
        print("최근 5개(monthly):")
        print(monthly.tail(), "\n")

    if return_data:
        return ax, monthly
    return ax
