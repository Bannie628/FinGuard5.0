import pandas as pd
import numpy as np


def safe_divide(a, b):
    """安全除法：避免除零错误"""
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def calculate_indicators(df):
    result = df.copy()
# 删除完全空白的行
    result = result.dropna(how="all")

# 删除没有年份和公司名称的备注行
    result = result.dropna(
    subset=["公司名称", "年份"],
    how="any"
)
    required_columns = [
        "公司名称", "年份",
        "总资产", "总负债",
        "流动资产", "流动负债",
        "营业收入", "净利润",
        "经营现金流", "应收账款"
    ]

    missing = [
        col for col in required_columns
        if col not in result.columns
    ]

    if missing:
        raise ValueError(f"缺少字段：{missing}")

    numeric_columns = required_columns[2:]

    for col in numeric_columns:
        result[col] = pd.to_numeric(
            result[col], errors="coerce"
        )

    result["年份"] = pd.to_numeric(
        result["年份"], errors="raise"
    ).astype(int)

    if result.duplicated(["公司名称", "年份"]).any():
        raise ValueError("同一公司同一年份存在重复记录")

    result = result.sort_values(
        ["公司名称", "年份"]
    ).reset_index(drop=True)

    # 1. 资产负债率
    result["资产负债率"] = result.apply(
        lambda row: safe_divide(
            row["总负债"], row["总资产"]
        ), axis=1
    )

    # 2. 流动比率
    result["流动比率"] = result.apply(
        lambda row: safe_divide(
            row["流动资产"], row["流动负债"]
        ), axis=1
    )

    # 3. 营业收入增长率
    result["营业收入增长率"] = (
        result.groupby("公司名称")["营业收入"]
        .pct_change(fill_method=None)
    )

    # 4. 净利润增长率
    result["净利润增长率"] = (
        result.groupby("公司名称")["净利润"]
        .pct_change(fill_method=None)
    )

    # 5. 应收账款增长率
    result["应收账款增长率"] = (
        result.groupby("公司名称")["应收账款"]
        .pct_change(fill_method=None)
    )

    # 6. 经营现金流利润比
    result["经营现金流利润比"] = result.apply(
        lambda row: safe_divide(
            row["经营现金流"], row["净利润"]
        ) if row["净利润"] > 0 else np.nan,
        axis=1
    )

    # 7. 净资产收益率（简化版）
    result["净资产"] = (
        result["总资产"] - result["总负债"]
    )

    result["ROE"] = result.apply(
        lambda row: safe_divide(
            row["净利润"], row["净资产"]
        ) if row["净资产"] > 0 else np.nan,
        axis=1
    )

    # 8. 总资产收益率（简化版）
    result["ROA"] = result.apply(
        lambda row: safe_divide(
            row["净利润"], row["总资产"]
        ), axis=1
    )

    return result