
import pandas as pd
import numpy as np

from financial_analysis import calculate_indicators
from risk_detection import detect_risks
from industry_analysis import (
    compare_industry_debt_ratio, identify_company_industry, normalize_industry
)
from finguard_zscore import calculate_z_score
# ==========================================
# 1. 财务指标查询
# ==========================================

def get_financial_indicators(
    company_name,
    year,
    financial_data
):
    """
    查询指定公司、指定年份的财务指标。

    financial_data:
        已上传并清理的财务数据DataFrame
    """

    if not isinstance(financial_data, pd.DataFrame):
        raise ValueError(
            "financial_data 必须是 pandas DataFrame。"
        )

    # 调用已有的财务指标计算模块
    result = calculate_indicators(
        financial_data
    )

    # 筛选公司与年份
    company_data = result[
        (result["公司名称"] == company_name)
        & (result["年份"] == int(year))
    ]

    if company_data.empty:
        return {
            "错误": (
                f"未找到{company_name}"
                f"{year}年的财务数据。"
            )
        }

    row = company_data.iloc[0]

    indicators = [
        "资产负债率",
        "流动比率",
        "营业收入增长率",
        "净利润增长率",
        "应收账款增长率",
        "经营现金流利润比"
    ]

    output = {
        "公司名称": company_name,
        "年份": int(year)
    }

    for indicator in indicators:

        value = row.get(
            indicator,
            np.nan
        )

        if pd.isna(value):
            output[indicator] = None
        else:
            output[indicator] = float(value)

    return output

if __name__ == "__main__":

    test_data = pd.DataFrame([
        {
            "公司名称": "测试公司A",
            "年份": 2024,
            "总资产": 1100,
            "总负债": 650,
            "流动资产": 420,
            "流动负债": 350,
            "营业收入": 660,
            "净利润": 75,
            "经营现金流": 60,
            "应收账款": 140
        },
        {
            "公司名称": "测试公司A",
            "年份": 2025,
            "总资产": 1200,
            "总负债": 900,
            "流动资产": 430,
            "流动负债": 500,
            "营业收入": 720,
            "净利润": 50,
            "经营现金流": 15,
            "应收账款": 210
        }
    ])

    result = get_financial_indicators(
        company_name="测试公司A",
        year=2025,
        financial_data=test_data
    )

    print("财务指标查询结果：")
    print(result)

def detect_financial_risks(
    company_name,
    year,
    financial_data
):
    """
    识别指定公司、指定年份的财务风险。
    """

    if not isinstance(financial_data, pd.DataFrame):
        raise ValueError(
            "financial_data 必须是 pandas DataFrame。"
        )

    result = calculate_indicators(financial_data)

    company_data = result[
        (result["公司名称"] == company_name)
        & (result["年份"] == int(year))
    ]

    if company_data.empty:
        return {
            "错误": f"未找到{company_name}{year}年的财务数据。"
        }

    latest = company_data.iloc[0]

    risks = detect_risks(latest)

    return {
        "公司名称": company_name,
        "年份": int(year),
        "风险数量": len(risks),
        "风险列表": risks
    }


def calculate_risk_score(risk_results, latest=None):
    """与 app.py 保持一致的五维加权评分；缺失维度按有效权重归一化。"""
    if isinstance(risk_results, dict):
        risks = risk_results.get("风险列表", [])
    else:
        risks = risk_results or []

    def value(key):
        if latest is None:
            return None
        try:
            result = float(latest.get(key))
            return result if np.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    def has_risk(keyword):
        return any(
            keyword in str(item.get("风险类型", ""))
            for item in risks if isinstance(item, dict)
        )

    debt, current = value("资产负债率"), value("流动比率")
    revenue, profit = value("营业收入增长率"), value("净利润增长率")
    receivable, cash = value("应收账款增长率"), value("经营现金流利润比")

    if debt is None: debt_score = None
    elif debt <= .5: debt_score = 0
    elif debt <= .6: debt_score = 5
    elif debt <= .7: debt_score = 10
    elif debt <= .85: debt_score = 15
    else: debt_score = 20
    if debt_score is not None and has_risk("偿债"): debt_score = max(debt_score, 10)

    if current is None: liquidity_score = None
    elif current >= 2: liquidity_score = 0
    elif current >= 1.5: liquidity_score = 5
    elif current >= 1.2: liquidity_score = 10
    elif current >= 1: liquidity_score = 15
    else: liquidity_score = 20
    if liquidity_score is not None and has_risk("流动"): liquidity_score = max(liquidity_score, 10)

    if revenue is None or profit is None: profit_score = None
    elif revenue >= 0 and profit >= 0: profit_score = 0
    elif revenue >= 0 or profit >= 0: profit_score = 10
    elif revenue > -.2 and profit > -.2: profit_score = 15
    else: profit_score = 20
    if profit_score is not None and (has_risk("盈利") or has_risk("成长")): profit_score = max(profit_score, 10)

    if receivable is None or revenue is None: receivable_score = None
    else:
        diff = receivable - revenue
        receivable_score = 0 if diff <= 0 else 5 if diff <= .1 else 10 if diff <= .2 else 15 if diff <= .3 else 20
    if receivable_score is not None and has_risk("应收"): receivable_score = max(receivable_score, 10)

    if cash is None: cash_score = None
    elif cash >= 1.2: cash_score = 0
    elif cash >= .8: cash_score = 5
    elif cash >= .5: cash_score = 10
    elif cash >= 0: cash_score = 15
    else: cash_score = 20
    if cash_score is not None and has_risk("现金流"): cash_score = max(cash_score, 15)

    weights = {"偿债风险": .25, "流动性风险": .20, "盈利风险": .20, "应收账款风险": .15, "现金流风险": .20}
    values = {"偿债风险": debt_score, "流动性风险": liquidity_score, "盈利风险": profit_score, "应收账款风险": receivable_score, "现金流风险": cash_score}
    details, weighted, available = [], 0.0, 0.0
    for name, raw in values.items():
        weight = weights[name]
        contribution = None if raw is None else round(raw / 20 * weight * 100, 2)
        if raw is not None:
            weighted += contribution
            available += weight
        details.append({"风险维度": name, "权重": f"{int(weight*100)}%", "风险得分": raw, "满分": 20, "加权得分": contribution, "状态": "数据不足" if raw is None else ("已触发" if raw > 0 else "正常")})

    if available <= 0:
        return {"错误": "有效财务指标不足，无法计算综合风险评分。", "风险维度详情": details}
    score = round(weighted / available, 2)
    level = "低风险" if score <= 20 else "中低风险" if score <= 40 else "中风险" if score <= 60 else "高风险" if score <= 80 else "极高风险"
    return {"综合风险评分": score, "最高分值": 100, "风险等级": level, "有效维度数": sum(v is not None for v in values.values()), "风险维度详情": details, "评分方法": "五维加权评分；缺失维度按有效权重归一化，与页面评分口径一致"}

def get_z_score(
    company_name,
    year,
    financial_data,
    industry=None,
    listing_status=None,
):
    """获取指定企业的 Altman Z-Score，并优先读取明确的行业/上市状态信息。"""
    if not isinstance(financial_data, pd.DataFrame):
        raise ValueError("financial_data 必须是 pandas DataFrame。")

    result = calculate_indicators(financial_data)
    company_data = result[
        (result["公司名称"].astype(str) == str(company_name))
        & (result["年份"] == int(year))
    ]
    if company_data.empty:
        return {"错误": f"未找到{company_name}{year}年的财务数据。"}

    latest = company_data.iloc[0]
    resolved_industry = identify_company_industry(latest, industry)
    resolved_listing = listing_status or latest.get("上市状态")
    if pd.isna(resolved_listing):
        resolved_listing = None

    z_result = calculate_z_score(
        latest,
        industry=resolved_industry,
        listing_status=resolved_listing,
    )
    return {
        "公司名称": company_name,
        "年份": int(year),
        **z_result,
    }


def compare_industry_financial_indicators(
    company_name,
    year,
    financial_data,
    industry=None,
    min_peer_samples=3,
):
    """使用上传文件中同年度、同明确行业的其他企业构建横截面比较。

    不存在可靠同行样本时返回数据不足，不回退到虚构的固定行业数组。
    """
    if not isinstance(financial_data, pd.DataFrame):
        return {"错误": "financial_data 必须是 pandas DataFrame。"}

    try:
        indicators = calculate_indicators(financial_data)
    except Exception as exc:
        return {"错误": f"财务指标计算失败：{exc}"}

    target = indicators[
        (indicators["公司名称"].astype(str) == str(company_name))
        & (indicators["年份"] == int(year))
    ]
    if target.empty:
        return {"错误": f"未找到{company_name}{year}年的财务数据。"}

    target_row = target.iloc[0]
    resolved_industry = identify_company_industry(target_row, industry)
    if not resolved_industry:
        return {
            "公司名称": company_name,
            "年份": int(year),
            "错误": "企业所属行业尚未确认。请在页面中选择行业，或在上传数据中提供“所属行业”字段。",
            "数据来源": "无：未进行行业比较",
        }

    industry_column = next(
        (col for col in ("所属行业", "行业", "证监会行业", "申万行业", "行业名称") if col in indicators.columns),
        None,
    )
    if industry_column is None:
        return {
            "公司名称": company_name,
            "年份": int(year),
            "所属行业": resolved_industry,
            "错误": "上传数据未包含其他企业的行业信息，无法构建同行业样本。",
            "数据来源": "无：缺少同行企业行业字段",
        }

    peers = indicators[
        (indicators["年份"] == int(year))
        & (indicators["公司名称"].astype(str) != str(company_name))
    ].copy()
    peers["_标准行业"] = peers[industry_column].apply(normalize_industry)
    peers = peers[peers["_标准行业"] == resolved_industry]
    peer_values = pd.to_numeric(peers["资产负债率"], errors="coerce")
    peer_values = peer_values.replace([np.inf, -np.inf], np.nan).dropna()

    debt_ratio = target_row.get("资产负债率")
    if pd.isna(debt_ratio):
        return {"公司名称": company_name, "年份": int(year), "所属行业": resolved_industry, "错误": "企业资产负债率缺失，无法比较。"}

    comparison = compare_industry_debt_ratio(
        company_name=company_name,
        debt_ratio=float(debt_ratio),
        industry_name=resolved_industry,
        industry_debt_ratios=peer_values.tolist(),
        data_source=f"用户上传数据：{int(year)}年度、同一标准行业的其他企业样本",
        min_samples=int(min_peer_samples),
    )
    return {"年份": int(year), **comparison}


# FinGuard 5.0：独立的多年度财务趋势查询工具
def get_financial_trends(company_name, start_year, end_year, financial_data):
    if not isinstance(financial_data, pd.DataFrame):
        raise TypeError("financial_data 必须是 DataFrame")
    start_year, end_year = int(start_year), int(end_year)
    if start_year > end_year or end_year - start_year > 20:
        raise ValueError("年份范围无效或超过20年")
    required = ["公司名称", "年份", "营业收入", "净利润", "经营现金流"]
    missing = [c for c in required if c not in financial_data.columns]
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    data = financial_data.loc[financial_data["公司名称"].astype(str).str.strip() == str(company_name).strip(), required].copy()
    data["年份"] = pd.to_numeric(data["年份"], errors="coerce")
    data = data.loc[data["年份"].between(start_year, end_year)].sort_values("年份")
    if data["年份"].duplicated().any():
        raise ValueError("同一企业同一年份存在重复记录")
    records = []
    previous = None
    for _, row in data.iterrows():
        record = {"年份": int(row["年份"])}
        for key in ["营业收入", "净利润", "经营现金流"]:
            number = pd.to_numeric(row[key], errors="coerce")
            record[key] = float(number) if pd.notna(number) and np.isfinite(number) else None
            old = previous.get(key) if previous else None
            record[key + "同比增长率"] = (
                (record[key] - old) / abs(old)
                if record[key] is not None and old is not None and old != 0
                and record["年份"] == previous["年份"] + 1 else None
            )
        records.append(record)
        previous = record
    present = {r["年份"] for r in records}
    missing_years = [y for y in range(start_year, end_year + 1) if y not in present]
    return {"公司名称": str(company_name), "查询起始年": start_year,
            "查询结束年": end_year, "年度数据": records, "缺失年份": missing_years,
            "说明": "同比增长率为小数；利润或现金流由负转正时同比增长率仅作数学变化参考。"}
