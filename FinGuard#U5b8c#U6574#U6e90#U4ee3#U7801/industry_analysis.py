# -*- coding: utf-8 -*-
"""行业识别与行业基准比较。

行业识别只规范化数据源中明确提供的行业信息，不根据公司名称或财务比率猜测行业。
行业比较优先使用用户上传数据中同年度、同行业的企业样本，不内置虚构行业数据。
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

INDUSTRY_ALIASES = {
    "制造业": ("制造业", "制造", "工业制造", "制造业企业"),
    "信息技术/软件服务业": ("信息技术", "软件和信息技术服务业", "软件服务", "软件业", "互联网和相关服务", "计算机服务"),
    "房地产": ("房地产", "房地产业", "房地产开发经营"),
    "建筑业": ("建筑业", "建筑", "土木工程建筑业"),
    "批发零售业": ("批发零售业", "批发业", "零售业", "批发和零售业"),
    "交通运输业": ("交通运输业", "交通运输、仓储和邮政业", "道路运输", "航空运输", "水上运输"),
    "住宿餐饮业": ("住宿餐饮业", "住宿业", "餐饮业"),
    "专业服务业": ("专业服务业", "科学研究和技术服务业", "租赁和商务服务业"),
    "公用事业": ("公用事业", "电力、热力、燃气及水生产和供应业", "电力", "燃气", "水务"),
    "金融业（银行/证券/保险）": ("金融业", "银行业", "银行", "证券业", "证券", "保险业", "保险", "其他金融业"),
    "农林牧渔业": ("农林牧渔业", "农、林、牧、渔业", "农业", "林业", "畜牧业", "渔业"),
    "采矿业": ("采矿业", "煤炭开采", "石油和天然气开采", "有色金属矿采选"),
    "文化传媒业": ("文化传媒业", "文化、体育和娱乐业", "传媒", "新闻和出版业"),
    "卫生和社会工作": ("卫生和社会工作", "医疗服务"),
    "其他非制造业": ("其他非制造业", "其他行业", "其他"),
}


def normalize_industry(raw_industry) -> Optional[str]:
    """将明确的行业名称映射到 FinGuard 统一分类；无法识别返回 None。"""
    if raw_industry is None or pd.isna(raw_industry):
        return None
    text = re.sub(r"\s+", "", str(raw_industry)).strip()
    if not text or text.lower() in {"nan", "none", "null", "未知", "不详"}:
        return None
    for canonical, aliases in INDUSTRY_ALIASES.items():
        if text == canonical or text in aliases:
            return canonical
    # 对国家行业分类中的较长名称做保守关键词匹配，金融行业优先避免误归类。
    for canonical, aliases in INDUSTRY_ALIASES.items():
        if any(alias in text for alias in aliases if len(alias) >= 3):
            return canonical
    return text  # 保留来源给出的细分行业，不擅自改成其他行业


def identify_company_industry(company_row, user_confirmed_industry=None) -> Optional[str]:
    """行业识别优先级：用户本次确认 > 数据中的所属行业/行业字段；不凭公司名推断。"""
    if user_confirmed_industry:
        return normalize_industry(user_confirmed_industry)
    if isinstance(company_row, (dict, pd.Series)):
        for field in ("所属行业", "行业", "证监会行业", "申万行业", "行业名称"):
            value = company_row.get(field)
            normalized = normalize_industry(value)
            if normalized:
                return normalized
    return None


def calculate_industry_percentile(company_value, industry_values: Iterable):
    """计算企业指标在同行业样本中的经验百分位。百分位只描述数值位置，不代表好坏。"""
    try:
        company_value = float(company_value)
    except (TypeError, ValueError):
        return {"错误": "企业指标值缺失或无效"}
    if not math.isfinite(company_value):
        return {"错误": "企业指标值不是有限数值"}
    values = pd.to_numeric(pd.Series(list(industry_values)), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {"错误": "行业样本数据不足"}
    return {
        "企业指标值": company_value,
        "行业中位数": float(values.median()),
        "行业25分位数": float(values.quantile(0.25)),
        "行业75分位数": float(values.quantile(0.75)),
        "企业行业百分位": round(float(values.le(company_value).mean() * 100), 2),
        "行业样本数量": int(len(values)),
    }


def compare_industry_debt_ratio(company_name, debt_ratio, industry_name, industry_debt_ratios, *, data_source="用户上传的同行业样本", min_samples=3):
    """仅在存在足够同行业样本时比较；不会用内置演示数值冒充真实基准。"""
    result = calculate_industry_percentile(debt_ratio, [] if industry_debt_ratios is None else industry_debt_ratios)
    if "错误" in result:
        return {"公司名称": company_name, "所属行业": industry_name, "分析指标": "资产负债率", **result}
    if result["行业样本数量"] < min_samples:
        return {
            "公司名称": company_name, "所属行业": industry_name, "分析指标": "资产负债率",
            "错误": f"同行业有效样本仅{result['行业样本数量']}家，至少需要{min_samples}家，暂不生成行业结论。",
            "行业样本数量": result["行业样本数量"], "数据来源": data_source,
        }
    median = result["行业中位数"]
    comparison = "高于行业中位数" if float(debt_ratio) > median else "低于行业中位数" if float(debt_ratio) < median else "等于行业中位数"
    return {
        "公司名称": company_name, "所属行业": industry_name, "分析指标": "资产负债率",
        **result, "行业比较结论": comparison, "数据来源": data_source,
        "口径说明": "百分位仅表示资产负债率在样本中的相对位置；资产负债率越高通常意味着杠杆水平越高。",
    }


if __name__ == "__main__":
    demo = pd.Series({"所属行业": "房地产业"})
    print("行业标准化测试：", identify_company_industry(demo))
    print("无样本保护测试：", compare_industry_debt_ratio("测试企业", 0.6, "制造业", []))
