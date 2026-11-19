import math


def _to_finite_float(data, key, label=None):
    """读取有限数值；缺失或无效时给出可读错误。"""
    value = data.get(key, None)
    if value is None:
        raise ValueError(f"缺少模型所需字段：{label or key}")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"字段“{label or key}”不是有效数字")
    if not math.isfinite(value):
        raise ValueError(f"字段“{label or key}”存在无效数值")
    return value


def _is_manufacturing_industry(industry):
    """仅将明确的制造业归为制造业；“非制造业”不得因包含“制造”二字而误匹配。"""
    text = str(industry or "").strip()
    if not text or "非制造" in text:
        return False
    return "制造业" in text or text == "制造"


def _risk_level(score, low_cut, high_cut):
    if score < low_cut:
        return (
            "高风险",
            "Z-Score 低于模型风险下限，企业可能存在较高的财务困境风险。"
        )
    if score < high_cut:
        return (
            "灰色区域",
            "Z-Score 位于模型灰色区域，建议结合现金流、债务期限及行业情况进一步判断。"
        )
    return (
        "低风险",
        "Z-Score 高于模型安全区间下限，但仍需结合其他风险指标综合判断。"
    )


def calculate_z_score(data, industry=None, listing_status=None):
    """
    按企业行业和上市状态选择 Altman 模型。

    - 上市制造业：经典 Altman Z 模型
    - 非上市制造业：Altman Z' 模型
    - 非金融非制造业：Altman Z'' 模型（作为非制造业参考模型）
    - 银行、证券、保险等金融机构：不套用普通企业 Altman 模型

    注意：Z-Score 是辅助预警工具，不是信用评级或违约概率。
    """
    base_result = {
        "Z-Score": None,
        "风险等级": "待确认",
        "风险说明": "",
        "模型名称": None,
        "所属行业": industry,
        "上市状态": listing_status,
        "模型公式": None,
        "风险阈值": None,
        "适用性说明": "",
        "指标明细": {},
    }

    try:
        if not industry:
            base_result["风险等级"] = "缺少行业信息"
            base_result["风险说明"] = "请先确认企业所属行业，再选择适用的 Z-Score 模型。"
            return base_result

        industry_text = str(industry).strip()
        listing_text = str(listing_status or "").strip()
        if _is_manufacturing_industry(industry_text) and listing_text not in (
            "上市", "上市公司", "是", "已上市", "public", "listed",
            "非上市", "未上市", "非上市公司", "否", "private", "unlisted"
        ):
            base_result["风险等级"] = "待确认"
            base_result["风险说明"] = "制造业企业需要确认上市状态，才能匹配上市制造业 Z 或非上市制造业 Z' 模型。"
            base_result["适用性说明"] = "请在企业行业与模型配置中选择“上市”或“非上市”。"
            return base_result

        finance_keywords = ("金融", "银行", "证券", "保险", "信托", "期货")
        if any(word in industry_text for word in finance_keywords):
            base_result["风险等级"] = "模型不适用"
            base_result["风险说明"] = (
                "银行、证券、保险等金融机构的资产负债结构和监管口径特殊，"
                "不应直接套用普通工商企业 Altman Z-Score。"
            )
            base_result["适用性说明"] = "建议改用适合金融机构的资本充足率、流动性及资产质量指标体系。"
            return base_result

        total_assets = _to_finite_float(data, "总资产")
        total_liabilities = _to_finite_float(data, "总负债")
        current_assets = _to_finite_float(data, "流动资产")
        current_liabilities = _to_finite_float(data, "流动负债")
        retained_earnings = _to_finite_float(data, "留存收益")
        ebit = _to_finite_float(data, "息税前利润")
        revenue = _to_finite_float(data, "营业收入")

        if total_assets <= 0:
            raise ValueError("总资产必须大于 0。")
        if total_liabilities <= 0:
            raise ValueError("总负债必须大于 0。")
        if current_assets < 0 or current_liabilities < 0 or revenue < 0:
            raise ValueError("流动资产、流动负债和营业收入不能为负数。")

        working_capital = current_assets - current_liabilities
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x5 = revenue / total_assets
        book_equity = total_assets - total_liabilities

        is_manufacturing = _is_manufacturing_industry(industry_text)
        is_listed = listing_text in ("上市", "上市公司", "是", "已上市", "public", "listed")

        if is_manufacturing and is_listed:
            # 上市制造业经典 Z 模型：X4 使用股权市场价值/总负债
            market_value = _to_finite_float(data, "股票市值")
            if market_value < 0:
                raise ValueError("股票市值不能为负数。")
            x4 = market_value / total_liabilities
            score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            model_name = "Altman Z（上市制造业模型）"
            formula = "Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5"
            low_cut, high_cut = 1.81, 2.99
            x4_name = "X4_权益市场价值负债比"
            details = {
                "X1_营运资本总资产比": round(x1, 4),
                "X2_留存收益总资产比": round(x2, 4),
                "X3_息税前利润总资产比": round(x3, 4),
                x4_name: round(x4, 4),
                "X5_资产周转率": round(x5, 4),
            }
            applicability = "适用于上市制造业企业；本模型使用股票市值作为 X4。"

        elif is_manufacturing and not is_listed:
            # 非上市制造业 Z'：X4 使用账面权益/总负债
            x4 = book_equity / total_liabilities
            score = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
            model_name = "Altman Z'（非上市制造业模型）"
            formula = "Z' = 0.717X1 + 0.847X2 + 3.107X3 + 0.420X4 + 0.998X5"
            low_cut, high_cut = 1.23, 2.90
            details = {
                "X1_营运资本总资产比": round(x1, 4),
                "X2_留存收益总资产比": round(x2, 4),
                "X3_息税前利润总资产比": round(x3, 4),
                "X4_账面权益总负债比": round(x4, 4),
                "X5_资产周转率": round(x5, 4),
            }
            applicability = "适用于非上市制造业企业；X4 使用账面权益（总资产－总负债）/总负债。"

        else:
            # 非制造业 Z''：不使用营业收入/总资产；X4 使用账面权益/总负债
            x4 = book_equity / total_liabilities
            score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
            model_name = "Altman Z''（非制造业参考模型）"
            formula = "Z'' = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4"
            low_cut, high_cut = 1.10, 2.60
            details = {
                "X1_营运资本总资产比": round(x1, 4),
                "X2_留存收益总资产比": round(x2, 4),
                "X3_息税前利润总资产比": round(x3, 4),
                "X4_账面权益总负债比": round(x4, 4),
            }
            applicability = (
                "非金融非制造业参考模型。行业间会计结构差异较大，"
                "结果应结合行业特征解释，不应视为经过本地样本校准的信用评级。"
            )
            if "其他" in industry_text:
                applicability += " 当前行业归类为“其他”，建议进一步核实行业分类及模型适用性。"

        score = round(float(score), 4)
        if not math.isfinite(score):
            raise ValueError("模型计算结果不是有限数值。")

        level, description = _risk_level(score, low_cut, high_cut)
        base_result.update({
            "Z-Score": score,
            "风险等级": level,
            "风险说明": description,
            "模型名称": model_name,
            "模型公式": formula,
            "风险阈值": {"高风险上限": low_cut, "低风险起点": high_cut},
            "适用性说明": applicability,
            "指标明细": details,
        })
        return base_result

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        base_result["风险等级"] = "数据不足"
        base_result["风险说明"] = str(exc)
        base_result["适用性说明"] = "请补齐当前模型要求的财务字段后重新计算。"
        return base_result
