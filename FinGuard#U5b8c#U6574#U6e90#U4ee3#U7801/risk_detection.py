import pandas as pd


def detect_risks(row):
    risks = []

    debt = row.get("资产负债率")
    current = row.get("流动比率")
    revenue = row.get("营业收入增长率")
    profit = row.get("净利润增长率")
    receivable = row.get("应收账款增长率")
    cash_ratio = row.get("经营现金流利润比")

    # 风险1：资产负债率较高
    if pd.notna(debt) and debt > 0.7:
        risks.append({
            "风险类型": "偿债风险",
            "风险等级": "中",
            "风险描述": "资产负债率超过70%，建议关注债务压力。"
        })

    # 风险2：流动比率较低
    if pd.notna(current) and current < 1:
        risks.append({
            "风险类型": "流动性风险",
            "风险等级": "高",
            "风险描述": "流动比率低于1，短期偿债能力值得关注。"
        })

    # 风险3：应收账款增速过快
    if (
        pd.notna(revenue)
        and pd.notna(receivable)
        and receivable > revenue + 0.2
    ):
        risks.append({
            "风险类型": "应收账款风险",
            "风险等级": "中",
            "风险描述": "应收账款增速高于营收增速20个百分点以上。"
        })

    # 风险4：净利润下降
    if pd.notna(profit) and profit < -0.2:
        risks.append({
            "风险类型": "盈利风险",
            "风险等级": "中",
            "风险描述": "净利润同比下降超过20%。"
        })

    # 风险5：利润现金保障不足
    if pd.notna(cash_ratio) and cash_ratio < 0.5:
        risks.append({
            "风险类型": "现金流风险",
            "风险等级": "高",
            "风险描述": "经营现金流与净利润之比低于0.5。"
        })

    return risks