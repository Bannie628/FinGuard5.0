# ==========================================
# FinGuard 3.0
# DeepSeek 多工具协同 Agent
# Streamlit 集成版本
# ==========================================

import json
import pandas as pd

from financial_agent import create_agent_client

from financial_tools import (
    get_financial_indicators,
    detect_financial_risks,
    calculate_risk_score,
    get_z_score,
    compare_industry_financial_indicators,
    get_financial_trends
)

# ==========================================
# 1. Agent 工具定义
# ==========================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_indicators",
            "description": "查询指定公司和年份的财务指标。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称"
                    },
                    "year": {
                        "type": "integer",
                        "description": "年份"
                    }
                },
                "required": ["company_name", "year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_financial_risks",
            "description": "识别指定公司和年份的财务风险。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称"
                    },
                    "year": {
                        "type": "integer",
                        "description": "年份"
                    }
                },
                "required": ["company_name", "year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk_score",
            "description": (
                "根据已识别的财务风险计算综合风险评分。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称"
                    },
                    "year": {
                        "type": "integer",
                        "description": "年份"
                    }
                },
                "required": ["company_name", "year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_industry_financial_indicators",
            "description": (
                "将指定公司某年度的财务指标与同行业样本进行比较，"
                "返回行业中位数、25分位数、75分位数、"
                "企业行业百分位以及行业比较结论。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称"
                    },
                    "year": {
                        "type": "integer",
                        "description": "分析年份"
                    }
                },
                "required": [
                    "company_name",
                    "year"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_z_score",
            "description": "使用Altman Z-Score模型分析上市制造业企业的财务困境风险。当用户询问财务困境、破产风险或Z-Score时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称"
                    },
                    "year": {
                        "type": "integer",
                        "description": "分析年份"
                    }
                },
                "required": ["company_name", "year"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_trends",
            "description": "获取指定企业连续多个年度的收入、利润和经营现金流及同比变化；分析多年趋势时优先调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "start_year": {"type": "integer"},
                    "end_year": {"type": "integer"}
                },
                "required": ["company_name", "start_year", "end_year"]
            }
        }
    }

]


# ==========================================
# 2. 工具执行函数
# ==========================================

def execute_tool(
    function_name,
    arguments,
    financial_data
):

    company_name = arguments["company_name"]
    year = int(arguments["year"]) if function_name != "get_financial_trends" else None

    if function_name == "get_financial_trends":
        return get_financial_trends(company_name, arguments["start_year"], arguments["end_year"], financial_data)

    if function_name == "get_financial_indicators":

        return get_financial_indicators(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )

    elif function_name == "detect_financial_risks":

        return detect_financial_risks(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )

    elif function_name == "calculate_risk_score":

        risk_result = detect_financial_risks(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )

        if isinstance(risk_result, dict) and "错误" in risk_result:
            return risk_result

        indicators = get_financial_indicators(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )
        if isinstance(indicators, dict) and "错误" in indicators:
            return indicators
        score_result = calculate_risk_score(
            risk_result,
            latest=indicators
        )

        return {
            "公司名称": company_name,
            "年份": year,
            **score_result
        }

    elif function_name == "get_z_score":

        return get_z_score(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )

    elif function_name == "compare_industry_financial_indicators":

        return compare_industry_financial_indicators(
            company_name=company_name,
            year=year,
            financial_data=financial_data
        )

    
    else:

        raise ValueError(
            f"不支持的工具：{function_name}"
        )


# ==========================================
# 3. 多工具 Agent 主函数
# ==========================================

def run_multi_agent(
    company_name,
    year,
    financial_data,
    user_question=None
):

    if not isinstance(financial_data, pd.DataFrame):
        raise TypeError(
            "financial_data 必须是 pandas DataFrame"
        )

    if financial_data.empty:
        raise ValueError("财务数据不能为空")

    company_name = str(company_name)
    year = int(year)

    # 防止工具查询其他公司或年份
    company_data = financial_data[
        financial_data["公司名称"].astype(str) == company_name
    ].copy()

    if company_data.empty:
        raise ValueError(
            f"没有找到公司：{company_name}"
        )

    available_years = pd.to_numeric(
        company_data["年份"],
        errors="coerce"
    )

    if not (available_years == year).any():
        raise ValueError(
            f"没有找到{company_name}在{year}年的数据"
        )

    if user_question is None:

        user_question = (
            f"请全面分析{company_name}在{year}年的财务风险，"
            "包括财务指标、风险识别、"
            "综合风险评分和管理建议。"
        )

    client = create_agent_client()

    messages = [
        {"role": "system", "content": """你是 FinGuard 5.0 金融财务分析智能体。
先理解用户问题，再自主选择相关 Python 工具。不要机械地调用全部工具。
多年收入、利润、现金流趋势问题优先调用 get_financial_trends；
偿债与盈利分析可调用 get_financial_indicators 和 detect_financial_risks；
综合风险评分调用 calculate_risk_score；财务困境调用 get_z_score；
同行业对比调用 compare_industry_financial_indicators。
仅当用户要求全面风险诊断时，才综合调用多项风险工具。
仅使用工具返回的真实数据，不得编造年份、指标、行业基准或评分。
历史年份允许查询同一家企业上传数据中的其他年份。
对缺失年度或工具失败必须明确说明；只报告已成功执行的工具。
根据用户的问题动态组织报告，避免套用固定九段模板。"""},
        {"role": "user", "content": (
            f"当前选中企业：{company_name}\n当前页面年度：{year}\n"
            f"用户实际问题：{user_question}\n"
            "请围绕用户实际问题选择工具，并依据工具结果作答。"
        )}
    ]

    tool_records = []

    latest_score = None

    # ======================================
    # 最多进行6轮工具调用
    # ======================================

    for step in range(8):
        print("\n当前注册的全部工具:")

        for tool in tools:
            print(tool["function"]["name"])
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2
        )

        message = response.choices[0].message

        # 没有工具调用，返回最终报告
        if not message.tool_calls:

            return {
                "公司名称": company_name,
                "年份": year,
                "分析报告": message.content or "",
                "工具调用记录": tool_records,
                "风险评分结果": latest_score,
                "执行状态": "完成" if (message.content or "").strip() else "未完成"
            }

        messages.append(
            message.model_dump(exclude_none=True)
        )

        # ==================================
        # 执行工具
        # ==================================

        for tool_call in message.tool_calls:

            function_name = tool_call.function.name
            arguments = {}
            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

                if not isinstance(arguments, dict):
                    raise ValueError(
                        "工具参数必须是JSON对象"
                    )

                # 只限制公司，不限制历史年份；具体年份由工具校验数据存在性。
                if str(arguments.get("company_name", "")).strip() != company_name.strip():
                    raise ValueError("工具请求的公司与当前选择不一致")
                if function_name == "get_financial_trends":
                    if "start_year" not in arguments or "end_year" not in arguments:
                        raise ValueError("趋势工具必须提供 start_year 和 end_year")
                elif "year" not in arguments:
                    raise ValueError("单年度工具必须提供 year")

                result = execute_tool(
                    function_name=function_name,
                    arguments=arguments,
                    financial_data=financial_data
                )

            except Exception as e:

                result = {
                    "错误": str(e)
                }

            # 保存工具执行记录
            tool_records.append({
                "工具名称": function_name,
                "工具参数": arguments,
                "执行结果": result
            })

            if (
                function_name == "calculate_risk_score"
                and isinstance(result, dict)
                and "综合风险评分" in result
            ):
                latest_score = result

            # 将工具结果返回DeepSeek
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str
                )
            })

    return {
        "公司名称": company_name,
        "年份": year,
        "分析报告": (
            "Agent已达到最大工具调用轮数，"
            "本次未能完成最终分析。"
        ),
        "工具调用记录": tool_records,
        "风险评分结果": latest_score,
        "执行状态": "未完成"
    }
# ==============================
# 程序启动入口
# ==============================

if __name__ == "__main__":
    print("FinGuard 4.0 多工具 Agent")
    print("请通过 multi_agent_test.py 启动测试")