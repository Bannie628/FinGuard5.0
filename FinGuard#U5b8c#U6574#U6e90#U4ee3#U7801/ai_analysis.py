# ==========================================
# FinGuard DeepSeek 财务风险分析模块
# 支持本地 .env 与 Streamlit Cloud Secrets
# ==========================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_deepseek_api_key():
    """读取 DeepSeek API Key：优先环境变量，其次 Streamlit Secrets。"""
    key = os.getenv("DEEPSEEK_API_KEY")
    if key and key.strip():
        return key.strip()

    try:
        import streamlit as st
        # 支持 Streamlit Cloud Secrets 顶层配置：DEEPSEEK_API_KEY = "..."
        key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if key and str(key).strip():
            return str(key).strip()

        # 兼容可选的分组配置：[deepseek] api_key = "..."
        deepseek_section = st.secrets.get("deepseek", {})
        if hasattr(deepseek_section, "get"):
            key = deepseek_section.get("api_key", "")
            if key and str(key).strip():
                return str(key).strip()
    except Exception:
        # 本地未安装/未配置 Streamlit Secrets 时，继续返回空值
        pass

    return None


def analyze_financial_risk(company_name, financial_data, risk_results):
    """根据企业财务指标和规则识别结果生成风险诊断报告。"""
    api_key = get_deepseek_api_key()
    if not api_key:
        return (
            "AI分析失败：未找到 DEEPSEEK_API_KEY。请检查 Streamlit Cloud 的 Secrets，"
            "配置格式应为 DEEPSEEK_API_KEY = \"你的密钥\"。"
        )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=60.0,
            max_retries=1,
        )

        financial_text = json.dumps(
            financial_data, ensure_ascii=False, indent=2, default=str
        )
        risk_text = json.dumps(
            risk_results, ensure_ascii=False, indent=2, default=str
        )

        prompt = f"""
请对以下上市公司进行财务风险分析。

公司名称：{company_name}

财务数据：
{financial_text}

已识别的财务风险：
{risk_text}

请严格依据提供的数据完成分析，报告包含：
一、财务状况总体评价
二、偿债能力分析
三、盈利能力分析
四、现金流风险分析
五、主要风险及形成原因
六、风险改善建议

要求：使用专业、审慎的金融分析语言；结合具体指标解释；不得编造数据；
对数据不足明确说明；区分已触发规则与潜在风险；建议应具体可执行；
不要把风险提示表述为确定发生的财务危机。
"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一名专业的上市公司财务风险分析师。"
                        "请根据财务指标和风险规则提供客观、审慎、有依据的分析。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=False,
        )

        result = response.choices[0].message.content
        if not result:
            return "AI分析失败：模型未返回有效内容。"
        return result

    except Exception as e:
        return f"AI分析失败：{type(e).__name__}：{str(e)}"
