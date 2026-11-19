# ==========================================
# FinGuard 金融风险诊断 AI Agent
# 支持本地 .env 与 Streamlit Cloud Secrets
# ==========================================

import os
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


def create_agent_client():
    api_key = get_deepseek_api_key()
    if not api_key:
        raise ValueError(
            "未检测到 DEEPSEEK_API_KEY。请在本地 .env 或 Streamlit Cloud → App settings → Secrets 中配置。"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=1,
    )


AGENT_SYSTEM_PROMPT = """
你是 FinGuard 金融风险智能诊断 Agent。
你的任务是协助用户分析上市公司的财务风险。
请优先使用系统提供的财务数据和分析工具，不得编造企业财务数据。
分析应关注偿债能力、流动性、盈利能力、现金流和应收账款风险。
数据不足时必须明确说明。风险评分是规则演示评分，不代表真实违约概率或信用评级。
所有结论仅供研究参考，不构成投资建议。
"""


def ask_financial_agent(question):
    client = create_agent_client()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask_financial_agent("请介绍你可以帮助用户完成哪些财务风险分析任务。"))
