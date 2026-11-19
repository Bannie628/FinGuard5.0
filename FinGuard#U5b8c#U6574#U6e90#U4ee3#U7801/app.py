import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from financial_analysis import calculate_indicators
from risk_detection import detect_risks
from ai_analysis import analyze_financial_risk
from multi_agent import run_multi_agent
from finguard_zscore import calculate_z_score
from industry_analysis import identify_company_industry

# =====================================
# 1. 页面基础设置
# =====================================

st.set_page_config(
    page_title="FinGuard金融风险智能诊断系统",
    page_icon="📊",
    layout="wide"
)
# =====================================
# FinGuard 深蓝科技风主题
# =====================================

st.markdown("""
<style>

/* 左侧工作台导航 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06172b 0%, #0a2340 100%);
    border-right: 1px solid #24527a;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.sidebar-brand { display:flex; align-items:center; gap:12px; padding:12px 10px 22px; border-bottom:1px solid rgba(100,170,220,.2); margin-bottom:22px; }
.sidebar-brand-icon { width:42px; height:42px; display:flex; align-items:center; justify-content:center; border-radius:13px; background:linear-gradient(145deg,#1479c8,#19b5cf); font-size:23px; box-shadow:0 6px 18px rgba(0,150,220,.25); }
.sidebar-brand-name { color:#f3f9ff; font-size:22px; font-weight:850; letter-spacing:.3px; }
.sidebar-brand-sub { color:#78a9cc; font-size:9px; letter-spacing:1.2px; margin-top:2px; }
.sidebar-section-label { color:#6f9bbd; font-size:10px; font-weight:800; letter-spacing:1.4px; margin:4px 8px 12px; }
/* 功能导航：卡片式交互 */
[data-testid="stSidebar"] [data-testid="stRadio"] > div,
[data-testid="stSidebar"] div[role="radiogroup"] {
    display:flex;
    flex-direction:column;
    gap:10px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label {
    display:flex;
    align-items:center;
    width:100%;
    min-height:54px;
    box-sizing:border-box;
    padding:14px 14px;
    border:1px solid rgba(105,158,202,.18);
    border-radius:13px;
    background:linear-gradient(115deg,rgba(17,43,70,.82),rgba(11,31,53,.72));
    transition:all .18s ease;
    cursor:pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:linear-gradient(115deg,rgba(24,79,119,.72),rgba(15,52,83,.82));
    border-color:rgba(68,180,230,.58);
    transform:translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(110deg,rgba(17,112,170,.42),rgba(10,66,105,.58));
    border:1px solid #36b9e8;
    box-shadow:inset 3px 0 0 #35c5ed,0 7px 20px rgba(0,128,190,.13);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color:#c9dced !important;
    font-size:14px;
    font-weight:680;
    letter-spacing:.1px;
    margin:0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color:#ffffff !important;
    font-weight:800;
}
.sidebar-nav-heading {
    color:#e8f5ff;
    font-size:14px;
    font-weight:800;
    margin:2px 8px 4px;
}
.sidebar-nav-caption {
    color:#769bb9;
    font-size:10px;
    line-height:1.7;
    margin:0 8px 14px;
}
.sidebar-nav-active {
    display:flex;
    align-items:center;
    gap:8px;
    margin:14px 2px 2px;
    padding:9px 11px;
    border-radius:10px;
    background:rgba(36,184,153,.09);
    border:1px solid rgba(58,196,163,.2);
    color:#73e0c2;
    font-size:11px;
    font-weight:750;
}
.sidebar-divider { height:1px; background:rgba(100,170,220,.2); margin:22px 4px 18px; }
.sidebar-company { color:#eaf6ff; font-size:15px; font-weight:750; padding:9px 10px; background:rgba(35,111,164,.18); border:1px solid rgba(80,158,210,.25); border-radius:9px; margin-top:7px; }
.sidebar-status { color:#55d9b2; font-size:12px; line-height:1.9; margin-top:14px; padding:11px; border-radius:9px; background:rgba(27,153,119,.10); border:1px solid rgba(57,194,157,.22); }
.sidebar-status span { color:#a8c4da; }
.sidebar-footnote { color:#688ba8; font-size:10px; line-height:1.8; margin:30px 5px 8px; padding-top:14px; border-top:1px solid rgba(100,170,220,.16); }
@media (min-width: 900px) { [data-testid="stSidebar"] { min-width:260px; max-width:260px; } }

/* 整体背景 */
.stApp {
    background: linear-gradient(
        135deg,
        #071426 0%,
        #0B1F3A 55%,
        #102D50 100%
    );
    color: #EAF2FF;
}

/* 页面主内容 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* 标题颜色 */
h1, h2, h3 {
    color: #F1F7FF !important;
    font-weight: 700;
}

/* 普通文字 */
.stApp p,
.stApp label {
    color: #DCE8F7;
}

/* 数据卡片 */
div[data-testid="stMetric"] {
    background: #102A4C;
    border: 1px solid #24476B;
    border-radius: 14px;
    padding: 20px;
}

/* 数据卡片数值 */
div[data-testid="stMetricValue"] {
    color: #56C5FF;
    font-weight: 800;
}

/* 按钮 */
.stButton > button {
    background: linear-gradient(
        90deg,
        #1677FF,
        #00A6FF
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 24px;
}

.stButton > button:hover {
    background: #005EDB;
    color: white;
}

/* 展开面板 */
div[data-testid="stExpander"] {
    background: #102A4C;
    border: 1px solid #24476B;
    border-radius: 12px;
}

/* 分割线 */
hr { border-color: #24476B; }

/* FinGuard 页面升级：统一科技风组件 */
[data-testid="stHeader"] { background: rgba(7,20,38,.82); }
[data-testid="stSidebar"] { background: #091a30; border-right: 1px solid #24476B; }
[data-testid="stFileUploader"] section {
    background: rgba(16,42,76,.72); border: 1px dashed #3989c9;
    border-radius: 14px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #24476B; border-radius: 12px; overflow: hidden;
}
[data-testid="stMetric"] {
    min-height: 118px; box-shadow: 0 8px 24px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] { color: #B8D1EC !important; font-size: .9rem; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.stButton > button, [data-testid="stDownloadButton"] button {
    min-height: 44px; box-shadow: 0 5px 16px rgba(0,116,255,.18);
}
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stExpander"] { box-shadow: 0 5px 18px rgba(0,0,0,.10); }
.stCaption { color: #9FB8D3 !important; }


/* 展示版布局增强 */
.block-container { max-width: 1500px; padding-top: 1.4rem; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #24476B; }
[data-testid="stTabs"] button[role="tab"] { background: #102A4C; border: 1px solid #24476B; border-radius: 10px 10px 0 0; padding: 12px 18px; color: #C8DDF2; }
[data-testid="stTabs"] button[aria-selected="true"] { background: #1768A8; color: #FFFFFF; border-color: #3C9FE8; }
.company-banner { display:flex; align-items:center; justify-content:space-between; gap:18px; flex-wrap:wrap; padding:20px 24px; margin:18px 0 22px; border:1px solid #2B6391; border-radius:16px; background:linear-gradient(110deg,rgba(19,58,96,.96),rgba(13,39,68,.96)); box-shadow:0 10px 28px rgba(0,0,0,.16); }
.company-banner span { color:#9FC7E8; font-size:13px; }
.company-banner h2 { margin:5px 0 0; color:#FFFFFF!important; }
.banner-meta { min-width:115px; display:flex; flex-direction:column; gap:5px; }
.banner-meta strong { font-size:23px; color:#63C9FF; }
.context-chip { display:inline-block; margin-top:26px; padding:10px 15px; border:1px solid #2D628D; border-radius:12px; color:#BFE6FF; background:#102A4C; }
.empty-state { text-align:center; padding:42px 20px; margin:18px 0; border:1px dashed #397BAA; border-radius:18px; background:rgba(16,42,76,.55); }
.empty-state h3 { color:#F1F7FF!important; margin:12px 0 6px; }
.empty-state p { color:#B8D1EC!important; }
.report-panel { padding:20px 22px; border:1px solid #2C628D; border-left:4px solid #35B8F4; border-radius:12px; background:rgba(13,39,68,.72); color:#EAF2FF; line-height:1.8; white-space:normal; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#102A4C,#12385D); border:1px solid #2A5B83; border-radius:14px; padding:16px 18px; }
[data-testid="stMetricValue"] { color:#65CFFF!important; }
[data-testid="stPlotlyChart"] { border:1px solid rgba(57,119,165,.45); border-radius:14px; padding:8px; background:rgba(10,30,52,.35); }

/* ===== FinGuard 竞赛展示强化层 ===== */
.stApp { background: radial-gradient(circle at 12% 0%, #153d63 0, transparent 30%), linear-gradient(145deg,#061321 0%,#091d33 52%,#0d2947 100%); }
.block-container { max-width: 1540px; padding-left: clamp(1rem,3vw,3rem); padding-right: clamp(1rem,3vw,3rem); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { background:rgba(7,23,40,.72); padding:7px; border:1px solid #24496b; border-radius:15px; gap:7px; }
[data-testid="stTabs"] button[role="tab"] { border:1px solid transparent; border-radius:10px; padding:12px 17px; font-weight:650; }
[data-testid="stTabs"] button[aria-selected="true"] { background:linear-gradient(110deg,#1768a8,#168fc0); box-shadow:0 5px 18px rgba(0,145,220,.22); }
[data-testid="stMetric"] { min-height:126px; display:flex; flex-direction:column; justify-content:center; background:linear-gradient(145deg,rgba(17,48,78,.98),rgba(13,37,62,.98)); border:1px solid #2b5c83; box-shadow:0 10px 28px rgba(0,0,0,.16); }
[data-testid="stMetricLabel"] p { color:#b8d1e8!important; font-weight:600; }
[data-testid="stMetricValue"] { color:#69d2ff!important; letter-spacing:-.5px; }
.stButton>button,[data-testid="stDownloadButton"] button { border-radius:11px; font-weight:700; transition:all .18s ease; }
.stButton>button:hover,[data-testid="stDownloadButton"] button:hover { transform:translateY(-1px); box-shadow:0 8px 22px rgba(0,145,220,.25); }
[data-testid="stFileUploader"] { padding:10px; border-radius:15px; background:rgba(12,38,64,.5); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #3979A8; border-radius:14px; overflow:hidden; background:rgba(29,83,128,.24); }
[data-testid="stPlotlyChart"] { border:1px solid rgba(73,151,207,.62); border-radius:16px; overflow:hidden; background:linear-gradient(145deg,rgba(27,76,119,.42),rgba(13,43,73,.48)); padding:8px; box-shadow:0 8px 24px rgba(0,0,0,.12); }
[data-testid="stExpander"] { border:1px solid #24496b; border-radius:13px; background:rgba(12,37,62,.55); }
.company-banner { margin:20px 0 24px; }
.section-kicker { color:#66cfff; font-size:12px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.hero-title { font-size:clamp(2rem,4vw,3rem); line-height:1.08; font-weight:850; letter-spacing:-1px; color:#fff; margin:8px 0; }
.hero-subtitle { color:#c1d7eb; font-size:15px; max-width:780px; line-height:1.7; }
.feature-card { height:100%; min-height:112px; padding:17px 18px; border-radius:14px; border:1px solid #28577d; background:linear-gradient(145deg,rgba(17,51,82,.9),rgba(11,32,54,.9)); }
.feature-card b { display:block; color:#eaf6ff; font-size:15px; margin:6px 0; }
.feature-card span { color:#a9c6df; font-size:12px; line-height:1.5; }
div[data-testid="stAlert"] { border-radius:12px; }
.stCaption, small { color:#9fb9d1!important; }
@media (max-width: 700px) { .block-container { padding-top:1rem; } .company-banner { padding:16px; } }

/* 财务分析 / 风险诊断：统一浅蓝科技风表格容器 */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    width: 100% !important;
    border: 1px solid rgba(91,177,230,.62) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    background: linear-gradient(145deg, rgba(26,75,116,.62), rgba(13,43,73,.78)) !important;
    box-shadow: 0 8px 22px rgba(0,0,0,.14);
}
[data-testid="stDataFrame"] > div { background: transparent !important; }
[data-testid="stDataFrame"] canvas { border-radius: 12px; }
[data-testid="stExpander"] details { background: rgba(22,62,96,.42); }
[data-testid="stExpander"] summary { color: #DDF1FF !important; font-weight: 700; }
.table-section-title {
    color: #EAF6FF; font-size: 16px; font-weight: 800;
    padding: 10px 14px; margin: 12px 0 8px;
    border-left: 4px solid #31B9ED; border-radius: 0 10px 10px 0;
    background: linear-gradient(90deg, rgba(37,126,184,.24), rgba(37,126,184,0));
}
.chart-note { color:#9FC5E2; font-size:12px; margin:3px 0 8px; }


/* ===== Sidebar navigation visibility and visual polish ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06172b 0%, #0b2745 58%, #0d3152 100%) !important;
    border-right: 1px solid rgba(72, 160, 218, .35) !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: .65rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] {
    width: 100%;
    padding: 4px 2px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #8eb8d8 !important;
    font-size: 11px !important;
    font-weight: 750 !important;
    letter-spacing: 1px;
    margin: 0 0 8px 4px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 9px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 50px !important;
    padding: 10px 12px !important;
    margin: 0 !important;
    border: 1px solid rgba(112, 173, 216, .24) !important;
    border-radius: 12px !important;
    background: rgba(15, 43, 70, .88) !important;
    box-sizing: border-box !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label * {
    color: #d8eafa !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(105deg, rgba(19, 119, 177, .55), rgba(13, 70, 111, .75)) !important;
    border-color: #39bce9 !important;
    box-shadow: inset 3px 0 0 #42d0f4, 0 6px 16px rgba(0, 135, 205, .16) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) * {
    color: #ffffff !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input {
    accent-color: #39bce9 !important;
}
@media (max-width: 900px) {
    [data-testid="stSidebar"] { min-width: 250px; }
}


/* ===== 高对比度文字修复：侧边导航、说明文字与图表标签 ===== */
/* 侧边栏品牌与辅助文字：提高亮度，避免深蓝背景上发灰 */
[data-testid="stSidebar"] .sidebar-brand-sub,
[data-testid="stSidebar"] .sidebar-section-label,
[data-testid="stSidebar"] .sidebar-nav-caption,
[data-testid="stSidebar"] .sidebar-footnote,
[data-testid="stSidebar"] .sidebar-status span {
    color: #B9DDF5 !important;
    opacity: 1 !important;
}

/* 左侧四个功能入口：未选中状态也保持清晰 */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label span,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label div {
    color: #F0F7FF !important;
    opacity: 1 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,.35);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(105deg, rgba(19,119,177,.72), rgba(13,70,111,.88)) !important;
    border-color: #55DDF5 !important;
}
[data-testid="stSidebar"] .sidebar-nav-active {
    color: #8FFFE0 !important;
}

/* Plotly 图表：坐标轴刻度、图例、标题与注释统一提亮 */
.js-plotly-plot .plotly text,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .legend text,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .annotation-text {
    fill: #F0F7FF !important;
    color: #F0F7FF !important;
    opacity: 1 !important;
}
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text {
    font-weight: 600 !important;
}
.js-plotly-plot .plotly .legend text {
    font-weight: 650 !important;
}
.js-plotly-plot .plotly .xaxislayer-above path,
.js-plotly-plot .plotly .yaxislayer-above path {
    stroke: rgba(190,220,245,.65) !important;
}
.js-plotly-plot .plotly .gridlayer path {
    stroke: rgba(175,210,238,.20) !important;
}

/* Streamlit 常规辅助说明文字也避免使用低对比灰色 */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
.stCaption,
small {
    color: #B9D5EA !important;
    opacity: 1 !important;
}



/* ===== 上传区域文字提亮：与“上传企业年度财务数据”说明文字统一 ===== */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label *,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploader"] section div {
    color: #B9D5EA !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #EAF6FF !important;
}


/* 导航改为原生按钮，直接设置按钮本体与内部文本的高对比度颜色 */
[data-testid="stSidebar"] [class*="st-key-fg_nav_"] button {
    width: 100% !important;
    min-height: 68px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 14px 16px !important;
    border-radius: 13px !important;
    border: 1px solid #51789B !important;
    background: #102D49 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [class*="st-key-fg_nav_"] button *,
[data-testid="stSidebar"] [class*="st-key-fg_nav_"] button p,
[data-testid="stSidebar"] [class*="st-key-fg_nav_"] button span {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 750 !important;
    text-shadow: none !important;
}
[data-testid="stSidebar"] [class*="st-key-fg_nav_"] button:hover {
    background: #18547B !important;
    border-color: #52D8F3 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# FinGuard 品牌顶部
# =====================================
st.markdown("""
<div style="padding:30px 32px 27px;border-radius:20px;background:linear-gradient(112deg,rgba(12,37,64,.98),rgba(19,83,132,.96) 68%,rgba(13,55,91,.98));border:1px solid #327caf;margin:4px 0 20px;box-shadow:0 18px 45px rgba(0,0,0,.22);position:relative;overflow:hidden;">
  <div style="position:absolute;right:-35px;top:-90px;width:310px;height:310px;border:1px solid rgba(113,208,255,.18);border-radius:50%;box-shadow:0 0 0 28px rgba(113,208,255,.035),0 0 0 58px rgba(113,208,255,.025);"></div>
  <div class="section-kicker">FINANCIAL INTELLIGENCE PLATFORM</div>
  <div class="hero-title">🛡️ FinGuard <span style="font-size:clamp(1rem,2vw,1.35rem);font-weight:600;color:#8bdcff;letter-spacing:0">金融风险智能诊断系统</span></div>
  <div class="hero-subtitle">融合多维财务指标、可解释风险规则与大语言模型智能体，构建企业财务风险识别、评估与辅助决策的一体化分析平台。</div>
  <div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:20px;">
    <span style="background:rgba(5,22,39,.38);border:1px solid #3979a6;border-radius:30px;padding:7px 13px;color:#c8efff;font-size:12px;">◈ 多维财务指标</span>
    <span style="background:rgba(5,22,39,.38);border:1px solid #3979a6;border-radius:30px;padding:7px 13px;color:#c8efff;font-size:12px;">◈ 五维风险评估</span>
    <span style="background:rgba(5,22,39,.38);border:1px solid #3979a6;border-radius:30px;padding:7px 13px;color:#c8efff;font-size:12px;">◈ Altman Z-Score</span>
    <span style="background:rgba(5,22,39,.38);border:1px solid #3979a6;border-radius:30px;padding:7px 13px;color:#c8efff;font-size:12px;">◈ DeepSeek · Multi-Agent</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================
# 2. 财务数据字段
# =====================================

REQUIRED_COLUMNS = [
    "公司名称",
    "年份",
    "总资产",
    "总负债",
    "流动资产",
    "流动负债",
    "营业收入",
    "净利润",
    "经营现金流",
    "应收账款",
    "留存收益",
    "息税前利润"
]

NUMERIC_COLUMNS = [
    "总资产",
    "总负债",
    "流动资产",
    "流动负债",
    "营业收入",
    "净利润",
    "经营现金流",
    "应收账款",
    "留存收益",
    "息税前利润"
]

INDICATOR_COLUMNS = [
    "资产负债率",
    "流动比率",
    "营业收入增长率",
    "净利润增长率",
    "应收账款增长率",
    "经营现金流利润比"
]


# 五类风险，每类20分
RISK_CATEGORIES = [
    ("偿债风险", ("资产负债", "偿债", "杠杆")),
    ("流动性风险", ("流动", "短期偿债")),
    ("应收账款风险", ("应收", "回款")),
    ("盈利风险", ("利润", "盈利", "业绩")),
    ("现金流风险", ("现金流", "现金保障"))
]

# =====================================
# 3. 数据清理
# =====================================

def clean_financial_data(df):

    df = df.copy()

    # 清理列名
    df.columns = df.columns.astype(str).str.strip()

    # 检查必要字段
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Excel缺少必要字段：" + "、".join(missing)
        )

    # 删除完全空白的行
    df = df.dropna(how="all")

    # 识别备注行
    company_text = (
        df["公司名称"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    note_mask = company_text.str.match(
        r"^(注[:：]|备注[:：]|说明[:：])"
    )

    df = df.loc[~note_mask].copy()

    # 清理公司名称
    df["公司名称"] = (
        df["公司名称"]
        .astype("string")
        .str.strip()
    )

    if df.empty:
        raise ValueError(
            "文件没有可供分析的财务数据。"
        )

    if (
        df["公司名称"].isna().any()
        or (df["公司名称"] == "").any()
    ):
        raise ValueError(
            "公司名称存在空值，请检查Excel。"
        )

    # 转换年份
    df["年份"] = pd.to_numeric(
        df["年份"],
        errors="coerce"
    )

    invalid_year = (
        df["年份"].isna()
        | ~np.isfinite(df["年份"])
        | (df["年份"] % 1 != 0)
    )

    if invalid_year.any():
        raise ValueError(
            "年份列存在空值或无效年份，请检查Excel。"
        )

    df["年份"] = df["年份"].astype(int)

    # 转换必需财务数据
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # 股票市值仅用于“上市制造业经典 Z 模型”，对其他模型不是必需字段。
    # 若上传了该列则进行数值清洗；未上传时保留缺失，由模型适用性检查决定能否计算。
    if "股票市值" in df.columns:
        df["股票市值"] = pd.to_numeric(df["股票市值"], errors="coerce")
        df["股票市值"] = df["股票市值"].replace([np.inf, -np.inf], np.nan)

    # 检查缺失值
    missing_values = df[NUMERIC_COLUMNS].isna()

    if missing_values.any().any():

        problem_columns = missing_values.any()

        problem_columns = (
            problem_columns[
                problem_columns
            ].index.tolist()
        )

        raise ValueError(
            "以下财务字段存在缺失或无效数据："
            + "、".join(problem_columns)
        )

    # 检查重复记录
    duplicates = df.duplicated(
        subset=["公司名称", "年份"]
    )

    if duplicates.any():
        raise ValueError(
            "Excel存在重复的公司和年份记录。"
        )

    # 排序
    df = df.sort_values(
        ["公司名称", "年份"]
    ).reset_index(drop=True)

    return df


# =====================================
# 4. 财务指标检查
# =====================================

def validate_indicators(df):

    df = df.copy()

    missing = [
        col for col in INDICATOR_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "财务分析模块缺少以下指标："
            + "、".join(missing)
        )

    for col in INDICATOR_COLUMNS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

    return df


# =====================================
# 5. 财务趋势图
# =====================================

def display_finguard_table(data):
    """统一渲染 FinGuard 深蓝浅蓝数据表；宽表采用横向滚动，避免列宽过窄导致文字竖排。"""
    from html import escape

    if data is None:
        st.info("暂无可展示的数据。")
        return
    table_data = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if table_data.empty:
        st.info("暂无可展示的数据。")
        return

    table_data.columns = [str(c) for c in table_data.columns]
    for col in table_data.columns:
        if pd.api.types.is_float_dtype(table_data[col]):
            if "评分" in col or "得分" in col:
                table_data[col] = table_data[col].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")
            else:
                table_data[col] = table_data[col].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
        else:
            table_data[col] = table_data[col].map(lambda v: "—" if pd.isna(v) else str(v))

    # 宽表按列数动态设置最小宽度，保留可读列宽并由外层容器横向滚动。
    column_count = len(table_data.columns)
    min_table_width = max(680, min(1900, column_count * 112))
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in table_data.columns)
    body_rows = []
    for row in table_data.itertuples(index=False, name=None):
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")

    table_html = f'''
    <div class="finguard-table-scroll" style="width:100%; max-width:100%; overflow-x:auto; overflow-y:hidden; margin:12px 0 20px 0; border:1px solid #3979A8; border-radius:12px; background:#102A4C;">
      <table class="finguard-data-table" style="width:100%; min-width:{min_table_width}px; table-layout:auto; border-collapse:separate; border-spacing:0; color:#EAF6FF; font-size:14px;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    <style>
      .finguard-data-table thead th {{ background:#1B5B88; color:#F2FAFF; font-weight:700; padding:13px 16px; text-align:center; border-bottom:1px solid #56A9D8; white-space:nowrap; min-width:92px; }}
      .finguard-data-table thead th:first-child {{ min-width:130px; }}
      .finguard-data-table tbody td {{ background:#102F52; color:#E1F1FF; padding:12px 16px; text-align:center; border-bottom:1px solid rgba(100,174,220,.20); white-space:nowrap; min-width:92px; }}
      .finguard-data-table tbody tr:nth-child(even) td {{ background:#174269; }}
      .finguard-data-table tbody tr:last-child td {{ border-bottom:none; }}
      .finguard-data-table tbody tr:hover td {{ background:#205783; }}
      .finguard-table-scroll::-webkit-scrollbar {{ height:9px; }}
      .finguard-table-scroll::-webkit-scrollbar-track {{ background:#0B2038; border-radius:8px; }}
      .finguard-table-scroll::-webkit-scrollbar-thumb {{ background:#3979A8; border-radius:8px; }}
    </style>
    '''
    st.markdown(table_html, unsafe_allow_html=True)


def apply_finguard_chart_theme(fig, height=360):
    """为财务与风险图表应用统一的深蓝/浅蓝视觉，避免白底突兀。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,39,66,.32)",
        font=dict(color="#DCEBFA", family="Arial, Microsoft YaHei, sans-serif", size=12),
        height=height,
        margin=dict(l=48, r=28, t=62, b=48),
        title=dict(x=0.03, xanchor="left", font=dict(size=17, color="#F1F8FF")),
        legend=dict(bgcolor="rgba(12,39,66,.45)", bordercolor="rgba(91,177,230,.25)", borderwidth=1),
        hoverlabel=dict(bgcolor="#123456", bordercolor="#55BDEB", font=dict(color="#F1F8FF")),
    )
    fig.update_xaxes(
        showgrid=False,
        zerolinecolor="rgba(132,185,222,.25)",
        linecolor="rgba(132,185,222,.35)",
        tickfont=dict(color="#C9DFF0"),
        title_font=dict(color="#B8D5EA"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(132,185,222,.16)",
        zerolinecolor="rgba(132,185,222,.25)",
        linecolor="rgba(132,185,222,.25)",
        tickfont=dict(color="#C9DFF0"),
        title_font=dict(color="#B8D5EA"),
    )
    return fig


def draw_financial_chart(company_data):

    st.subheader("二、财务变化趋势")

    chart_data = company_data.copy()

    chart_data["年份"] = pd.to_numeric(
        chart_data["年份"],
        errors="coerce"
    )

    chart_data = chart_data.sort_values("年份")

    indicators = [
        "资产负债率",
        "营业收入增长率",
        "净利润增长率"
    ]

    for col in indicators:

        chart_data[col] = pd.to_numeric(
            chart_data[col],
            errors="coerce"
        )

        chart_data[col] = chart_data[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # 查看趋势图原始数据
    with st.expander("查看趋势图原始数据"):

        display_finguard_table(chart_data[["年份"] + indicators])

    # 创建趋势图
    fig = go.Figure()

    trace_colors = {"资产负债率": "#35C5F0", "营业收入增长率": "#43D6A5", "净利润增长率": "#F3B65B"}
    for col in indicators:

        valid_data = chart_data[
            ["年份", col]
        ].dropna()

        if valid_data.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=valid_data["年份"].tolist(),
                y=valid_data[col].tolist(),
                mode="lines+markers",
                name=col,
                line=dict(width=3, color=trace_colors.get(col, "#35C5F0")),
                marker=dict(size=8, color=trace_colors.get(col, "#35C5F0"), line=dict(width=1, color="#EAF6FF")),
                hovertemplate="年份：%{x}<br>指标值：%{y:.2f}<extra>%{fullData.name}</extra>"
            )
        )

    fig.update_layout(
        title="主要财务指标变化趋势",
        xaxis_title="报告年份",
        yaxis_title="指标数值",
        hovermode="x unified",
        hoverdistance=100,
        bargap=0.25,
    )
    apply_finguard_chart_theme(fig, height=390)

    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=chart_data["年份"].dropna().tolist(),
        tickformat="d"
    )

    if len(fig.data) == 0:

        st.warning(
            "没有可用于绘图的有效财务指标。"
        )

    else:

        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 五维财务风险自动诊断
# ==========================================
def draw_risk_insight(details):
    st.markdown("### 🔍 五维财务风险自动诊断")
    recommendations = {
        "偿债风险": "建议优化资本结构，控制负债规模，增强长期偿债能力。",
        "流动性风险": "建议加强现金流管理，合理安排短期债务，提高流动资产保障能力。",
        "盈利风险": "建议分析营业收入和净利润变化原因，优化成本结构，提高盈利能力。",
        "应收账款风险": "建议加强客户信用管理，完善应收账款催收机制，降低坏账风险。",
        "现金流风险": "建议改善经营性现金流，关注现金回款情况，提高现金流稳定性。"
    }
    if not details:
        st.warning("暂无风险评分数据，无法生成诊断。")
        return
    valid_details = [
        item for item in details
        if item.get("风险得分") is not None
        and np.isfinite(pd.to_numeric(item.get("风险得分"), errors="coerce"))
    ]
    if not valid_details:
        st.warning("有效风险评分不足，无法生成诊断。")
        return
    sorted_details = sorted(
        valid_details,
        key=lambda x: float(x["风险得分"]),
        reverse=True
    )
    highest = sorted_details[0]
    st.info(
        f"当前最突出的风险维度："
        f"{highest['风险维度']}，"
        f"风险得分：{highest['风险得分']}/20"
    )
    st.markdown("#### 各维度风险分析")
    for item in sorted_details:
        name = item["风险维度"]
        score = float(item["风险得分"])
        if score >= 15:
            level = "高风险"
            st.error(f"🔴 {name}：{score:.1f}/20 —— {level}")
        elif score >= 10:
            level = "中风险"
            st.warning(f"🟠 {name}：{score:.1f}/20 —— {level}")
        elif score >= 5:
            level = "较低风险"
            st.info(f"🟡 {name}：{score:.1f}/20 —— {level}")
        else:
            level = "低风险"
            st.success(f"🟢 {name}：{score:.1f}/20 —— {level}")
        if score >= 10:
            st.write(
                f"**改进建议：** {recommendations.get(name, '建议进一步分析该维度的财务指标。')}"
            )
    st.caption(
        "说明：以上诊断依据系统预设的五维风险评分规则生成，"
        "仅供企业财务风险辅助分析参考。"
    )

# =====================================
# 6. 风险结果标准化
# =====================================

def normalize_risks(risks):

    if risks is None:
        return []

    if isinstance(risks, pd.DataFrame):
        return risks.to_dict("records")

    if isinstance(risks, dict):
        return [risks]

    if isinstance(risks, (list, tuple)):
        return list(risks)

    raise ValueError(
        "风险识别模块返回了不支持的数据格式。"
    )

# =====================================
# 7. 现金流风险补充检查
# =====================================

def check_cashflow_risk(latest, risks):

    cash_ratio = latest.get(
        "经营现金流利润比",
        np.nan
    )

    if pd.isna(cash_ratio):
        return risks

    cash_ratio = float(cash_ratio)

    # 避免重复添加现金流风险
    has_cashflow_risk = any(
        isinstance(risk, dict)
        and "现金流" in str(
            risk.get("风险类型", "")
        )
        for risk in risks
    )

    if cash_ratio < 0.5 and not has_cashflow_risk:

        risks.append({
            "风险类型": "现金流风险",
            "风险等级": "高",
            "风险描述": (
                "经营现金流利润比低于0.5，"
                "利润现金保障能力需要关注。"
            )
        })

    return risks


# =====================================
# 8. 风险展示
# =====================================

def show_risks(risks):

    st.subheader("三、财务风险识别")

    if not risks:

        st.success(
            "当前未触发预设财务风险规则。"
        )

        return

    for risk in risks:

        if isinstance(risk, dict):

            risk_type = risk.get(
                "风险类型",
                "未知风险"
            )

            risk_level = risk.get(
                "风险等级",
                "待评估"
            )

            risk_description = risk.get(
                "风险描述",
                ""
            )

            message = (
                f"【{risk_level}】"
                f"{risk_type}："
                f"{risk_description}"
            )

        else:

            message = str(risk)

        st.warning(message)


# =====================================
# 9. 五维加权动态风险评分模型
# =====================================
def calculate_risk_score(latest, risks):
    """
    FinGuard 五维加权风险评分模型
    风险权重：
    偿债风险：25%
    流动性风险：20%
    盈利风险：20%
    应收账款风险：15%
    现金流风险：20%
    每个维度原始得分0-20分。
    综合得分0-100分，分数越高表示风险越大。
    """
    def get_value(column):
        value = latest.get(column, np.nan)
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    # 获取财务指标
    debt_ratio = get_value("资产负债率")
    current_ratio = get_value("流动比率")
    revenue_growth = get_value("营业收入增长率")
    profit_growth = get_value("净利润增长率")
    receivable_growth = get_value("应收账款增长率")
    cash_ratio = get_value("经营现金流利润比")
    # 检查风险识别结果
    def has_risk(keyword):
        for risk in risks:
            if isinstance(risk, dict):
                risk_type = str(risk.get("风险类型", ""))
            else:
                risk_type = str(risk)
            if keyword in risk_type:
                return True
        return False
    details = []
    # ---------------------------------
    # 1. 偿债风险（权重25%）
    # ---------------------------------
    if debt_ratio is None:
        debt_score = None
    elif debt_ratio <= 0.5:
        debt_score = 0
    elif debt_ratio <= 0.6:
        debt_score = 5
    elif debt_ratio <= 0.7:
        debt_score = 10
    elif debt_ratio <= 0.85:
        debt_score = 15
    else:
        debt_score = 20
    if debt_score is not None and has_risk("偿债"):
        debt_score = max(debt_score, 10)
    details.append({
        "风险维度": "偿债风险",
        "权重": "25%",
        "风险得分": debt_score,
        "满分": 20,
        "状态": "数据不足" if debt_score is None else
                ("已触发" if debt_score > 0 else "正常")
    })
    # ---------------------------------
    # 2. 流动性风险（权重20%）
    # ---------------------------------
    if current_ratio is None:
        liquidity_score = None
    elif current_ratio >= 2:
        liquidity_score = 0
    elif current_ratio >= 1.5:
        liquidity_score = 5
    elif current_ratio >= 1.2:
        liquidity_score = 10
    elif current_ratio >= 1:
        liquidity_score = 15
    else:
        liquidity_score = 20
    if liquidity_score is not None and has_risk("流动"):
        liquidity_score = max(liquidity_score, 10)
    details.append({
        "风险维度": "流动性风险",
        "权重": "20%",
        "风险得分": liquidity_score,
        "满分": 20,
        "状态": "数据不足" if liquidity_score is None else
                ("已触发" if liquidity_score > 0 else "正常")
    })
    # ---------------------------------
    # 3. 盈利风险（权重20%）
    # ---------------------------------
    if revenue_growth is None or profit_growth is None:
        profit_score = None
    else:
        if revenue_growth >= 0 and profit_growth >= 0:
            profit_score = 0
        elif revenue_growth >= 0 or profit_growth >= 0:
            profit_score = 10
        elif revenue_growth > -0.2 and profit_growth > -0.2:
            profit_score = 15
        else:
            profit_score = 20
    if profit_score is not None and has_risk("成长"):
        profit_score = max(profit_score, 10)
    details.append({
        "风险维度": "盈利风险",
        "权重": "20%",
        "风险得分": profit_score,
        "满分": 20,
        "状态": "数据不足" if profit_score is None else
                ("已触发" if profit_score > 0 else "正常")
    })
    # ---------------------------------
    # 4. 应收账款风险（权重15%）
    # ---------------------------------
    if receivable_growth is None or revenue_growth is None:
        receivable_score = None
    else:
        difference = receivable_growth - revenue_growth
        if difference <= 0:
            receivable_score = 0
        elif difference <= 0.1:
            receivable_score = 5
        elif difference <= 0.2:
            receivable_score = 10
        elif difference <= 0.3:
            receivable_score = 15
        else:
            receivable_score = 20
    if receivable_score is not None and has_risk("应收"):
        receivable_score = max(receivable_score, 10)
    details.append({
        "风险维度": "应收账款风险",
        "权重": "15%",
        "风险得分": receivable_score,
        "满分": 20,
        "状态": "数据不足" if receivable_score is None else
                ("已触发" if receivable_score > 0 else "正常")
    })
    # ---------------------------------
    # 5. 现金流风险（权重20%）
    # ---------------------------------
    if cash_ratio is None:
        cash_score = None
    elif cash_ratio >= 1.2:
        cash_score = 0
    elif cash_ratio >= 0.8:
        cash_score = 5
    elif cash_ratio >= 0.5:
        cash_score = 10
    elif cash_ratio >= 0:
        cash_score = 15
    else:
        cash_score = 20
    if cash_score is not None and has_risk("现金流"):
        cash_score = max(cash_score, 15)
    details.append({
        "风险维度": "现金流风险",
        "权重": "20%",
        "风险得分": cash_score,
        "满分": 20,
        "状态": "数据不足" if cash_score is None else
                ("已触发" if cash_score > 0 else "正常")
    })
    # ---------------------------------
    # 6. 计算加权综合评分
    # ---------------------------------
    weights = {
        "偿债风险": 0.25,
        "流动性风险": 0.20,
        "盈利风险": 0.20,
        "应收账款风险": 0.15,
        "现金流风险": 0.20
    }
    available_weight = 0
    weighted_score = 0
    for item in details:
        raw_score = item["风险得分"]
        if raw_score is None:
            item["加权得分"] = None
            continue
        weight = weights[item["风险维度"]]
        weighted_score += raw_score / 20 * weight * 100
        available_weight += weight
        item["加权得分"] = round(
            raw_score / 20 * weight * 100, 2
        )
    if available_weight == 0:
        raise ValueError("有效财务指标不足，无法计算综合风险评分。")
    # 缺失维度不直接按零分处理，按有效权重归一化
    score = round(weighted_score / available_weight, 2)
    # ---------------------------------
    # 7. 风险等级判定
    # ---------------------------------
    if score <= 20:
        level, color = "低风险", "#2ca02c"
    elif score <= 40:
        level, color = "中低风险", "#8dbb35"
    elif score <= 60:
        level, color = "中风险", "#e6a700"
    elif score <= 80:
        level, color = "高风险", "#ff7f0e"
    else:
        level, color = "极高风险", "#d62728"
    return score, level, color, details

# =====================================
# 10. 风险仪表盘
# =====================================

# ==========================================
# 五维财务风险雷达图
# ==========================================

def _risk_chart_theme(fig, height=350):
    """统一风险图表主题：透明底色、浅蓝卡片、统一字体与边距。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCEBFA", family="Arial, Microsoft YaHei, sans-serif", size=12),
        height=height,
        margin=dict(l=28, r=28, t=62, b=35),
        showlegend=False,
        title=dict(x=0.04, xanchor="left", font=dict(size=17, color="#F1F8FF")),
    )
    return fig


def draw_risk_dashboard(score, level, color, details):
    """对称式综合风险驾驶舱：上方仪表盘/风险摘要，下方柱状图/雷达图。"""
    st.subheader("企业综合风险评估")
    st.caption(
        "基于偿债、流动性、盈利、应收账款和现金流五个维度进行综合评价。"
    )

    # 整理五维评分；缺失值保留为空，不伪造为 0 分。
    dimensions = ["偿债风险", "流动性风险", "盈利风险", "应收账款风险", "现金流风险"]
    score_map = {}
    for item in details:
        name = item.get("风险维度", "")
        value = item.get("风险得分")
        if name in dimensions and value is not None:
            try:
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    score_map[name] = max(0.0, min(20.0, numeric_value))
            except (TypeError, ValueError):
                pass

    # 第一行：仪表盘与风险摘要，保持 1:1 对称。
    gauge_col, summary_col = st.columns([1, 1])
    with gauge_col:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={
                "suffix": " / 100",
                "font": {"size": 42, "color": "#F4FAFF", "family": "Arial, Microsoft YaHei, sans-serif"},
                "valueformat": ".1f",
            },
            title={
                "text": "综合财务风险评分",
                "font": {"size": 18, "color": "#EAF5FF"},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "tickfont": {"size": 11, "color": "#C7DDF0"},
                    "tickcolor": "#83AFCF",
                },
                "bar": {"color": color, "thickness": 0.27},
                "bgcolor": "rgba(17,55,91,.55)",
                "borderwidth": 1,
                "bordercolor": "rgba(96,163,211,.35)",
                "steps": [
                    {"range": [0, 20], "color": "#164F65"},
                    {"range": [20, 40], "color": "#236A75"},
                    {"range": [40, 60], "color": "#8B783D"},
                    {"range": [60, 80], "color": "#A65D3D"},
                    {"range": [80, 100], "color": "#873D4B"},
                ],
            },
            domain={"x": [0.08, 0.92], "y": [0.04, 0.83]},
        ))
        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=22, r=22, t=65, b=18),
            font=dict(color="#DCEBFA"),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with summary_col:
        triggered_count = sum(
            d.get("风险得分") is not None and float(d["风险得分"]) > 0
            for d in details
        )
        level_color = color if color else "#26B9E8"
        st.markdown(
            f"""
            <div style="height:340px;box-sizing:border-box;padding:25px 26px;border-radius:18px;
                        border:1px solid #2674A9;background:linear-gradient(145deg,rgba(20,65,108,.78),rgba(11,39,70,.88));
                        box-shadow:0 10px 28px rgba(0,0,0,.16);display:flex;flex-direction:column;justify-content:center;">
              <div style="font-size:20px;font-weight:800;color:#EAF6FF;margin-bottom:22px;">当前风险等级
                <span style="display:inline-block;margin-left:10px;padding:7px 15px;border-radius:9px;background:{level_color};color:#fff;">{level}</span>
              </div>
              <div style="display:flex;align-items:baseline;gap:8px;margin:7px 0 17px;">
                <span style="font-size:14px;color:#B9D4EA;">风险得分</span>
                <strong style="font-size:30px;color:#49D8C2;">{float(score):.2f}</strong>
                <span style="font-size:15px;color:#DCEBFA;">/ 100</span>
              </div>
              <div style="display:flex;align-items:baseline;gap:8px;margin-bottom:24px;">
                <span style="font-size:14px;color:#B9D4EA;">已触发评分维度</span>
                <strong style="font-size:25px;color:#49D8C2;">{triggered_count}</strong>
                <span style="color:#DCEBFA;">/ 5</span>
              </div>
              <div style="padding:15px 16px;border-radius:12px;background:rgba(65,137,190,.18);border:1px solid rgba(99,177,226,.28);">
                <div style="font-weight:750;color:#DDF1FF;margin-bottom:7px;">风险等级划分</div>
                <div style="font-size:13px;line-height:1.9;color:#C2D9EC;">0–20 低风险　|　21–40 中低风险<br>41–60 中风险　|　61–80 高风险　|　81–100 极高风险</div>
              </div>
              <div style="font-size:11px;color:#91B3CE;margin-top:13px;line-height:1.5;">本评分为可解释的竞赛演示模型，不等同于经统计验证的企业信用评级。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 第二行：维度柱状图与雷达图，统一高度、透明背景和浅蓝卡片。
    bar_col, radar_col = st.columns([1, 1])
    with bar_col:
        bar_names = [name for name in dimensions if name in score_map]
        bar_values = [score_map[name] for name in bar_names]
        missing_names = [name for name in dimensions if name not in score_map]
        chart_names = bar_names + missing_names
        chart_values = bar_values + [None] * len(missing_names)
        bar_colors = ["#28B8E8" if value is not None and value < 10 else "#F0A64A" if value is not None and value < 15 else "#E56B62" for value in chart_values]
        bar = go.Figure(go.Bar(
            x=chart_names,
            y=[value if value is not None else 0 for value in chart_values],
            marker=dict(color=bar_colors, line=dict(color="rgba(160,220,255,.55)", width=1)),
            text=[f"{value:.1f}" if value is not None else "数据不足" for value in chart_values],
            textposition="outside",
            textfont=dict(color="#EAF6FF", size=12),
            cliponaxis=False,
        ))
        bar.update_layout(
            title="五维风险得分",
            yaxis=dict(title="维度得分（0–20）", range=[0, 23], gridcolor="rgba(130,180,220,.18)", zerolinecolor="rgba(130,180,220,.25)"),
            xaxis=dict(tickfont=dict(size=11, color="#D7E9F8"), linecolor="rgba(130,180,220,.3)"),
            bargap=0.34,
        )
        _risk_chart_theme(bar, height=360)
        st.plotly_chart(bar, use_container_width=True)

    with radar_col:
        radar_names = [name for name in dimensions if name in score_map]
        radar_values = [score_map[name] for name in radar_names]
        if len(radar_names) >= 3:
            radar = go.Figure(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_names + [radar_names[0]],
                fill="toself",
                name="风险得分",
                line=dict(color="#23C4F2", width=3),
                fillcolor="rgba(35,196,242,.28)",
                marker=dict(size=7, color="#5DE0FF", line=dict(color="#D7F7FF", width=1)),
            ))
            radar.update_layout(
                title="五维风险分布",
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 20], tickvals=[0, 5, 10, 15, 20],
                                    tickfont=dict(size=10, color="#BBD4E8"), gridcolor="rgba(130,180,220,.22)",
                                    linecolor="rgba(130,180,220,.28)"),
                    angularaxis=dict(tickfont=dict(size=11, color="#DCEBFA"), gridcolor="rgba(130,180,220,.22)",
                                     linecolor="rgba(130,180,220,.28)"),
                ),
            )
            _risk_chart_theme(radar, height=360)
            st.plotly_chart(radar, use_container_width=True)
        else:
            st.markdown(
                "<div style='height:360px;display:flex;align-items:center;justify-content:center;border:1px solid #2674A9;border-radius:16px;background:rgba(28,78,120,.25);color:#BBD4E8;'>有效风险维度少于 3 项，暂不绘制雷达图。</div>",
                unsafe_allow_html=True,
            )

    with st.expander("查看评分明细与规则"):
        display_finguard_table(pd.DataFrame(details))
        st.write("评分方法：五维加权模型，偿债25%、流动性20%、盈利20%、应收账款15%、现金流20%。")
        st.write("各维度原始得分为0—20分，再根据对应权重计算综合得分。")
        st.write("缺失指标不直接按零分处理，综合评分按有效维度权重归一化。")


def draw_risk_radar(details):
    """兼容旧调用的独立雷达图入口；主驾驶舱已将雷达图并入对称图表区。"""
    return None

# ==========================================
# Altman Z-Score 财务预警仪表盘
# ==========================================

def draw_z_score_dashboard(latest, industry, listing_status, result=None):
    """根据企业行业和上市状态展示匹配的 Altman 模型结果。"""
    st.markdown("---")
    st.subheader("📊 Altman Z-Score 企业财务预警")

    if result is None:
        result = calculate_z_score(latest, industry=industry, listing_status=listing_status)

    z_score = result.get("Z-Score")
    risk_level = result.get("风险等级", "待确认")
    description = result.get("风险说明", "")
    model_name = result.get("模型名称")
    thresholds = result.get("风险阈值") or {}

    st.markdown(
        f"<div style='padding:12px 16px;margin:8px 0 14px;border:1px solid #2875a8;"
        f"border-radius:12px;background:rgba(13,55,91,.48);color:#d9efff;'>"
        f"<b>行业：</b>{industry}　 <b>上市状态：</b>{listing_status}　 "
        f"<b>匹配模型：</b>{model_name or '暂未匹配'}</div>",
        unsafe_allow_html=True,
    )

    if z_score is None:
        if risk_level == "模型不适用":
            st.warning(description)
        elif risk_level in ("缺少行业信息", "待确认"):
            st.info(description)
        else:
            st.warning(f"Z-Score 暂时无法计算：{description}")
        if result.get("适用性说明"):
            st.caption(result["适用性说明"])
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Altman Z-Score", f"{z_score:.4f}")
    with col2:
        st.metric("Z-Score 风险等级", risk_level)

    if risk_level == "高风险":
        st.error(description)
    elif risk_level == "灰色区域":
        st.warning(description)
    else:
        st.success(description)

    low_cut = float(thresholds.get("高风险上限", 1.81))
    high_cut = float(thresholds.get("低风险起点", 2.99))
    st.info(
        f"当前模型参考区间：Z < {low_cut:g} 为高风险；"
        f"{low_cut:g} ≤ Z < {high_cut:g} 为灰色区域；"
        f"Z ≥ {high_cut:g} 为相对低风险。"
    )

    fig = go.Figure()
    bar_color = {"高风险": "#ef4444", "灰色区域": "#f59e0b", "低风险": "#22c55e"}
    x_max = max(float(z_score) * 1.15, high_cut * 1.25, 3.5)
    fig.add_trace(go.Bar(
        x=[z_score], y=["Z-Score"], orientation="h",
        marker_color=bar_color.get(risk_level, "#35C5F0"),
        text=[f"{z_score:.4f}"], textposition="auto",
    ))
    fig.add_vline(
        x=low_cut, line_dash="dash", line_color="#ef4444",
        annotation_text=f"{low_cut:g}",
    )
    fig.add_vline(
        x=high_cut, line_dash="dash", line_color="#22c55e",
        annotation_text=f"{high_cut:g}",
    )
    fig.update_layout(
        title=f"{model_name} 风险区间",
        xaxis_title="Z-Score", yaxis_title="",
        xaxis=dict(range=[0, x_max]),
        height=300, showlegend=False,
        margin=dict(l=35, r=30, t=58, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,39,66,.32)",
        font=dict(color="#DCEBFA", family="Arial, Microsoft YaHei, sans-serif"),
    )
    apply_finguard_chart_theme(fig, height=310)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 模型核心指标")
    detail_df = pd.DataFrame(
        list(result.get("指标明细", {}).items()),
        columns=["Z-Score 指标名称", "计算结果"],
    )
    if not detail_df.empty:
        st.caption("指标数值保留 4 位小数。")
        display_finguard_table(detail_df)

    st.caption(f"模型公式：{result.get('模型公式', '—')}")
    st.caption(f"适用性说明：{result.get('适用性说明', '—')}")
    st.caption(
        "风险提示：Altman Z-Score 为财务困境筛查工具，不等同于信用评级或违约概率；"
        "行业差异、会计口径及数据质量均可能影响结果。"
    )


def draw_risk_trend(company_data):
    st.subheader("五、多年度财务风险趋势分析")
    trend_data = company_data.copy().sort_values("年份")
    if len(trend_data) < 2:
        st.warning("至少需要两个年度的财务数据，才能进行风险趋势分析。")
        return
    trend_scores = []
    # 保存五个风险维度的年度评分
    dimension_trends = []

    for _, row in trend_data.iterrows():
        year_risks = normalize_risks(
            detect_risks(row)
        )
        year_risks = check_cashflow_risk(
            row, year_risks
        )
        year_score, year_level, _, year_details = calculate_risk_score(
            row, year_risks
        )
        trend_scores.append({
            "年份": int(row["年份"]),
            "综合风险评分": year_score,
            "风险等级": year_level
        })
        # 每个年度都保存自己的五维评分，不能放在年度循环之外
        for item in year_details:
            dimension_trends.append({
                "年份": int(row["年份"]),
                "风险维度": item["风险维度"],
                "风险得分": item["风险得分"]
            })
    score_df = pd.DataFrame(trend_scores)
    score_df = score_df[["年份", "综合风险评分", "风险等级"]]
    st.markdown("#### 📋 多年度风险评分汇总")
    st.caption("按年份排列，综合风险评分保留 2 位小数；不显示无意义的行索引。")
    display_finguard_table(score_df)

    # =====================================
    # 绘制多年度风险趋势折线图
    # =====================================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=score_df["年份"].tolist(),
            y=score_df["综合风险评分"].tolist(),

            mode="lines+markers+text",

            name="综合风险评分",

            text=[
                f"{x:.2f}"
                for x in score_df["综合风险评分"]
            ],

            textposition="top center",

            line=dict(
                width=4,
                color="#35C5F0"
            ),

            marker=dict(
                size=11,
                color="#35C5F0",
                line=dict(width=2, color="#EAF6FF")
            )
        )
    )

    fig.update_layout(
        title="企业多年度综合财务风险变化趋势",

        xaxis_title="年份",

        yaxis_title="综合风险评分",

        yaxis=dict(
            range=[0, 100],
            dtick=20
        ),

        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),

        height=410,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,39,66,.32)",
        font=dict(color="#DCEBFA", family="Arial, Microsoft YaHei, sans-serif"),
        showlegend=True,
        margin=dict(l=50, r=30, t=65, b=48),
    )
    apply_finguard_chart_theme(fig, height=410)

    st.plotly_chart(fig, use_container_width=True)

    # 风险变化判断
    first_score = score_df.iloc[0]["综合风险评分"]
    latest_score = score_df.iloc[-1]["综合风险评分"]
    change = round(latest_score - first_score, 2)
    st.markdown("### 风险趋势诊断")
    if change > 5:
        st.error(
            f"风险恶化：综合风险评分上升了{change}分，"
            "建议重点关注企业财务状况。"
        )
    elif change < -5:
        st.success(
            f"风险改善：综合风险评分下降了{abs(change)}分，"
            "企业财务风险有所缓解。"
        )
    else:
        st.info(
            f"风险基本稳定：综合风险评分变化为{change}分。"
        )
# =====================================
# 11. 主程序
# =====================================

def main():
    # 左侧功能导航始终先渲染，避免未上传数据或数据校验失败时工具栏消失。
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🛡️</div>
            <div>
                <div class="sidebar-brand-name">FinGuard</div>
                <div class="sidebar-brand-sub">FINANCIAL INTELLIGENCE</div>
            </div>
        </div>
        <div class="sidebar-nav-heading">功能导航 / WORKSPACE</div>
        <div class="sidebar-nav-caption">选择模块，查看企业经营、财务表现与风险情况</div>
        """, unsafe_allow_html=True)
        # 使用原生按钮替代 radio：避免不同 Streamlit 版本的 radio 内部 DOM 样式覆盖文字颜色。
        nav_items = [
            ("总览驾驶舱", "📊  总览驾驶舱  ·  全局概览"),
            ("财务分析", "📈  财务分析  ·  指标与趋势"),
            ("风险诊断", "🛡️  风险诊断  ·  预警与评分"),
            ("AI 智能中心", "🤖  AI 智能中心  ·  智能分析"),
        ]
        if "finguard_active_page" not in st.session_state:
            st.session_state["finguard_active_page"] = "总览驾驶舱"
        for idx, (page_name, button_text) in enumerate(nav_items):
            if st.button(button_text, key=f"fg_nav_{idx}"):
                st.session_state["finguard_active_page"] = page_name
        active_page = st.session_state["finguard_active_page"]
        st.markdown(
            f"<div class='sidebar-nav-active'>● 当前模块：{active_page}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.caption("FinGuard · 企业财务风险智能诊断")
        st.caption("风险分析仅供研究与辅助决策参考")

    # 页面导览与数据接入
    st.markdown("## 📂 数据接入与分析配置")
    st.caption("上传企业年度财务数据，FinGuard 将自动完成数据校验、指标计算与智能风险诊断。")
    uploaded_file = st.file_uploader("上传企业财务数据（Excel / CSV）", type=["xlsx", "csv"], help="请确保文件包含页面下方模板所列字段。")

    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
          <div style="font-size:42px">📁</div>
          <h3>等待接入企业财务数据</h3>
          <p>上传 Excel 或 CSV 文件，即可开启财务分析、风险评估与 AI 诊断。</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("查看数据模板字段要求"):
            st.write("必需字段：" + "、".join(REQUIRED_COLUMNS))
            st.caption("可选字段：所属行业、上市状态、股票市值。若提供“所属行业”和“上市状态”，系统会优先读取；缺失时可在页面中手动确认。")
            st.caption("年份请填写整数；财务数值字段请填写有效数字，不要混入单位或备注。")
        return

    try:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))

        df = clean_financial_data(df)
        result = validate_indicators(calculate_indicators(df))
        company_list = result["公司名称"].dropna().unique().tolist()
        if not company_list:
            raise ValueError("没有可供选择的公司。")

        st.success(f"数据接入成功：{len(company_list)} 家企业，{len(result)} 条年度记录。")
        selector_col, info_col = st.columns([2, 3])
        with selector_col:
            company = st.selectbox("选择分析企业", company_list, key="finguard_company_select")
        with info_col:
            st.markdown("<div class='context-chip'>🛡️ 当前工作区：企业财务风险诊断</div>", unsafe_allow_html=True)

        company_data = result[result["公司名称"] == company].copy().sort_values("年份")
        latest = company_data.iloc[-1]

        # 企业行业与上市状态：优先读取上传字段，缺失时由用户手动确认。
        industry_options = [
            "请选择行业", "制造业", "信息技术/软件服务业", "批发零售业",
            "房地产", "建筑业", "交通运输业", "住宿餐饮业",
            "专业服务业", "公用事业", "金融业（银行/证券/保险）", "其他非制造业",
        ]
        detected_industry = identify_company_industry(latest)
        raw_industry = str(latest.get("所属行业", "")).strip()
        if detected_industry:
            if detected_industry not in industry_options:
                industry_options.append(detected_industry)
            industry_default = detected_industry
            industry_source = f"已从上传数据识别并标准化：{raw_industry or detected_industry}"
        else:
            industry_default = "请选择行业"
            industry_source = "上传数据未提供可确认的行业信息；请人工选择，不会根据公司名称猜测"

        raw_listing = str(latest.get("上市状态", "")).strip()
        listing_aliases = {
            "是": "上市", "已上市": "上市", "上市公司": "上市",
            "public": "上市", "listed": "上市",
            "否": "非上市", "未上市": "非上市", "非上市公司": "非上市",
            "private": "非上市", "unlisted": "非上市",
            "1": "上市", "0": "非上市", "true": "上市", "false": "非上市",
        }
        normalized_listing = listing_aliases.get(raw_listing.lower(), raw_listing)
        if normalized_listing not in ("上市", "非上市"):
            normalized_listing = "请选择上市状态"
            listing_source = "上传数据未提供，请手动确认"
        else:
            listing_source = "已从上传数据读取"

        with st.expander("🏢 企业行业与模型配置（用于匹配 Z-Score 模型）", expanded=True):
            industry_col, listing_col = st.columns(2)
            with industry_col:
                selected_industry = st.selectbox(
                    "企业所属行业", industry_options,
                    index=industry_options.index(industry_default),
                    key=f"finguard_industry_{hashlib.sha256(file_bytes).hexdigest()[:8]}_{company}",
                )
                st.caption(industry_source)
            with listing_col:
                listing_options = ["请选择上市状态", "上市", "非上市"]
                selected_listing_status = st.selectbox(
                    "企业上市状态", listing_options,
                    index=listing_options.index(normalized_listing),
                    key=f"finguard_listing_{hashlib.sha256(file_bytes).hexdigest()[:8]}_{company}",
                )
                st.caption(listing_source)
            st.caption(
                "系统将根据行业和上市状态匹配模型。金融机构不使用普通企业 Altman 模型；"
                "未确认行业或上市状态时，不会擅自给出 Z-Score 风险结论。"
            )

        if selected_industry == "请选择行业":
            zscore_result = {
                "Z-Score": None, "风险等级": "缺少行业信息",
                "风险说明": "请先确认企业所属行业，再计算适用的 Z-Score 模型。",
                "模型名称": None, "模型公式": None, "风险阈值": None,
                "适用性说明": "行业信息未确认，系统不会默认使用制造业模型。",
                "指标明细": {},
            }
        elif (
            "制造" in selected_industry
            and "非制造" not in selected_industry
            and selected_listing_status == "请选择上市状态"
        ):
            zscore_result = {
                "Z-Score": None, "风险等级": "待确认",
                "风险说明": "制造业企业需要确认上市状态，才能匹配上市制造业 Z 或非上市制造业 Z' 模型。",
                "模型名称": None, "模型公式": None, "风险阈值": None,
                "适用性说明": "请确认企业是否上市。",
                "指标明细": {},
            }
        else:
            zscore_result = calculate_z_score(
                latest, industry=selected_industry,
                listing_status=selected_listing_status,
            )

        risks = normalize_risks(detect_risks(latest))
        risks = check_cashflow_risk(latest, risks)
        score, level, color, details = calculate_risk_score(latest, risks)

        st.markdown(f"<div class='company-banner'><div><span>正在分析企业</span><h2>{company}</h2></div><div class='banner-meta'><span>最新报告年度</span><strong>{int(latest['年份'])}</strong></div><div class='banner-meta'><span>数据覆盖年度</span><strong>{len(company_data)} 年</strong></div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 14px;padding:11px 15px;border:1px solid #28577d;border-radius:12px;background:rgba(9,29,49,.72);">
          <span style="color:#62d5b5;font-weight:800;font-size:12px;">● 数据分析已就绪</span>
          <span style="color:#8eabc5;font-size:12px;">当前企业数据已完成字段校验与指标计算</span>
          <span style="margin-left:auto;color:#8eabc5;font-size:12px;">分析口径：最新年度 · 五维加权模型</span>
        </div>
        """, unsafe_allow_html=True)

        # 数据载入后补充当前企业与报告年度信息；功能导航已在 main() 开始处渲染。
        with st.sidebar:
            st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
            st.markdown("**当前分析企业**")
            st.markdown(f"<div class='sidebar-company'>{company}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='sidebar-status'>● 数据已就绪<br><span>报告年度：{int(latest['年份'])}</span></div>",
                unsafe_allow_html=True,
            )

        if active_page == "总览驾驶舱":
            st.markdown("### 企业风险总览")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("综合风险评分", f"{float(score):.1f} / 100")
            m2.metric("风险等级", level)
            m3.metric("触发风险规则", f"{len(risks)} 项")
            m4.metric("分析年度", f"{int(latest['年份'])} 年")
            draw_risk_dashboard(score, level, color, details)
            draw_risk_insight(details)
            st.caption("说明：风险评分基于 FinGuard 预设规则模型，仅供辅助分析与竞赛演示，不构成投资建议。")

        elif active_page == "财务分析":
            st.markdown("### 财务指标与经营趋势")
            st.caption("查看企业年度财务指标，并观察关键指标随时间的变化。")
            with st.expander("查看年度财务指标明细", expanded=True):
                display_finguard_table(company_data)
            draw_financial_chart(company_data)
            with st.expander("查看原始上传数据"):
                display_finguard_table(df[df["公司名称"] == company])

        elif active_page == "风险诊断":
            st.markdown("### 风险识别与预警结果")
            show_risks(risks)
            draw_z_score_dashboard(latest, selected_industry, selected_listing_status, zscore_result)
            draw_risk_trend(company_data)

        elif active_page == "AI 智能中心":
            st.markdown("### DeepSeek 智能财务诊断")
            st.caption("结合企业财务指标与规则识别结果，生成结构化风险诊断报告。")
            st.info("如果点击后提示 API Key、鉴权或网络错误，请检查项目根目录的 .env 配置及 ai_analysis.py 中的模型调用设置。")
            financial_data = latest.to_dict()
            financial_data["综合风险评分"] = score
            financial_data["综合风险等级"] = level
            financial_data["评分方法"] = "五类风险各20分，触发即计分，总分0至100；演示模型，非信用评级"
            financial_data["所属行业"] = selected_industry
            financial_data["上市状态"] = selected_listing_status
            financial_data["Z-Score模型结果"] = zscore_result

            report_key = "{}_{}_{}_{}_{}".format(hashlib.sha256(file_bytes).hexdigest(), company, int(latest["年份"]), selected_industry, selected_listing_status)
            if st.session_state.get("finguard_report_key") != report_key:
                st.session_state["finguard_report_key"] = report_key
                st.session_state["finguard_ai_report"] = None

            if st.button("✨ 生成 AI 财务风险诊断报告", key="finguard_ai_button"):
                try:
                    with st.spinner("DeepSeek 正在分析企业财务数据，请稍候..."):
                        ai_report = analyze_financial_risk(company_name=company, financial_data=financial_data, risk_results=risks)
                    if not isinstance(ai_report, str):
                        ai_report = str(ai_report)
                    if not ai_report.strip():
                        raise ValueError("AI 返回了空报告，请稍后重试。")
                    st.session_state["finguard_ai_report"] = ai_report
                    st.success("AI 财务风险分析完成！")
                except Exception as ai_error:
                    st.error(f"AI 分析失败：{ai_error}")
                    with st.expander("查看 AI 错误详情"):
                        st.exception(ai_error)

            report = st.session_state.get("finguard_ai_report")
            if report:
                st.markdown("#### 📋 AI 财务风险诊断报告")
                with st.container():
                    st.markdown(report)
                st.download_button("📥 下载 AI 诊断报告", data=report, file_name=f"{company}_AI财务风险诊断报告.md", mime="text/markdown")
                st.caption("免责声明：本报告由人工智能辅助生成，仅供研究与分析参考，不构成投资建议。")

            st.markdown("---")
            st.markdown("### 🧠 FinGuard 多工具协同 Agent")
            st.caption("Agent 可根据问题自主调用财务指标查询、风险识别及综合评分工具。")
            agent_year = int(company_data["年份"].max())
            user_question = st.text_area(
                "向 FinGuard Agent 提问",
                value=f"请全面分析{company}在{agent_year}年的财务风险，包括关键指标、风险识别、综合评分和管理建议。",
                height=105,
                key="finguard_agent_question"
            )
            if st.button("🚀 启动 Agent 分析", key="finguard_agent_button"):
                try:
                    with st.spinner("FinGuard Agent 正在规划任务并调用分析工具..."):
                        agent_financial_data = result.copy()
                    selected_rows = agent_financial_data["公司名称"].astype(str) == str(company)
                    agent_financial_data.loc[selected_rows, "所属行业"] = selected_industry
                    agent_financial_data.loc[selected_rows, "上市状态"] = selected_listing_status
                    agent_result = run_multi_agent(company_name=company, year=agent_year, financial_data=agent_financial_data, user_question=user_question)
                    if agent_result.get("执行状态") != "完成":
                        st.warning("Agent 尚未完成分析，请检查工具调用记录。")
                    else:
                        st.success("Agent 分析完成！")
                    score_result = agent_result.get("风险评分结果")
                    if score_result:
                        a, b = st.columns(2)
                        a.metric("Agent 综合风险评分", f"{score_result['综合风险评分']} / 100")
                        b.metric("Agent 风险等级", score_result["风险等级"])
                    st.markdown("#### Agent 财务风险诊断报告")
                    st.markdown(agent_result.get("分析报告", "暂无报告内容。"))
                    with st.expander("🔍 查看 Agent 工具调用轨迹"):
                        for index, record in enumerate(agent_result.get("工具调用记录", []), start=1):
                            st.markdown(f"**第 {index} 次调用：{record.get('工具名称', '未知工具')}**")
                            st.json({"工具参数": record.get("工具参数"), "执行结果": record.get("执行结果")})
                    agent_report = agent_result.get("分析报告", "")
                    if agent_report:
                        st.download_button("📥 下载 Agent 分析报告", data=agent_report, file_name=f"{company}_{agent_year}_Agent风险报告.md", mime="text/markdown")
                except Exception as e:
                    st.error(f"Agent 运行失败：{e}")
                    with st.expander("查看详细错误"):
                        st.exception(e)

    except Exception as e:
        st.error(f"系统运行错误：{e}")
        with st.expander("查看详细错误信息"):
            st.exception(e)

# =====================================
# FinGuard 展示页脚
# =====================================
st.markdown("""
<div style="margin-top:34px;padding:16px 4px 6px;border-top:1px solid #24496b;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#7898b5;font-size:11px;">
  <span>© FinGuard · Financial Risk Intelligence</span>
  <span>风险分析仅供研究与辅助决策参考，不构成投资、信贷或审计意见。</span>
</div>
""", unsafe_allow_html=True)

# =====================================
# 12. 程序入口
# =====================================

if __name__ == "__main__":
    main()