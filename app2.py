import pandas as pd
import streamlit as st
import re

# 页面配置
st.set_page_config(page_title="GPU 硬件架构与显存决策看板", layout="wide")


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    # 新增 year_month 字段方便过滤历史数据
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    return df


try:
    df = load_data("shs.csv")
except FileNotFoundError:
    st.error("未找到 shs.csv 文件，请确保数据文件与脚本在同一目录下。")
    st.stop()

latest_date = df["date"].max()


# ==========================================
# 核心分类算法 (提取为全局函数以便复用)
# ==========================================
def classify_nv_gpu(gpu_name):
    gpu_name = str(gpu_name).upper()
    # 指标1：20系及以上
    is_20_plus = any(series in gpu_name for series in ["RTX 20", "RTX 30", "RTX 40", "RTX 50"])

    # 指标2：性能及格线
    high_perf_keywords = ["2080", "3060", "3070", "3080", "3090", "4060", "4070", "4080", "4090", "5060", "5070", "5080", "5090"]
    is_high_perf = any(kw in gpu_name for kw in high_perf_keywords)

    # 指标3：40系及以上 (独占帧生成, SER, 神经网络渲染等特性)
    is_40_plus = any(series in gpu_name for series in ["RTX 40", "RTX 50"])

    return is_20_plus, is_high_perf, is_40_plus


def classify_vram_group(vram_name):
    vram_str = str(vram_name).upper()
    match = re.search(r"(\d+)\s*(MB|GB)", vram_str)
    if not match:
        return "未知"

    amount = int(match.group(1))
    unit = match.group(2)
    mb = amount if unit == "MB" else amount * 1024

    if mb < 3900:
        return "< 4 GB"
    elif 3900 <= mb <= 5000:
        return "4 GB"
    elif 5000 < mb <= 7000:
        return "6 GB"
    elif 7000 < mb <= 8500:
        return "8 GB"
    elif mb > 8500:
        return "> 8 GB"
    else:
        return "未知"


# ==========================================
# 提取指定时间序列数据的统计算法
# ==========================================
def get_trend_data(df_source, target_months):
    records = []
    for ym in target_months:
        df_m = df_source[df_source["year_month"] == ym]
        if df_m.empty:
            continue

        # 1. 计算 NVIDIA 架构与性能指标
        df_gpu = df_m[df_m["category"].isin(["Video Card Description", "DirectX 12 GPUs", "DirectX 11 GPUs"])].copy()
        df_nv = df_gpu[df_gpu["name"].str.contains("NVIDIA|GeForce", case=False, na=False)].copy()

        if not df_nv.empty:
            total_nv = df_nv["percentage"].sum()
            res = df_nv["name"].apply(classify_nv_gpu)
            df_nv["is_20_plus"] = res.apply(lambda x: x[0])
            df_nv["is_high_perf"] = res.apply(lambda x: x[1])
            df_nv["is_40_plus"] = res.apply(lambda x: x[2])

            ratio_20 = df_nv[df_nv["is_20_plus"]]["percentage"].sum() / total_nv if total_nv > 0 else 0
            ratio_high = df_nv[df_nv["is_high_perf"]]["percentage"].sum() / total_nv if total_nv > 0 else 0
            ratio_40 = df_nv[df_nv["is_40_plus"]]["percentage"].sum() / total_nv if total_nv > 0 else 0
        else:
            ratio_20, ratio_high, ratio_40 = 0, 0, 0

        # 2. 计算显存分布指标 (全平台)
        df_vram = df_m[df_m["category"].isin(["VRAM", "Video Card RAM"])].copy()
        df_vram["vram_group"] = df_vram["name"].apply(classify_vram_group)
        vram_sum = df_vram[df_vram["vram_group"] != "未知"].groupby("vram_group")["percentage"].sum()
        total_vram = vram_sum.sum()

        record = {
            "月份": ym,
            "Turing 及以上架构占比 (N卡)": ratio_20,
            "≥ 2080/3060 性能占比 (N卡)": ratio_high,
            "Ada/Blackwell (40系+) 占比 (N卡)": ratio_40,
        }
        if total_vram > 0:
            for vg in ["< 4 GB", "4 GB", "6 GB", "8 GB", "> 8 GB"]:
                record[vg] = vram_sum.get(vg, 0) / total_vram
        records.append(record)

    return pd.DataFrame(records)


# ==========================================
# 页面渲染：全局与最新月概览
# ==========================================
st.title(f"🛠️ GPU 硬件架构与显存决策看板 ({latest_date.strftime('%Y-%m')})")
st.markdown("基于 Steam 硬件调查的宏观数据提取。用于辅助决定渲染管线技术栈（如 Mesh Shader、帧生成、SER）、最低配置基准以及优化侧重点。")
st.divider()

# 获取当前最新月的数据用于顶部展示
current_month_data = get_trend_data(df, [latest_date.strftime("%Y-%m")])

if not current_month_data.empty:
    curr = current_month_data.iloc[0]
    st.header("📌 核心决策指标总览 (最新月)")
    # 改为 4 列布局
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("问题一：20系及以上占比 (N卡内)", f"{curr['Turing 及以上架构占比 (N卡)']:.2%}")
    col2.metric("问题二：≥ 3060/2080 占比 (N卡内)", f"{curr['≥ 2080/3060 性能占比 (N卡)']:.2%}")
    col3.metric("问题三：40系及以上占比 (N卡内)", f"{curr['Ada/Blackwell (40系+) 占比 (N卡)']:.2%}")

    # 找出当月占比最大的显存
    vram_cols = ["< 4 GB", "4 GB", "6 GB", "8 GB", "> 8 GB"]
    top_vram_name = current_month_data[vram_cols].iloc[0].idxmax()
    top_vram_val = current_month_data[vram_cols].iloc[0].max()
    col4.metric(f"问题四：主流显存容量 (全平台)", f"{top_vram_name} ({top_vram_val:.2%})")

st.divider()

# ==========================================
# 第四部分：历史趋势对比
# ==========================================
st.header("📈 第四部分：历史趋势对比与增长斜率")
st.markdown("观察特性的普及速度。如果趋势线呈现陡峭上升，说明该硬件特性（如帧生成）正在形成绝对主流，我们的技术管线可以随之激进；如果是平缓上升或停滞，则说明下沉市场仍不可忽视。")

# --- 数据集 1：2025 全年趋势 ---
months_2025 = [f"2025-{str(m).zfill(2)}" for m in range(3, 13)] + ["2026-01", "2026-02"]
df_2025 = get_trend_data(df, months_2025)

# --- 数据集 2：2021 - 2026 年度趋势 ---
years_jan = [f"{y}-02" for y in range(2021, 2027)]
df_yearly = get_trend_data(df, years_jan)

tab1, tab2 = st.tabs(["📊 近一年趋势 (2025.03 - 2026.02)", "📅 跨年度对比 (2021.02 - 2026.02)"])


def render_trend_charts(trend_df):
    if trend_df.empty:
        st.warning("所选时间段内暂无数据或数据缺失。")
        return

    trend_df = trend_df.set_index("月份")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. GPU 架构与性能普及走势")
        # 将新指标加入折线图
        st.line_chart(trend_df[["Turing 及以上架构占比 (N卡)", "≥ 2080/3060 性能占比 (N卡)", "Ada/Blackwell (40系+) 占比 (N卡)"]])

    with col_b:
        st.subheader("2. 显存 (VRAM) 分布演进走势")
        # 面积图非常适合看“各个容量占比的此消彼长”
        st.area_chart(trend_df[["< 4 GB", "4 GB", "6 GB", "8 GB", "> 8 GB"]])

    with st.expander("查看原始对比数据 (小数格式，1.0 = 100%)"):
        st.dataframe(trend_df)


with tab1:
    render_trend_charts(df_2025)

with tab2:
    render_trend_charts(df_yearly)
