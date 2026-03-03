import pandas as pd
import streamlit as st
import re

# 页面配置
st.set_page_config(page_title="GPU 硬件架构与显存决策看板", layout="wide")


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


try:
    df = load_data("shs.csv")
except FileNotFoundError:
    st.error("未找到 shs.csv 文件，请确保数据文件与脚本在同一目录下。")
    st.stop()

latest_date = df["date"].max()

st.title(f"🛠️ GPU 硬件架构与显存决策看板 ({latest_date.strftime('%Y-%m')})")
st.markdown("基于 Steam 硬件调查的宏观数据提取。用于辅助决定渲染管线技术栈、最低配置基准以及优化侧重点。")
st.divider()

# ==========================================
# 数据清洗与预处理
# ==========================================
# 1. 显卡型号数据 (仅限 NVIDIA)
df_gpu_all = df[(df["date"] == latest_date) & (df["category"].isin(["Video Card Description", "DirectX 12 GPUs", "DirectX 11 GPUs"]))].copy()
df_nv = df_gpu_all[df_gpu_all["name"].str.contains("NVIDIA|GeForce", case=False, na=False)].copy()

# 2. 计算 NVIDIA 显卡的总大盘占比作为分母
total_nv_share = df_nv["percentage"].sum()


# 3. 标记代际与性能
def classify_nv_gpu(gpu_name):
    gpu_name = str(gpu_name).upper()

    # 指标1：20系及以上 (Turing 及以上架构，这里主要统计 RTX 系列以确保包含 RT/Tensor/Mesh Shader 完整特性)
    # 注：若需包含阉割版 Turing 的 GTX 16 系，可在此列表加入 'GTX 16'
    is_20_plus = any(series in gpu_name for series in ["RTX 20", "RTX 30", "RTX 40", "RTX 50"])

    # 指标2：性能 >= 2080 / 3060
    high_perf_keywords = ["2080", "3060", "3070", "3080", "3090", "4060", "4070", "4080", "4090", "5060", "5070", "5080", "5090"]
    # 排除一些可能造成误判的低端型号，比如 3050 等 (如果只用关键词的话，目前关键词没有冲突，可以安全匹配)
    is_high_perf = any(kw in gpu_name for kw in high_perf_keywords)

    return pd.Series([is_20_plus, is_high_perf])


df_nv[["is_20_plus", "is_high_perf"]] = df_nv["name"].apply(classify_nv_gpu)

# 计算指标 1 和 2 的比例
share_20_plus = df_nv[df_nv["is_20_plus"]]["percentage"].sum()
share_high_perf = df_nv[df_nv["is_high_perf"]]["percentage"].sum()

ratio_20_plus_in_nv = share_20_plus / total_nv_share if total_nv_share > 0 else 0
ratio_high_perf_in_nv = share_high_perf / total_nv_share if total_nv_share > 0 else 0


# 4. 显存数据清洗 (全平台)
df_vram = df[(df["date"] == latest_date) & (df["category"].isin(["VRAM", "Video Card RAM"]))].copy()


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


df_vram["vram_group"] = df_vram["name"].apply(classify_vram_group)
vram_summary = df_vram[df_vram["vram_group"] != "未知"].groupby("vram_group")["percentage"].sum().reset_index()

# 为了保证图表顺序正确
vram_order = ["< 4 GB", "4 GB", "6 GB", "8 GB", "> 8 GB"]
vram_summary["vram_group"] = pd.Categorical(vram_summary["vram_group"], categories=vram_order, ordered=True)
vram_summary = vram_summary.sort_values("vram_group")
# 归一化百分比（避免 Steam 原数据加起来不是 100% 的误差）
vram_total = vram_summary["percentage"].sum()
vram_summary["normalized_percentage"] = (vram_summary["percentage"] / vram_total * 100).round(2)


# ==========================================
# 页面布局：三大核心问题解答
# ==========================================

st.header("📌 核心决策指标总览")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("问题一：20系及以上占比 (N卡内)", f"{ratio_20_plus_in_nv:.2%}")
with col2:
    st.metric("问题二：≥ 3060/2080 占比 (N卡内)", f"{ratio_high_perf_in_nv:.2%}")
with col3:
    # 找出显存占比最大的区间
    top_vram = vram_summary.iloc[vram_summary["normalized_percentage"].argmax()]
    st.metric(f"问题三：主流显存容量 (全平台)", f"{top_vram['vram_group']} ({top_vram['normalized_percentage']}%)")

st.divider()

# --- 详情模块 1 ---
st.subheader("1. 架构代际：Turing 及以上架构占比")
st.markdown(
    f"""
**🎯 决策影响 (技术栈与架构)：**
在所有 NVIDIA 用户中，有 **{ratio_20_plus_in_nv:.2%}** 的设备属于 20 系及以上的现代架构。
*如果该比例超过你的业务阈值（通常为 60%-70%），则可以激进地将 Mesh Shader、Hardware Ray Tracing (DXR) 等现代 API 特性作为底层架构的强依赖，而无须为其编写高昂的 Fallback 兼容代码。*
"""
)
st.progress(ratio_20_plus_in_nv)

with st.expander("查看 N 卡架构明细数据"):
    df_nv_20 = df_nv[df_nv["is_20_plus"]].sort_values(by="percentage", ascending=False)
    st.dataframe(df_nv_20[["name", "percentage"]].style.format({"percentage": "{:.2%}"}))

# --- 详情模块 2 ---
st.subheader("2. 性能标杆：大于等于 RTX 2080 / 3060 的占比")
st.markdown(
    f"""
**🎯 决策影响 (最低配置基准)：**
在所有 NVIDIA 用户中，有 **{ratio_high_perf_in_nv:.2%}** 的设备跨越了 2080 / 3060 这一性能门槛。
*这代表着有多少玩家能以“主机级别 (PS5 性能对标)”的体验运行你们的项目。如果以此作为最低配置，你需要接受会流失掉剩余那部分使用如 1060, 1650, 2060 等老旧型号的下沉市场玩家。*
"""
)
st.progress(ratio_high_perf_in_nv)

with st.expander("查看达标 GPU 列表明细"):
    df_nv_high = df_nv[df_nv["is_high_perf"]].sort_values(by="percentage", ascending=False)
    st.dataframe(df_nv_high[["name", "percentage"]].style.format({"percentage": "{:.2%}"}))

# --- 详情模块 3 ---
st.subheader("3. 显存分布：全平台 VRAM 容量概览")
st.markdown(
    """
**🎯 决策影响 (优化侧重点 - GPU 性能 vs 显存)：**
如果 `< 4GB` 和 `4GB` 占比较小，而 `8GB` 及以上成为绝对主流，那么在优化策略上，可以**适度放宽显存预算**（如使用更高分辨率的贴图、更激进的预加载机制），将精力更多地集中在着色器（Shader）复杂度和 Draw Call 的 GPU 性能优化上。反之则需要死磕显存压缩。
"""
)

col_chart, col_data = st.columns([2, 1])
with col_chart:
    st.bar_chart(vram_summary.set_index("vram_group")["normalized_percentage"])
with col_data:
    st.dataframe(vram_summary[["vram_group", "normalized_percentage"]].rename(columns={"vram_group": "显存容量", "normalized_percentage": "全平台占比 (%)"}), hide_index=True)
