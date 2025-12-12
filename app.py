# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------- 1. 基础配置 ----------
st.set_page_config(
    page_title="CSD疗效预测系统",
    page_icon="🩺",
    layout="wide", # 使用宽屏模式，展示更多信息
    initial_sidebar_state="expanded"
)

# 自定义 CSS 优化细节
st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    .stAlert {margin-top: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 2.5rem;}
    </style>
""", unsafe_allow_html=True)

# ---------- 2. 模型加载与工具函数 ----------
@st.cache_resource
def load_model():
    try:
        return joblib.load("lgb_best.pkl")
    except FileNotFoundError:
        st.error("未找到模型文件 `lgb_best.pkl`，请确保文件在同级目录下。")
        # 返回一个伪造模型用于UI调试 (正式使用请删除此逻辑)
        class DummyModel:
            def predict_proba(self, X): return np.array([[0.2, 0.45]]) # 模拟输出
        return DummyModel()

model = load_model()

def plot_gauge(prob):
    """绘制风险仪表盘（修正版：解决标题遮挡问题）"""
    # 颜色逻辑
    if prob < 0.3: color = "#28a745" # Green
    elif prob < 0.7: color = "#ffc107" # Yellow
    else: color = "#dc3545" # Red
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "疗效不佳风险 (Outcome=1)", 
            'font': {'size': 18},
            'align': 'center'
        },
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(40, 167, 69, 0.1)'},
                {'range': [30, 70], 'color': 'rgba(255, 193, 7, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(220, 53, 69, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    
    # --- 关键修改点 ---
    # 1. height: 从 250 改为 300，增加整体高度
    # 2. margin: t (top) 从 30 改为 80，给标题留出足够空间
    # 3. margin: b (bottom) 设为 10，减少底部空白
    fig.update_layout(
        height=300, 
        margin=dict(l=30, r=30, t=80, b=10),
        font={'family': "Arial"} # 确保字体渲染正常
    )
    return fig

# ---------- 3. 侧边栏：参数输入 ----------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/doctor-male--v1.png", width=60) # 示例图标
    st.title("参数配置")
    st.info("请根据术前检查结果录入数据")

    st.markdown("### 🧬 解剖结构")
    length = st.number_input("憩室长度 (cm)", 0.0, 5.0, 0.8, 0.1, help="长轴最大径")
    rmt    = st.number_input("残余肌层厚度 (cm)", 0.0, 5.0, 0.3, 0.01, help="底部到浆膜面最短距离")

    st.divider()

    st.markdown("### 🧪 临床指标")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        pre_hb = st.number_input("Hb (g/L)", 50, 200, 115, 1, help="术前血红蛋白")
        post_wbc = st.number_input("术后 WBC", 1.0, 30.0, 5.5, 0.1, help="×10⁹/L")
    with col_sb2:
        pre_alb = st.number_input("Alb (g/L)", 20, 60, 40, 1, help="术前白蛋白")
        bmi     = st.number_input("BMI", 10.0, 60.0, 23.0, 0.1)

    # 构造输入数据
    input_df = pd.DataFrame([[
        length, rmt, pre_hb, pre_alb, post_wbc, bmi
    ]], columns=['Length', 'RMT', 'Pre_Hb', 'Pre_Alb', 'Post_WBC', 'BMI']).astype("float32")

# ---------- 4. 主界面 ----------
st.title("🔍 宫腔镜手术修复 CSD 疗效预测系统")
st.markdown("基于机器学习模型 (`LightGBM`) 预测手术疗效不佳的概率。")

# 使用 Tabs 分离功能
tab1, tab2 = st.tabs(["👤 单例智能诊断", "📂 批量数据分析"])

# === Tab 1: 单例预测 ===
with tab1:
    col_main, col_chart = st.columns([1, 1.5], gap="large")

    with col_main:
        st.markdown("#### 当前输入概览")
        st.dataframe(input_df.T.style.format("{:.2f}"), use_container_width=True, height=250)
        
        predict_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("模型计算中..."):
            prob = float(model.predict_proba(input_df)[0, 1])
        
        # 结果展示区
        with col_chart:
            st.plotly_chart(plot_gauge(prob), use_container_width=True)

        # 风险解释区（跨栏展示）
        st.divider()
        if prob < 0.3:
            st.success(f"**低风险 (概率: {prob:.1%})**：预后良好的可能性较大。")
        elif prob < 0.7:
            st.warning(f"**中风险 (概率: {prob:.1%})**：处于临界范围，建议结合临床综合判断。")
        else:
            st.error(f"**高风险 (概率: {prob:.1%})**：疗效不佳风险较高，需重点关注。")

# === Tab 2: 批量预测 ===
with tab2:
    st.markdown("#### 📤 上传 CSV 文件")
    st.markdown("文件需包含以下列：`Length`, `RMT`, `Pre_Hb`, `Pre_Alb`, `Post_WBC`, `BMI`")
    
    uploaded = st.file_uploader("拖拽文件到此处", type=["csv"])
    
    if uploaded:
        batch = pd.read_csv(uploaded)
        required_cols = set(input_df.columns)
        miss = required_cols - set(batch.columns)
        
        if miss:
            st.error(f"❌ 文件格式错误，缺少列：{', '.join(miss)}")
        else:
            with st.spinner("正在批量计算..."):
                batch["Pred_Prob"] = model.predict_proba(batch[list(input_df.columns)])[:, 1]
                
                # 统计概览
                st.success(f"✅ 成功处理 {len(batch)} 条数据")
                
                col_b1, col_b2 = st.columns([2, 1])
                with col_b1:
                    fig_hist = px.histogram(
                        batch, x="Pred_Prob", nbins=20, 
                        title="预测概率分布",
                        color_discrete_sequence=['#636EFA'],
                        labels={"Pred_Prob": "风险概率"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_b2:
                    st.markdown("##### 快速统计")
                    st.write(batch["Pred_Prob"].describe())
                    
                    csv = batch.to_csv(index=False).encode('utf-8-sig') # 使用 sig 解决中文乱码
                    st.download_button(
                        label="📥 下载预测结果",
                        data=csv,
                        file_name="CSD_Prediction_Results.csv",
                        mime="text/csv",
                        type="primary"
                    )
                
                with st.expander("查看详细数据"):
                    st.dataframe(batch.style.background_gradient(subset=['Pred_Prob'], cmap="RdYlGn_r"))

# ---------- 底部声明 ----------
st.markdown("---")

st.caption("⚠️ **免责声明**：本工具仅供科研辅助参考，不能替代医生的专业临床诊断。")
