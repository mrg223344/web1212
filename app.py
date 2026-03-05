# -*- coding: utf-8 -*-
"""
Streamlit 应用：宫腔镜手术治疗CSD疗效预测平台（V1.6）

特点：
- 自动从当前目录加载模型文件 calibrated_rf_model.pkl
- 仅支持单例预测
- 标题居中加大字号，美观；下方小字说明为“宫腔镜手术治疗CSD疗效结局预测”；五个指标在显示时保留两位小数；结果在页面正中以大号百分比 + 进度条形式直观呈现；概率显示为“手术后疗效显著(有效)的预测概率”

运行：
streamlit run app.py

注意：预测仅供参考，不能替代临床决策。
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 页面配置
st.set_page_config(page_title="宫腔镜手术治疗CSD疗效预测平台", page_icon="🩺", layout="centered")

# 标题与介绍（居中、美化）
st.markdown("""
<div style='text-align: center;'>
  <h1 style='color: #1E90FF; font-size:48px; margin-bottom: 0.2rem;'>🩺 宫腔镜手术治疗CSD疗效预测平台</h1>
  <p style='color: gray; font-size:16px; margin-top: 0.1rem;'>基于机器学习模型，提供个体化宫腔镜手术治疗CSD疗效预测结果</p>
</div>
<hr style='border:1px solid #1E90FF;'>
""", unsafe_allow_html=True)

# 固定模型文件名（当前目录）
MODEL_FILENAME = 'calibrated_rf_model.pkl'
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)

# 侧边栏输入区域
st.sidebar.markdown("<h3 style='color:#1E90FF;'>请输入患者特征</h3>", unsafe_allow_html=True)

bmi = st.sidebar.number_input("BMI（体质指数）", min_value=0.0, max_value=100.0, value=22.0, step=0.01, format="%.2f")
rmt = st.sidebar.number_input("RMT（残余肌层厚度，mm）", min_value=0.0, max_value=50.0, value=4.00, step=0.01, format="%.2f")
length = st.sidebar.number_input("Length（憩室长度，mm）", min_value=0.0, max_value=200.0, value=12.00, step=0.01, format="%.2f")
pre_hb = st.sidebar.number_input("Pre_Hb（术前血红蛋白，g/L）", min_value=0.0, max_value=300.0, value=120.00, step=0.01, format="%.2f")
pre_alb = st.sidebar.number_input("Pre_Alb（术前白蛋白，g/L）", min_value=0.0, max_value=100.0, value=40.00, step=0.01, format="%.2f")

predict_button = st.sidebar.button("🔍 立即预测")

# 模型加载函数
@st.cache_data(show_spinner=False)
def load_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到模型文件：{path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

# 预测函数
def predict_single(model, features_dict):
    df = pd.DataFrame([features_dict], columns=['BMI','RMT','Length','Pre_Hb','Pre_Alb']).astype(float)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(df)[:, 1][0]
    else:
        pred = model.predict(df)[0]
        prob = float(pred)
    label = int(prob >= 0.5)
    return label, prob

# 主体内容：如果点击预测则显示输入（两位小数）与居中百分比
if predict_button:
    features = {
        'BMI': bmi,
        'RMT': rmt,
        'Length': length,
        'Pre_Hb': pre_hb,
        'Pre_Alb': pre_alb
    }

    # 显示输入值（保留两位小数）
    st.markdown("<h3 style='color:#1E90FF;'>🔹 患者输入（保留两位小数）</h3>", unsafe_allow_html=True)
    df_inputs = pd.DataFrame.from_dict(features, orient='index', columns=['值'])
    df_inputs['值'] = df_inputs['值'].astype(float).map(lambda x: f"{x:.2f}")
    st.table(df_inputs)

    try:
        label, prob = predict_single(model, features)
        prob_pct = prob * 100
        prob_text = f"{prob_pct:.2f}%"

        # 居中显示大号百分比并配合进度条
        st.markdown("<hr style='border:1px solid #1E90FF;'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; padding: 10px;'>", unsafe_allow_html=True)
        if label == 1:
            st.markdown(f"<h2 style='color:#FF4500; margin:0;'>预测结果：宫腔镜手术治疗CSD有效 </h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#2E8B57; margin:0;'>预测结果：宫腔镜手术治疗CSD有效 </h2>", unsafe_allow_html=True)

        # 大号百分比
        st.markdown(f"<div style='font-size:56px; font-weight:700; color:#333; margin-top:8px;'>{prob_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 进度条（居中）
        st.progress(min(max(prob, 0.0), 1.0))

        # 补充说明
        st.write(f"宫腔镜手术治疗CSD有效的预测概率：{prob:.4f}")
        st.info("⚠️ 说明：预测结果仅供科研与教学参考，不能替代临床判断。若用于临床请做严格外部验证与合规审查。")

    except Exception as e:
        st.error(f"预测失败：{e}")
else:
    st.markdown("<p style='text-align:center;color:gray;'>请在左侧输入患者特征后点击“立即预测”。</p>", unsafe_allow_html=True)

# 页脚
st.sidebar.markdown("<hr style='border:1px solid #1E90FF;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align:center;color:gray;'>开发：DiagnoML 平台 ｜ 版本 V1.6</p>", unsafe_allow_html=True)
