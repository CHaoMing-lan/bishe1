# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt

# # Load the model
# model = joblib.load('rf.pkl')

# # Define feature options
# cp_options = {
#     1: 'Typical angina (1)',
#     2: 'Atypical angina (2)',
#     3: 'Non-anginal pain (3)',
#     4: 'Asymptomatic (4)'
# }

# restecg_options = {
#     0: 'Normal (0)',
#     1: 'ST-T wave abnormality (1)',
#     2: 'Left ventricular hypertrophy (2)'
# }

# slope_options = {
#     1: 'Upsloping (1)',
#     2: 'Flat (2)',
#     3: 'Downsloping (3)'
# }

# thal_options = {
#     1: 'Normal (1)',
#     2: 'Fixed defect (2)',
#     3: 'Reversible defect (3)'
# }

# # Define feature names
# feature_names = [
#     "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol",
#     "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate", "Exercise Induced Angina",
#     "ST Depression", "Slope", "Number of Vessels", "Thal"
# ]

# # Streamlit user interface
# st.title("AuraGluco")

# # age: numerical input
# age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# # sex: categorical selection
# sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# # cp: categorical selection
# cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# # trestbps: numerical input
# trestbps = st.number_input("Resting blood pressure (trestbps):", min_value=50, max_value=200, value=120)

# # chol: numerical input
# chol = st.number_input("Serum cholesterol in mg/dl (chol):", min_value=100, max_value=600, value=200)

# # fbs: categorical selection
# fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# # restecg: categorical selection
# restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# # thalach: numerical input
# thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)

# # exang: categorical selection
# exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# # oldpeak: numerical input
# oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# # slope: categorical selection
# slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# # ca: numerical input
# ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)

# # thal: categorical selection
# thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# # Process inputs and make predictions
# feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
# features = np.array([feature_values])

# if st.button("Predict"):
#     # Predict class and probabilities
#     predicted_class = model.predict(features)[0]
#     predicted_proba = model.predict_proba(features)[0]

#     # Display prediction results
#     st.write(f"**Predicted Class:** {predicted_class}")
#     st.write(f"**Prediction Probabilities:** {predicted_proba}")

#     # Generate advice based on prediction results
#     probability = predicted_proba[predicted_class] * 100

#     if predicted_class == 1:
#         advice = (
#             f"According to our model, you have a high risk of heart disease. "
#             f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
#             "While this is just an estimate, it suggests that you may be at significant risk. "
#             "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
#             "to ensure you receive an accurate diagnosis and necessary treatment."
#         )
#     else:
#         advice = (
#             f"According to our model, you have a low risk of heart disease. "
#             f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
#             "However, maintaining a healthy lifestyle is still very important. "
#             "I recommend regular check-ups to monitor your heart health, "
#             "and to seek medical advice promptly if you experience any symptoms."
#         )

#     st.write(advice)

#     # Calculate SHAP values and display force plot
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

#     shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
#     plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

#     st.image("shap_force_plot.png")

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于Windows
plt.rcParams['axes.unicode_minus'] = False

# --- 页面配置 ---
st.set_page_config(
    page_title="个人消费贷款违约识别",
    page_icon="❤️",
    layout="wide"
)

# --- CSS样式 ---
st.markdown("""
<style>
    .header {
        font-size: 24px;
        color: #FFFFFF;
        background-color: #D32F2F;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
    }
    .divider {
        border-top: 2px solid #D32F2F;
        margin: 10px 0;
    }
    .result-card {
        border-left: 5px solid #D32F2F;
        padding: 15px;
        background-color: #F5F5F5;
        border-radius: 5px;
        margin: 15px 0;
    }
    .high-risk { color: #D32F2F; font-weight: bold; }
    .low-risk { color: #388E3C; font-weight: bold; }
    .feature-importance {
        font-size: 14px;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# --- 标题和说明 ---
st.markdown('<div class="header">❤️ 个人消费违约风险评估系统</div>', unsafe_allow_html=True)
st.markdown("""
<div style="color: #616161; font-size: 14px;">
本系统基于机器学习模型，用于评估个人消费信贷违约风险.
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --- 加载模型 ---
@st.cache_resource
def load_model():
    return joblib.load('rf.pkl')  # 替换为你的模型路径

model = load_model()

# --- 特征选项 ---
credit_history_options = {
    1: '小于1年 (1)',
    2: '1-3年 (2)',
    3: '3-5年 (3)',
    4: '5年以上 (4)'
}

credit_score_options = {
    1: '优秀 (750+) (1)',
    2: '良好 (700-749) (2)',
    3: '一般 (650-699) (3)',
    4: '较差 (600-649) (4)',
    5: '差 (<600) (5)'
}

loan_term_options = {
    1: '短期 (<1年) (1)',
    2: '中期 (1-3年) (2)',
    3: '长期 (>3年) (3)'
}

job_type_options = {
    1: '公务员/事业单位 (1)',
    2: '企业员工 (2)',
    3: '自由职业 (3)',
    4: '其他 (4)'
}

# --- 中文化特征名称 ---
feature_names = [
    "年龄", "性别", "信用历史长度", "月收入(元)", "月负债(元)",
    "有逾期记录", "信用评分等级", "信用卡数量", "有房贷",
    "贷款金额(万元)", "贷款期限", "已有贷款笔数", "职业类型"
]

# --- 输入表单 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("基础信息")
    age = st.slider("年龄 (岁)", 18, 100, 50, 
                   help="根据WHO标准，40岁以上建议定期筛查")
    sex = st.radio("性别", ["女", "男"], index=1, horizontal=True,
                  help="男性冠心病风险约为女性的2倍")
    
    st.subheader("财务指标")
    trestbps = st.slider("月收入 (元)", 80, 200, 120, 
                        help="请输入税后月收入")
    chol = st.number_input("月负债（元）", min_value=100, max_value=600, value=200,
                         help="理想值<200mg/dL")
    fbs = st.checkbox("有逾期记录")

with col2:
    st.subheader("信用历史")
    cp = st.selectbox("信用历史长度", options=list(cp_options.keys()), 
                     format_func=lambda x: cp_options[x],
                     help="信用历史良好")
    restecg = st.selectbox("信用评分等级", options=list(restecg_options.keys()),
                          format_func=lambda x: restecg_options[x])
    thalach = st.slider("信用卡数量", 60, 220, 150,
                       help="数量较多")
    exang = st.checkbox("有房贷")
    oldpeak = st.slider("贷款金额 (万元)", 0.0, 6.0, 1.0, step=0.1,
                       help="运动后相对于静息的变化值")
    slope = st.selectbox("贷款期限", options=list(slope_options.keys()),
                        format_func=lambda x: slope_options[x])
    ca = st.slider("已有贷款笔数", 0, 3, 0)
    thal = st.selectbox("职业类型", options=list(thal_options.keys()),
                       format_func=lambda x: thal_options[x])

# --- 预测逻辑 ---
if st.button("开始信用评估", type="primary", use_container_width=True):
    # 转换输入数据
    sex_num = 1 if sex == "男" else 0
    fbs_num = 1 if fbs else 0
    exang_num = 1 if exang else 0
    
    feature_values = [age, sex_num, cp, trestbps, chol, fbs_num, 
                     restecg, thalach, exang_num, oldpeak, slope, ca, thal]
    features_df = pd.DataFrame([feature_values], columns=feature_names)
    
    with st.spinner("正在分析信用数据....."):
        # 预测结果
        predicted_class = model.predict(features_df)[0]
        predicted_proba = model.predict_proba(features_df)[0]
        probability = predicted_proba[predicted_class] * 100
        
        # --- 结果展示 ---
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        if predicted_class == 1:
            st.markdown(f"""
            <div class="result-card">
                <h3 class="high-risk">🔴 高风险预警 (违约概率: {probability:.1f}%)</h3>
                <p><b>信贷建议：</b></p>
                <ol>
                    <li>建议降低授信额度</li>
                    <li>增加担保措施</li>
                    <li>缩短贷款期限</li>
                    <li>提高利率定价</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h3 class="low-risk">🟢 低风险 (违约概率: {probability:.1f}%)</h3>
                <p><b>健康建议：</b></p>
                <ol>
                    <li>可适当提高授信额度</li>
                    <li>可考虑优惠利率</li>
                    <li>建议定期(每半年)更新信用评估</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # --- SHAP解释 ---
        st.subheader("风险因素贡献度分析")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][0,:],
            features_df.iloc[0,:],
            matplotlib=True,
            show=False,
            text_rotation=15
        )
        ax.set_title("各因素对违约风险的影响", fontsize=14)
        st.pyplot(fig)
        
        # --- 风险因素提示 ---
        top_3_risks = pd.Series(np.abs(shap_values[1][0]), index=feature_names).nlargest(3)
        st.markdown("""
        <div class="feature-importance">
        <b>主要风险因素：</b><br>
        """ + "<br>".join([f"• {name}: {value:.2f}" for name, value in top_3_risks.items()]) + """
        </div>
        """, unsafe_allow_html=True)

# --- 预测合规声明 ---
st.markdown("""
<div style="font-size:12px; color:#757575; margin-top:50px;">
<hr>
<b>免责声明：</b>本系统预测结果基于机器学习模型，仅供参考，不能替代专业信贷评审。实际信贷决策需结合客户面谈、资产证明等综合判断。数据采集符合相关隐私保护法规，
所有计算均在本地完成。
</div>
""", unsafe_allow_html=True)
