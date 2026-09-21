# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_PATH = "model/final_model.pkl"
THRESHOLD_PATH = "model/best_threshold.pkl"
INPUT_COLUMNS_PATH = "model/input_columns.pkl"

TEXT_COL = "Daily_Journal_Text"


# ============================================================
# 2. WORD LISTS GIỐNG FILE TRAIN
# ============================================================

NEGATIVE_WORDS = [
    "buồn", "áp lực", "mệt", "mệt mỏi", "tuyệt vọng", "cô đơn",
    "lo lắng", "căng thẳng", "chán", "kiệt sức", "sợ", "tệ",
    "stress", "stressed", "tired", "hopeless", "lonely",
    "anxious", "depressed", "exhausted"
]

POSITIVE_WORDS = [
    "vui", "ổn", "bình tĩnh", "có động lực", "hy vọng", "thư giãn",
    "tự tin", "tốt", "hạnh phúc", "khỏe", "happy", "good",
    "calm", "motivated", "hopeful", "relaxed", "confident"
]

NEGATION_WORDS = ["không", "chưa", "chẳng", "chả", "never", "not", "no"]

SLEEP_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
}


# ============================================================
# 3. SENTIMENT FUNCTION
# ============================================================

def calculate_sentiment_score(text):
    text = str(text).lower()
    tokens = text.split()

    pos_count = 0
    neg_count = 0

    for i, token in enumerate(tokens):
        window = tokens[max(0, i - 2): i]
        is_negated = any(w in NEGATION_WORDS for w in window)

        if any(pw in token for pw in POSITIVE_WORDS):
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1

        elif any(nw in token for nw in NEGATIVE_WORDS):
            if is_negated:
                pos_count += 1
            else:
                neg_count += 1

    score = (pos_count - neg_count) / (pos_count + neg_count + 1)
    return round(score, 3)


# ============================================================
# 4. CUSTOM TRANSFORMER GIỐNG TRAIN
# ============================================================

class DepressionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, text_col, sleep_map):
        self.text_col = text_col
        self.sleep_map = sleep_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.text_col in X.columns:
            X["Sentiment_Score_From_Text"] = (
                X[self.text_col]
                .fillna("")
                .astype(str)
                .apply(calculate_sentiment_score)
            )

        if "Sleep Duration" in X.columns:
            X["Sleep_Duration_Numeric"] = (
                X["Sleep Duration"].map(self.sleep_map).fillna(1)
            )

        if "Academic Pressure" in X.columns and "Financial Stress" in X.columns:
            X["Mental_Stress_Score"] = (
                pd.to_numeric(X["Academic Pressure"], errors="coerce") +
                pd.to_numeric(X["Financial Stress"], errors="coerce")
            )

        if "Academic Pressure" in X.columns and "Study Hours" in X.columns:
            X["Academic_Burden_Index"] = (
                pd.to_numeric(X["Academic Pressure"], errors="coerce") *
                pd.to_numeric(X["Study Hours"], errors="coerce")
            )

        return X


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        input_columns = joblib.load(INPUT_COLUMNS_PATH)
        return model, threshold, input_columns

    except FileNotFoundError as e:
        st.error("Không tìm thấy file model. Hãy chạy train.py trước.")
        st.code(str(e))
        st.stop()


model, high_threshold, input_columns = load_artifacts()
moderate_threshold = max(0.30, float(high_threshold) - 0.20)


# ============================================================
# 6. LẤY CỘT NUMERIC / CATEGORICAL TỪ MODEL
# ============================================================

def get_model_columns(model):
    preprocessor = model.named_steps["preprocessor"]

    numeric_cols = []
    categorical_cols = []

    for name, transformer, cols in preprocessor.transformers:
        if name == "num":
            numeric_cols = list(cols)
        elif name == "cat":
            categorical_cols = list(cols)

    return numeric_cols, categorical_cols


numeric_cols_from_model, categorical_cols_from_model = get_model_columns(model)


# ============================================================
# 7. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Youth Mental Health Guardian",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Youth Mental Health Guardian")
st.subheader("AI-based Depression Risk Prediction and Explainable Mental Health Analysis System")
st.markdown("---")


# ============================================================
# 8. INPUT FORM
# ============================================================

left, right = st.columns([1.25, 0.75])

with left:
    st.header("📝 Thông tin khảo sát sinh viên")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 15, 40, 20)
        academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
        study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            [
                "Less than 5 hours",
                "5-6 hours",
                "7-8 hours",
                "More than 8 hours"
            ]
        )

    with col2:
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy"]
        )

        suicidal_thoughts = st.selectbox(
            "Have you ever had suicidal thoughts ?",
            ["No", "Yes"]
        )

        study_hours = st.slider("Study Hours", 0, 12, 5)

        financial_stress = st.slider("Financial Stress", 0, 5, 3)

        family_history = st.selectbox(
            "Family History of Mental Illness",
            ["No", "Yes"]
        )

    daily_text = st.text_area(
        "Daily Journal Text",
        "I feel exhausted and overwhelmed with academic pressure"
    )

with right:
    st.header("📌 Hệ thống")
    st.info(
        """
        Hệ thống gồm:

        - Machine Learning
        - NLP TF-IDF
        - Feature Engineering
        - Threshold Optimization
        - SHAP Explainable AI

        Đầu ra:

        - Risk Level
        - Risk Score
        - Sentiment Score
        - SHAP Top Factors
        - Personalized Recommendation
        """
    )


# ============================================================
# 9. BUILD INPUT DATAFRAME — BẢN SỬA LỖI MEDIAN
# ============================================================

def build_input_dataframe():
    input_dict = {
        "Gender": gender,
        "Age": age,
        "Academic Pressure": academic_pressure,
        "Study Satisfaction": study_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Study Hours": study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
        "Daily_Journal_Text": daily_text
    }

    input_df = pd.DataFrame([input_dict])

    for col in input_columns:
        if col not in input_df.columns:
            if col in numeric_cols_from_model:
                input_df[col] = np.nan
            elif col in categorical_cols_from_model:
                input_df[col] = "Unknown"
            elif col == TEXT_COL:
                input_df[col] = ""
            else:
                input_df[col] = np.nan

    input_df = input_df[input_columns]

    for col in numeric_cols_from_model:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    for col in categorical_cols_from_model:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna("Unknown").astype(str)

    if TEXT_COL in input_df.columns:
        input_df[TEXT_COL] = input_df[TEXT_COL].fillna("").astype(str)

    return input_df


# ============================================================
# 10. RECOMMENDATION
# ============================================================

def generate_recommendations(
    probability,
    academic_pressure,
    study_satisfaction,
    sleep_duration,
    dietary_habits,
    financial_stress,
    suicidal_thoughts,
    sentiment_score
):
    recommendations = []

    if academic_pressure >= 4:
        recommendations.append(
            "📚 Áp lực học tập cao: nên chia nhỏ thời gian học, dùng Pomodoro và tránh học liên tục quá lâu."
        )

    if study_satisfaction <= 2:
        recommendations.append(
            "😊 Mức hài lòng học tập thấp: nên xem lại phương pháp học, mục tiêu học tập và trao đổi với cố vấn học tập."
        )

    if sleep_duration in ["Less than 5 hours", "5-6 hours"]:
        recommendations.append(
            "😴 Giấc ngủ chưa đủ: nên duy trì 7-8 giờ ngủ mỗi ngày và hạn chế dùng màn hình trước khi ngủ."
        )

    if dietary_habits == "Unhealthy":
        recommendations.append(
            "🥗 Thói quen ăn uống chưa tốt: nên ăn đủ bữa, uống đủ nước và hạn chế bỏ bữa."
        )

    if financial_stress >= 4:
        recommendations.append(
            "💰 Áp lực tài chính cao: nên lập kế hoạch chi tiêu hoặc tìm hỗ trợ học bổng, trợ cấp sinh viên."
        )

    if suicidal_thoughts == "Yes":
        recommendations.append(
            "⚠️ Có suy nghĩ tiêu cực/tự tử: nên liên hệ ngay người thân, cố vấn học đường hoặc chuyên gia tâm lý."
        )

    if sentiment_score < 0:
        recommendations.append(
            "💬 Nhật ký cảm xúc có xu hướng tiêu cực: nên theo dõi cảm xúc hằng ngày và chia sẻ với người đáng tin cậy."
        )

    if probability >= high_threshold:
        recommendations.append(
            "🚨 Mức rủi ro cao: nên ưu tiên gặp chuyên viên tâm lý học đường hoặc cơ sở hỗ trợ sức khỏe tinh thần."
        )

    if not recommendations:
        recommendations.append(
            "✅ Các chỉ số hiện tại tương đối ổn định. Nên tiếp tục duy trì giấc ngủ, học tập và sinh hoạt cân bằng."
        )

    return recommendations


# ============================================================
# 11. PREDICT
# ============================================================

st.markdown("---")

if st.button("🔍 Dự đoán nguy cơ trầm cảm"):

    input_data = build_input_dataframe()

    probability = model.predict_proba(input_data)[0][1]
    risk_score = round(probability * 100, 2)
    sentiment_score = calculate_sentiment_score(daily_text)

    st.header("📊 Kết quả dự đoán")

    if probability >= high_threshold:
        risk_level = "High Risk"
        st.error("🚨 NGUY CƠ TRẦM CẢM CAO")
    elif probability >= moderate_threshold:
        risk_level = "Moderate Risk"
        st.warning("⚠️ NGUY CƠ TRẦM CẢM TRUNG BÌNH")
    else:
        risk_level = "Low Risk"
        st.success("✅ NGUY CƠ TRẦM CẢM THẤP")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{risk_score}/100")
    c2.metric("Probability", round(probability, 4))
    c3.metric("High Threshold", round(float(high_threshold), 2))
    c4.metric("Sentiment Score", sentiment_score)

    st.markdown("---")

    # ========================================================
    # 12. SHAP
    # ========================================================

    st.header("🧠 SHAP Explainable AI")

    try:
        feature_engineer = model.named_steps["feature_engineer"]
        preprocessor = model.named_steps["preprocessor"]
        ml_model = model.named_steps["model"]

        input_fe = feature_engineer.transform(input_data)
        input_processed = preprocessor.transform(input_fe)

        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer.shap_values(input_processed)

        if isinstance(shap_values, list):
            shap_for_class = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            if len(shap_values.shape) == 3:
                shap_for_class = shap_values[0, :, 1]
            else:
                shap_for_class = shap_values[0]

            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_for_class,
            "Absolute Impact": np.abs(shap_for_class)
        }).sort_values("Absolute Impact", ascending=False)

        st.subheader("Top Risk Factors")
        st.dataframe(
            shap_df.head(10)[["Feature", "SHAP Value"]],
            use_container_width=True
        )

        st.subheader("SHAP Waterfall Plot")

        explanation = shap.Explanation(
            values=shap_for_class,
            base_values=base_value,
            data=input_processed[0],
            feature_names=feature_names
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning("SHAP chưa hiển thị được với model hiện tại.")
        st.code(str(e))

    st.markdown("---")

    # ========================================================
    # 13. PERSONALIZED RECOMMENDATION
    # ========================================================

    st.header("📌 Personalized Recommendation")

    recommendations = generate_recommendations(
        probability=probability,
        academic_pressure=academic_pressure,
        study_satisfaction=study_satisfaction,
        sleep_duration=sleep_duration,
        dietary_habits=dietary_habits,
        financial_stress=financial_stress,
        suicidal_thoughts=suicidal_thoughts,
        sentiment_score=sentiment_score
    )

    for rec in recommendations:
        st.write("- " + rec)


# ============================================================
# 14. FOOTER
# ============================================================

st.markdown("---")
st.caption("Youth Mental Health Guardian | Machine Learning + NLP + SHAP XAI")