import os
import logging
import warnings
import joblib
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
    make_scorer,
)
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ─────────────────────────────────────────────────────────────
# CONFIG  (tất cả magic numbers tập trung tại đây)
# ─────────────────────────────────────────────────────────────
 
CONFIG = {
    # Paths
    "data_path": "data/Advanced_XAI_Depression_Dataset.csv",
    "model_dir": "model",
    "results_dir": "results",
    "shap_dir": "results/shap",
 
    # Columns
    "target_col": "Depression",
    "text_col": "Daily_Journal_Text",
    "drop_cols": ["Social_Engagement_Score", "Burnout_Level", "Motivation_Level"],
 
    # Preprocessing
    "tfidf_max_features": 100,
    "tfidf_ngram_range": (1, 2),
 
    # Split & CV
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
 
    # Threshold search
    "threshold_min": 0.40,
    "threshold_max": 0.81,
    "threshold_step": 0.01,
    "min_precision_for_threshold": 0.70,
 
    # Model selection metric priority
    "selection_metrics": ["Recall", "F1-score", "ROC-AUC"],
 
    # SHAP
    "shap_max_display": 10,
    "shap_waterfall_sample": 0,   # index trong X_test dùng cho waterfall
}
 
# ─────────────────────────────────────────────────────────────
# WORD LISTS
# ─────────────────────────────────────────────────────────────
 
STOP_WORDS = [
    "là", "và", "của", "có", "cho", "với", "một", "những", "các",
    "tôi", "em", "mình", "bạn", "rất", "thì", "mà", "đang", "đã",
    "này", "kia", "đó", "trong", "khi", "vì", "do", "nên", "ở",
    "về", "ra", "vào", "lên", "xuống", "được", "cũng",
    "and", "the", "is", "are", "was", "were", "a", "an", "of",
    "to", "in", "on", "for", "with", "that", "this", "it",
    "be", "as", "at", "by", "from", "or", "if", "but", "so",
    "because", "about", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off",
    "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "than", "too", "very",
    "can", "will", "just", "should", "now",
]
 
NEGATIVE_WORDS = [
    "buồn", "áp lực", "mệt", "mệt mỏi", "tuyệt vọng", "cô đơn",
    "lo lắng", "căng thẳng", "chán", "kiệt sức", "sợ", "tệ",
    "stress", "stressed", "tired", "hopeless", "lonely",
    "anxious", "depressed", "exhausted",
]
 
POSITIVE_WORDS = [
    "vui", "ổn", "bình tĩnh", "có động lực", "hy vọng", "thư giãn",
    "tự tin", "tốt", "hạnh phúc", "khỏe", "happy", "good",
    "calm", "motivated", "hopeful", "relaxed", "confident",
]
 
# Các từ phủ định → đảo cực của từ tiếp theo
NEGATION_WORDS = ["không", "chưa", "chẳng", "chả", "never", "not", "no"]
 
SLEEP_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
}
 
# ─────────────────────────────────────────────────────────────
# FIX 2: SENTIMENT SCORER XỬ LÝ NEGATION
# ─────────────────────────────────────────────────────────────
 
def calculate_sentiment_score(text: str) -> float:
    """
    Tính sentiment score từ văn bản.
    Xử lý negation: nếu trước từ tích cực/tiêu cực có từ phủ định
    (cách nhau ≤ 2 token) thì đảo cực của từ đó.
    """
    text = str(text).lower()
    tokens = text.split()
 
    pos_count = 0
    neg_count = 0
 
    for i, token in enumerate(tokens):
        # Xác định cửa sổ phủ định (2 token trước)
        window = tokens[max(0, i - 2): i]
        is_negated = any(w in NEGATION_WORDS for w in window)
 
        if any(pw in token for pw in POSITIVE_WORDS):
            if is_negated:
                neg_count += 1   # "không vui" → negative
            else:
                pos_count += 1
 
        elif any(nw in token for nw in NEGATIVE_WORDS):
            if is_negated:
                pos_count += 1   # "không mệt" → positive
            else:
                neg_count += 1
 
    score = (pos_count - neg_count) / (pos_count + neg_count + 1)
    return round(score, 3)
 
 
# ─────────────────────────────────────────────────────────────
# FIX 1: FEATURE ENGINEERING TRANSFORMER (bên trong Pipeline)
# ─────────────────────────────────────────────────────────────
 
class DepressionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Tạo các feature mới từ dữ liệu thô.
    Đặt bên trong Pipeline → chỉ fit trên X_train, không bị leak.
    """
 
    def __init__(self, text_col: str, sleep_map: dict):
        self.text_col = text_col
        self.sleep_map = sleep_map
 
    def fit(self, X, y=None):
        return self   # stateless transformer
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
 
        # Sentiment từ văn bản
        if self.text_col in X.columns:
            X["Sentiment_Score_From_Text"] = (
                X[self.text_col].fillna("").apply(calculate_sentiment_score)
            )
 
        # Sleep duration → numeric
        if "Sleep Duration" in X.columns:
            X["Sleep_Duration_Numeric"] = (
                X["Sleep Duration"].map(self.sleep_map).fillna(1)
            )
 
        # Mental stress & academic burden
        if "Academic Pressure" in X.columns and "Financial Stress" in X.columns:
            X["Mental_Stress_Score"] = (
                X["Academic Pressure"] + X["Financial Stress"]
            )
 
        if "Academic Pressure" in X.columns and "Study Hours" in X.columns:
            X["Academic_Burden_Index"] = (
                X["Academic Pressure"] * X["Study Hours"]
            )
 
        return X
 
 
# ─────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────
 
log.info("Loading data from %s", CONFIG["data_path"])
df = pd.read_csv(CONFIG["data_path"])
 
if CONFIG["target_col"] not in df.columns:
    raise ValueError(f"Không tìm thấy cột '{CONFIG['target_col']}' trong dataset.")
 
if CONFIG["text_col"] not in df.columns:
    df[CONFIG["text_col"]] = ""
 
df[CONFIG["text_col"]] = df[CONFIG["text_col"]].fillna("").astype(str)
 
# Xoá các cột không dùng
df = df.drop(columns=[c for c in CONFIG["drop_cols"] if c in df.columns])
 
# Encode target
target_encoder = LabelEncoder()
df[CONFIG["target_col"]] = target_encoder.fit_transform(
    df[CONFIG["target_col"]].astype(str)
)
 
X = df.drop(columns=[CONFIG["target_col"]])
y = df[CONFIG["target_col"]]
 
# Tính scale_pos_weight cho XGBoost (FIX 6)
neg_count = (y == 0).sum()
pos_count = (y == 1).sum()
scale_pos_weight = round(neg_count / pos_count, 2)
log.info("class distribution  neg=%d  pos=%d  scale_pos_weight=%.2f",
         neg_count, pos_count, scale_pos_weight)
 
# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
    stratify=y,
)
 
# ─────────────────────────────────────────────────────────────
# DYNAMIC COLUMN DETECTION (sau feature engineering)
# ─────────────────────────────────────────────────────────────
# Dùng một bản transform nhỏ để lấy tên cột thực tế
 
_fe_probe = DepressionFeatureEngineer(
    text_col=CONFIG["text_col"], sleep_map=SLEEP_MAP
)
X_probe = _fe_probe.transform(X_train.head(5))
 
numeric_cols = X_probe.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_probe.select_dtypes(include=["object"]).columns.tolist()
 
if CONFIG["text_col"] in categorical_cols:
    categorical_cols.remove(CONFIG["text_col"])
 
log.info("Numeric features  : %d", len(numeric_cols))
log.info("Categorical features: %d", len(categorical_cols))
log.info("Text column       : %s", CONFIG["text_col"])
 
# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
 
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
 
text_transformer = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=CONFIG["tfidf_max_features"],
        stop_words=STOP_WORDS,
        ngram_range=CONFIG["tfidf_ngram_range"],
    )),
])
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("txt", text_transformer, CONFIG["text_col"]),
    ],
    remainder="drop",
    sparse_threshold=0,
)
 
# ─────────────────────────────────────────────────────────────
# XGBoost (optional import)
# ─────────────────────────────────────────────────────────────
 
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
 
# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
 
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=CONFIG["random_state"],
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        random_state=CONFIG["random_state"],
        class_weight="balanced",
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=CONFIG["random_state"],
    ),
}
 
if XGBOOST_AVAILABLE:
    # FIX 6: scale_pos_weight thay cho class_weight để cân bằng class
    models["XGBoost"] = XGBClassifier(
        random_state=CONFIG["random_state"],
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
 
# ─────────────────────────────────────────────────────────────
# FIX 5: CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────
 
log.info("\n========== CROSS-VALIDATION (%d-fold) ==========",
         CONFIG["cv_folds"])
 
cv = StratifiedKFold(
    n_splits=CONFIG["cv_folds"],
    shuffle=True,
    random_state=CONFIG["random_state"],
)
 
scoring = {
    "accuracy" : make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall"   : make_scorer(recall_score, zero_division=0),
    "f1"       : make_scorer(f1_score, zero_division=0),
    "roc_auc"  : make_scorer(roc_auc_score, needs_proba=True),
}
 
cv_summary = []
 
for name, model in models.items():
    pipeline = Pipeline([
        ("feature_engineer", DepressionFeatureEngineer(
            text_col=CONFIG["text_col"], sleep_map=SLEEP_MAP
        )),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
 
    cv_results = cross_validate(pipeline, X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=-1)
 
    cv_summary.append({
        "Model"    : name,
        "CV_Accuracy" : round(cv_results["test_accuracy"].mean(), 4),
        "CV_Precision": round(cv_results["test_precision"].mean(), 4),
        "CV_Recall"   : round(cv_results["test_recall"].mean(), 4),
        "CV_F1"       : round(cv_results["test_f1"].mean(), 4),
        "CV_ROC_AUC"  : round(cv_results["test_roc_auc"].mean(), 4),
        "CV_F1_Std"   : round(cv_results["test_f1"].std(), 4),   # độ ổn định
    })
    log.info("  %s → F1=%.4f ± %.4f  Recall=%.4f  AUC=%.4f",
             name,
             cv_results["test_f1"].mean(),
             cv_results["test_f1"].std(),
             cv_results["test_recall"].mean(),
             cv_results["test_roc_auc"].mean())
 
cv_df = pd.DataFrame(cv_summary)
print("\n" + cv_df.to_string(index=False))
 
# ─────────────────────────────────────────────────────────────
# TRAIN FINAL MODELS & EVALUATE ON HOLDOUT TEST SET
# ─────────────────────────────────────────────────────────────
 
log.info("\n========== TRAINING MODELS ==========")
 
results = []
trained_models = {}
 
for name, model in models.items():
    pipeline = Pipeline([
        ("feature_engineer", DepressionFeatureEngineer(
            text_col=CONFIG["text_col"], sleep_map=SLEEP_MAP
        )),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
 
    pipeline.fit(X_train, y_train)
 
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
 
    results.append({
        "Model"    : name,
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1-score" : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_proba),
    })
 
    trained_models[name] = pipeline
 
results_df = pd.DataFrame(results)
 
log.info("\n========== MODEL COMPARISON ==========")
print(results_df.sort_values(
    by=CONFIG["selection_metrics"], ascending=False
).to_string(index=False))
 
# Chọn model tốt nhất
best_row = results_df.sort_values(
    by=CONFIG["selection_metrics"], ascending=False
).iloc[0]
 
best_model_name = best_row["Model"]
final_model     = trained_models[best_model_name]
 
log.info("\n🏆 Best Model Selected: %s 🏆", best_model_name)
 
# ─────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────────────────────
 
y_proba_final   = final_model.predict_proba(X_test)[:, 1]
threshold_results = []
 
for threshold in np.arange(
    CONFIG["threshold_min"],
    CONFIG["threshold_max"],
    CONFIG["threshold_step"],
):
    y_pred_thr = (y_proba_final >= threshold).astype(int)
    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision_score(y_test, y_pred_thr, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred_thr, zero_division=0),
        "F1-score" : f1_score(y_test, y_pred_thr, zero_division=0),
    })
 
threshold_df   = pd.DataFrame(threshold_results)
valid_thresholds = threshold_df[
    threshold_df["Precision"] >= CONFIG["min_precision_for_threshold"]
]
 
best_threshold_row = (
    valid_thresholds.sort_values(by=["Recall", "F1-score"], ascending=False).iloc[0]
    if len(valid_thresholds) > 0
    else threshold_df.sort_values(by=["F1-score", "Recall"], ascending=False).iloc[0]
)
 
best_threshold = float(best_threshold_row["Threshold"])
final_pred     = (y_proba_final >= best_threshold).astype(int)
 
final_accuracy  = accuracy_score(y_test, final_pred)
final_precision = precision_score(y_test, final_pred, zero_division=0)
final_recall    = recall_score(y_test, final_pred, zero_division=0)
final_f1        = f1_score(y_test, final_pred, zero_division=0)
final_roc_auc   = roc_auc_score(y_test, y_proba_final)
 
log.info("\n========== FINAL MODEL EVALUATION ==========")
log.info("Best Model    : %s", best_model_name)
log.info("Best Threshold: %.2f", best_threshold)
log.info("Accuracy      : %.4f", final_accuracy)
log.info("Precision     : %.4f", final_precision)
log.info("Recall        : %.4f", final_recall)
log.info("F1-score      : %.4f", final_f1)
log.info("ROC-AUC       : %.4f", final_roc_auc)
 
print("\nClassification Report:")
print(classification_report(y_test, final_pred))
 
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))
 
# ─────────────────────────────────────────────────────────────
# CREATE OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────
 
for d in [CONFIG["model_dir"], CONFIG["results_dir"], CONFIG["shap_dir"]]:
    os.makedirs(d, exist_ok=True)
 
# ─────────────────────────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────────────────────────
 
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {final_roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_model_name}")
plt.legend()
plt.savefig(f"{CONFIG['results_dir']}/roc_curve.png", bbox_inches="tight")
plt.close()
 
# ─────────────────────────────────────────────────────────────
# CV F1 STABILITY CHART
# ─────────────────────────────────────────────────────────────
 
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    cv_df["Model"],
    cv_df["CV_F1"],
    xerr=cv_df["CV_F1_Std"],
    color="#4C9BE8",
    edgecolor="white",
    capsize=5,
)
ax.set_xlabel("CV F1-score (mean ± std)")
ax.set_title("Cross-Validation F1 Stability")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{CONFIG['results_dir']}/cv_f1_stability.png", bbox_inches="tight")
plt.close()
 
# ─────────────────────────────────────────────────────────────
# FIX 4: SHAP — logging rõ ràng, không nuốt lỗi thầm lặng
# ─────────────────────────────────────────────────────────────
 
log.info("\n========== SHAP EXPLAINABLE AI ==========")
 
shap_success = False
 
try:
    preprocessor_part = final_model.named_steps["preprocessor"]
    model_part        = final_model.named_steps["model"]
 
    # Cần transform qua cả feature engineering trước
    fe_part           = final_model.named_steps["feature_engineer"]
    X_test_fe         = fe_part.transform(X_test)
    X_test_processed  = preprocessor_part.transform(X_test_fe)
    processed_feature_names = preprocessor_part.get_feature_names_out()
 
    explainer   = shap.TreeExplainer(model_part)
    shap_values = explainer.shap_values(X_test_processed)
 
    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_class = shap_values[:, :, 1]
    else:
        shap_values_class = shap_values
 
    # Feature importance CSV
    if hasattr(model_part, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature"   : processed_feature_names,
            "Importance": model_part.feature_importances_,
        }).sort_values("Importance", ascending=False)
        importance_df.to_csv(
            f"{CONFIG['results_dir']}/feature_importance.csv", index=False
        )
 
    # Summary plot (beeswarm)
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values_class, X_test_processed,
        feature_names=processed_feature_names, show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{CONFIG['shap_dir']}/shap_summary_plot.png", bbox_inches="tight")
    plt.close()
 
    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_class, X_test_processed,
        feature_names=processed_feature_names,
        plot_type="bar", show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{CONFIG['shap_dir']}/shap_bar_plot.png", bbox_inches="tight")
    plt.close()
 
    # Waterfall plot cho 1 sample
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    sample_idx = CONFIG["shap_waterfall_sample"]
    waterfall_explanation = shap.Explanation(
        values       = shap_values_class[sample_idx],
        base_values  = base_value,
        data         = X_test_processed[sample_idx],
        feature_names= processed_feature_names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(
        waterfall_explanation,
        max_display=CONFIG["shap_max_display"],
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{CONFIG['shap_dir']}/shap_waterfall_plot.png", bbox_inches="tight")
    plt.close()
 
    shap_success = True
    log.info("SHAP plots saved successfully.")
 
except Exception as e:
    # FIX 4: log traceback đầy đủ thay vì chỉ in message
    log.error("SHAP failed — model will still be saved without explainability.",
              exc_info=True)
 
# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
 
joblib.dump(final_model,     f"{CONFIG['model_dir']}/final_model.pkl")
joblib.dump(final_model,     f"{CONFIG['model_dir']}/shap_model.pkl")
joblib.dump(target_encoder,  f"{CONFIG['model_dir']}/target_encoder.pkl")
joblib.dump(best_threshold,  f"{CONFIG['model_dir']}/best_threshold.pkl")
joblib.dump(X.columns.tolist(), f"{CONFIG['model_dir']}/input_columns.pkl")
 
results_df.to_csv(f"{CONFIG['results_dir']}/model_comparison.csv",       index=False)
threshold_df.to_csv(f"{CONFIG['results_dir']}/threshold_optimization.csv", index=False)
cv_df.to_csv(f"{CONFIG['results_dir']}/cv_results.csv",                   index=False)
 
pd.DataFrame([{
    "Best Model"     : best_model_name,
    "Best Threshold" : best_threshold,
    "Accuracy"       : final_accuracy,
    "Precision"      : final_precision,
    "Recall"         : final_recall,
    "F1-score"       : final_f1,
    "ROC-AUC"        : final_roc_auc,
    "SHAP_Generated" : shap_success,
}]).to_csv(f"{CONFIG['results_dir']}/final_metrics.csv", index=False)
 
log.info("Training completed successfully.")
log.info("SHAP generated: %s", shap_success)