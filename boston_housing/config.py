"""
全局配置文件（已更新为最优超参数，基于 42 组调优实验）
"""
import os

# ========== 路径配置 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "..", "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
for d in [MODEL_DIR, OUTPUT_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ========== 数据集配置 ==========
CSV_FILE = os.path.join(DATA_DIR, "boston.csv")
TARGET_COL = "MEDV"
FEATURE_COLS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PIRATIO", "B", "LSTAT"
]
FEATURE_NAMES_DISPLAY = [
    "CRIM(人均犯罪率)", "ZN(住宅用地比例)", "INDUS(非零售商业用地)",
    "CHAS(临河=1)", "NOX(氮氧化物)", "RM(房间数)", "AGE(1940前建成)",
    "DIS(就业中心距离)", "RAD(高速指数)", "TAX(财产税率)",
    "PIRATIO(师生比)", "B(黑人比例)", "LSTAT(低收入%)"
]

# ========== 数据预处理 ==========
TEST_SIZE = 0.2
RANDOM_STATE = 42
NORMALIZE_METHOD = "standard"

# ========== BP 最优超参数（调优结果）==========
# 最佳组合: lr=0.01, hidden=(64,32,16), act=relu, alpha=1e-4, solver=adam
# R²=0.8882, RMSE=2.86, MAE=1.88, 训练时间 0.08s
HIDDEN_LAYER_SIZES = (64, 32, 16)
ACTIVATION = "relu"
SOLVER = "adam"
LEARNING_RATE_INIT = 0.01
MAX_EPOCHS = 500
BATCH_SIZE = "auto"
EARLY_STOPPING = True
EARLY_STOPPING_TOLERANCE = 10
ALPHA = 0.0001

# ========== 扩展 ==========
EPOCH_COMPARISON_LIST = [50, 100, 200, 500]

# ========== 文件 ==========
MODEL_FILENAME = "bp_model.pkl"
SCALER_FILENAME = "scaler.pkl"
PREDICTIONS_FILENAME = "predictions.csv"
HISTORY_FILENAME = "history.csv"
