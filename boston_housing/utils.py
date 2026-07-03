"""
工具函数模块：反归一化、格式转换等通用功能
"""
import numpy as np
import joblib
import os
from config import MODEL_DIR, MODEL_FILENAME, SCALER_FILENAME


def save_model(model, scaler=None):
    """
    保存训练好的模型和归一化器到文件
    """
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"[保存] 模型已保存至: {model_path}")

    if scaler is not None:
        scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)
        joblib.dump(scaler, scaler_path)
        print(f"[保存] 归一化器已保存至: {scaler_path}")


def load_model():
    """
    从文件加载已训练的模型和归一化器
    返回: (model, scaler) 或 (None, None)
    """
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(MODEL_DIR, SCALER_FILENAME)

    model = None
    scaler = None

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"[加载] 模型已加载: {model_path}")
    else:
        print("[警告] 未找到已保存的模型文件")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[加载] 归一化器已加载: {scaler_path}")

    return model, scaler


def inverse_transform_target(scaler, y_normalized, feature_index=-1):
    """
    仅反归一化目标值
    scaler: StandardScaler 或 MinMaxScaler
    y_normalized: 归一化后的目标值
    feature_index: 目标值在原始特征矩阵中的索引，默认最后一列
    """
    # 创建一个临时数组，形状与原始特征匹配
    # 使用 scaler 的 n_features_in_ 来确定维度
    n_features = scaler.n_features_in_
    dummy = np.zeros((len(y_normalized), n_features))
    dummy[:, feature_index] = y_normalized.flatten()
    dummy_inversed = scaler.inverse_transform(dummy)
    return dummy_inversed[:, feature_index].reshape(-1, 1)


def format_price(price):
    """格式化房价输出，保留 2 位小数"""
    return f"${price:,.2f}"


def print_separator(char="=", width=60):
    """打印分隔线"""
    print(char * width)
