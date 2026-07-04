"""
数据管理模块：从本地 CSV 加载波士顿房价数据集、数据清洗、归一化
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CSV_FILE, FEATURE_COLS, TARGET_COL,
                    TEST_SIZE, RANDOM_STATE, NORMALIZE_METHOD,
                    FEATURE_NAMES_DISPLAY)


def load_local_csv():
    """
    从本地 CSV 文件加载波士顿房价数据集
    返回: (X, y, feature_names)
    """
    csv_path = CSV_FILE
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件未找到: {csv_path}")

    print(f"[数据] 正在加载本地 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 检查列是否存在
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV 中缺少以下列: {missing_cols}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # 确保都是数值类型
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    print(f"[数据] 加载成功！特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
    print(f"[数据] 特征列: {list(X.columns)}")
    print(f"[数据] 目标列: {TARGET_COL}")
    return X, y, FEATURE_COLS


def display_data_info(X, y, feature_names=None):
    names = feature_names if feature_names is not None else list(X.columns)
    print()
    print("=" * 60)
    print("数据集基本信息")
    print("=" * 60)
    print(f"样本总数: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"特征名称: {names}")
    preview = X.head().copy()
    preview[TARGET_COL] = y.head().values
    print(f"\n前 5 条数据预览:")
    print(preview.to_string())
    print(f"\n基本统计信息:")
    print(X.describe().to_string())
def clean_data(X, y):
    """
    数据清洗：检查缺失值、处理异常值
    返回: (X_clean, y_clean)
    """
    print("\n[清洗] 开始数据清洗...")
    data = pd.concat([X, y.rename(TARGET_COL)], axis=1)

    # 1. 缺失值
    missing_counts = data.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        print(f"[清洗] 发现缺失值:\n{missing_cols}")
        for col in missing_cols.index:
            if col == TARGET_COL:
                data = data.dropna(subset=[TARGET_COL])
                print("[清洗] 目标值缺失的样本已被删除")
            else:
                mean_val = data[col].mean()
                data[col] = data[col].fillna(mean_val)
                print(f"[清洗] 特征 '{col}' 缺失值已用均值 {mean_val:.4f} 填充")
    else:
        print("[清洗] 未发现缺失值")

    # 2. 异常值检测（3-sigma）
    print("[清洗] 使用 3-sigma 原则检测异常值...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        mean = data[col].mean()
        std = data[col].std()
        outliers = data[(np.abs(data[col] - mean) > 3 * std)]
        if len(outliers) > 0:
            outlier_info[col] = len(outliers)
            print(f"[清洗] 特征 '{col}': {len(outliers)} 个异常值 ({len(outliers)/len(data)*100:.1f}%)")
    if not outlier_info:
        print("[清洗] 未发现显著异常值")

    # 3. 去除目标值极端异常 (4-sigma)
    y_mean = data[TARGET_COL].mean()
    y_std = data[TARGET_COL].std()
    before_drop = len(data)
    data = data[np.abs(data[TARGET_COL] - y_mean) <= 4 * y_std]
    dropped = before_drop - len(data)
    if dropped > 0:
        print(f"[清洗] 去除了 {dropped} 个目标值极端异常样本")

    X_clean = data.drop(columns=[TARGET_COL])
    y_clean = data[TARGET_COL]
    print(f"[清洗] 完成！剩余样本数: {len(data)}")
    return X_clean, y_clean


def normalize_data(X, y, fit=True, scaler=None):
    """
    数据归一化/标准化处理
    fit=True: 拟合+转换（训练集）；fit=False: 仅转换（测试集/新数据）
    返回: (X_norm, y_norm, scaler)
    """
    if NORMALIZE_METHOD == "minmax":
        from sklearn.preprocessing import MinMaxScaler as Scaler
    else:
        from sklearn.preprocessing import StandardScaler as Scaler

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    if fit:
        scaler = Scaler()
        data_combined = np.column_stack([X_arr, y_arr])
        data_norm = scaler.fit_transform(data_combined)
        X_norm = data_norm[:, :-1]
        y_norm = data_norm[:, -1]
        method_name = "MinMaxScaler" if NORMALIZE_METHOD == "minmax" else "StandardScaler"
        print(f"\n[归一化] 使用 {method_name} 完成归一化 (拟合+转换)")
    else:
        if scaler is None:
            raise ValueError("转换模式需要提供已拟合的 scaler")
        data_combined = np.column_stack([X_arr, y_arr.reshape(-1, 1)])
        data_norm = scaler.transform(data_combined)
        X_norm = data_norm[:, :-1]
        y_norm = data_norm[:, -1]
        print("[归一化] 使用已拟合的 scaler 完成转换")

    return X_norm, y_norm, scaler


def prepare_datasets():
    """
    一键完成: 加载 CSV → 显示信息 → 清洗 → 划分 → 归一化
    返回: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    # 1. 加载数据
    X, y, feature_names = load_local_csv()
    display_data_info(X, y, feature_names)

    # 2. 清洗
    X_clean, y_clean = clean_data(X, y)

    # 3. 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[划分] 训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 4. 归一化
    X_train_norm, y_train_norm, scaler = normalize_data(X_train, y_train, fit=True)
    X_test_norm, y_test_norm, _ = normalize_data(X_test, y_test, fit=False, scaler=scaler)

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, scaler, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, features = prepare_datasets()
    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print("数据准备完成！")
