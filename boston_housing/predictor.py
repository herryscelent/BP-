"""
房价预测模块：批量预测和单条数据预测
"""
import numpy as np
import pandas as pd
import os

from config import OUTPUT_DIR, PREDICTIONS_FILENAME


def batch_predict(model, X_test, scaler=None):
    """
    对测试集进行批量预测

    返回:
        y_pred: 预测值（原始尺度）
    """
    y_pred_norm = model.predict(X_test)

    if scaler is not None:
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(y_pred_norm), n_features))
        dummy[:, -1] = y_pred_norm
        y_pred = scaler.inverse_transform(dummy)[:, -1]
    else:
        y_pred = y_pred_norm

    return y_pred


def predict_single(model, features_array, scaler=None):
    """
    输入单个样本的 13 个特征，预测房价

    参数:
        model: 已训练的模型
        features_array: 13 个特征的数组或列表
        scaler: 归一化器

    返回:
        预测的房价（原始尺度）
    """
    # 转为 numpy 数组并 reshape
    features = np.array(features_array).reshape(1, -1)

    # 归一化（使用训练集的 scaler）
    if scaler is not None:
        # 需要构建包含目标列的完整数组以匹配 scaler 的维度
        n_features = scaler.n_features_in_
        dummy = np.zeros((1, n_features))
        dummy[:, :-1] = features
        features_norm = scaler.transform(dummy)[:, :-1]
    else:
        features_norm = features

    # 预测
    pred_norm = model.predict(features_norm)

    # 反归一化
    if scaler is not None:
        dummy_pred = np.zeros((1, n_features))
        dummy_pred[:, -1] = pred_norm
        pred = scaler.inverse_transform(dummy_pred)[0, -1]
    else:
        pred = pred_norm[0]

    return pred


def save_predictions(y_true, y_pred, feature_names=None, X_test=None):
    """
    将预测结果保存为 CSV 文件

    参数:
        y_true: 真实值
        y_pred: 预测值
        feature_names: 特征名称列表
        X_test: 测试特征（可选，用于保存输入特征）
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    pct_errors = (abs_errors / y_true) * 100

    # 构建结果 DataFrame
    results = pd.DataFrame({
        "真实房价": y_true,
        "预测房价": y_pred,
        "绝对误差": abs_errors,
        "相对误差(%)": pct_errors
    })

    # 可选：包含输入特征
    if X_test is not None and feature_names is not None:
        features_df = pd.DataFrame(X_test, columns=feature_names)
        results = pd.concat([features_df, results], axis=1)

    # 排序显示
    results = results.sort_values("真实房价")

    output_path = os.path.join(OUTPUT_DIR, PREDICTIONS_FILENAME)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n[保存] 预测结果已保存至: {output_path}")
    print(f"[保存] 共 {len(results)} 条预测记录")

    return results


def print_prediction_samples(y_true, y_pred, n_samples=10):
    """
    打印预测结果样本对比
    """
    print("\n" + "-" * 60)
    print("预测结果样本 (前 {} 条)".format(min(n_samples, len(y_true))))
    print("-" * 60)
    print(f"{'序号':<6}{'真实房价':<12}{'预测房价':<12}{'误差':<12}{'误差率(%)':<12}")
    print("-" * 60)

    for i in range(min(n_samples, len(y_true))):
        error = abs(y_true[i] - y_pred[i])
        pct_error = error / y_true[i] * 100
        print(f"{i+1:<6}{y_true[i]:<12.2f}{y_pred[i]:<12.2f}{error:<12.2f}{pct_error:<12.2f}")

    print("-" * 60)


if __name__ == "__main__":
    # 测试预测模块
    from data_loader import prepare_datasets
    from utils import load_model

    X_train, X_test, y_train, y_test, scaler, features = prepare_datasets()
    model, _ = load_model()

    if model is not None:
        y_pred = batch_predict(model, X_test, scaler)

        # 反归一化 y_test
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(y_test), n_features))
        dummy[:, -1] = y_test
        y_true = scaler.inverse_transform(dummy)[:, -1]

        print_prediction_samples(y_true, y_pred)

        # 测试单条预测
        sample_feat = X_test[0]
        single_pred = predict_single(model, sample_feat, scaler)
        print(f"\n单条预测测试:")
        print(f"特征: {sample_feat}")
        print(f"预测房价: ${single_pred:.2f}")
    else:
        print("请先训练模型！")
