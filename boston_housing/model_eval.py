"""
模型评估模块：计算 MAE, MSE, RMSE, R^2 等评估指标
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, scaler=None, verbose=True):
    """
    评估模型性能

    参数:
        model: 已训练的模型
        X_test: 测试特征 (已归一化)
        y_test: 测试目标 (已归一化)
        scaler: 归一化器（用于反归一化计算原始尺度的指标）
        verbose: 是否打印评估报告

    返回:
        metrics_dict: 包含各项评估指标的字典
    """
    # 预测
    y_pred_norm = model.predict(X_test)

    # 如果在归一化器前传入了 y_test 的归一化值，需要反归一化
    if scaler is not None:
        # 构建完整数组进行反归一化
        n_features = scaler.n_features_in_
        # 创建包含特征和目标的数组
        test_combined = np.zeros((len(y_test), n_features))
        # 填充特征（使用测试集的均值作为占位）
        # 实际上只需要反归一化 y
        dummy_pred = np.zeros((len(y_pred_norm), n_features))
        dummy_pred[:, -1] = y_pred_norm
        dummy_true = np.zeros((len(y_test), n_features))
        dummy_true[:, -1] = y_test

        y_pred = scaler.inverse_transform(dummy_pred)[:, -1]
        y_true = scaler.inverse_transform(dummy_true)[:, -1]
    else:
        y_pred = y_pred_norm
        y_true = y_test

    # 计算评估指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # 计算 MAPE（平均绝对百分比误差）
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    if verbose:
        print("\n" + "=" * 60)
        print("模型评估报告")
        print("=" * 60)
        print(f"平均绝对误差 (MAE)   : {mae:.4f}")
        print(f"均方误差 (MSE)       : {mse:.4f}")
        print(f"均方根误差 (RMSE)    : {rmse:.4f}")
        print(f"决定系数 (R^2)        : {r2:.4f}")
        print(f"平均绝对百分比误差    : {mape:.2f}%")
        print("=" * 60)

        if r2 >= 0.7:
            print("模型评价: 模型表现良好，预测精度较高 (R^2 ≥ 0.7)")
        elif r2 >= 0.5:
            print("模型评价: 模型表现一般，可尝试调优 (0.5 ≤ R^2 < 0.7)")
        else:
            print("模型评价: 模型表现欠佳，建议调参或增加训练数据 (R^2 < 0.5)")

    return metrics, (y_true, y_pred)


if __name__ == "__main__":
    # 测试评估模块
    from data_loader import prepare_datasets
    from utils import load_model

    X_train, X_test, y_train, y_test, scaler, features = prepare_datasets()
    model, _ = load_model()
    if model is not None:
        metrics, (y_true, y_pred) = evaluate_model(model, X_test, y_test, scaler)
        print(f"评估指标: {metrics}")
    else:
        print("请先训练模型！")
