"""
扩展功能：比较不同训练轮数 (Epoch) 对模型性能的影响
"""
import time
import numpy as np

from model_builder import build_bp_model
from model_eval import evaluate_model
from visualizer import plot_epoch_comparison

from config import EPOCH_COMPARISON_LIST


def compare_epochs(X_train, y_train, X_test, y_test, scaler=None,
                   epoch_list=None, verbose=True):
    """
    比较不同 Epoch 数量对模型性能的影响

    参数:
        epoch_list: 要比较的 epoch 值列表

    返回:
        results: [{'epochs': n, 'mae': v, 'mse': v, 'rmse': v, 'r2': v, 'time': t}, ...]
    """
    if epoch_list is None:
        epoch_list = EPOCH_COMPARISON_LIST

    results = []

    print("\n" + "=" * 60)
    print("Epoch 性能对比分析")
    print("=" * 60)

    for epochs in epoch_list:
        print(f"\n[对比] 训练 Epoch = {epochs}...")

        # 构建模型（仅修改 epoch 数量）
        model = build_bp_model(max_epochs=epochs)

        # 训练
        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        # 评估
        metrics, _ = evaluate_model(model, X_test, y_test, scaler, verbose=False)

        result = {
            "epochs": epochs,
            "mae": metrics["MAE"],
            "mse": metrics["MSE"],
            "rmse": metrics["RMSE"],
            "r2": metrics["R2"],
            "mape": metrics["MAPE"],
            "time": training_time
        }
        results.append(result)

        if verbose:
            print(f"  -> 用时: {training_time:.2f}s | MAE: {metrics['MAE']:.4f} | R^2: {metrics['R2']:.4f}")

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("Epoch 性能对比汇总表")
    print("=" * 80)
    print(f"{'Epoch':<10}{'MAE':<12}{'RMSE':<12}{'R^2':<12}{'MAPE(%)':<12}{'用时(s)':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r['epochs']:<10}{r['mae']:<12.4f}{r['rmse']:<12.4f}"
              f"{r['r2']:<12.4f}{r['mape']:<12.2f}{r['time']:<12.2f}")

    print("=" * 80)

    # 给出建议
    best_r2 = max(results, key=lambda x: x['r2'])
    print(f"\n[建议] 最佳性能: Epoch={best_r2['epochs']}, R^2={best_r2['r2']:.4f}, MAE={best_r2['mae']:.4f}")

    # 生成对比图表
    plot_epoch_comparison(results)

    return results


if __name__ == "__main__":
    # 测试 Epoch 对比
    from data_loader import prepare_datasets
    X_train, X_test, y_train, y_test, scaler, features = prepare_datasets()
    results = compare_epochs(X_train, y_train, X_test, y_test, scaler)
    print("\nEpoch 对比完成！")
    print(f"结果: {results}")
