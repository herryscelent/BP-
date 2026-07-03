"""
数据可视化模块：Loss曲线、真实vs预测对比图、残差图等
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from config import FIGURES_DIR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_loss_curve(loss_curve, title="训练损失变化曲线", save=True):
    """
    绘制训练过程中 Loss 随 Epoch 变化的曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, 'b-', linewidth=2, alpha=0.8)
    plt.xlabel("迭代次数 (Epoch)", fontsize=12)
    plt.ylabel("损失值 (Loss)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "loss_curve.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[图表] Loss 曲线已保存至: {path}")

    plt.close()


def plot_comparison(y_true, y_pred, title="真实房价 vs 预测房价对比", save=True):
    """
    绘制真实房价与预测房价的散点对比图，含 y=x 参考线
    """
    plt.figure(figsize=(10, 6))

    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=30, c='steelblue', edgecolors='white', linewidth=0.5)

    # y=x 参考线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='理想预测 (y=x)')

    plt.xlabel("真实房价", fontsize=12)
    plt.ylabel("预测房价", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "prediction_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[图表] 预测对比图已保存至: {path}")

    plt.close()


def plot_residuals(y_true, y_pred, title="预测误差残差分布", save=True):
    """
    绘制预测误差（残差）散点图
    """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))

    # 残差散点图
    plt.scatter(y_pred, residuals, alpha=0.6, s=30, c='coral', edgecolors='white', linewidth=0.5)

    # 零误差参考线
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='零误差线')

    plt.xlabel("预测房价", fontsize=12)
    plt.ylabel("残差 (真实值 - 预测值)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "residuals.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[图表] 残差分布图已保存至: {path}")

    plt.close()


def plot_feature_distribution(X, feature_names, title="特征分布直方图", save=True):
    """
    绘制各特征分布直方图（用于数据分析展示）
    """
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        ax.hist(X[:, i], bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("值", fontsize=8)
        ax.set_ylabel("频数", fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "feature_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[图表] 特征分布图已保存至: {path}")

    plt.close()


def plot_epoch_comparison(epoch_results, title="不同训练轮数对模型性能的影响", save=True):
    """
    绘制不同 Epoch 下 MAE 和 R^2 的对比柱状图

    epoch_results: [{'epochs': n, 'mae': v, 'r2': v}, ...]
    """
    epochs = [r['epochs'] for r in epoch_results]
    maes = [r['mae'] for r in epoch_results]
    r2s = [r['r2'] for r in epoch_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MAE 对比
    bars1 = ax1.bar(range(len(epochs)), maes, color='steelblue', alpha=0.8, width=0.6)
    ax1.set_xticks(range(len(epochs)))
    ax1.set_xticklabels([f"{e}" for e in epochs])
    ax1.set_xlabel("训练轮数 (Epoch)", fontsize=12)
    ax1.set_ylabel("MAE", fontsize=12)
    ax1.set_title("不同 Epoch 的 MAE 对比", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # 在柱状图上显示数值
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # R^2 对比
    bars2 = ax2.bar(range(len(epochs)), r2s, color='coral', alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(epochs)))
    ax2.set_xticklabels([f"{e}" for e in epochs])
    ax2.set_xlabel("训练轮数 (Epoch)", fontsize=12)
    ax2.set_ylabel("R^2", fontsize=12)
    ax2.set_title("不同 Epoch 的 R^2 对比", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "epoch_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[图表] Epoch 对比图已保存至: {path}")

    plt.close()


def plot_all_visualizations(model, y_true, y_pred, X, feature_names, loss_curve=None, epoch_results=None):
    """
    一键生成所有可视化图表
    """
    if loss_curve:
        plot_loss_curve(loss_curve)

    plot_comparison(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_feature_distribution(X, feature_names)

    if epoch_results:
        plot_epoch_comparison(epoch_results)

    print("\n[图表] 所有可视化图表已生成完毕！")


if __name__ == "__main__":
    # 测试可视化模块
    print("可视化模块测试 - 请先训练模型并生成数据")
    print(f"图表将保存至: {FIGURES_DIR}")
