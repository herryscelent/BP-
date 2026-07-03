"""
模型构建模块：创建 BP 神经网络模型
"""
from sklearn.neural_network import MLPRegressor

from config import (HIDDEN_LAYER_SIZES, ACTIVATION, SOLVER,
                    LEARNING_RATE_INIT, MAX_EPOCHS, BATCH_SIZE,
                    EARLY_STOPPING, EARLY_STOPPING_TOLERANCE, ALPHA,
                    RANDOM_STATE)


def build_bp_model(hidden_layer_sizes=None, max_epochs=None,
                   learning_rate_init=None, activation=None, solver=None):
    """
    构建 BP 神经网络模型 (MLPRegressor)

    参数:
        hidden_layer_sizes: 隐藏层结构，如 (64, 32, 16)
        max_epochs: 最大训练轮数
        learning_rate_init: 初始学习率
        activation: 激活函数 ('relu', 'tanh', 'logistic')
        solver: 优化器 ('adam', 'sgd', 'lbfgs')

    返回:
        MLPRegressor 实例
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = HIDDEN_LAYER_SIZES
    if max_epochs is None:
        max_epochs = MAX_EPOCHS
    if learning_rate_init is None:
        learning_rate_init = LEARNING_RATE_INIT
    if activation is None:
        activation = ACTIVATION
    if solver is None:
        solver = SOLVER

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        learning_rate_init=learning_rate_init,
        max_iter=max_epochs,
        shuffle=True,
        random_state=RANDOM_STATE,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=EARLY_STOPPING,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=EARLY_STOPPING_TOLERANCE,
    )

    # 打印模型结构信息
    print("\n[模型] BP 神经网络结构:")
    print(f"  - 输入层: 13 个神经元 (13 个特征)")
    print(f"  - 隐藏层: {list(hidden_layer_sizes)}")
    print(f"  - 输出层: 1 个神经元 (房价预测值)")
    print(f"  - 激活函数: {activation}")
    print(f"  - 优化器: {solver}")
    print(f"  - 学习率: {learning_rate_init}")
    print(f"  - 最大训练轮数: {max_epochs}")
    print(f"  - L2 正则化: alpha={ALPHA}")
    if EARLY_STOPPING:
        print(f"  - Early Stopping: 启用 (容忍 {EARLY_STOPPING_TOLERANCE} 轮)")

    return model


def print_model_summary(model):
    """打印训练后的模型摘要"""
    print("\n[模型] 训练完成摘要:")
    print(f"  - 实际迭代轮数: {model.n_iter_}")
    print(f"  - 最终损失值: {model.loss_:.6f}")
    if hasattr(model, 'best_loss_') and model.best_loss_ is not None:
        print(f"  - 最优损失值: {model.best_loss_:.6f}")
    print(f"  - 网络层数: {model.n_layers_}")
    print(f"  - 各层神经元数: {model.n_features_in_}, ", end="")
    if hasattr(model, 'hidden_layer_sizes'):
        print(f"{list(model.hidden_layer_sizes)}, 1")
    print(f"  - 训练样本数: {model.n_iter_ * len(getattr(model, 'loss_curve_', [0]))}")


if __name__ == "__main__":
    # 测试模型构建
    model = build_bp_model()
    print(model)
    print("模型构建成功！")
