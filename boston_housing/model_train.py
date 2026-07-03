"""
模型训练模块：训练 BP 神经网络、保存/加载模型
"""
import time

from model_builder import build_bp_model, print_model_summary
from utils import save_model


def train_model(X_train, y_train, hidden_layer_sizes=None,
                max_epochs=None, learning_rate_init=None,
                activation=None, solver=None, verbose=True):
    """
    构建并训练 BP 神经网络模型

    参数:
        X_train: 训练特征 (已归一化)
        y_train: 训练目标 (已归一化)
        verbose: 是否打印训练过程信息

    返回:
        (trained_model, training_history)
        training_history 包含 loss_curve_ 和训练时间等信息
    """
    # 构建模型
    model = build_bp_model(
        hidden_layer_sizes=hidden_layer_sizes,
        max_epochs=max_epochs,
        learning_rate_init=learning_rate_init,
        activation=activation,
        solver=solver
    )

    # 训练模型
    print("\n[训练] 开始训练 BP 神经网络...")
    print("[训练] 训练过程中...")
    start_time = time.time()

    # MLPRegressor 在 fit 过程中会自动打印 verbose 信息
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # 收集训练历史
    history = {
        "n_iter": model.n_iter_,
        "loss": model.loss_,
        "loss_curve": list(model.loss_curve_),
        "training_time": training_time
    }

    if verbose:
        print(f"\n[训练] 训练完成！用时: {training_time:.2f} 秒")
        print_model_summary(model)

    return model, history


def train_and_save_model(X_train, y_train, scaler, **kwargs):
    """
    训练模型并保存到文件
    返回: (model, history)
    """
    model, history = train_model(X_train, y_train, **kwargs)
    save_model(model, scaler)
    return model, history


if __name__ == "__main__":
    # 测试训练模块
    from data_loader import prepare_datasets
    X_train, X_test, y_train, y_test, scaler, features = prepare_datasets()
    model, history = train_and_save_model(X_train, y_train, scaler)
    print(f"训练历史 key: {history.keys()}")
    print("模型训练与保存成功！")
