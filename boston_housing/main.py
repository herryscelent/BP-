"""
主程序入口：波士顿房价预测系统 - 命令行菜单交互界面

使用方法:
    python main.py

功能菜单:
    1. 加载数据并显示基本信息
    2. 训练 BP 神经网络模型
    3. 模型评估
    4. 输入特征预测房价
    5. 可视化展示
    6. 对比不同 Epoch 性能
    7. 退出系统
"""
import sys
import numpy as np
import pandas as pd

# ========== 全局状态 ==========
class AppState:
    """保存系统运行时的全局状态"""
    def __init__(self):
        # 数据相关
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_test_original = None  # 反归一化后的真实值
        self.scaler = None
        self.feature_names = None
        self.data_loaded = False

        # 模型相关
        self.model = None
        self.model_trained = False
        self.training_history = None

        # 评估相关
        self.metrics = None
        self.y_pred = None

state = AppState()


# ========== 辅助函数 ==========

def print_header():
    """打印系统标题"""
    print("\n" + "=" * 60)
    print("     基于 BP 神经网络的波士顿房价预测系统")
    print("=" * 60)
    print(f"  数据状态: {'已加载' if state.data_loaded else '未加载'}")
    print(f"  模型状态: {'已训练' if state.model_trained else '未训练'}")
    print("=" * 60)


def print_menu():
    """打印功能菜单"""
    print("\n" + "-" * 60)
    print("功能菜单:")
    print("-" * 60)
    print("  1. 加载数据并显示基本信息")
    print("  2. 训练 BP 神经网络模型")
    print("  3. 模型评估 (MAE/MSE/RMSE/R^2)")
    print("  4. 输入特征预测房价")
    print("  5. 可视化展示")
    print("  6. 对比不同 Epoch 性能")
    print("  7. 退出系统")
    print("-" * 60)


def wait_for_enter():
    """等待用户按 Enter 继续"""
    input("\n按 Enter 键返回主菜单...")


# ========== 菜单功能实现 ==========

def menu_load_data():
    """菜单 1: 加载数据并显示基本信息"""
    print("\n" + "=" * 60)
    print(">> 1. 加载数据")
    print("=" * 60)

    try:
        from data_loader import prepare_datasets
        result = prepare_datasets()
        X_train, X_test, y_train, y_test, scaler, feature_names = result

        # 保存状态
        state.X_train = X_train
        state.X_test = X_test
        state.y_train = y_train
        state.y_test = y_test
        state.scaler = scaler
        state.feature_names = feature_names
        state.data_loaded = True

        # 保存反归一化后的真实值
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(y_test), n_features))
        dummy[:, -1] = y_test
        state.y_test_original = scaler.inverse_transform(dummy)[:, -1]

        print(f"\n✓ 数据加载成功！训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    except Exception as e:
        print(f"\n✗ 数据加载失败: {e}")

    wait_for_enter()


def menu_train_model():
    """菜单 2: 训练 BP 神经网络模型"""
    print("\n" + "=" * 60)
    print(">> 2. 训练 BP 神经网络模型")
    print("=" * 60)

    if not state.data_loaded:
        print("⚠ 请先加载数据！（请执行功能 1）")
        wait_for_enter()
        return

    try:
        from model_train import train_and_save_model
        from model_eval import evaluate_model

        # 询问是否自定义参数
        choice = input("\n是否使用默认参数？(y/n, 默认 y): ").strip().lower()

        kwargs = {}
        if choice == 'n':
            try:
                epochs = int(input("请输入训练轮数 (Epoch, 默认 500): ") or "500")
                lr = float(input("请输入学习率 (Learning Rate, 默认 0.001): ") or "0.001")
                hidden = input("请输入隐藏层结构 (逗号分隔, 默认 64,32,16): ") or "64,32,16"
                hidden_tuple = tuple(int(x.strip()) for x in hidden.split(","))
                kwargs['max_epochs'] = epochs
                kwargs['learning_rate_init'] = lr
                kwargs['hidden_layer_sizes'] = hidden_tuple
            except ValueError as e:
                print(f"输入格式错误: {e}，将使用默认参数")
                kwargs = {}

        model, history = train_and_save_model(
            state.X_train, state.y_train, state.scaler, **kwargs
        )

        state.model = model
        state.model_trained = True
        state.training_history = history

        # 自动评估
        metrics, (y_true, y_pred) = evaluate_model(
            model, state.X_test, state.y_test, state.scaler
        )
        state.metrics = metrics
        state.y_pred = y_pred

        print("\n✓ 模型训练与评估完成！")

    except Exception as e:
        print(f"\n✗ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()

    wait_for_enter()


def menu_evaluate():
    """菜单 3: 模型评估"""
    print("\n" + "=" * 60)
    print(">> 3. 模型评估")
    print("=" * 60)

    if not state.model_trained or state.model is None:
        print("⚠ 模型未训练或未加载，正在尝试加载已有模型...")
        from utils import load_model
        model, scaler = load_model()
        if model is None:
            print("✗ 没有可用模型，请先训练！(执行功能 2)")
            wait_for_enter()
            return
        state.model = model
        if scaler is not None:
            state.scaler = scaler
        state.model_trained = True

    try:
        from model_eval import evaluate_model
        from predictor import batch_predict, print_prediction_samples

        metrics, (y_true, y_pred) = evaluate_model(
            state.model, state.X_test, state.y_test, state.scaler
        )
        state.metrics = metrics
        state.y_pred = y_pred

        # 打印预测样本
        print_prediction_samples(y_true[:20], y_pred[:20], n_samples=10)

    except Exception as e:
        print(f"\n✗ 评估失败: {e}")

    wait_for_enter()


def menu_predict():
    """菜单 4: 输入特征预测房价"""
    print("\n" + "=" * 60)
    print(">> 4. 输入特征预测房价")
    print("=" * 60)

    if not state.model_trained or state.model is None:
        print("⚠ 模型未训练，正在尝试加载已有模型...")
        from utils import load_model
        model, _ = load_model()
        if model is None:
            print("✗ 没有可用模型，请先训练！(执行功能 2)")
            wait_for_enter()
            return
        state.model = model
        state.model_trained = True

    if not state.data_loaded or state.scaler is None:
        print("⚠ 归一化器未加载，正在尝试从文件加载...")
        from utils import load_model
        _, scaler = load_model()
        if scaler is None:
            print("✗ 无法获取归一化器，请先加载数据！(执行功能 1)")
            wait_for_enter()
            return
        state.scaler = scaler

    from predictor import predict_single

    print("\n请输入预测参数（波士顿房价 13 个特征）:")
    print("-" * 60)
    print("CRIM:   城镇人均犯罪率")
    print("ZN:     占地面积超过 25000 平方英尺的住宅用地比例")
    print("INDUS:  非零售商业用地比例")
    print("CHAS:   是否靠近查尔斯河 (1=是, 0=否)")
    print("NOX:    氮氧化物浓度 (每千万分之一)")
    print("RM:     每住宅平均房间数")
    print("AGE:    1940 年前建成的自住单位比例")
    print("DIS:    到波士顿就业中心的加权距离")
    print("RAD:    径向高速公路可达性指数")
    print("TAX:    每万美元的全额财产税率")
    print("PIRATIO: 城镇学生与教师比例")
    print("B:      城镇黑人比例 Bk × 1000")
    print("LSTAT:  低收入人口比例 (%)")
    print("-" * 60)
    print("参考范围: CRIM(0~100), ZN(0~100), INDUS(0~30), CHAS(0/1),")
    print("         NOX(0.3~0.9), RM(3~9), AGE(0~100), DIS(1~13),")
    print("         RAD(1~25), TAX(100~800), PIRATIO(12~22), B(0~400),")
    print("         LSTAT(0~40)")
    print("-" * 60)

    try:
        features = []
# 使用 config 中定义的 FEATURE_COLS
        from config import FEATURE_COLS
        feature_names_display = FEATURE_COLS

        for name in feature_names_display:
            val = input(f"  {name} = ").strip()
            if val == "":
                print(f"  [提示] 输入为空，使用默认值 0")
                features.append(0.0)
            else:
                features.append(float(val))

        pred_price = predict_single(state.model, features, state.scaler)

        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"  预测房价: ${pred_price:,.2f} (千美元)")
        print(f"  即约: {pred_price * 1000:,.0f} 美元")
        print("=" * 60)

    except ValueError as e:
        print(f"\n[错误] 输入格式不正确: {e}")
        print("请确保所有输入均为有效数字！")
    except Exception as e:
        print(f"\n[错误] 预测失败: {e}")

    wait_for_enter()


def menu_visualize():
    """菜单 5: 可视化展示"""
    print("\n" + "=" * 60)
    print(">> 5. 可视化展示")
    print("=" * 60)

    if not state.data_loaded:
        print("⚠ 请先加载数据！(执行功能 1)")
        wait_for_enter()
        return

    if not state.model_trained:
        print("⚠ 请先训练模型！(执行功能 2)")
        wait_for_enter()
        return

    try:
        from visualizer import plot_all_visualizations

        # 获取反归一化后的预测值和真实值
        from model_eval import evaluate_model
        metrics, (y_true, y_pred) = evaluate_model(
            state.model, state.X_test, state.y_test, state.scaler, verbose=False
        )

        loss_curve = state.training_history.get("loss_curve") if state.training_history else None

        # 获取原始 X (未归一化) 用于特征分布图
        # 从 scaler 反归一化得到
        n_features = state.scaler.n_features_in_
        dummy_X = np.zeros((state.X_test.shape[0], n_features))
        dummy_X[:, :-1] = state.X_test
        X_original = state.scaler.inverse_transform(dummy_X)[:, :-1]

        print("\n正在生成可视化图表...")
        plot_all_visualizations(
            state.model, y_true, y_pred, X_original,
            state.feature_names, loss_curve
        )

        print("\n✓ 所有图表已生成，请查看 output/figures/ 目录")

    except Exception as e:
        print(f"\n✗ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()

    wait_for_enter()


def menu_epoch_comparison():
    """菜单 6: 对比不同 Epoch 性能"""
    print("\n" + "=" * 60)
    print(">> 6. 对比不同 Epoch 性能")
    print("=" * 60)

    if not state.data_loaded:
        print("⚠ 请先加载数据！(执行功能 1)")
        wait_for_enter()
        return

    try:
        from epoch_comparison import compare_epochs

        # 询问是否自定义 Epoch 列表
        choice = input("\n是否使用默认 Epoch 列表 [50, 100, 200, 500]？(y/n, 默认 y): ").strip().lower()

        epoch_list = None
        if choice == 'n':
            try:
                user_input = input("请输入 Epoch 值列表 (逗号分隔，如 50,100,200,500): ")
                epoch_list = [int(x.strip()) for x in user_input.split(",")]
                print(f"使用自定义 Epoch 列表: {epoch_list}")
            except:
                print("输入格式错误，使用默认列表")

        results = compare_epochs(
            state.X_train, state.y_train,
            state.X_test, state.y_test,
            state.scaler, epoch_list
        )

        print("\n✓ Epoch 对比完成！详细结果如上所示。")

    except Exception as e:
        print(f"\n✗ Epoch 对比失败: {e}")
        import traceback
        traceback.print_exc()

    wait_for_enter()


def menu_exit():
    """菜单 7: 退出系统"""
    print("\n" + "=" * 60)
    print("感谢使用基于 BP 神经网络的波士顿房价预测系统！")
    print("=" * 60)
    sys.exit(0)


# ========== 主循环 ==========

def main():
    """主函数：显示菜单并处理用户选择"""

    # 尝试自动加载已保存的模型
    from utils import load_model
    model, scaler = load_model()
    if model is not None:
        state.model = model
        state.model_trained = True
    if scaler is not None:
        state.scaler = scaler

    # 菜单映射
    menu_actions = {
        '1': menu_load_data,
        '2': menu_train_model,
        '3': menu_evaluate,
        '4': menu_predict,
        '5': menu_visualize,
        '6': menu_epoch_comparison,
        '7': menu_exit,
    }

    while True:
        print_header()
        print_menu()

        choice = input("请输入选项 (1-7): ").strip()

        if choice in menu_actions:
            menu_actions[choice]()
        else:
            print("无效输入，请输入 1-7 之间的数字。")


if __name__ == "__main__":
    main()
