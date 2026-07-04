"""
波士顿房价预测系统 - Tkinter GUI 界面
支持：批量粘贴预测 / 逐条输入预测 / 模型管理 / 可视化
"""
import sys, os, warnings, threading
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np

from config import FEATURE_COLS, FEATURE_NAMES_DISPLAY, FIGURES_DIR
from data_loader import prepare_datasets
from model_train import train_and_save_model
from model_eval import evaluate_model
from predictor import predict_single, predict_bulk, batch_predict
from visualizer import plot_all_visualizations
from epoch_comparison import compare_epochs
from utils import load_model


class AppState:
    def __init__(self):
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = self.feature_names = None
        self.model = None
        self.data_loaded = False
        self.model_trained = False
        self.y_pred = self.y_true_orig = None

state = AppState()


class BostonHousingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("波士顿房价预测系统")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self._build_toolbar()
        self._build_statusbar()
        self._build_main_area()
        self._update_status()
        self._try_load_saved_model()

    # ==== Toolbar ====
    def _build_toolbar(self):
        frame = ttk.Frame(self.root, padding=5)
        frame.pack(fill="x")
        actions = [
            ("加载数据", self._load_data),
            ("训练模型", self._train_model),
            ("模型评估", self._evaluate),
            ("可视化展示", self._visualize),
            ("Epoch对比", self._epoch_compare),
        ]
        for text, cmd in actions:
            ttk.Button(frame, text=text, command=cmd).pack(side="left", padx=3)

    # ==== Status Bar ====
    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="就绪")
        bar = ttk.Label(self.root, textvariable=self._status_var,
                        relief="sunken", anchor="w", padding=3)
        bar.pack(fill="x", padx=5)

    def _update_status(self):
        data_s = "已加载" if state.data_loaded else "未加载"
        model_s = "已训练" if state.model_trained else "未训练"
        self._status_var.set(f"数据: {data_s} | 模型: {model_s}")

    # ==== Main Area (Notebook) ====
    def _build_main_area(self):
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self._build_batch_tab()
        self._build_sequential_tab()
        self._build_log_tab()

    # ---- Tab 1: 批量预测 ----
    def _build_batch_tab(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="批量预测")

        lbl = ttk.Label(tab, text="将数据粘贴到下方，每行一个样本，"
                        "用Tab/逗号/空格分隔13个特征值:",
                        font=("", 10))
        lbl.pack(anchor="w", padx=5, pady=5)

        self._batch_text = scrolledtext.ScrolledText(tab, height=8, font=("Consolas", 9))
        self._batch_text.pack(fill="both", expand=True, padx=5)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="从剪贴板粘贴",
                   command=self._paste_clipboard).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="清空", command=lambda: self._batch_text.delete("1.0", "end")).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="开始预测", command=self._batch_predict).pack(side="right", padx=2)

        # 预测结果 table
        tree_frame = ttk.Frame(tab)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        cols = ("#",) + tuple(FEATURE_COLS[:4]) + ("...", "Price($K)", "折合金额")
        self._batch_tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                                         height=8, selectmode="browse")
        for c in cols:
            w = 60 if c in FEATURE_COLS[:4] else 40 if c == "#" else 100
            self._batch_tree.heading(c, text=c)
            self._batch_tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._batch_tree.yview)
        self._batch_tree.configure(yscrollcommand=vsb.set)
        self._batch_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    # ---- Tab 2: 逐条输入 ----
    def _build_sequential_tab(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="逐条输入")

        self._seq_entries = {}
        input_frame = ttk.LabelFrame(tab, text="输入13个特征", padding=10)
        input_frame.pack(fill="x", padx=5, pady=5)

        for i, name in enumerate(FEATURE_COLS):
            lbl = ttk.Label(input_frame, text=name)
            lbl.grid(row=i//4, column=(i%4)*2, sticky="w", padx=5)
            ent = ttk.Entry(input_frame, width=14)
            ent.grid(row=i//4, column=(i%4)*2+1, padx=5, pady=2)
            self._seq_entries[name] = ent

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_frame, text="加入队列", command=self._add_to_queue).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="清空 Queue", command=self._clear_queue).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="开始预测", command=self._predict_sequential_queue).pack(side="right", padx=2)

        self._queue_label = ttk.Label(tab, text="队列: 0 个样本", font=("", 9))
        self._queue_label.pack(anchor="w", padx=5)

        # Queue list & results
        paned = ttk.PanedWindow(tab, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=5, pady=5)

        q_frame = ttk.LabelFrame(paned, text="待预测队列", padding=5)
        self._queue_list = tk.Listbox(q_frame, height=6, font=("Consolas", 8))
        self._queue_list.pack(fill="both", expand=True)
        paned.add(q_frame, weight=1)

        r_frame = ttk.LabelFrame(paned, text="预测结果", padding=5)
        cols2 = ("#", "CRIM", "RM", "Price($K)", "折合金额")
        self._seq_tree = ttk.Treeview(r_frame, columns=cols2, show="headings", height=6)
        for c in cols2:
            self._seq_tree.heading(c, text=c)
            self._seq_tree.column(c, width=70 if c in ("CRIM","RM") else 40 if c=="#" else 100)
        self._seq_tree.pack(fill="both", expand=True)
        paned.add(r_frame, weight=2)

        self._seq_queue = []

    # ---- Tab 3: Log ----
    def _build_log_tab(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="系统日志")

        self._log_text = scrolledtext.ScrolledText(tab, height=10, font=("Consolas", 9),
                                                    state="disabled")
        self._log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def log(self, msg):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.root.update_idletasks()

    # ==== Clipboard ====
    def _paste_clipboard(self):
        try:
            text = self.root.clipboard_get()
            self._batch_text.insert("end", text + "\n")
        except:
            messagebox.showwarning("Clipboard", "剪贴板中没有文本。")

    # ==== Core operations ====
    def _try_load_saved_model(self):
        model, scaler = load_model()
        if model is not None:
            state.model = model
            state.model_trained = True
            state.scaler = scaler
            self._update_status()
            self.log("已加载 saved model from disk.")

    def _load_data(self):
        def task():
            try:
                r = prepare_datasets()
                state.X_train, state.X_test = r[0], r[1]
                state.y_train, state.y_test = r[2], r[3]
                state.scaler, state.feature_names = r[4], r[5]
                state.data_loaded = True
                nf = state.scaler.n_features_in_
                dummy = np.zeros((len(state.y_test), nf))
                dummy[:, -1] = state.y_test
                state.y_true_orig = state.scaler.inverse_transform(dummy)[:, -1]
            except Exception as e:
                self.log(f"加载失败: {e}")
            self._update_status()
        threading.Thread(target=task, daemon=True).start()

    def _train_model(self):
        if not state.data_loaded:
            messagebox.showwarning("提示", "请先加载数据！")
            return
        def task():
            try:
                state.model, hist = train_and_save_model(
                    state.X_train, state.y_train, state.scaler)
                state.model_trained = True
                self.log(f"已训练: {hist['n_iter']} 轮, 损失={hist['loss']:.4f}")
                metrics, (yt, yp) = evaluate_model(
                    state.model, state.X_test, state.y_test, state.scaler)
                self.log(f"R^2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
            except Exception as e:
                self.log(f"训练出错: {e}")
            self._update_status()
        threading.Thread(target=task, daemon=True).start()

    def _evaluate(self):
        if not state.model_trained:
            messagebox.showwarning("提示", "请先训练或加载模型！")
            return
        try:
            metrics, (yt, yp) = evaluate_model(
                state.model, state.X_test, state.y_test, state.scaler)
            msg = (f"MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, "
                   f"RMSE={metrics['RMSE']:.4f}, R^2={metrics['R2']:.4f}, "
                   f"MAPE={metrics['MAPE']:.2f}%")
            self.log(msg)
            messagebox.showinfo("评估结果", msg)
        except Exception as e:
            self.log(f"评估出错: {e}")

    def _visualize(self):
        if not state.model_trained:
            messagebox.showwarning("提示", "请先训练模型！")
            return
        def task():
            try:
                metrics, (yt, yp) = evaluate_model(
                    state.model, state.X_test, state.y_test, state.scaler, verbose=False)
                nf = state.scaler.n_features_in_
                dummy_x = np.zeros((state.X_test.shape[0], nf))
                dummy_x[:, :-1] = state.X_test
                x_orig = state.scaler.inverse_transform(dummy_x)[:, :-1]
                plot_all_visualizations(state.model, yt, yp, x_orig,
                    state.feature_names, None)
                self.log("图表已保存至 output/figures/ 目录")
            except Exception as e:
                self.log(f"可视化出错: {e}")
        threading.Thread(target=task, daemon=True).start()

    def _epoch_compare(self):
        if not state.data_loaded:
            messagebox.showwarning("提示", "请先加载数据！")
            return
        def task():
            try:
                compare_epochs(state.X_train, state.y_train,
                               state.X_test, state.y_test,
                               state.scaler, verbose=True)
                self.log("Epoch对比完成。")
            except Exception as e:
                self.log(f"Epoch对比出错: {e}")
        threading.Thread(target=task, daemon=True).start()

    # ==== 批量预测 ====
    def _parse_pasted_data(self, text):
        """解析粘贴的文本，返回特征列表"""
        rows = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            parts = [p for p in parts if p]
            if len(parts) < 13:
                continue
            try:
                feats = [float(p) for p in parts[:13]]
                rows.append(feats)
            except:
                continue
        return rows

    def _batch_predict(self):
        raw = self._batch_text.get("1.0", "end").strip()
        if not raw:
            messagebox.showwarning("提示", "请先粘贴数据！")
            return
        if not state.model_trained:
            messagebox.showwarning("提示", "请先训练模型！")
            return
        rows = self._parse_pasted_data(raw)
        if not rows:
            self.log("未找到有效的数值行。")
            return
        self.log(f"正在预测 {len(rows)} 个样本...")
        try:
            prices = predict_bulk(rows, state.model, state.scaler)
            # 清空 tree
            for item in self._batch_tree.get_children():
                self._batch_tree.delete(item)
            for i, (feats, price) in enumerate(zip(rows, prices)):
                vals = (i+1, feats[0], feats[1], feats[2], feats[3],
                        "...", f"{price:.2f}", f"${price*1000:.0f}")
                self._batch_tree.insert("", "end", values=vals)
            self.log(f"预测完成 {len(rows)} houses.")
        except Exception as e:
            self.log(f"批量预测出错: {e}")

    # ==== 逐条输入 ====
    def _add_to_queue(self):
        feats = []
        for name in FEATURE_COLS:
            val = self._seq_entries[name].get().strip()
            if not val:
                messagebox.showwarning("提示", f"缺少: {name}")
                return
            try:
                feats.append(float(val))
            except:
                messagebox.showwarning("提示", f"无效: {name}={val}")
                return
        self._seq_queue.append(feats)
        self._queue_list.insert("end", f"#{len(self._seq_queue)}: " +
                                " ".join(f"{v:8.4f}" for v in feats[:4]) + " ...")
        self._queue_label.config(text=f"Queue: {len(self._seq_queue)} items")
        for ent in self._seq_entries.values():
            ent.delete(0, "end")

    def _clear_queue(self):
        self._seq_queue.clear()
        self._queue_list.delete(0, "end")
        self._queue_label.config(text="队列: 0 个样本")
        for item in self._seq_tree.get_children():
            self._seq_tree.delete(item)

    def _predict_sequential_queue(self):
        if not self._seq_queue:
            messagebox.showwarning("提示", "队列为空！")
            return
        if not state.model_trained:
            messagebox.showwarning("提示", "请先训练模型！")
            return
        try:
            prices = predict_bulk(self._seq_queue, state.model, state.scaler)
            for item in self._seq_tree.get_children():
                self._seq_tree.delete(item)
            for i, (feats, price) in enumerate(zip(self._seq_queue, prices)):
                self._seq_tree.insert("", "end", values=(
                    i+1, feats[0], feats[5], f"{price:.2f}", f"${price*1000:.0f}"))
            self.log(f"逐条预测完成: {len(prices)} houses.")
        except Exception as e:
            self.log(f"逐条预测出错: {e}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BostonHousingGUI()
    app.run()
