import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

# ================= 配置区域 =================
DATA_DIR_PATTERN = "./records/fixed4/r_r/sim_*/all_agents_trajectories.json"
ASYM_DATA_PATTER = "./records/asym/r_a/sim_*/all_agents_trajectories.json"
# 纳什均衡点
NE_VECTOR = np.array([2.06, 2.51, 2.97, 3.42, 3.88]).reshape(-1, 1)

# 梯度计算参数
VALUE_INDEX = np.array([5, 5.5, 6, 6.5, 7])

# 全局绘图样式
X_LIMIT = 40
Y_LIMIT_TOP = 4.5
LINE_ALPHA = 0.4
LINE_WIDTH = 1.5

# ==================== 用户自定义区域 ====================
# 在这里修改每个图的：
# 1. theoretical_T: 理论收敛时间
# 2. t_label: 竖线在图例中显示的标签 (支持 LaTeX)
# 3. color: 线条颜色 (R, G, B) 元组，范围 0-1
# =======================================================
PLOT_CONFIGS = {
    "Verification_NE": {
        "theoretical_T": 8.2*2 + 13.988,  # 修改这里的数值
        "t_label": '$T=30.39s$',            # 修改这里的标签
        "color": (1.0, 0.0, 0.0)          # 红色
    },
    "Verification_Z_Consensus": {
        "theoretical_T": 8.2,            # 修改这里的数值
        "t_label": '$T_1=8.2s$',     # 修改这里的标签
        "color": (0.0, 1.0, 0.0)          # 绿色
    },
    "Verification_V_Gradient": {
        "theoretical_T": 8.2*2,            # 修改这里的数值
        "t_label": '$T_1+T_2=16.4s$',      # 修改这里的标签
        "color": (0.0, 0.0, 1.0)          # 蓝色
    }
}

# ================= 核心计算逻辑 =================

def calculate_gradient_vectorized(x_t):
    sum_x = np.sum(x_t, axis=1, keepdims=True)
    grad = 0.1 * sum_x + 1.25 + 1.1 * x_t - VALUE_INDEX
    return grad

# ================= 通用绘图函数 =================

def plot_generic_metric(metric_name, y_label_latex, calc_error_func, 
                        theoretical_T, t_label, line_color_rgb):
    """
    通用绘图函数，支持自定义颜色和理论时间T，并正确显示图例
    """
    files = glob.glob(DATA_DIR_PATTERN)
    files_asym = glob.glob(ASYM_DATA_PATTER)

    if not files and not files_asym:
        print(f"Error: No files found in {DATA_DIR_PATTERN} or {ASYM_DATA_PATTER}")
        return

    print(f"[{metric_name}] Found fixed={len(files)}, asym={len(files_asym)}. Plotting with T={theoretical_T:.2f}...")

    fig, ax = plt.subplots()

    # 只让第一条曲线进入 legend，避免重复
    fixed_labeled = False
    asym_labeled = False

    # ------- Asymptotic 算法 -------
    for file_path in files_asym:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            time_steps, error_norm = calc_error_func(data)
            log_error = np.log10(error_norm + 1e-12)

            ax.plot(
                time_steps, log_error,
                color=(0.0, 0.0, 1.0),         # 你也可以改成传参
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
                label="Asymptotic algorithm [24]" if not asym_labeled else None
            )
            asym_labeled = True

        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    # ------- Fixed-time 算法 -------
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            time_steps, error_norm = calc_error_func(data)
            log_error = np.log10(error_norm + 1e-12)  # 防止 log10(0)

            ax.plot(
                time_steps, log_error,
                color=line_color_rgb,          # 使用传入颜色
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
                label="Fixed-time algorithm (6)-(11)" if not fixed_labeled else None
            )
            fixed_labeled = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Skipping {file_path}: {e}")


    # ------- 理论时间竖线 -------
    if theoretical_T is not None:
        ax.axvline(
            x=theoretical_T,
            color='black',
            linestyle='--',
            linewidth=2.5,
            label=t_label
        )

    # ------- 图表样式 -------
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(y_label_latex, fontsize=14)
    ax.set_xlim(left=0, right=X_LIMIT)
    ax.set_ylim(top=Y_LIMIT_TOP)

    # ------- 图例：直接 legend 即可（因为我们已避免重复） -------
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    save_name = f"Figure_{metric_name}.png"
    plt.savefig(save_name, dpi=300)
    print(f"Saved: {save_name}\n")
    plt.close()

# ================= 各个指标计算回调 =================

def _calc_state_error(data):
    time_steps = np.array(data['time_steps'])
    x_matrix = np.array(data['trajectories']['x']).squeeze(-1)
    error_matrix = x_matrix - NE_VECTOR
    dist = np.linalg.norm(error_matrix, axis=0)
    return time_steps, dist

def _calc_consensus_error(data):
    time_steps = np.array(data['time_steps'])
    x_matrix = np.array(data['trajectories']['x']).squeeze(-1)
    z_tensor = np.array(data['trajectories']['z'])
    true_state_at_t = x_matrix.T
    z_error_tensor = z_tensor - true_state_at_t[np.newaxis, :, :]
    dist = np.linalg.norm(z_error_tensor, axis=(0, 2))
    return time_steps, dist

def _calc_gradient_error(data):
    time_steps = np.array(data['time_steps'])
    x_matrix = np.array(data['trajectories']['x']).squeeze(-1)
    v_tensor = np.array(data['trajectories']['v'])
    true_state_at_t = x_matrix.T
    true_gradients = calculate_gradient_vectorized(true_state_at_t)
    v_error_tensor = v_tensor - true_gradients[np.newaxis, :, :]
    dist = np.linalg.norm(v_error_tensor, axis=(0, 2))
    return time_steps, dist

# ================= 主程序 =================

if __name__ == "__main__":
    
    # 1. 纳什均衡误差图
    cfg_ne = PLOT_CONFIGS["Verification_NE"]
    plot_generic_metric(
        metric_name="Verification_Merge_NE", 
        y_label_latex=r'$\log_{10}(\|x - x^*\|)$', 
        calc_error_func=_calc_state_error,
        theoretical_T=cfg_ne["theoretical_T"],
        t_label=cfg_ne["t_label"],
        line_color_rgb=cfg_ne["color"]
    )
