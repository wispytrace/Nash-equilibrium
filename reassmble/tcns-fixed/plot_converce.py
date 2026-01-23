import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

# ================= 配置区域 =================
DATA_DIR_PATTERN = "./records/fixed4/r_r_d/sim_*/all_agents_trajectories.json"

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
        "t_label": '$T=33.26(s)$',            # 修改这里的标签
        "color": (1.0, 0.0, 0.0)          # 红色
    },
    "Verification_Z_Consensus": {
        "theoretical_T": 8.2,            # 修改这里的数值
        "t_label": '$T_1=7.3(s)$',     # 修改这里的标签
        "color": (0.0, 0.6, 0.0)          # 绿色
    },
    "Verification_V_Gradient": {
        "theoretical_T": 7.3*2,            # 修改这里的数值
        "t_label": '$T_1+T_2=14.6(s)$',      # 修改这里的标签
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
    通用绘图函数，支持自定义颜色和理论时间T
    """
    files = glob.glob(DATA_DIR_PATTERN)
    if not files:
        print(f"Error: No files found in {DATA_DIR_PATTERN}")
        return

    print(f"[{metric_name}] Found {len(files)} files. Plotting with T={theoretical_T:.2f}...")

    fig, ax = plt.subplots()

    valid_count = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 计算误差
            time_steps, error_norm = calc_error_func(data)
            # 取对数
            log_error = np.log10(error_norm)
            
            # 绘制线条
            ax.plot(time_steps, log_error, 
                    color=line_color_rgb,  # 使用传入的 RGB 颜色
                    linewidth=LINE_WIDTH, 
                    alpha=LINE_ALPHA)
            
            valid_count += 1
            
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    # === 绘制理论时间竖线 ===
    if theoretical_T is not None:
        ax.axvline(x=theoretical_T, color='black', linestyle='--', 
                   linewidth=2.5, label=t_label)

    # === 图表样式设置 ===
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(y_label_latex, fontsize=14)
    ax.set_xlim(left=0, right=X_LIMIT)
    ax.set_ylim(top=Y_LIMIT_TOP)
    
    # === 图例处理 (去重) ===
    # 获取所有图柄和标签
    handles, labels = ax.get_legend_handles_labels()
    # 使用字典去重 (因为有多条仿真线，但只需要显示T的图例，
    # 如果想显示仿真线的图例，可以在 ax.plot 中添加 label，但通常不需要)
    by_label = dict(zip(labels, handles))
    
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='upper right')

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
        metric_name="Verification_NE", 
        y_label_latex=r'$\log_{10}(\|x(t) - x^*\|)$', 
        calc_error_func=_calc_state_error,
        theoretical_T=cfg_ne["theoretical_T"],
        t_label=cfg_ne["t_label"],
        line_color_rgb=cfg_ne["color"]
    )

    # 2. 一致性误差图
    cfg_z = PLOT_CONFIGS["Verification_Z_Consensus"]
    plot_generic_metric(
        metric_name="Verification_Z_Consensus", 
        y_label_latex=r'$\log_{10}(\|z - \mathbf{1}_N \otimes x\|)$', 
        calc_error_func=_calc_consensus_error,
        theoretical_T=cfg_z["theoretical_T"],
        t_label=cfg_z["t_label"],
        line_color_rgb=cfg_z["color"]
    )

    # 3. 梯度跟踪误差图
    cfg_v = PLOT_CONFIGS["Verification_V_Gradient"]
    plot_generic_metric(
        metric_name="Verification_V_Gradient", 
        y_label_latex=r'$\log_{10}(\|v - \mathbf{1}_N \otimes \nabla F(x)\|)$', 
        calc_error_func=_calc_gradient_error,
        theoretical_T=cfg_v["theoretical_T"],
        t_label=cfg_v["t_label"],
        line_color_rgb=cfg_v["color"]
    )