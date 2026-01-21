import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch, Rectangle

# ================= 配置区域 =================
DATA_DIR_PATTERN = "./records/fixed4/r_r/sim_*/all_agents_trajectories.json"

# 纳什均衡点
NE_VECTOR = np.array([2.06, 2.51, 2.97, 3.42, 3.88]).reshape(-1, 1)

# 梯度计算参数
VALUE_INDEX = np.array([5, 5.5, 6, 6.5, 7])

# 理论时间
THEORETICAL_T_MAX = 38.34

# 绘图配置
X_LIMIT = 50
Y_LIMIT_TOP = 4.5
LINE_COLOR = 'blue'
LINE_ALPHA = 0.3
LINE_WIDTH = 1.0

# 插图 (Inset) 配置
INSET_X_RANGE = (0, 10)
# 插图位置 [x, y, width, height]
INSET_POSITION = [0.25, 0.35, 0.4, 0.4]

# ================= 核心计算逻辑 =================

def calculate_gradient_vectorized(x_t):
    sum_x = np.sum(x_t, axis=1, keepdims=True)
    grad = 0.1 * sum_x + 1.25 + 1.1 * x_t - VALUE_INDEX
    return grad

# ================= 通用绘图函数 =================

def plot_generic_metric(metric_name, y_label_latex, calc_error_func):
    files = glob.glob(DATA_DIR_PATTERN)
    if not files:
        print(f"Error: No files found in {DATA_DIR_PATTERN}")
        return

    print(f"[{metric_name}] Found {len(files)} files. Plotting...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # === 创建局部放大子图 ===
    axins = ax.inset_axes(INSET_POSITION)

    valid_count = 0
    y_min_inset = float('inf')
    y_max_inset = float('-inf')

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            time_steps, error_norm = calc_error_func(data)
            log_error = np.log10(error_norm)
            
            # 主图绘制
            ax.plot(time_steps, log_error, 
                     color=LINE_COLOR, linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
            
            # 插图绘制
            axins.plot(time_steps, log_error, 
                       color=LINE_COLOR, linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
            
            # 统计极值
            mask = (time_steps >= INSET_X_RANGE[0]) & (time_steps <= INSET_X_RANGE[1])
            if np.any(mask):
                local_data = log_error[mask]
                y_min_inset = min(y_min_inset, np.min(local_data))
                y_max_inset = max(y_max_inset, np.max(local_data))

            valid_count += 1
            
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    # === 主图样式 ===
    ax.axvline(x=THEORETICAL_T_MAX, color='red', linestyle='--', linewidth=2.5, label='$T_{max}$')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(y_label_latex, fontsize=14)
    ax.set_xlim(left=0, right=X_LIMIT)
    ax.set_ylim(top=Y_LIMIT_TOP)
    
    # 图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12, loc='center right')

    # === 插图样式 ===
    axins.set_xlim(INSET_X_RANGE[0], INSET_X_RANGE[1])
    if valid_count > 0:
        y_margin = (y_max_inset - y_min_inset) * 0.1
        axins.set_ylim(y_min_inset - y_margin, y_max_inset + y_margin)
    axins.tick_params(axis='both', which='major', labelsize=10)

    # === 绘制指引箭头和框 (关键修改) ===
    
    xlims = axins.get_xlim()
    ylims = axins.get_ylim()

    # # 1. 主图上的矩形框：改为实线 '-'
    # rect = Rectangle((xlims[0], ylims[0]), xlims[1]-xlims[0], ylims[1]-ylims[0],
    #                  linewidth=1.2, edgecolor='black', facecolor='none', linestyle='-') # <--- 这里改成了实线
    # ax.add_patch(rect)

    # 2. 箭头配置：实线箭头
    # arrow_props = dict(arrowstyle="->", linestyle="-", color="black", lw=1.2, mutation_scale=15)

    # # 箭头 1 (左上)
    # con1 = ConnectionPatch(xyA=(xlims[0], ylims[1]), coordsA=ax.transData,
    #                        xyB=(0, 1), coordsB=axins.transAxes,
    #                        **arrow_props)
    # fig.add_artist(con1)

    # # 箭头 2 (右下)
    # con2 = ConnectionPatch(xyA=(xlims[1], ylims[0]), coordsA=ax.transData,
    #                        xyB=(1, 0), coordsB=axins.transAxes,
    #                        **arrow_props)
    # fig.add_artist(con2)

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
    plot_generic_metric("Verification_NE", r'$\log_{10}(\|x(t) - x^*\|)$', _calc_state_error)
    plot_generic_metric("Verification_Z_Consensus", r'$\log_{10}(\|z - \mathbf{1}_N \otimes x\|)$', _calc_consensus_error)
    plot_generic_metric("Verification_V_Gradient", r'$\log_{10}(\|v - \mathbf{1}_N \otimes \nabla F(x)\|)$', _calc_gradient_error)