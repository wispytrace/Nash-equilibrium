import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# ================= 配置区域 =================
DATA_DIR_PATTERN = "./records/fixed4/r_r_d5/sim_*/all_agents_trajectories.json"

# 纳什均衡点
NE_VECTOR = np.array(np.array([2.0596, 2.5142, 2.9687, 3.4232, 3.8778])).reshape(-1, 1)

# 梯度计算参数
VALUE_INDEX = np.array([5, 5.5, 6, 6.5, 7])

# 全局绘图样式
X_LIMIT = 20
Y_LIMIT_TOP = 4.5
LINE_ALPHA = 0.7
LINE_WIDTH = 1.5

# ==================== 用户自定义区域 ====================
# 在这里修改每个图的：
# 1. theoretical_T: 理论收敛时间
# 2. t_label: 竖线在图例中显示的标签 (支持 LaTeX)
# 3. color: 线条颜色 (R, G, B) 元组，范围 0-1
# =======================================================
PLOT_CONFIGS = {
    "Verification_NE": {
        "theoretical_T": 1.572*2 + 7.478,  # 修改这里的数值
        "t_label": '$T=T_1+T_2+T_3=10.62(s)$',            # 修改这里的标签
        "color": (1.0, 0.0, 0.0)          # 红色
    },
    "Verification_Z_Consensus": {
        "theoretical_T": 1.57,            # 修改这里的数值
        "t_label": '$T_1=1.57(s)$',     # 修改这里的标签
        "color": (0.0, 0.6, 0.0)          # 绿色
    },
    "Verification_V_Gradient": {
        "theoretical_T": 1.57*2,            # 修改这里的数值
        "t_label": '$T_1+T_2=3.14(s)$',      # 修改这里的标签
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
                        theoretical_T, t_label, line_color_rgb, xlim=None, zomm_config=None):
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
    
    if zomm_config:
        bounds = [0.5, 0.1, 0.4, 0.3] 
        axins = ax.inset_axes(bounds)

        # 设置放大框所关注的数据范围
        axins.set_xlim(0,5)
        axins.set_ylim(-4, 0)
        
        # 设置放大框的刻度字体大小，避免太挤
        axins.tick_params(axis='both', labelsize=10)

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
    ax.set_xlabel('Time(s)', fontsize=14)
    ax.set_ylabel(y_label_latex, fontsize=14)
    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)
    else:
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

def plot_multi_file_comparison(file_list, save_name="Comparison_NE_Error.png"):
    """
    比较多个轨迹文件的 NE 误差收敛情况
    样式复刻：使用 TABLEAU_COLORS，并计算 5e-4 收敛时间
    """
    if not file_list:
        print("Error: No files provided for comparison.")
        return

    print(f"[Comparison] Plotting {len(file_list)} files...")

    # === 1. 清除当前画布 (参考 plt.clf()) ===
    plt.clf()
    # 使用大一点的图幅，或者你可以去掉 figsize 使用默认
    plt.figure() 

    # === 2. 颜色设置 (参考 mcolors.TABLEAU_COLORS) ===
    colors_keys = list(mcolors.TABLEAU_COLORS.keys())
    
    # 收敛阈值
    THRESHOLD = 5e-4

    for count, file_path in enumerate(file_list):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 提取标签：使用文件夹名称 (例如 sim_01)
            dir_name = os.path.basename(os.path.dirname(file_path))
            label_name = f"Set {count+1}"

            # === 3. 数据处理与收敛检测 ===
            # 复用之前的计算函数获取 time 和 error_norm
            time_steps, dist = _calc_state_error(data)
            
            # 参考代码的收敛检测逻辑 (is_cite)
            # 逻辑：寻找首次进入阈值且稳定的时间点。如果后续反弹超出阈值，之前的记录会被视为无效。
            is_cite = False
            convergence_time = None
            
            for i in range(len(dist)):
                if dist[i] < THRESHOLD:
                    if not is_cite:
                        is_cite = True
                        convergence_time = time_steps[i]
                        # 打印第一次进入该区域的时间
                        print(f"[{label_name}] Reached {THRESHOLD} at t = {convergence_time:.4f} s")
                else:
                    # 如果误差反弹回阈值以上，重置状态
                    if is_cite:
                        # 你可以选择是否取消之前的打印，或者只打印“反弹了”
                        # 这里严格按照参考代码逻辑：del is_cite[config_index] 意味着之前的收敛不算数
                        is_cite = False 
                        convergence_time = None 

            # 取对数用于绘图
            log_error = np.log10(dist)

            # === 4. 绘图 (参考 plt.plot) ===
            color_key = colors_keys[count % len(colors_keys)]
            plt.plot(time_steps, log_error, 
                     color=mcolors.TABLEAU_COLORS[color_key], 
                     label=label_name,
                     linewidth=1.3) # 可选：保持一定的线宽
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            print(f"Skipping {file_path}: {e}")

    # === 5. 样式设置 (完全复刻参考代码) ===
    # plt.xlim(timescale[0], timescale[1]) # 如果你需要限制X轴范围，可以在配置里加，这里暂时用默认
    plt.xlim(left=0, right=4) # 保持你原本的 X_LIMIT
    
    plt.xlabel('Time(s)', fontsize=15)
    plt.ylabel("$log_{10}(||x - x*||)$", fontsize=15)
    
    # 图例设置
    plt.legend(loc='upper right', fontsize=12)
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Saved comparison plot: {save_name}\n")
    plt.close()

def plot_status_graph_from_file(file_path):
    """
    读取轨迹文件并绘制状态曲线 (x vs time)
    完全复刻参考代码的样式
    """
    # 1. 读取数据
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取时间轴
    time_steps = np.array(data['time_steps'])
    
    # 提取状态向量 x (假设结构为 [agent][time][dim])
    # squeeze(-1) 是为了把 (N, T, 1) 变成 (N, T)
    status_vector = np.array(data['trajectories']['x']).squeeze(-1)
    
    # 2. 定义 NE 目标值 (根据你之前的上下文硬编码，或者你可以作为参数传入)
    opt_value = [2.0596, 2.5142, 2.9687, 3.4232, 3.8778]
    
    # 3. 绘图逻辑 (复刻参考代码)
    plt.clf()
    plt.figure() # 稍微设置一个默认大小

    colors = list(mcolors.TABLEAU_COLORS.keys())
    num_agents = len(status_vector)

    # 这里的 self.charater 替换为默认的下标生成逻辑 (1, 2, 3...)
    # 假设 self.charater[i+1] 对应的是 index i+1
    
    for i in range(num_agents):
        # 颜色逻辑：
        # 实线 (实际轨迹) 使用 colors列表的 2*i
        # 虚线 (目标值) 使用 colors列表的 2*i + 1
        # 加上 % len(colors) 防止越界
        color_idx_traj = (2 * i) % len(colors)
        color_idx_opt = (2 * i + 1) % len(colors)

        # 绘制实际轨迹
        plt.plot(time_steps, status_vector[i],
                 color=mcolors.TABLEAU_COLORS[colors[color_idx_traj]], 
                 label="x$_{" + str(i+1) + "}$") # LaTeX 下标格式
        
        # 绘制目标值虚线
        plt.plot(time_steps, [opt_value[i]] * len(time_steps), '--', 
                 color=mcolors.TABLEAU_COLORS[colors[color_idx_opt]], 
                 label="x$_{" + str(i+1) + "}^*$") # LaTeX 下标格式

    # 4. 样式设置 (保持原样)
    # 注意：如果你的仿真时间很长，这里原本的 limit(0, 2.5) 可能需要调整
    # 这里我保留原代码的 2.5，或者你可以改为 max(time_steps)
    plt.xlim(0, 2.5) 
    
    # Y轴范围自动适应 opt_value
    plt.ylim(0, max(opt_value) + 4)
    
    plt.legend(loc='upper right', ncol=2, fontsize=12)
    plt.xlabel('Time(s)', fontsize=15)
    plt.ylabel('Action', fontsize=15)

    # 5. 保存
    # 自动保存到文件所在的目录
    save_dir = os.path.dirname(file_path)
    save_name = os.path.join(save_dir, "status_reproduced.png")
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"save to {save_name}")
    plt.close()

def plot_ui_graph_from_file(file_path):
    """
    读取轨迹文件并绘制控制输入曲线 (u vs time)
    复刻 plot_ui_graph 的样式
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. 提取数据
    time_steps = np.array(data['time_steps'])
    
    # 检查是否存在 'u' 数据
    if 'u' not in data['trajectories']:
        print(f"Error: Key 'u' not found in {file_path}. Please ensure simulation records 'u'.")
        return

    # 提取控制输入 u
    # 假设结构为 [agent][time][dim] -> (N, T)
    ui_vector = np.array(data['trajectories']['u']).squeeze(-1)
    
    num_agents = len(ui_vector)

    # 2. 绘图初始化
    plt.clf()
    plt.figure() # 使用默认大小

    colors = list(mcolors.TABLEAU_COLORS.keys())

    # 3. 循环绘图
    for i in range(num_agents):
        # 颜色逻辑复刻：colors[2*i]
        # 添加取余保护防止越界
        color_idx = (2 * i) % len(colors)
        
        plt.plot(time_steps, ui_vector[i],
                 color=mcolors.TABLEAU_COLORS[colors[color_idx]], 
                 label="u$_{" + str(i+1) + "}$") # LaTeX 下标 u_1, u_2...

    # 4. 样式复刻
    plt.xlim(0, 2.5)
    # plt.ylim(bottom=0) # 原代码注释掉了，这里也保持注释
    
    plt.legend(loc='upper right', fontsize=12)
    plt.xlabel('Time(s)', fontsize=15)
    plt.ylabel('Control input', fontsize=15)

    # 5. 保存
    save_dir = os.path.dirname(file_path)
    save_name = os.path.join(save_dir, "status_update.png")
    
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"save to {save_name}")
    plt.close()
# ================= 主程序 =================

if __name__ == "__main__":
    plot_status_graph_from_file("/app/reassmble/tcns-fixed/records/fixed4/r_r_d5/sim_101/all_agents_trajectories.json")
    plot_ui_graph_from_file("/app/reassmble/tcns-fixed/records/fixed4/r_r_d5/sim_101/all_agents_trajectories.json")
    # plot_multi_file_comparison(["/app/reassmble/tcns-fixed/records/fixed4/r_r_d1/sim_101/all_agents_trajectories.json", "/app/reassmble/tcns-fixed/records/fixed4/r_r_d2/sim_101/all_agents_trajectories.json", "/app/reassmble/tcns-fixed/records/fixed4/r_r_d3/sim_101/all_agents_trajectories.json"
    #                             ,"/app/reassmble/tcns-fixed/records/fixed4/r_r_d4/sim_101/all_agents_trajectories.json"])
    # 1. 纳什均衡误差图
    cfg_ne = PLOT_CONFIGS["Verification_NE"]
    plot_generic_metric(
        metric_name="Verification_NE", 
        y_label_latex=r'$\log_{10}(\||x - x^*\||)$', 
        calc_error_func=_calc_state_error,
        theoretical_T=cfg_ne["theoretical_T"],
        t_label=cfg_ne["t_label"],
        line_color_rgb=cfg_ne["color"],
        xlim=15
    )

    # 2. 一致性误差图
    cfg_z = PLOT_CONFIGS["Verification_Z_Consensus"]
    plot_generic_metric(
        metric_name="Verification_Z_Consensus", 
        y_label_latex=r'$\log_{10}(\||z - 1_N \otimes x\||)$', 
        calc_error_func=_calc_consensus_error,
        theoretical_T=cfg_z["theoretical_T"],
        t_label=cfg_z["t_label"],
        line_color_rgb=cfg_z["color"],
        xlim = 5
    )

    # 3. 梯度跟踪误差图
    cfg_v = PLOT_CONFIGS["Verification_V_Gradient"]
    plot_generic_metric(
        metric_name="Verification_V_Gradient", 
        y_label_latex=r'$\log_{10}(\||v - 1_N \otimes F(x)\||)$', 
        calc_error_func=_calc_gradient_error,
        theoretical_T=cfg_v["theoretical_T"],
        t_label=cfg_v["t_label"],
        line_color_rgb=cfg_v["color"],
        xlim=5
    )