import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv  # 新增：用于保存csv文件

# ================= 配置区域 =================
# DATA_DIR_PATTERN = "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/nd-fixed/records/euler_constraint/f1/sim_*/all_agents_trajectories.json"
DATA_DIR_PATTERN = "/app/reassmble/nd-fixed/records/euler_constraint_asym/a1/sim_*/all_agents_trajectories.json"

# 纳什均衡点
NE_VECTOR = np.array(np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]])).reshape(-1, 1)

# 收敛判定阈值
# CONVERGENCE_THRESHOLD = 7e-4 
CONVERGENCE_THRESHOLD = 2.5e-3

# 输出结果的文件名
RESULT_CSV_NAME = "convergence_times.csv"
PLOT_SAVE_NAME = "Comparison_Convergence.png"

# ================= 核心逻辑 =================

def calc_state_error(data):
    """
    计算状态误差 ||x - x*||
    """
    time_steps = np.array(data['time_steps'])
    x_matrix = np.array(data['trajectories']['x'])

        # 定义 NE 点
    NE_vector = np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]])

    print(np.linalg.norm(NE_vector.flatten()))
    # 计算距离
    error_matrix = x_matrix - NE_vector[:,None,:]
    error_swapped = np.swapaxes(error_matrix, 0, 1)
    
    # 第二步：把 Agents 和 Coords 维度合并（展平）
    # 我们希望每个时间步对应一个长度为 10 的向量 (5个智能体 * 2个坐标)
    # 变换后: (2001, 10)
    error_flattened = error_swapped.reshape(error_swapped.shape[0], -1)
    
    # 第三步：沿着展平后的维度 (axis=1) 计算范数
    # 结果 shape: (2001,) -> 这是一个随时间变化的一维数组
    dist = np.linalg.norm(error_flattened, axis=1)

    return time_steps, dist

def find_convergence_time(time_steps, error_dist, threshold):
    """
    寻找收敛时间（含反弹检测）。
    """
    is_converged = False
    convergence_time = None
    
    for i in range(len(error_dist)):
        if error_dist[i] < threshold:
            # 如果当前点小于阈值，且之前标记为未收敛，则标记此刻为收敛开始
            if not is_converged:
                is_converged = True
                convergence_time = time_steps[i]
        else:
            # 如果当前点大于阈值，说明之前的收敛不稳定（反弹了），重置状态
            if is_converged:
                is_converged = False
                convergence_time = None
                
    return convergence_time if is_converged else None

def main():
    # 1. 搜索文件
    files = glob.glob(DATA_DIR_PATTERN)
    files.sort() # 排序，确保列表顺序一致
    
    if not files:
        print(f"Error: No files found in {DATA_DIR_PATTERN}")
        return

    print(f"[Comparison] Found {len(files)} simulation files. Processing...")

    # 初始化绘图
    plt.figure(figsize=(10, 6))
    colors_keys = list(mcolors.TABLEAU_COLORS.keys())
    
    # --- 新增：用于存储结果的列表 ---
    results_record = []

    # 2. 遍历文件处理
    for count, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 获取文件夹名 (sim_xx)
            dir_name = os.path.basename(os.path.dirname(file_path))
            label_name = f"{dir_name}"

            # 计算误差
            time_steps, dist = calc_state_error(data)
            
            # 计算收敛时间
            t_conv = find_convergence_time(time_steps, dist, CONVERGENCE_THRESHOLD)
            
            # --- 记录数据 ---
            # 如果 t_conv 是 None，标记为 "Failed"
            record_time = t_conv if t_conv is not None else -1 
            status_str = "Converged" if t_conv is not None else "Not Converged"
            
            results_record.append({
                "Simulation": label_name,
                "Status": status_str,
                "Convergence_Time(s)": record_time,
                "File_Path": file_path
            })

            # 打印与设置图例标签
            if t_conv is not None:
                print(f"[{label_name}] Converged to {CONVERGENCE_THRESHOLD} at t = {t_conv:.4f} s")
                plot_label = f"{label_name} ($T={t_conv:.2f}s$)"
            else:
                print(f"[{label_name}] Did not converge steadily.")
                plot_label = f"{label_name} (N/A)"

            # 绘图 (取对数)
            log_error = np.log10(dist)
            color_key = colors_keys[count % len(colors_keys)]
            plt.plot(time_steps, log_error, 
                     color=mcolors.TABLEAU_COLORS[color_key], 
                     label=plot_label,
                     linewidth=1.5,
                     alpha=0.8)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    # 3. 保存 CSV 结果文件
    if results_record:
        # 字段名
        headers = ["Simulation", "Status", "Convergence_Time(s)", "File_Path"]
        try:
            with open(RESULT_CSV_NAME, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in results_record:
                    writer.writerow(row)
            print(f"\n[Success] Convergence times saved to: {RESULT_CSV_NAME}")
        except IOError as e:
            print(f"Error saving CSV: {e}")

    # 4. 图表美化与保存
    plt.axhline(y=np.log10(CONVERGENCE_THRESHOLD), color='black', linestyle='--', alpha=0.5, label='Threshold 5e-4')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel(r'$log_{10}(||x(t) - x^*||)$', fontsize=14)
    plt.title('Nash Equilibrium Seeking Error Comparison', fontsize=16)
    plt.xlim(left=0)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(PLOT_SAVE_NAME, dpi=300)
    print(f"[Success] Plot saved to: {PLOT_SAVE_NAME}")
    plt.close()

if __name__ == "__main__":
    main()