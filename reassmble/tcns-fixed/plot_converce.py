import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

# ================= 配置区域 =================
# 1. 数据路径模式 (匹配所有 sim_ 开头的文件夹)
DATA_DIR_PATTERN = "./records/fixed4/r_r/sim_*/all_agents_trajectories.json"

# 2. 纳什均衡点 (跟你代码里一致)
NE_VECTOR = np.array([2.06, 2.51, 2.97, 3.42, 3.88]).reshape(-1, 1)

# 3. 理论固定时间 T_max 计算
# 根据你的 Config: a=0.2, b=2.5, p=0.8, q=1.2
# 标准固定时间上界公式: T_max <= 1/(a*(1-p)) + 1/(b*(q-1))
# 请务必根据你论文中的 Theorem 1 核对这个公式！
# 计算过程:
# Term 1: 1 / (0.2 * (1 - 0.8)) = 1 / 0.04 = 25
# Term 2: 1 / (2.5 * (1.2 - 1)) = 1 / 0.5 = 2
# T_max = 27.0
THEORETICAL_T_MAX = 38.34

# ================= 绘图逻辑 =================

def plot_convergence_verification():
    # 查找所有数据文件
    files = glob.glob(DATA_DIR_PATTERN)
    
    if not files:
        print("Error: No data files found! Please check the path pattern.")
        print(f"Searching in: {DATA_DIR_PATTERN}")
        return

    print(f"Found {len(files)} simulation files. Processing...")

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 循环读取并绘制每一条线
    for idx, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 1. 获取时间轴
            time_steps = np.array(data['time_steps'])
            
            # 2. 获取 x 轨迹 (Shape: [Agents, Time, Dim])
            x_trajs = data['trajectories']['x']
            # 转为 numpy: (Agents, Time) - 假设 Dim=1并squeeze掉
            x_matrix = np.array(x_trajs).squeeze(-1)
            
            # 3. 计算误差范数 ||x(t) - x*||
            # x_matrix shape: (5, T), NE_vector shape: (5, 1) -> 广播相减
            error_matrix = x_matrix - NE_VECTOR
            # 按列求范数 (axis=0), 得到 (T,)
            dist_to_NE = np.linalg.norm(error_matrix, axis=0)

            # 4. 绘图 (Semilog Y)
            # alpha设置低一点(0.3)，因为有100条线，透明度可以显示出"重叠"的密度感
            ax.plot(time_steps, np.log10(dist_to_NE), color='blue', linewidth=1.0, alpha=0.3)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(files)}...")
                
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    # ================= 审稿人要求的关键元素 =================
    
    # 1. 绘制理论时间上界 (红色虚线)
    ax.axvline(x=THEORETICAL_T_MAX, color='red', linestyle='--', linewidth=2.5)
    ax.set_xlim(left=0, right=50)  # X轴从0开始
    # 2. 设置坐标轴和标签
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\|x(t) - x^*\|)$', fontsize=14) # 使用 LaTeX 格式
    ax.set_title(f'Fixed-Time Convergence Verification\n({len(files)} simulations with initial norms $10^1 \sim 10^4$)', fontsize=14)
    
    # 3. 细节调整
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=12, loc='upper right')
    
    # 设置 Y 轴范围，确保能看到 10^-5 以下的收敛，同时也能看到 10^4 的起点
    # 根据你的数据情况微调
    ax.set_ylim(top=4.5) 

    plt.tight_layout()
    
    # 保存图片
    save_name = "Figure_Reviewer_Verification.png"
    plt.savefig(save_name, dpi=300)
    print(f"\nPlot saved successfully to: {save_name}")
    # plt.show() # 如果在服务器上运行，请注释掉这一行

def plot_convergence_verification2():
    # 查找所有数据文件
    files = glob.glob(DATA_DIR_PATTERN)
    
    if not files:
        print("Error: No data files found! Please check the path pattern.")
        print(f"Searching in: {DATA_DIR_PATTERN}")
        return

    print(f"Found {len(files)} simulation files. Processing...")

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 循环读取并绘制每一条线
    for idx, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 1. 获取时间轴
            time_steps = np.array(data['time_steps'])
            
            # 2. 获取 x 轨迹 (Shape: [Agents, Time, Dim])
            x_trajs = data['trajectories']['x']
            z_trajs = data['trajectories']['z']
            # 转为 numpy: (Agents, Time) - 假设 Dim=1并squeeze掉
            x_matrix = np.array(x_trajs).squeeze(-1)
            
            z_tensor = np.array(z_trajs)
            true_state_at_t = x_matrix.T
            z_error_tensor = z_tensor - true_state_at_t[np.newaxis, :, :]
            # 3. 计算误差范数 ||x(t) - x*||
            # x_matrix shape: (5, T), NE_vector shape: (5, 1) -> 广播相减
            dist_to_NE = np.linalg.norm(z_error_tensor, axis=(0, 2)) # 对 Agent和Dimension求范数，保留Time

            # 4. 绘图 (Semilog Y)
            # alpha设置低一点(0.3)，因为有100条线，透明度可以显示出"重叠"的密度感
            ax.plot(time_steps, np.log10(dist_to_NE), color='blue', linewidth=1.0, alpha=0.3)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(files)}...")
                
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    # ================= 审稿人要求的关键元素 =================
    
    # 1. 绘制理论时间上界 (红色虚线)
    # ax.axvline(x=THEORETICAL_T_MAX, color='red', linestyle='--', linewidth=2.5)

    # 2. 设置坐标轴和标签
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\|z - 1_N \otimes x\|)$', fontsize=14) # 使用 LaTeX 格式
    ax.set_title(f'Fixed-Time Convergence Verification\n({len(files)} simulations with initial norms $10^1 \sim 10^4$)', fontsize=14)
    ax.set_xlim(left=0, right=50)  # X轴从0开始
    # 3. 细节调整
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=12, loc='upper right')
    
    # 设置 Y 轴范围，确保能看到 10^-5 以下的收敛，同时也能看到 10^4 的起点
    # 根据你的数据情况微调
    ax.set_ylim(top=4.5) 

    plt.tight_layout()
    
    # 保存图片
    save_name = "Figure_Reviewer_Verification2.png"
    plt.savefig(save_name, dpi=300)
    print(f"\nPlot saved successfully to: {save_name}")
    # plt.show() # 如果在服务器上运行，请注释掉这一行


def get_partial_value(x, k):
    value_index = [5, 5.5, 6, 6.5, 7]
    x_sum = np.sum(x)*0.1 + 1.25
    x_sum += 1.1*x[k] - value_index[k]
    # print(x, k ,x_sum)
    return x_sum

def plot_convergence_verification3():
    # 查找所有数据文件
    files = glob.glob(DATA_DIR_PATTERN)
    
    if not files:
        print("Error: No data files found! Please check the path pattern.")
        print(f"Searching in: {DATA_DIR_PATTERN}")
        return

    print(f"Found {len(files)} simulation files. Processing...")

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6))

    # 循环读取并绘制每一条线
    for idx, file_path in enumerate(files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 1. 获取时间轴
            time_steps = np.array(data['time_steps'])
            
            # 2. 获取 x 轨迹 (Shape: [Agents, Time, Dim])
            x_trajs = data['trajectories']['x']
            v_trajs = data['trajectories']['v']
            # 转为 numpy: (Agents, Time) - 假设 Dim=1并squeeze掉
            x_matrix = np.array(x_trajs).squeeze(-1)
            true_state_at_t = x_matrix.T
            # print(true_state_at_t.shape, true_state_at_t[0])
            partial_martix = np.zeros(true_state_at_t.shape)
            for i in range(true_state_at_t.shape[0]):
                for j in range(true_state_at_t.shape[1]):
                    partial_martix[i,j] = get_partial_value(true_state_at_t[i], j)
            
            v_tensor = np.array(v_trajs)
            
            v_error_tensor = v_tensor - partial_martix[np.newaxis, :, :]
            # 3. 计算误差范数 ||x(t) - x*||
            # x_matrix shape: (5, T), NE_vector shape: (5, 1) -> 广播相减
            dist_to_NE = np.linalg.norm(v_error_tensor, axis=(0, 2)) # 对 Agent和Dimension求范数，保留Time

            # 4. 绘图 (Semilog Y)
            # alpha设置低一点(0.3)，因为有100条线，透明度可以显示出"重叠"的密度感
            ax.plot(time_steps, np.log10(dist_to_NE), color='blue', linewidth=1.0, alpha=0.3)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(files)}...")
                
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    # ================= 审稿人要求的关键元素 =================
    
    # 1. 绘制理论时间上界 (红色虚线)
    # ax.axvline(x=THEORETICAL_T_MAX, color='red', linestyle='--', linewidth=2.5)

    # 2. 设置坐标轴和标签
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\|v - 1_N \otimes \nabla_i f_i(x)\|)$', fontsize=14) # 使用 LaTeX 格式
    ax.set_title(f'Fixed-Time Convergence Verification\n({len(files)} simulations with initial norms $10^1 \sim 10^4$)', fontsize=14)
    ax.set_xlim(left=0, right=50)  # X轴从0开始
    # 3. 细节调整
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=12, loc='upper right')
    
    # 设置 Y 轴范围，确保能看到 10^-5 以下的收敛，同时也能看到 10^4 的起点
    # 根据你的数据情况微调
    ax.set_ylim(top=4.5) 

    plt.tight_layout()
    
    # 保存图片
    save_name = "Figure_Reviewer_Verification3.png"
    plt.savefig(save_name, dpi=300)
    print(f"\nPlot saved successfully to: {save_name}")
    # plt.show() # 如果在服务器上运行，请注释掉这一行

if __name__ == "__main__":
    plot_convergence_verification()
    plot_convergence_verification2()
    plot_convergence_verification3()