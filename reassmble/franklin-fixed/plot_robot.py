import matplotlib as mpl
import numpy as np
import json

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
COLORS = list(mcolors.TABLEAU_COLORS.keys())
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_with_inset(time, data_dict):
    """
    time: 时间数组 (x轴)
    data_dict: 字典，键为标签(如 'u11')，值为对应的数组 (y轴)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 绘制主图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (label, data), color in zip(data_dict.items(), colors):
        ax.plot(time, data, label=f'${label}$', color=color)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('$u_{i1}$', fontsize=14)
    ax.set_xlim(0, 5)
    ax.set_ylim(-170, 100)
    ax.legend(loc='upper right', ncol=2, fontsize=12)

    # 2. 创建局部放大图 (Inset Axes)
    # width, height 可以是比例 (如 "30%") 或英寸
    # loc=4 表示右下角 (lower right)
    ax_ins = inset_axes(ax, width="40%", height="30%", loc=4, borderpad=3)

    # 3. 在放大图中再次绘制数据
    for (label, data), color in zip(data_dict.items(), colors):
        ax_ins.plot(time, data, color=color)

    # 4. 设置放大图的显示范围 (这里设为展示 3s-5s 的微小震动)
    x1, x2 = 3.0, 5.0
    y1, y2 = 15, 25  # 根据你数据的具体波动范围调整
    ax_ins.set_xlim(x1, x2)
    ax_ins.set_ylim(y1, y2)
    
    # 隐藏放大图的坐标刻度（可选，为了简洁）
    # ax_ins.set_xticks([])
    # ax_ins.set_yticks([])

    # 5. 在主图中画一个矩形框并连线到放大图
    # fc="none" 矩形不填充，ec="0.5" 灰色边框
    mark_inset(ax, ax_ins, loc1=2, loc2=1, fc="none", ec="0.5", ls="--")

    plt.show()

def plot_simulation_result(centralized_data=None):
    if centralized_data is None:
        return

    sim_id = centralized_data.get('sim_id', 0)
    time_steps = np.array(centralized_data["time_steps"])
    
    # 定义 NE 点 (N个玩家, 2个维度)
    NE_vector = np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]])
    tau_max = 2

    # 数据处理
    try:
        x_matrix = np.array(centralized_data["trajectories"]["y"])
        u_matrix = np.array(centralized_data["trajectories"]["u"])
        b_matrix = np.array(centralized_data["trajectories"]["b"])
        num_agents = x_matrix.shape[0]
    except Exception as e:
        print(f"数据解析失败: {e}")
        return

    # 维度标签
    dim_labels = ["1", "2"]
    
    for dim in range(2): # 遍历两个维度，分别生成图片
        # 创建新的图片，使用默认长宽比 (6.4, 4.8)
        plt.figure()
        
        # 1. 绘制每个玩家的轨迹
        for i in range(num_agents):
            plt.plot(time_steps, x_matrix[i, :, dim], color=COLORS[i % len(COLORS)],
                     label=f"$y_{{{i+1}{dim_labels[dim]}}}$", linewidth=1.2)
            
            # 2. 绘制对应的 NE 点 (虚线)
            # 仅为第一个玩家的虚线添加图例标签，防止图例冗余
            ne_label = f"$y_{{{i+1}{dim_labels[dim]}}}^{{\star}}$"
            # plt.axhline(y=NE_vector[i, dim], color=COLORS[i % len(COLORS)], linestyle='--', alpha=0.6, label=ne_label)

        # 3. 绘制控制边界 tau_max (红色点划线)
        plt.axhline(y=tau_max, color=COLORS[6], linestyle='-.', linewidth=1, label=r"$\tau_{max}$")
        plt.axhline(y=-tau_max, color=COLORS[7], linestyle='-.', linewidth=1, label=r"$-\tau_{max}$")

        # 样式设置
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"$y_{{i{dim_labels[dim]}}}$", fontsize=15)
        # plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 将图例放在右侧外部
        plt.legend(loc='upper right',  fontsize=12, ncol=3)
        
        # 保存图片
        file_name = f"transient_response_dim{dim+1}_sim_{sim_id}.png"
        full_path = os.path.join( file_name)
        plt.xlim(left=0,right=5)
        plt.ylim(-tau_max-0.5, tau_max+2.5)
        plt.savefig(full_path, dpi=300)
        plt.show()
        plt.close() # 显式关闭，释放内存
        
        print(f"维度 {dim+1} 的图像已保存至: {full_path}")

    for dim in range(2): # 遍历两个维度，分别生成图片
        # 创建新的图片，使用默认长宽比 (6.4, 4.8)
        plt.figure()
        
        # 1. 绘制每个玩家的轨迹
        for i in range(num_agents):
            plt.plot(time_steps, u_matrix[i, :, dim], color=COLORS[i % len(COLORS)],
                     label=f"$u_{{{i+1}{dim_labels[dim]}}}$", linewidth=1.2)
            


        # 样式设置
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"$u_{{i{dim_labels[dim]}}}$", fontsize=15)
        # plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 将图例放在右侧外部
        plt.legend(loc='upper right',  fontsize=12, ncol=2)
        
        # 保存图片
        file_name = f"transient_response_control_dim{dim+1}_sim_{sim_id}.png"
        full_path = os.path.join( file_name)
        plt.xlim(left=0,right=5)
        plt.ylim(top=100)
        plt.savefig(full_path, dpi=300)
        plt.show()
        plt.close() # 显式关闭，释放内存
        
        print(f"维度 {dim+1} 的图像已保存至: {full_path}")

    for dim in range(2): # 遍历两个维度，分别生成图片
        # 创建新的图片，使用默认长宽比 (6.4, 4.8)
        plt.figure()
        
        # 1. 绘制每个玩家的轨迹
        for i in range(num_agents):
            plt.plot(time_steps, b_matrix[i, :, dim], color=COLORS[i % len(COLORS)],
                     label=f"$b_{{{i+1}{dim_labels[dim]}}}$", linewidth=1.2)
            


        # 样式设置
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"$b_{{i{dim_labels[dim]}}}$", fontsize=15)
        # plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 将图例放在右侧外部
        plt.legend(loc='upper right',  fontsize=12, ncol=2)
        
        # 保存图片
        file_name = f"transient_response_disturbance_dim{dim+1}_sim_{sim_id}.png"
        full_path = os.path.join( file_name)
        plt.xlim(left=0,right=5)
        plt.ylim(top=10)
        plt.savefig(full_path, dpi=300)
        plt.show()
        plt.close() # 显式关闭，释放内存
        
        print(f"维度 {dim+1} 的图像已保存至: {full_path}")


def plot_coupled_constraints(centralized_data=None):
    if centralized_data is None:
        return

    time_steps = np.array(centralized_data["time_steps"])
    
    # 1. 获取所有智能体的轨迹数据 (Shape: Num_Agents, Time, Dimension)
    x_matrix = np.array(centralized_data["trajectories"]["y"])
    
    # 2. 计算耦合项之和: 对所有智能体(axis=0)求和
    # sum_x_dim1 shape: (Time,) 代表 sum(x_i1)
    # sum_x_dim2 shape: (Time,) 代表 sum(x_i2)
    sum_x_dim1 = np.sum(x_matrix[:, :, 0], axis=0)
    sum_x_dim2 = np.sum(x_matrix[:, :, 1], axis=0)
    
    # 3. 从配置中获取耦合约束边界 (g_l, g_u)
    # 假设你的配置文件中有这些值
    g1_l, g1_u = -0.4, 0.4
    g2_l, g2_u = -0.4, 0.4

    # --- 开始绘制第1个维度的耦合约束图 ---
    plt.plot(time_steps, sum_x_dim1, label=r'$\sum_{i=1}^5 y_{i1}$', color='blue', linewidth=2)
    
    # 绘制耦合边界线 (红色粗虚线)
    plt.axhline(y=g1_u, color='red', linestyle='--', linewidth=2, label=r'$g_1^u$ (Upper Bound)')
    plt.axhline(y=g1_l, color='red', linestyle='--', linewidth=2, label=r'$g_1^l$ (Lower Bound)')
    
    # 填充可行域背景色（可选，增加视觉效果）
    plt.fill_between(time_steps, g1_l, g1_u, color='green', alpha=0.1, label='Feasible Region')

    # plt.title("Total Coupled State Evolution ($y_{i1}$)")
    plt.xlabel("Time (s)")
    plt.ylabel(r" $\sum_{i=1}^5 y_{i1}$")
    plt.xlim(left=0, right=time_steps[-1])
    # plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    # 自动保存
    plt.savefig(os.path.join( "coupled_constraint_dim1.png"), dpi=300, bbox_inches='tight')
    print(f"图像已保存至: coupled_constraint_dim1.png")
    plt.clf()

    # --- 开始绘制第2个维度的耦合约束图 ---
    plt.plot(time_steps, sum_x_dim2, label=r'$\sum_{i=1}^5 y_{i1}$', color='darkgreen', linewidth=2)
    plt.axhline(y=g2_u, color='red', linestyle='--', linewidth=2, label=r'$g_2^u$')
    plt.axhline(y=g2_l, color='red', linestyle='--', linewidth=2, label=r'$g_2^l$')
    plt.fill_between(time_steps, g1_l, g1_u, color='green', alpha=0.1, label='Feasible Region')
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\sum_{i=1}^5 y_{i1}$")
    plt.xlim(left=0, right=time_steps[-1])
    # plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join( "coupled_constraint_dim2.png"), dpi=300, bbox_inches='tight')
    print(f"图像已保存至: coupled_constraint_dim2.png")

def plot_compare(centralized_data_list, labels):
    NE_vector = np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]])

    for i, centralized_data in enumerate(centralized_data_list):
        time_steps = np.array(centralized_data["time_steps"])
        x_matrix = np.array(centralized_data["trajectories"]["x"])
        # 计算距离
        error_matrix = x_matrix - NE_vector[:,None,:]
        error_swapped = np.swapaxes(error_matrix, 0, 1)
        
        # 第二步：把 Agents 和 Coords 维度合并（展平）
        # 我们希望每个时间步对应一个长度为 10 的向量 (5个智能体 * 2个坐标)
        # 变换后: (2001, 10)
        error_flattened = error_swapped.reshape(error_swapped.shape[0], -1)
        dist_to_NE = np.linalg.norm(error_flattened, axis=1)
        plt.plot(time_steps, dist_to_NE, linewidth=1.2, color=COLORS[i % len(COLORS)], label=labels[i])

    # plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    plt.xlim(left=0,right=5)
    plt.show()
    plt.legend(loc='upper right',  fontsize=12)

    file_name = f"transient_response_compare.png"
    full_path = os.path.join( file_name)
    plt.xlim(left=0,right=5)
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel('$\|x(t) - x^*\|$')
    plt.ylim(bottom=0)
    plt.savefig(full_path, dpi=300)
    print(f"图像已保存至: {full_path}")

if __name__ == "__main__":
    with open('/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint/f1/sim_101/all_agents_trajectories.json') as f:
        centralized_data = json.load(f)
        plot_simulation_result(centralized_data)
        plot_coupled_constraints(centralized_data)
    
    files_list = ["/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/nd-fixed/records/euler_constraint_asym/a1/sim_1/all_agents_trajectories.json",
                   "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/nd-fixed/records/euler_constraint/f2/sim_1/all_agents_trajectories.json",
                    "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/nd-fixed/records/euler_constraint/f1/sim_1/all_agents_trajectories.json"]
    label_list = ["Asymptotical algorithm", "Finite-time algorithm", "Fixed-time algorithm"]

    centralized_data_list = []
    for file in files_list:
        with open(file) as f:
            centralized_data_list.append(json.load(f))
    plt.clf()
    plot_compare(centralized_data_list, label_list)