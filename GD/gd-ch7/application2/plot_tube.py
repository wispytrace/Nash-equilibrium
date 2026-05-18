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

def plot_simulation_result(centralized_data=None):
    if centralized_data is None:
        return

    sim_id = centralized_data.get('sim_id', 0)
    time_steps = np.array(centralized_data["time_steps"])
    
    # 定义 NE 点 (N个玩家, 2个维度)
    NE_vector = np.array([6.14369664e-13, 6.59648787e-01, 1.64974270e+00, 2.63984995e+00,
 3.62994579e+00, 4.62004803e+00, 5.61015032e+00, 6.60024727e+00,
 7.59034764e+00 ,8.00000000e+00, 8.00000000e+00, 8.00000000e+00,
 8.00000000e+00])
    tau_max = 8

    # 数据处理
    try:
        x_matrix = np.array(centralized_data["trajectories"]["y"])
        # u_matrix = np.array(centralized_data["trajectories"]["u"])
        # b_matrix = np.array(centralized_data["trajectories"]["b"])
        num_agents = x_matrix.shape[0]
    except Exception as e:
        print(f"数据解析失败: {e}")
        return

    print(x_matrix[:,-1])

    # 维度标签
    print(x_matrix[:,-1])
    dim_labels = ["1"]
    
    for dim in range(1): # 遍历两个维度，分别生成图片
        # 创建新的图片，使用默认长宽比 (6.4, 4.8)
        plt.figure()
        
        # 1. 绘制每个玩家的轨迹
        for i in range(num_agents):
            plt.plot(time_steps, x_matrix[i, :]- NE_vector[i], color=COLORS[i % len(COLORS)],
                     label=f"$x_{i}-x_{i}^{{\star}}$", linewidth=1.2)
            
            # 2. 绘制对应的 NE 点 (虚线)
            # 仅为第一个玩家的虚线添加图例标签，防止图例冗余
            ne_label = f"$y_{{{i+1}{dim_labels[dim]}}}^{{\star}}$"
            # plt.axhline(0, color=COLORS[i % len(COLORS)], linestyle='--', alpha=0.6, label=ne_label)

        # 3. 绘制控制边界 tau_max (红色点划线)
        plt.axhline(0, color='black', linestyle='-.', linewidth=1)
        # plt.axhline(y=-tau_max, color=COLORS[8], linestyle='-.', linewidth=1, label=r"$-\tau_{max}$")

        # 样式设置
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"$x_i-x_i^{{\star}}$", fontsize=15)
        # plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 将图例放在右侧外部
        plt.legend(loc='upper right',  fontsize=12, ncol=2)
        
        # 保存图片
        file_name = f"tube_transient_response_dim{dim+1}_sim_{sim_id}.png"
        full_path = os.path.join( file_name)
        plt.xlim(left=0,right=5)
        plt.ylim(top=2)
        plt.savefig(full_path, dpi=300)
        plt.show()
        plt.close() # 显式关闭，释放内存
        
        print(f"维度 {dim+1} 的图像已保存至: {full_path}")

# def plot_coupled_constraints(centralized_data=None):
#     if centralized_data is None:
#         return

#     time_steps = np.array(centralized_data["time_steps"])
    
#     # 1. 获取所有智能体的轨迹数据 (Shape: Num_Agents, Time, Dimension)
#     x_matrix = np.array(centralized_data["trajectories"]["y"])
    
#     # 2. 计算耦合项之和: 对所有智能体(axis=0)求和
#     # sum_x_dim1 shape: (Time,) 代表 sum(x_i1)
#     # sum_x_dim2 shape: (Time,) 代表 sum(x_i2)
#     sum_x_dim1 = np.sum(x_matrix[:, :], axis=0)
    
#     # 3. 从配置中获取耦合约束边界 (g_l, g_u)
#     # 假设你的配置文件中有这些值
#     c = 40

#     plt.plot(time_steps, sum_x_dim1, label=r'$\sum_{i=1}^6 y_{i}$', color='darkgreen', linewidth=2)
#     # 绘制耦合边界线 (红色粗虚线)
#     plt.axhline(y=c, color='red', linestyle='--', linewidth=2, label=r'$\sum_{i=1}^6 g_i=40$')
    

#     # plt.title("Total Coupled State Evolution ($y_{i1}$)")
#     plt.xlabel("Time (s)")
#     plt.ylabel(r" $\sum_{i=1}^6 y_{i}$")
#     # plt.grid(True, linestyle=':', alpha=0.6)
#     plt.legend(loc='upper right')
#     plt.ylim(top=50)
    
#     # 自动保存
#     plt.savefig(os.path.join( "coupled_constraint_dim1.png"), dpi=300, bbox_inches='tight')
#     print(f"图像已保存至: coupled_constraint_dim1.png")
#     plt.clf()


def plot_compare(centralized_data_list, labels):
    plt.clf()
    NE_vector = np.array([2.01980647, 3.00989722 ,4.00000317, 4.99010216, 5.98019429])

    for i, centralized_data in enumerate(centralized_data_list):
        time_steps = np.array(centralized_data["time_steps"])
        
        # 获取轨迹矩阵，形状应为 (n_agents, n_timesteps)，即 (5, len(time_steps))
        x_matrix = np.array(centralized_data["trajectories"]["x"])
        
        x_matrix_cleaned = np.array(centralized_data["trajectories"]["x"]).squeeze() 
        print(x_matrix_cleaned[:,-1])

        # 2. 检查智能体数量是否匹配
        n_agents_data = x_matrix_cleaned.shape[0]
        n_agents_ne = NE_vector.shape[0]

        if n_agents_data != n_agents_ne:
            print(f"警告：数据中有 {n_agents_data} 个智能体，但 NE 向量只有 {n_agents_ne} 个！")
            # 截取匹配的部分进行计算（仅用于调试）
            x_matrix_cleaned = x_matrix_cleaned[:n_agents_ne, :]

        # 3. 重新计算 diff
        diff = x_matrix_cleaned - NE_vector.reshape(-1, 1)
        dist_to_NE = np.linalg.norm(diff, axis=0)

        # 绘制曲线
        plt.plot(time_steps, dist_to_NE, linewidth=1.2, color=COLORS[i % len(COLORS)], label=labels[i])

    plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    plt.xlim(left=0,right=5)
    plt.show()
    plt.legend(loc='upper right',  fontsize=12)

    file_name = f"tube_transient_response_compare.png"
    full_path = os.path.join( file_name)
    plt.xlim(left=0,right=7)
    plt.ylim(bottom=0)
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel('$\|x - x^*\|$')
    # plt.ylim(top=10)
    plt.savefig(full_path, dpi=300)
    print(f"图像已保存至: {full_path}")

def plot_simulation_result2(centralized_data=None):
    if centralized_data is None:
        return

    sim_id = centralized_data.get('sim_id', 0)
    time_steps = np.array(centralized_data["time_steps"])
    
    # 定义 NE 点 (N个玩家, 2个维度)
    NE_vector = np.array([4.98628387, 5.97637428 ,6.96648187, 7.95657234, 7.99999998])
    tau_max = 8

    # 数据处理
    try:
        x_matrix = np.array(centralized_data["trajectories"]["y"])
        # u_matrix = np.array(centralized_data["trajectories"]["u"])
        # b_matrix = np.array(centralized_data["trajectories"]["b"])
        num_agents = x_matrix.shape[0]
    except Exception as e:
        print(f"数据解析失败: {e}")
        return

    print(x_matrix[:,-1])

    # 维度标签
    print(x_matrix[:,-1])
    dim_labels = ["1"]
    
    for dim in range(1): # 遍历两个维度，分别生成图片
        # 创建新的图片，使用默认长宽比 (6.4, 4.8)
        plt.figure()
        
        # 1. 绘制每个玩家的轨迹
        for i in range(num_agents):
            plt.plot(time_steps, x_matrix[i, :]- NE_vector[i], color=COLORS[i % len(COLORS)],
                     label=f"$x_{i+1}-x_{i+1}^{{\star}}$", linewidth=1.2)
            
            # 2. 绘制对应的 NE 点 (虚线)
            # 仅为第一个玩家的虚线添加图例标签，防止图例冗余
            ne_label = f"$y_{{{i+1}{dim_labels[dim]}}}^{{\star}}$"
            # plt.axhline(0, color=COLORS[i % len(COLORS)], linestyle='--', alpha=0.6, label=ne_label)

        # 3. 绘制控制边界 tau_max (红色点划线)
        plt.axhline(0, color='black', linestyle='-.', linewidth=1)
        # plt.axhline(y=-tau_max, color=COLORS[8], linestyle='-.', linewidth=1, label=r"$-\tau_{max}$")

        # 样式设置
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"$x_i-x_i^{{\star}}$", fontsize=15)
        # plt.grid(True, which='both', linestyle=':', alpha=0.5)
        
        # 将图例放在右侧外部
        plt.legend(loc='upper right',  fontsize=12, ncol=2)
        
        # 保存图片
        file_name = f"tube_transient_response_dim{dim+1}_sim_{sim_id}_2.png"
        full_path = os.path.join( file_name)
        plt.xlim(left=0,right=5)
        plt.ylim(top=3)
        plt.savefig(full_path, dpi=300)
        plt.show()
        plt.close() # 显式关闭，释放内存
        
        print(f"维度 {dim+1} 的图像已保存至: {full_path}")

def plot_compare2(centralized_data_list, labels):
    plt.clf()
    NE_vector = np.array([4.98628387, 5.97637428 ,6.96648187, 7.95657234, 7.99999998])

    for i, centralized_data in enumerate(centralized_data_list):
        time_steps = np.array(centralized_data["time_steps"])
        
        # 获取轨迹矩阵，形状应为 (n_agents, n_timesteps)，即 (5, len(time_steps))
        x_matrix = np.array(centralized_data["trajectories"]["x"])
        
        x_matrix_cleaned = np.array(centralized_data["trajectories"]["x"]).squeeze() 
        print(x_matrix_cleaned[:,-1])

        # 2. 检查智能体数量是否匹配
        n_agents_data = x_matrix_cleaned.shape[0]
        n_agents_ne = NE_vector.shape[0]

        if n_agents_data != n_agents_ne:
            print(f"警告：数据中有 {n_agents_data} 个智能体，但 NE 向量只有 {n_agents_ne} 个！")
            # 截取匹配的部分进行计算（仅用于调试）
            x_matrix_cleaned = x_matrix_cleaned[:n_agents_ne, :]

        # 3. 重新计算 diff
        diff = x_matrix_cleaned - NE_vector.reshape(-1, 1)
        dist_to_NE = np.linalg.norm(diff, axis=0)

        # 绘制曲线
        plt.plot(time_steps, dist_to_NE, linewidth=1.2, color=COLORS[i % len(COLORS)], label=labels[i])

    plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    plt.xlim(left=0,right=5)
    plt.show()
    plt.legend(loc='upper right',  fontsize=12)

    file_name = f"tube_transient_response_compare2.png"
    full_path = os.path.join( file_name)
    plt.xlim(left=0,right=7)
    plt.ylim(bottom=0)
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel('$\|x - x^*\|$')
    # plt.ylim(top=10)
    plt.savefig(full_path, dpi=300)
    print(f"图像已保存至: {full_path}")

if __name__ == "__main__":
    with open('/app/GD/gd-ch7/application2/records/euler_constraint/r_0/sim_101/all_agents_trajectories.json') as f:
        centralized_data = json.load(f)
        plot_simulation_result(centralized_data)
        # plot_coupled_constraints(centralized_data)
        # plot_coupled_constraints(centralized_data)
    
    # files_list = ["/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint_tube_1/r_0/sim_101/all_agents_trajectories.json",
    #                "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint/r_0/sim_101/all_agents_trajectories.json"]
    # label_list = ["Asymptotical algorithm", "Fixed-time algorithm"]

    # centralized_data_list = []
    # for file in files_list:
    #     with open(file) as f:
    #         centralized_data_list.append(json.load(f))
    
    # plot_compare(centralized_data_list, label_list)


    # with open('/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint/a_1/sim_101/all_agents_trajectories.json') as f:
    #     centralized_data = json.load(f)
    #     plot_simulation_result2(centralized_data)
    #     # plot_coupled_constraints(centralized_data)
    #     # plot_coupled_constraints(centralized_data)
    
    # files_list = ["/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint/a_2/sim_101/all_agents_trajectories.json",
    #                "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/franklin-fixed/records/euler_constraint/a_1/sim_101/all_agents_trajectories.json"]
    # label_list = ["Asymptotical algorithm", "Fixed-time algorithm"]

    # centralized_data_list = []
    # for file in files_list:
    #     with open(file) as f:
    #         centralized_data_list.append(json.load(f))
    
    # plot_compare2(centralized_data_list, label_list)