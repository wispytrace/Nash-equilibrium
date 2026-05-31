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

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d plots

mpl.rcParams['figure.dpi'] = 600 
# mpl.rcParams['lines.linewidth'] = 1
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, '/app/GD/resource/font/Times New Roman.ttf')
plt.rcParams['mathtext.fontset'] = 'stix'

from matplotlib.font_manager import FontProperties, fontManager

prop = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = prop.get_name()
fontManager.addfont(font_path)


def plot_simulation_result(centralized_data=None):
    if centralized_data is None:
        return

    sim_id = centralized_data.get('sim_id', 0)
    time_steps = np.array(centralized_data["time_steps"])
    
    # 尝试解析数据
    try:
        q_matrix = np.array(centralized_data["trajectories"]["x"]).squeeze()     # qi
        dotq_matrix = np.array(centralized_data["trajectories"]["dot_x"]).squeeze() # dotqi
        u_matrix = np.array(centralized_data["trajectories"]["ui"]).squeeze()      # ui
        virtual = np.array(centralized_data["trajectories"]["y"]).squeeze()  # 虚拟信号 y
        esitimate = np.array(centralized_data["trajectories"]["z"]).squeeze()
        num_agents = q_matrix.shape[0]
    except Exception as e:
        print(f"数据解析失败: {e}")
        return

    # ====== 引入给定参数 ======
    D = 80            # 总需求 Demand
    q_i_min = 0        # 发电下限
    q_i_max = 11       # 发电上限
    a = 0.02           # 价格系数
    p0 = 50            # 基础价格
    T_max = 5.0        # 假设你理论推导的固定时间收敛上限为 T_max

    # ==========================================
    # 自定义 13 种高对比度、区分度明显的颜色
    # ==========================================
    distinct_colors = [
        '#e6194b', # 红色 (Red)
        '#3cb44b', # 绿色 (Green)
        '#4363d8', # 蓝色 (Blue)
        '#f58231', # 橙色 (Orange)
        '#911eb4', # 紫色 (Purple)
        '#00ced1', # 深青色 (Dark Turquoise)
        '#f032e6', # 洋红色 (Magenta)
        '#8b4513', # 棕色 (Saddle Brown)
        '#000075', # 藏青色 (Navy)
        '#808000', # 橄榄绿 (Olive)
        '#ff1493', # 深粉色 (Deep Pink)
        '#008080', # 水鸭色 (Teal)
        '#daa520'  # 金麒麟色 (Goldenrod)
    ]

    # ---------------------------------------------------------
    # 图 1: 发电量 / 策略状态 (x_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    
    for i in range(num_agents):
        c = distinct_colors[i % len(distinct_colors)] 
        # 1. 绘制实际功率演化曲线
        line, = plt.plot(time_steps, q_matrix[i, :], linewidth=1.5, color=c, label=f'Firm {i+1}')
        
        # 2. 提取时间切片的最后一个值作为该智能体的理论纳什均衡点 q_i^*
        q_gne = q_matrix[i, -1]
        
        # 3. 绘制对应智能体的纳什均衡虚线
        plt.axhline(y=q_gne, color=c, linestyle=':', alpha=0.6, linewidth=1.2)
    
    # 4. 绘制全局容量约束的上下界
    # plt.axhline(y=q_i_max, color='red', linestyle='--', linewidth=1.5, label=f' ($x^{{max}}={q_i_max}$)')
    # plt.axhline(y=q_i_min, color='black', linestyle='--', linewidth=1.5, label=f' ($x^{{min}}={q_i_min}$)')

    plt.xlabel('Time (s)')
    plt.ylabel('$x_i$')
    plt.xlim(left=0, right=7)
    plt.ylim(bottom=0, top=15)
    # 使用 ncol=4 将图例排成四列，避免 13+2 个图例太长挡住图像
    plt.legend(loc='upper right', ncol=4, fontsize=11, frameon=True) 
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_q_outputs.png")


    plt.figure()
    
    for i in range(num_agents):
        c = distinct_colors[i % len(distinct_colors)] 
        # 1. 绘制实际功率演化曲线
        line, = plt.plot(time_steps, virtual[i], linewidth=1.5, color=c, label=f'Firm {i+1}')
        
        # 2. 提取时间切片的最后一个值作为该智能体的理论纳什均衡点 q_i^*
        virtual_star = virtual[i, -1]
        
        # 3. 绘制对应智能体的纳什均衡虚线
        plt.axhline(y=virtual_star, color=c, linestyle=':', alpha=0.6, linewidth=1.2)
    
    plt.xlabel('Time (s)')
    plt.ylabel('$\omega_i$')
    plt.xlim(left=0, right=7)
    plt.ylim(bottom=0, top=15)
    # 使用 ncol=4 将图例排成四列，避免 13+2 个图例太长挡住图像
    plt.legend(loc='upper right', ncol=4, fontsize=11, frameon=True) 
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_v_outputs.png")

    # ---------------------------------------------------------
    # 图 2: 系统供需平衡误差 (Sum(q_i) - D) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    total_generation = np.sum(q_matrix, axis=0) 
    mismatch = total_generation - D
    
    plt.plot(time_steps, mismatch, color='blue', linewidth=2, label=r'$\sum x_i - D$')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Supply-Demand Mismatch')
    plt.legend(loc='upper right', ncol=3, fontsize=15, frameon=True) 
    plt.tight_layout()
    plt.xlim(left=0, right=7)
    plt.savefig(f"mismatch.png")

    # ---------------------------------------------------------
    # 图 3: 实时电价演化 (Price) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    clearing_price = p0 - a * total_generation
    
    plt.plot(time_steps, clearing_price, color='darkgreen', linewidth=2, label=r'$p_0-b\sum_{i=1}^N x_i$')
    plt.xlabel('Time (s)')
    plt.ylabel('Electricity Price')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.xlim(left=0, right=7)
    plt.savefig(f"price.png")

    # ---------------------------------------------------------
    # 图 4: 发电机转速偏差/导数动态 (dot_q_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    for i in range(num_agents):
        c = distinct_colors[i % len(distinct_colors)]
        plt.plot(time_steps, dotq_matrix[i, :], linewidth=1.5, color=c, label=f'Firm {i+1}')
        
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{x}_i$')
    plt.tight_layout()
    plt.xlim(left=0, right=7)
    plt.legend(loc='upper right', ncol=4, fontsize=11)
    plt.savefig(f"sim_{sim_id}_dot_q.png")

    # ---------------------------------------------------------
    # 图 5: 物理层控制输入 (u_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    for i in range(num_agents):
        c = distinct_colors[i % len(distinct_colors)]
        plt.plot(time_steps, u_matrix[i, :], linewidth=1.5, color=c, label=f'Firm {i+1}')
        
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input $u_i$')
    plt.tight_layout()
    plt.xlim(left=0, right=7)
    plt.legend(loc='upper right', ncol=4, fontsize=11)
    plt.savefig(f"control_input.png")
    
    # ---------------------------------------------------------
    # 图 6: 纳什均衡收敛误差 (Convergence Error) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    error_matrix = np.abs(q_matrix - virtual)
    
    for i in range(num_agents):
        c = distinct_colors[i % len(distinct_colors)]
        plt.plot(time_steps, error_matrix[i, :] + 1e-12, linewidth=1.5, color=c, label=f'Firm {i+1}')
    
    plt.xlim(left=0, right=7)
    plt.ylim(bottom=0)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Tracking Error $|x_i - \omega_i|$')
    plt.legend(loc='upper right', ncol=4, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_convergence_error.png")

    # ---------------------------------------------------------
    # 图 6: 纳什均衡收敛误差 (Convergence Error) vs 时间
    # ---------------------------------------------------------
    # plt.figure()
    
    # for i in range(num_agents):
    #     c = distinct_colors[i % len(distinct_colors)]
    #     for j in range(esitimate.shape[1]):

    #         error = esitimate[i,j,:] - virtual
    #     plt.plot(time_steps, error_matrix[i, :] + 1e-12, linewidth=1.5, color=c, label=f'Firm {i+1}')
    
    # plt.xlim(left=0, right=7)
    # plt.ylim(bottom=0)
    # plt.xlabel('Time (s)')
    # plt.ylabel(r'Tracking Error $|x_i - \omega_i|$')
    # plt.legend(loc='upper right', ncol=4, fontsize=11)
    # plt.tight_layout()
    # plt.savefig(f"sim_{sim_id}_convergence_error.png")

def plot_compare2(centralized_data_list, labels):
    plt.clf()
    NE_vector =  np.array([0.5445739, 1.53467401, 2.52477077, 3.51486522, 4.50496602, 5.4950933, 6.48516622, 7.47526981, 8.46533868, 9.45547493, 10.00000049, 10.00000033, 10.00000043])

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
    plt.ylabel(r"$||\mathbf{x} - \mathbf{x}^\star||$", fontsize=15)
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
    
    files_list = ["/app/GD/gd-ch7/application2/records/euler_constraint/r_1/sim_101/all_agents_trajectories.json", "/app/GD/gd-ch7/application2/records/euler_constraint/r_0/sim_101/all_agents_trajectories.json"]
    label_list = ["Asymptotical algorithm", "Fixed-time algorithm"]

    centralized_data_list = []
    for file in files_list:
        with open(file) as f:
            centralized_data_list.append(json.load(f))
    
    plot_compare2(centralized_data_list, label_list)