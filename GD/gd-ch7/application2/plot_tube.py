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

def plot_simulation_result(centralized_data=None):
    if centralized_data is None:
        return

    sim_id = centralized_data.get('sim_id', 0)
    time_steps = np.array(centralized_data["time_steps"])
    
    # 尝试解析数据
    try:
        # 使用 squeeze() 去除最后一维 Dimension (假设 Dimension=1)，使得矩阵变为 (Num_Agents, Time)
        q_matrix = np.array(centralized_data["trajectories"]["x"]).squeeze()     # qi
        dotq_matrix = np.array(centralized_data["trajectories"]["dot_x"]).squeeze() # dotqi
        u_matrix = np.array(centralized_data["trajectories"]["ui"]).squeeze()      # ui
        virtual = np.array(centralized_data["trajectories"]["y"]).squeeze()  # 虚拟信号 y
        num_agents = q_matrix.shape[0]
    except Exception as e:
        print(f"数据解析失败: {e}")
        return

    # ====== 引入给定参数 ======
    D = 65            # 总需求 Demand
    q_i_min = 0       # 发电下限
    q_i_max = 8       # 发电上限
    a = 0.02          # 价格系数 (对应原模型的 b)
    p0 = 40           # 基础价格
    # 假设你理论推导的固定时间收敛上限为 T_max (你可以根据实际情况修改)
    T_max = 5.0       

    # ====== 全局绘图样式设置 (适合学术论文) ======
    plt.rcParams.update({
        'font.size': 12, 
        'font.family': 'serif', # 论文常用衬线字体
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.alpha': 0.7
    })

    # ---------------------------------------------------------
    # 图 1: 发电机输出功率 (q_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    
    for i in range(num_agents):
        # 1. 绘制实际功率演化曲线，并获取当前曲线的颜色
        line, = plt.plot(time_steps, q_matrix[i, :], linewidth=1.5, label=f'Agent {i+1}')
        
        # 2. 提取时间切片的最后一个值作为该智能体的理论纳什均衡点 q_i^*
        q_gne = q_matrix[i, -1]
        
        # 3. 绘制对应智能体的纳什均衡虚线
        # 使用 line.get_color() 确保虚线颜色与曲线完美对应；ls=':' 表示细点虚线；alpha=0.6 降低透明度避免抢戏
        # 不设置 label，避免图例成倍增加
        plt.axhline(y=q_gne, color=line.get_color(), linestyle=':', alpha=0.6, linewidth=1.2)
    
    # 4. 绘制全局容量约束的上下界（保持作为参考背景）
    plt.axhline(y=q_i_max, color='red', linestyle='--', linewidth=1.5, label=f'Capacity Max ($q^{{max}}={q_i_max}$)')
    plt.axhline(y=q_i_min, color='black', linestyle='--', linewidth=1.5, label=f'Capacity Min ($q^{{min}}={q_i_min}$)')
    
    # 为了在图例中向审稿人说明“细点虚线代表纳什均衡”，我们可以手动添加一个辅助图例项
    plt.plot([], [], color='gray', linestyle=':', alpha=0.6, linewidth=1.2, label='Nash Equilibrium ($q_i^*$)')

    plt.xlabel('Time (s)')
    plt.ylabel('Power Output $q_i$ (p.u.)')
    
    # 使用 ncol=3 将图例排成三列，紧凑地放在上方或右侧
    plt.legend(loc='upper right', ncol=3, fontsize=10, frameon=True) 
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_q_outputs.pdf", format='pdf', bbox_inches='tight')

    # ---------------------------------------------------------
    # 图 2: 系统供需平衡误差 (Sum(q_i) - D) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    # 沿着智能体维度(axis=0)求和
    total_generation = np.sum(q_matrix, axis=0) 
    mismatch = total_generation - D
    
    plt.plot(time_steps, mismatch, color='blue', linewidth=2, label='$\sum q_i - D$')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Supply-Demand Mismatch')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_mismatch.pdf", format='pdf')

    # ---------------------------------------------------------
    # 图 3: 实时电价演化 (Price) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    # 计算价格 p = p0 - a * sum(q_i)
    clearing_price = p0 - a * total_generation
    
    plt.plot(time_steps, clearing_price, color='darkgreen', linewidth=2, label='Clearing Price $p(\sigma(\mathbf{q}))$')
    plt.xlabel('Time (s)')
    plt.ylabel('Electricity Price')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_price.pdf", format='pdf')

    # ---------------------------------------------------------
    # 图 4: 发电机转速偏差/导数动态 (dot_q_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    for i in range(num_agents):
        plt.plot(time_steps, dotq_matrix[i, :], linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Rate of Power Change $\dot{q}_i$')
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_dot_q.pdf", format='pdf')

    # ---------------------------------------------------------
    # 图 5: 物理层控制输入 (u_i) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    for i in range(num_agents):
        plt.plot(time_steps, u_matrix[i, :], linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Input $u_i / \\tau_i$')
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_control_input.pdf", format='pdf')

    # 将所有生成的图展示出来
    plt.show()

# ---------------------------------------------------------
    # 图 6: 纳什均衡收敛误差 (Convergence Error) vs 时间
    # ---------------------------------------------------------
    plt.figure()
    
    # 1. 提取理论纳什均衡点 (取仿真最后一步的值作为稳态 GNE)
    # 使用 keepdims 保持二维结构 (Num_Agents, 1)，利用 numpy 广播机制计算差值
    
    # 2. 计算每个智能体的绝对误差 e_i(t) = |q_i(t) - q_i^*|
    error_matrix = np.abs(q_matrix - virtual)
    
    # 也可以计算系统的总体范数误差 (可选)：
    # system_error = np.linalg.norm(q_matrix - q_star, axis=0)
    # plt.plot(time_steps, system_error, color='black', linewidth=2, label='System Norm Error')

    # 3. 绘制每个智能体的误差曲线
    for i in range(num_agents):
        # 为了防止 log 坐标系下报错（log(0) 是无意义的），可以给误差加上一个极小的值 1e-12
        plt.plot(time_steps, error_matrix[i, :] + 1e-12, linewidth=1.5, label=f'Player {i+1}')
    
    # ====== 高级技巧：使用对数坐标系 ======
    # 固定时间算法的误差在对数系下会呈现“断崖式下跌”，而渐近收敛只是一条平缓斜线
    # plt.yscale('log')
    
    # 设置 y 轴显示范围，过滤掉 1e-12 以下毫无意义的数值噪声，让图面更干净
    plt.ylim(bottom=1e-6) 
    
    # 标注理论的固定时间收敛上界 T_max
    # plt.axvline(x=T_max, color='purple', linestyle='-.', linewidth=1.5, label='$T_{max}$')

    plt.xlabel('Time (s)')
    plt.ylabel('Convergence Error $|q_i(t) - q_i^*|$ (Log Scale)')
    plt.legend(loc='upper right', ncol=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"sim_{sim_id}_convergence_error.pdf", format='pdf', bbox_inches='tight')

if __name__ == "__main__":
    with open('/mnt/binghao/NESeeking/Nash-equilibrium/GD/gd-ch7/application2/records/euler_constraint/r_0/sim_101/all_agents_trajectories.json') as f:
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