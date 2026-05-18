import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import copy
import json
import scipy.special as sp
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
# from standard import *
from config import config

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir+"/../../")
from utilis import *

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10


def get_multi_initi_value_convergence_time(base_path, num_agents, opt_value=None):
    convergence_time_dict = {}
    for simu_folder in sorted(os.listdir(base_path)):
        simu_path = os.path.join(base_path, simu_folder)
        if os.path.isdir(simu_path) and simu_folder.startswith("simu_"):
            try:
                simu_id = int(simu_folder.split("_")[1])
            except ValueError:
                continue
            convergence_time_dict[simu_id] = {"convergence_time": None, "init_value": None}
            memory = get_records_memory(simu_path, num_agents)
            status_vector = align_list(memory['x'])
            time = np.array(memory['time'][-1][:len(status_vector[0])])
            if opt_value is None:
                opt_value = np.zeros(status_vector.shape)
                NE_point = np.array([0.12815768398783586, 0.6012151511176591, 1.0233494776122871, 1.2797928121775644, 2.000099746727306])
                for i in range(len(time)):
                    for j in range(status_vector.shape[0]):
                        opt_value[j, i, :] = NE_point[j]

            convergence_time_dict[simu_id]["convergence_time"] = get_convergence_time(status_vector, opt_value, time, error=1e-2)
            convergence_time_dict[simu_id]["init_value"] = np.linalg.norm(status_vector[:, 0, :].flatten())  # 假设 x 的初始值在这里
            sorted_convergence_time = sorted(convergence_time_dict.items(), key=lambda x: x[1]["init_value"])
    
    print("Convergence Time for each simulation:")
    print(sorted_convergence_time)


def plot_graph(memory, record_path):
    figure_dir = record_path + "/figure"
    result_dir = record_path + "/result"
    
    z_vector = align_list(memory['z'])
    x_vector = align_list(memory['y'])
    partial_cost = align_list(memory['partial_cost'])
    update_value = align_list(memory['update_value'])
    cost = align_list(memory['cost'])
    time = np.array(memory['time'][-1][:len(x_vector[0])])
    print(x_vector.shape)
    opt_value = np.zeros(x_vector.shape)
    for i in range(x_vector.shape[0]):
        opt_value[i] = x_vector[i,-1,0]
    print(opt_value)
    # opt_value = np.zeros(x_vector.shape)
    # NE_point = np.array([0.12815768398783586, 0.6012151511176591, 1.0233494776122871, 1.2797928121775644, 2.000099746727306])
    # for i in range(len(time)):
    #     for j in range(x_vector.shape[0]):
    #         opt_value[j, i, :] = NE_point[j]
    
    # y_opt_value = np.zeros_like(y_vector)
    # N = y_vector.shape[0]

    # # 利用数学性质：相对旋转项求和为0，y 的理论最优解其实就是纯净的全局宏观导数
    # # 利用 NumPy 的向量化特性，直接对整个 time 数组进行一次性计算
    # opt_x = -1.5 * np.sin(0.5 * time)
    # opt_y =  1.5 * np.cos(0.5 * time)
    # opt_z =  0.5 * np.ones_like(time)  # 生成与 time 长度相同的 0.5 数组

    # 所有智能体 (j) 追踪的理论最优 y 值都是一样的
    # for j in range(N):
    #     y_opt_value[j, :, 0] = opt_x
    #     y_opt_value[j, :, 1] = opt_y
    #     y_opt_value[j, :, 2] = opt_z
    
    z_opt_value = np.zeros(z_vector.shape)
    for i in range(len(time)):
        for j in range(z_vector.shape[0]):
            z_opt_value[j, i, :] = x_vector[:, i, :]
    
    z_error = np.zeros((x_vector.shape[0], x_vector.shape[1]))

    for i in range(len(time)):
        for j in range(z_vector.shape[0]):
            z_error[j,i] = np.linalg.norm(z_vector[j,i]-z_opt_value[j,i])
    

    z_error = z_error.reshape(z_error.shape[0], z_error.shape[1], 1)
    cost = cost.reshape(cost.shape[0], cost.shape[1], 1)
    DoS_interval = config['c_0']['agent_config']['model_config']['DoS_interval']
    dos_interval = []
    for key, intervals in DoS_interval.items():
        dos_interval.extend(intervals)
    
    print(partial_cost.shape)
    partial_cost = partial_cost.reshape(partial_cost.shape[0], partial_cost.shape[1], 1)
    # plot_3d_trajectory_global_graph(x_vector, figure_dir, global_target=pg_opt_value)
    plot_status_converge_graph(time, x_vector, opt_value, figure_dir, file_name_prefix='status_convergence', ylabel=r"$x_i$")
    plot_status_graph(time, cost, figure_dir, file_name_prefix='cost_convergence', ylabel=r"$f_i(x)$", x_labels=["$f_1(x)$", "$f_2(x)$", "$f_3(x)$", "$f_4(x)$", "$f_5(x)$"])
    plot_dos_status_norm_converge_graph(time, partial_cost, figure_dir, file_name_prefix="partial_cost", ylabel="$log_{10}(||\\nabla_i\ f_i(x)||)$",xlabel_list=["$log_{10}(||\\nabla_1\ f_1(x)||)$", "$log_{10}(||\\nabla_2\ f_2(x)||)$", "$log_{10}(||\\nabla_3\ f_3(x)||)$", "$log_{10}(||\\nabla_4\ f_4(x)||)$", "$log_{10}(||\\nabla_5\ f_5(\omega)||)$"], dos_interval=dos_interval)
    plot_dos_status_norm_converge_graph(time, z_error, figure_dir, file_name_prefix="z_error", ylabel="$log_{10}(||z_i - x||)$",xlabel_list=["$log_{10}(||z_1 - x||)$", "$log_{10}(||z_2 - x||)$", "$log_{10}(||z_3 - x||)$", "$log_{10}(||z_4 - x||)$", "$log_{10}(||z_5 - x||)$"], dos_interval=dos_interval)
    # plot_status_graph(time, partial_cost, figure_dir, file_name_prefix='partial_cost', ylabel=r"$\nabla_if_i(x)$", x_labels=[r"$\nabla_1f_1(x)$", r"$\nabla_2f_2(x)$", r"$\nabla_3f_3(x)$", r"$\nabla_4f_4(x)$", r"$\nabla_5f_5(x)$"], equilibrium_value=0,ylim=(-1,1))
    # plot_status_graph(time, update_value, figure_dir, file_name_prefix='update_value', var_name='u', equilibrium_value=0)
    # plot_error_value_graph(time, y_vector, y_opt_value, figure_dir, ylabel_list=[r"$||y_1 - \bar{y}||$", r"$||y_2 - \bar{y}||$", r"$||y_3 - \bar{y}||$", r"$||y_4 - \bar{y}||$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.05))
    plot_error_value_graph(time, z_vector, z_opt_value, figure_dir, ylabel_list=[r"$||z_1 - x||$", r"$||z_2 - x||$", r"$||z_3 - x||$", r"$||z_4 - x||$", r"$||z_5 - x||$"], y_title=r"$||z_i - x||$", file_name_prefix='zi_status_error', xlim=(0, 0.05))

    initial_value_norms = [5, 15, 25, 35, 45, 55, 65, 75]
    asym_convergence_times = [1.84, 2.37, 2.57, 2.69, 2.79, 2.865, 2.925, 2.975] 
    fixed_convergence_times = [1.125, 1.524, 1.670, 1.76, 1.825, 1.875, 1.910, 1.945]
    finite_convergence_times = [1.252, 1.824, 2.075, 2.195, 2.435, 2.54, 2.695, 2.79]
    # # finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    # # finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    # asym_convergence_times = [value+0.04*i for i, value in enumerate(asym_convergence_times)]
    plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, finite_convergence_times, fixed_convergence_times], "$||e_x(0)||$", legneds=["Asymptotic algorithm", "Finite-time algorithm",  "Fixed-time algorithm"])





def plot_compare_graph(config_list):
    num_agents = 5
    model = "fixed_high_order"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    print(record_root_path)
    # 定义比对图的保存路径，可以放在 records/model 根目录下
    figure_dir = f"{current_dir}/records/{model}/compare_figures"
    os.makedirs(figure_dir, exist_ok=True)
    
    error_vectors = []
    labels = []
    common_time = None

    for config_index in config_list:
        record_path = record_root_path + str(config_index)
        memory = get_records_memory(record_path, num_agents)
        
        # 提取状态变量 x 和 time
        # 如果您想比较的是 vr，可以将 'x' 改为 'vr'
        x_vector = np.array(align_list(memory['x']))
        time = np.array(memory['time'][-1][:len(x_vector[0])])
        print(len(time))
        
        if common_time is None:
            common_time = time
            
        # 生成对应的最优理论轨迹 opt_value
        NE = np.array([0.12815768398783586, 0.6012151511176591, 1.0233494776122871, 1.2797928121775644, 2.000099746727306])
        opt_value = np.zeros(x_vector.shape)
        for i in range(len(time)):
            for j in range(x_vector.shape[0]):
                opt_value[j, i, :] = NE[j]
                
        # 计算该 config 下，每个时间步的全局误差 ||x - x*||
        diff_value_array = np.zeros(len(time))
        for i in range(len(time)):
            # x_vector[:, i, :] - opt_value[:, i, :] 是一个 (num_agents, dim) 的矩阵
            # np.linalg.norm 计算 Frobenius 范数，即所有智能体误差的平方和再开根号
            diff_matrix = x_vector[:, i, :] - opt_value[:, i, :]
            diff_value_array[i] = np.log10(np.linalg.norm(diff_matrix))
            
        error_vectors.append(diff_value_array)
        labels.append(f"Config {config_index}")
        
    # 调用之前编写的多组误差直接绘制函数
    plot_compare_direct_errors_graph(
        time=common_time,
        error_vectors=error_vectors,
        figure_dir=figure_dir,
        labels=["Asymptotic algorithm", "Finite-time algorithm", "Fixed-time algorithm"],
        ylabel=r"$||\mathbf{x} - \mathbf{x}^\star||$",
        file_name_prefix="x_status"
    )

if __name__ == "__main__":
    from config import config
    config_list = [ "c_0"]
    # config_index = "r_0"
    model = "fixed_linear"
    num_agents = 5
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        record_path = record_root_path + config_index
        memory = get_records_memory(record_path, num_agents)
        plot_graph(memory, record_path)

    # get_multi_initi_value_convergence_time(record_root_path + config_list[0], num_agents)

    # plot_compare_graph(["c_2", "c_0", "c_1"])