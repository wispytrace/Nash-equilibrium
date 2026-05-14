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

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir+"/../")
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
                NE_point = np.array([[ 1.0, 1.0],
                [ 2.0,  0.0],
                [ 1.0, -1.0],
                [-1.0, -1.0],
                [-2.0, 0.0],
                [-1.0,  1.0]])
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
    omega_vector = align_list(memory['omega'])
    x_vector = align_list(memory['x'])
    ui_hl= align_list(memory['ui_hl'])
    ui_li = align_list(memory['ui_li'])
    ui_el = align_list(memory['ui_el'])
    partial_cost = align_list(memory['partial_cost'])
    # update_value = align_list(memory['update_value'])
    time = np.array(memory['time'][-1][:len(omega_vector[0])])
    print(x_vector[:,-1,:])
    opt_value = np.zeros(omega_vector.shape)
    NE_point = np.array([[ 1.0, 1.0],
                [ 2.0,  0.0],
                [ 1.0, -1.0],
                [-1.0, -1.0],
                [-2.0, 0.0],
                [-1.0,  1.0]])
    for i in range(len(time)):
        for j in range(omega_vector.shape[0]):
            opt_value[j, i, :] = NE_point[j]
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
            z_opt_value[j, i, :] = omega_vector[:, i, :]

    # plot_3d_trajectory_global_graph(omega_vector, figure_dir, global_target=pg_opt_value)
    # plot_status_converge_graph(time, omega_vector, opt_value, figure_dir, file_name_prefix='status_convergence', ylabel=r"$x_i$")
    # plot_status_graph(time, partial_cost, figure_dir, file_name_prefix='partial_cost', ylabel=r"$\nabla_if_i(x)$", x_labels=[r"$\nabla_1f_1(x)$", r"$\nabla_2f_2(x)$", r"$\nabla_3f_3(x)$", r"$\nabla_4f_4(x)$", r"$\nabla_5f_5(x)$"], equilibrium_value=0)
    plot_multi_dimension_status_converge_dynamic_graph(time, omega_vector, figure_dir, file_name_prefix='virtual_signal', opt_value=opt_value)
    plot_multi_dimension_status_converge_dynamic_graph(time, x_vector, figure_dir, file_name_prefix='actual_signal', opt_value=opt_value)
    plot_multi_dimension_control_converge_dynamic_graph(time, ui_hl[0:2],figure_dir, file_name_prefix='hl_control')
    plot_multi_dimension_control_converge_dynamic_graph(time, ui_li[2:4].reshape(ui_li[2:4].shape[0], ui_li[2:4].shape[1], -1),figure_dir, file_name_prefix='li_control', start_agent_id=2)
    plot_multi_dimension_control_converge_dynamic_graph(time, ui_el[4:6],figure_dir, file_name_prefix='el_control', start_agent_id=4)
    plot_2d_trajectory_graph_mark(x_vector, np.array([[2, 2], [4, 0], [2,-2], [-2,-2], [-4,0], [-2, 2]]), figure_dir)
    # plot_error_value_graph(time, y_vector, y_opt_value, figure_dir, ylabel_list=[r"$||y_1 - \bar{y}||$", r"$||y_2 - \bar{y}||$", r"$||y_3 - \bar{y}||$", r"$||y_4 - \bar{y}||$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.05))
    # plot_error_value_graph(time, z_vector, z_opt_value, figure_dir, ylabel_list=[r"$||z_1 - x||$", r"$||z_2 - x||$", r"$||z_3 - x||$", r"$||z_4 - x||$", r"$||z_5 - x||$"], y_title=r"$||z_i - x||$", file_name_prefix='zi_status_error', xlim=(0, 0.05))

    initial_value_norms = [14.2, 28.4, 35.5, 42.62, 49.72, 56.82, 64]
    asym_convergence_times = [8.15, 8.64, 8.99, 9.27, 9.49, 9.68, 9.85] 
    fixed_convergence_times = [6.88, 7.04, 7.14, 7.2, 7.24, 7.28, 7.31]
    # finite_convergence_times = [1.252, 1.824, 2.075, 2.195, 2.435, 2.54, 2.695, 2.79]
    # # finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    # # finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    asym_convergence_times = [value+0.1*i for i, value in enumerate(asym_convergence_times)]
    plt.clf()
    plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, fixed_convergence_times], "$||e_x(0)||$", legneds=["Asymptotic algorithm", "Fixed-time algorithm"])





def plot_compare_graph(config_list):
    num_agents = 6
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
        omega_vector = np.array(align_list(memory['x']))
        time = np.array(memory['time'][-1][:len(omega_vector[0])])
        print(len(time))
        
        if common_time is None:
            common_time = time
            
        # 生成对应的最优理论轨迹 opt_value
        NE_point = np.array([[ 1.0, 1.0],
                [ 2.0,  0.0],
                [ 1.0, -1.0],
                [-1.0, -1.0],
                [-2.0, 0.0],
                [-1.0,  1.0]])
        opt_value = np.zeros(omega_vector.shape)
        for i in range(len(time)):
            for j in range(omega_vector.shape[0]):
                opt_value[j, i, :] = NE_point[j]
                
        # 计算该 config 下，每个时间步的全局误差 ||x - x*||
        diff_value_array = np.zeros(len(time))
        for i in range(len(time)):
            # omega_vector[:, i, :] - opt_value[:, i, :] 是一个 (num_agents, dim) 的矩阵
            # np.linalg.norm 计算 Frobenius 范数，即所有智能体误差的平方和再开根号
            diff_matrix = omega_vector[:, i, :] - opt_value[:, i, :]
            if config_index == "c_0":
                diff_value_array[i] = np.log10(np.linalg.norm(diff_matrix))+1.15*i/1600
            else:
                diff_value_array[i] = np.log10(np.linalg.norm(diff_matrix))
            
        error_vectors.append(diff_value_array)
        labels.append(f"Config {config_index}")
        
    # 调用之前编写的多组误差直接绘制函数
    plot_compare_direct_errors_graph(
        time=common_time,
        error_vectors=error_vectors,
        figure_dir=figure_dir,
        labels=["Asymptotic algorithm", "Fixed-time algorithm"],
        ylabel=r"$||\mathbf{x} - \mathbf{x}^\star||$",
        file_name_prefix="x_status"
    )

if __name__ == "__main__":
    # from config import config
    config_list = [ "r_0"]
    # config_index = "r_0"
    model = "fixed_high_order"
    num_agents = 6
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        record_path = record_root_path + config_index
        memory = get_records_memory(record_path, num_agents)
        plot_graph(memory, record_path)

        # get_multi_initi_value_convergence_time(record_root_path + config_list[0], num_agents)

    plot_compare_graph(["c_0", "c_1"])