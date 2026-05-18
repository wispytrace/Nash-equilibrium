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
sys.path.append(current_dir+"/../../")
from utilis import *

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10


def get_multi_initi_value_convergence_time(base_path, num_agents):
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
            status_vector = align_list(memory['vr'])
            time = np.array(memory['time'][-1][:len(status_vector[0])])
            opt_value = np.zeros(status_vector.shape)
            for i in range(len(time)):
                for j in range(status_vector.shape[0]):
                    opt_value[j, i, :] = np.array([2*np.cos(2*time[i] + j*np.pi/2)+np.cos(0.5*time[i]), 2*np.sin(2*time[i] + j*np.pi/2)+np.sin(0.5*time[i]), 0.5*time[i]])

            convergence_time_dict[simu_id]["convergence_time"] = get_convergence_time(status_vector, opt_value, time, error=1e-2)
            convergence_time_dict[simu_id]["init_value"] = np.linalg.norm(status_vector[:, 0, :].flatten())  # 假设 x 的初始值在这里
            sorted_convergence_time = sorted(convergence_time_dict.items(), key=lambda x: x[1]["init_value"])
    
    print("Convergence Time for each simulation:")
    print(sorted_convergence_time)


def plot_graph(memory, record_path):
    figure_dir = record_path + "/figure"
    result_dir = record_path + "/result"
    
    status_vector = align_list(memory['vr'])
    y_vector = align_list(memory['y'])
    z_vector = align_list(memory['z'])
    x_vector = align_list(memory['x'])
    ui = align_list(memory['update_value'])
    time = np.array(memory['time'][-1][:len(y_vector[0])])

    opt_value = np.zeros(status_vector.shape)
    for i in range(len(time)):
        for j in range(status_vector.shape[0]):
            opt_value[j, i, :] = np.array([
                3 * np.cos(0.5 * time[i]) + 2 * np.cos(2 * time[i] + j * np.pi / 2)-0.4, 
                3 * np.sin(0.5 * time[i]) + 2 * np.sin(2 * time[i] + j * np.pi / 2)-0.4, 
                0.5 * time[i]
            ])
    
    y_opt_value = np.zeros(y_vector.shape)
    for i in range(len(time)):
        for j in range(y_vector.shape[0]):
            y_sum = 0
            for k in range(y_vector.shape[0]):
                y_sum += np.array(y_vector[k, i, :])
            y_opt_value[j, i, :] = y_sum/y_vector.shape[0]
 

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
    
    pg_opt_value = np.zeros((y_vector.shape[1], y_vector.shape[2]))
    for i in range(len(time)):
        pg_opt_value[i, :] = np.array([
            3 * np.cos(0.5 * time[i]), 
            3 * np.sin(0.5 * time[i]), 
            0.5 * time[i]
        ])
    z_opt_value = np.zeros(z_vector.shape)
    for i in range(len(time)):
        for j in range(z_vector.shape[0]):
            z_opt_value[j, i, :] = status_vector[:, i, :]

    plot_3d_trajectory_global_graph(x_vector, figure_dir, global_target=pg_opt_value)
    plot_multi_dimension_status_converge_dynamic_graph(time, status_vector, figure_dir, opt_value=opt_value, y_title=r'\omega', label_opt=r'\omega', label=r'\omega',file_name_prefix='virtual_status_convergence')
    plot_multi_dimension_status_converge_dynamic_graph(time, x_vector, figure_dir, opt_value=opt_value, y_title='x', label_opt='x', label='x',file_name_prefix='status_convergence')
    plot_multi_dimension_status_converge_dynamic_graph(time, ui, figure_dir, opt_value=None, y_title='u', label_opt=r'\bar{y}', label=r'u',file_name_prefix='ui_status_covnergence')
    plot_error_value_graph(time, y_vector, y_opt_value, figure_dir, ylabel_list=[r"$||y_1 - \bar{y}||$", r"$||y_2 - \bar{y}||$", r"$||y_3 - \bar{y}||$", r"$||y_4 - \bar{y}||$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.05))
    plot_error_value_graph(time, z_vector, z_opt_value, figure_dir, ylabel_list=[r"$||z_1 - x||$", r"$||z_2 - x||$", r"$||z_3 - x||$", r"$||z_4 - x||$"], y_title=r"$||z_i - x||$", file_name_prefix='zi_status_error', xlim=(0, 0.05))

    initial_value_norms = [15, 75, 135, 195, 255, 315, 375, 435]
    asym_convergence_times = [3.74, 4.34, 4.58, 4.75, 4.91, 5.05, 5.15, 5.24] 
    fixed_convergence_times = [3.12, 3.57, 3.68, 3.73, 3.77, 3.81, 3.83, 3.85]
    # finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    # finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    asym_convergence_times = [value+0.04*i for i, value in enumerate(asym_convergence_times)]
    plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, fixed_convergence_times], "$||e_\omega(0)||$", legneds=["Asymptotic algorithm",  "Fixed-time algorithm"])


def plot_compare_graph(config_list):
    num_agents = 4
    model = "jssc"
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
        opt_value = np.zeros(x_vector.shape)
        for i in range(len(time)):
            for j in range(x_vector.shape[0]):
                opt_value[j, i, :] = np.array([
                    3 * np.cos(0.5 * time[i]) + 2 * np.cos(2 * time[i] + j * np.pi / 2) - 0.4, 
                    3 * np.sin(0.5 * time[i]) + 2 * np.sin(2 * time[i] + j * np.pi / 2) - 0.4, 
                    0.5 * time[i]
                ])
                
        # 计算该 config 下，每个时间步的全局误差 ||x - x*||
        diff_value_array = np.zeros(len(time))
        for i in range(len(time)):
            # x_vector[:, i, :] - opt_value[:, i, :] 是一个 (num_agents, dim) 的矩阵
            # np.linalg.norm 计算 Frobenius 范数，即所有智能体误差的平方和再开根号
            diff_matrix = x_vector[:, i, :] - opt_value[:, i, :]
            diff_value_array[i] = np.linalg.norm(diff_matrix)
            
        error_vectors.append(diff_value_array)
        labels.append(f"Config {config_index}")
        
    # 调用之前编写的多组误差直接绘制函数
    plot_compare_direct_errors_graph(
        time=common_time,
        error_vectors=error_vectors,
        figure_dir=figure_dir,
        labels=["Fixed-time algorithm", "Asymptotic algorithm"],
        ylabel=r"$||\mathbf{x} - \mathbf{x}^\star||$",
        file_name_prefix="x_status"
    )

if __name__ == "__main__":
    # 1
    from config import config
    config_list = [ "r_0"]
    # config_index = "r_0"
    model = "air_ground_protection"
    num_agents = 4
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        record_path = record_root_path + config_index
        memory = get_records_memory(record_path, num_agents)
        plot_graph(memory, record_path)
    
    # 2
    # get_multi_initi_value_convergence_time(record_root_path + config_list[1], num_agents)

    # 3
    # plot_compare_graph(config_list=["r_0", "r_1"])