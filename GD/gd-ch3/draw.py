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
    x_vector = align_list(memory['x'])
    partial_cost = align_list(memory['partial_cost'])
    update_value = align_list(memory['update_value'])
    time = np.array(memory['time'][-1][:len(x_vector[0])])

    opt_value = np.zeros(x_vector.shape)
    NE_point = np.array([0.12815768398783586, 0.6012151511176591, 1.0233494776122871, 1.2797928121775644, 2.000099746727306])
    for i in range(len(time)):
        for j in range(x_vector.shape[0]):
            opt_value[j, i, :] = NE_point[j]
    
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

    # plot_3d_trajectory_global_graph(x_vector, figure_dir, global_target=pg_opt_value)
    plot_status_converge_graph(time, x_vector, opt_value, figure_dir, file_name_prefix='status_convergence', ylabel=r"$x_i$")
    plot_status_graph(time, partial_cost, figure_dir, file_name_prefix='partial_cost', ylabel=r"$\nabla_if_i(x)$", x_labels=[r"$\nabla_1f_1(x)$", r"$\nabla_2f_2(x)$", r"$\nabla_3f_3(x)$", r"$\nabla_4f_4(x)$", r"$\nabla_5f_5(x)$"], equilibrium_value=0)
    plot_status_graph(time, update_value, figure_dir, file_name_prefix='update_value', var_name='u', equilibrium_value=0)
    # plot_error_value_graph(time, y_vector, y_opt_value, figure_dir, ylabel_list=[r"$||y_1 - \bar{y}||$", r"$||y_2 - \bar{y}||$", r"$||y_3 - \bar{y}||$", r"$||y_4 - \bar{y}||$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.05))
    plot_error_value_graph(time, z_vector, z_opt_value, figure_dir, ylabel_list=[r"$||z_1 - x||$", r"$||z_2 - x||$", r"$||z_3 - x||$", r"$||z_4 - x||$", r"$||z_5 - x||$"], y_title=r"$||z_i - x||$", file_name_prefix='zi_status_error', xlim=(0, 0.05))

    # initial_value_norms = [15, 75, 135, 195, 255, 315, 375, 435]
    # asym_convergence_times = [3.74, 4.34, 4.58, 4.75, 4.91, 5.05, 5.15, 5.24] 
    # fixed_convergence_times = [3.12, 3.57, 3.68, 3.73, 3.77, 3.81, 3.83, 3.85]
    # # finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    # # finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    # asym_convergence_times = [value+0.04*i for i, value in enumerate(asym_convergence_times)]
    # plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, fixed_convergence_times], "$||e_x(0)||$", legneds=["Asymptotic algorithm",  "Fixed-time algorithm"])

if __name__ == "__main__":
    from config import config
    config_list = [ "r_0"]
    # config_index = "r_0"
    model = "fixed_high_order"
    num_agents = 5
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    # for config_index in config_list:
    #     print(f"Running configuration: {config_index}")
    #     record_path = record_root_path + config_index
    #     memory = get_records_memory(record_path, num_agents)
    #     plot_graph(memory, record_path)

    get_multi_initi_value_convergence_time(record_root_path + config_list[0], num_agents)