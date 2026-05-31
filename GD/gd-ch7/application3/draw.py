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
    
    status_vector = align_list(memory['vr']) # vr为NE seeking的参考信号
    y_vector = align_list(memory['y'])  # 所有智能体的私有目标的速度的平均值的估计
    z_vector = align_list(memory['z']) # 状态估计，（agents, time, status_vector.shape）
    x_vector = align_list(memory['x']) # 位置
    v_vector = align_list(memory['v']) # 速度
    NE_vector = align_list(memory['NE'])
    ui = align_list(memory['update_value']) # 控制输入
    time = np.array(memory['time'][-1][:len(y_vector[0])])

    y_opt_value = np.zeros(y_vector.shape)
    for i in range(len(time)):
        for j in range(y_vector.shape[0]):
            y_sum = 0
            for k in range(y_vector.shape[0]):
                y_sum += np.array(y_vector[k, i, :])
            y_opt_value[j, i, :] = y_sum/y_vector.shape[0]
    
    pg_opt_value = np.zeros((y_vector.shape[1], y_vector.shape[2]))
    for i in range(len(time)):
        target_pos = np.array([0.5*time[i],0 , 0])
        pg_opt_value[i, :] = target_pos
    
    z_opt_value = np.zeros(z_vector.shape)
    for i in range(len(time)):
        for j in range(z_vector.shape[0]):
            z_opt_value[j, i, :] = status_vector[:, i, :]
    

    # plot_multiple_time_slices(x_vector, figure_dir, global_target=pg_opt_value)
    plot_multi_dimension_status_converge_dynamic_graph(time, status_vector, figure_dir, opt_value=NE_vector, y_title=r'\omega', label_opt=r'\omega', label=r'\omega',file_name_prefix='virtual_status_convergence')


    target_times = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 10.0]

    # 转换为对应的数组索引 (四舍五入转为整数)
    time_indices = [int(round(t / 0.05)) for t in target_times]

    # 生成漂亮的时间标签，放在图片上
    time_labels = [
        r"$t=0$ s (Start)", 
        r"$t=1.57$ s (Max Spread)", 
        r"$t=3.14$ s (Half Cycle)", 
        r"$t=4.71$ s (Min Spread)", 
        r"$t=6.28$ s (Full Cycle)",
        r"$t=10.0$ s (Steady)"
    ]
    # plot_3d_separate_time_slices(
    #     status_vector=x_vector, 
    #     time_indices=time_indices, 
    #     figure_dir=figure_dir, 
    #     global_target=pg_opt_value,
    #     time_labels=time_labels,  # 传入对应的时间文本
    #     file_tag="experiment1_",
    #     draw_formation_net=True   # 开启阵型连线，强烈建议！
    # )
    plot_multi_dimension_status_converge_dynamic_graph(time, x_vector, figure_dir, opt_value=status_vector, y_title='x', label_opt='x', label='x',file_name_prefix='status_convergence')
    plot_3d_trajectory_global_graph(x_vector, figure_dir, pg_opt_value)
    plot_multi_dimension_status_converge_dynamic_graph(time, ui, figure_dir, opt_value=None, y_title='u', label_opt=r'\bar{y}', label=r'u',file_name_prefix='ui_status_covnergence')
    plot_error_value_graph(time, y_vector, y_opt_value, figure_dir, 
                           ylabel_list=[r"$||y_1 - \bar{y}||$", r"$||y_2 - \bar{y}||$", r"$||y_3 - \bar{y}||$", r"$||y_4 - \bar{y}||$", r"$||y_5 - \bar{y}||$", r"$||y_6 - \bar{y}||$", r"$||y_7 - \bar{y}||$", r"$||y_8 - \bar{y}||$", r"$||y_9 - \bar{y}||$"], 
                           y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.5))
    
    plot_error_value_graph(
    time, 
    z_vector, 
    z_opt_value, 
    figure_dir, 
    ylabel_list=[
        r"$||z_1 - \omega||$", 
        r"$||z_2 - \omega||$", 
        r"$||z_3 - \omega||$", 
        r"$||z_4 - \omega||$", 
        r"$||z_5 - \omega||$", 
        r"$||z_6 - \omega||$", 
        r"$||z_7 - \omega||$", 
        r"$||z_8 - \omega||$"
    ], 
    y_title=r"$||z_i - \omega||$", 
    file_name_prefix='zi_omega_error', 
    xlim=(0, 0.5)
)
    
    plot_error_value_graph(
    time, 
    x_vector, 
    status_vector, 
    figure_dir, 
    ylabel_list=[
        r"$||x_1 - \omega_1||$", 
        r"$||x_2 - \omega_2||$", 
        r"$||x_3 - \omega_3||$", 
        r"$||x_4 - \omega_4||$", 
        r"$||x_5 - \omega_5||$", 
        r"$||x_6 - \omega_6||$", 
        r"$||x_7 - \omega_7||$", 
        r"$||x_8 - \omega_8||$"
    ], 
    y_title=r"$||x_i - \omega_i||$", 
    file_name_prefix='xi_omega_error',  # 建议将文件名也同步修改
    xlim=(0, 5)
)

    # initial_value_norms = [15, 75, 135, 195, 255, 315, 375, 435]
    # asym_convergence_times = [3.74, 4.34, 4.58, 4.75, 4.91, 5.05, 5.15, 5.24] 
    # fixed_convergence_times = [3.12, 3.57, 3.68, 3.73, 3.77, 3.81, 3.83, 3.85]
    # # finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    # # finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    # asym_convergence_times = [value+0.04*i for i, value in enumerate(asym_convergence_times)]
    # plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, fixed_convergence_times], "$||e_\omega(0)||$", legneds=["Asymptotic algorithm",  "Fixed-time algorithm"])


def plot_compare_graph(config_list):
    num_agents = 8
    model = "air_ground_protection"
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
        print(x_vector.shape)
        NE_vector = align_list(memory['NE'])
        print(len(time))
        
        if common_time is None:
            common_time = time
            
        NE_vector = np.array(NE_vector)
        print(NE_vector.shape)
        # 计算该 config 下，每个时间步的全局误差 ||x - x*||
        diff_value_array = np.zeros(len(time))
        for i in range(len(time)):
            # x_vector[:, i, :] - opt_value[:, i, :] 是一个 (num_agents, dim) 的矩阵
            # np.linalg.norm 计算 Frobenius 范数，即所有智能体误差的平方和再开根号
            diff_matrix = x_vector[:, i, :2] - NE_vector[:, i, :2]
            diff_value_array[i] = np.linalg.norm(diff_matrix.flatten())
            # if config_index == "r_2":
            #     if diff_value_array[i] < 8e-2:
            #         diff_value_array[i] = 8e-2
            # diff_value_array[i] = np.linalg.norm(diff_matrix.flatten())
            
        error_vectors.append(diff_value_array)
        labels.append(f"Config {config_index}")
        
    # 调用之前编写的多组误差直接绘制函数
    plot_compare_direct_errors_graph(
        time=common_time,
        error_vectors=error_vectors,
        figure_dir=figure_dir,
        labels=["Asymptotic algorithm", "Fixed-time algorithm" ],
        ylabel=r"$||\mathbf{x} - \mathbf{x}^\star||$",
        file_name_prefix="x_status"
    )

if __name__ == "__main__":
    # 1
    from config import config
    config_list = [ "r_2"]
    # config_index = "r_0"
    model = "air_ground_protection"
    num_agents = 8
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
    # plot_compare_graph(config_list=["r_2", "r_1"])