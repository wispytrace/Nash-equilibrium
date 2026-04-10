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
    time = np.array(memory['time'][-1][:len(y_vector[0])])

    opt_value = np.zeros(status_vector.shape)
    for i in range(len(time)):
        for j in range(status_vector.shape[0]):
            opt_value[j, i, :] = np.array([2*np.cos(2*time[i] + j*np.pi/2)+np.cos(0.5*time[i]), 2*np.sin(2*time[i] + j*np.pi/2)+np.sin(0.5*time[i]), 0.5*time[i]])
    
    y_opt_value = np.zeros(y_vector.shape)
    for i in range(len(time)):
        for j in range(y_vector.shape[0]):
            y_sum = 0
            for k in range(y_vector.shape[0]):
                y_sum += np.array([-4*np.sin(2*time[i] + k*np.pi/2), 4*np.cos(2*time[i] + k*np.pi/2), 0.5])
            y_opt_value[j, i, :] = y_sum/y_vector.shape[0]
    
    z_opt_value = np.zeros(z_vector.shape)
    for i in range(len(time)):
        for j in range(z_vector.shape[0]):
            z_opt_value[j, i, :] = status_vector[:, i, :]

    plot_3d_trajectory_graph(x_vector, figure_dir, "status")
    plot_multi_dimension_status_converge_dynamic_graph(time, status_vector, figure_dir, opt_value=opt_value, y_title=r'\omega', label_opt=r'\omega', label=r'\omega',file_name_prefix='virtual_status_convergence')
    plot_multi_dimension_status_converge_dynamic_graph(time, x_vector, figure_dir, opt_value=opt_value, y_title='x', label_opt='x', label='x',file_name_prefix='status_convergence')
    plot_error_value_graph(time, y_vector,y_opt_value, figure_dir, ylabel_list=[r"$y_1 - \bar{y}$", r"$y_2 - \bar{y}$", r"$y_3 - \bar{y}$", r"$y_4 - \bar{y}$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.5))
    plot_error_value_graph(time, z_vector,z_opt_value, figure_dir, ylabel_list=[r"$z_1 - \omega$", r"$z_2 - \omega$", r"$z_3 - \omega$", r"$z_4 - \omega$"], y_title=r"$||z_i - \omega||$", file_name_prefix='zi_status_error', xlim=(0, 1))


if __name__ == "__main__":
    from config import config
    config_list = [ "r_0", "r_1"]
    # config_index = "r_0"
    model = "jssc"
    num_agents = 4
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    # for config_index in config_list:
    #     print(f"Running configuration: {config_index}")
    #     record_path = record_root_path + config_index
    #     memory = get_records_memory(record_path, num_agents)
    #     plot_graph(memory, record_path)

    get_multi_initi_value_convergence_time(record_root_path + config_list[1], num_agents)