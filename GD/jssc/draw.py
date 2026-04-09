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



def plot_graph(memory, record_path):
    figure_dir = record_path + "/figure"
    result_dir = record_path + "/result"
    
    status_vector = align_list(memory['vr'])
    y_vector = align_list(memory['y'])
    z_vector = align_list(memory['z'])

    time = np.array(memory['time'][-1][:len(y_vector[0])])
    # ui = self.align_list(memory['ui'])


    # oi = self.align_list(memory['oi'])
    # track_error = self.align_list(memory['track_error'])
    # dot_x = self.align_list(memory['dot_x'])
    # dot_y = self.align_list(memory['doty'])
    # dot_track_error = self.align_list(memory['dot_track_error'])
    # partial_cost = self.align_list(memory['partial_cost'])
            
    # config_index_list = ["0","0_5", "0_3", "0_4"]
    # self.plot_compared_graph(config_index_list,figure_dir)
    # opt_value = np.array([[-0.9999385195244029, 0.16660663117309651, 0.3333955560508185], [4.8152365222943084e-05, -0.8334287020540452, 0.3334118895597235], [1.0001074180788625, 0.16654159249067055, 0.33345503463020215], [4.815265801524303e-05, 1.166571297885446, 0.33341188945528427], [2.0000614803817807, 0.16660663112190888, 3.333395556015464], [-1.999953791549649, 0.16663322749224166, 3.3333734276470937]])
    # plot_status_error_graph(time, virtual_vector, figure_dir, ylabel_list=["$\omega_{i1} - y_{i1}^*$", "$\omega_{i2} - y_{i2}^*$", "$\omega_{i3} - y_{i3}^*$"], opt_value=opt_value)
    # plot_status_error_graph(time, valid_status_vector, figure_dir, var_name='y', file_name_prefix='actual', opt_value=opt_value)
    # plot_3d_trajectory_graph(valid_status_vector, figure_dir, "status", p_center=np.array([0, 0.5, 2]), var_name='y')
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

    # plot_3d_trajectory_graph(status_vector, figure_dir, "virtual_status")
    plot_multi_dimension_status_converge_dynamic_graph(time, status_vector, figure_dir, opt_value=opt_value, y_title='\omega', label_opt='\omega', label='\omega',file_name_prefix='virtual_status_convergence')
    plot_error_value_graph(time, y_vector,y_opt_value, figure_dir, ylabel_list=[r"$y_1 - \bar{y}$", r"$y_2 - \bar{y}$", r"$y_3 - \bar{y}$", r"$y_4 - \bar{y}$"], y_title=r"$||y_i - \bar{y}||$", file_name_prefix='yi_status_error', xlim=(0, 0.5))
    plot_error_value_graph(time, z_vector,z_opt_value, figure_dir, ylabel_list=[r"$z_1 - \omega$", r"$z_2 - \omega$", r"$z_3 - \omega$", r"$z_4 - \omega$"], y_title=r"$||z_i - \omega||$", file_name_prefix='zi_status_error', xlim=(0, 1))
    # plot_status_graph(time, valid_speed_vector[2:, :], figure_dir, file_name_prefix="speed", ylabel_list=["$x_{i21}$", "$x_{i22}$", "$x_{i23}$"],xlabel_list=["Player 3", "Player 4", "Player 5", "Player 6"])



    # self.plot_assemble_estimation_graph(time, [estimate_vector], [virtual_vector], figure_dir, "virtual_status_estimate")
    

if __name__ == "__main__":
    from config import config
    config_list = [ "r_0"]
    # config_index = "r_0"
    model = "jssc"
    num_agents = 4
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records/{model}/"
    
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        record_path = record_root_path + config_index
        memory = get_records_memory(record_path, num_agents)
        plot_graph(memory, record_path)
