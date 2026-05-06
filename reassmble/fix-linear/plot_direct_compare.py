
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
sys.path.append(current_dir+"/../../GD")
from utilis import *

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

def plot_compare_direct_errors_graph(time, error_vectors, figure_dir, labels=None, ylabel=None, file_name_prefix=None):
    """
    直接绘制并比对多组误差数据的收敛曲线。
    
    参数:
        time: 共享的时间数组 (1D array)
        error_vectors: 包含多组误差数据的列表或数组 (例如: [error_algo1, error_algo2, ...])
        figure_dir: 图片保存路径
        labels: 图例标签列表
        ylabel: Y轴标签
        file_name_prefix: 保存文件名的前缀
    """
    os.makedirs(figure_dir, exist_ok=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    print(f"Plotting {len(error_vectors)} error trajectories...")
    
    # 如果没有传入标签，自动生成 Test 1, Test 2...
    labels = ["Test " + str(i+1) for i in range(len(error_vectors))] if labels is None else labels

    # 直接遍历并绘制传入的误差向量
    for i, error_vector in enumerate(error_vectors):
        error_vector = np.array(error_vector)
        # 确保 error_vector 是一维的
        if error_vector.ndim > 1:
            error_vector = error_vector.flatten()
            
        plt.plot(time, error_vector, color=colors[i % len(colors)], label=labels[i], linewidth=1.5)

    # 设置坐标轴标签 (如果需要自定义字体，可在此处加上 fontproperties=prop)
    plt.xlabel('Time (sec)', fontsize=15)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    else:
        plt.ylabel("Convergence Error", fontsize=15)
        
    plt.legend(fontsize=12, loc='upper right')
    
    # 动态设置 X 轴边界为时间的起止点
    plt.xlim(left=time[0] if len(time) > 0 else 0, right=time[-1] if len(time) > 0 else 8)
    # plt.ylim(bottom=0)
    
    plt.tight_layout()

    # 构造文件名并保存
    if file_name_prefix:
        fname = f"{file_name_prefix}_compare_error.png"
    else:
        fname = "compare_error.png"
        
    path = os.path.join(figure_dir, fname)
    plt.savefig(path, dpi=300) # 建议加上 dpi=300 保证论文插图清晰度
    plt.close()
    print(f"Saved figure: {path}")

def plot_compare_graph(config_list):
    num_agents = 4
    model = "fixed_linear"
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
        x_vector = np.array(align_list(memory['y']))
        time = np.array(memory['time'][-1][:len(x_vector[0])])
        print(len(time))
        
        if common_time is None:
            common_time = time
            
        # 生成对应的最优理论轨迹 opt_value
        NE = np.array([5.748618334849947, 15.552539709004664, 25.35645648812881, 35.16037820240961]).reshape(-1, 1)
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
            diff_value_array[i] = np.linalg.norm(diff_matrix)
            
        error_vectors.append(diff_value_array)
        labels.append(f"Config {config_index}")
        
    # 调用之前编写的多组误差直接绘制函数
    plot_compare_direct_errors_graph(
        time=common_time,
        error_vectors=error_vectors,
        figure_dir=figure_dir,
        labels=["Exponential algorithm", "Finite-time algorithm", "Fixed-time algorithm"],
        ylabel=r"$||\mathbf{x} - \mathbf{x}^\star||$",
        file_name_prefix="x_status"
    )



if __name__ == "__main__":
    # from config import config
    # config_list = [ "r_0"]
    # # config_index = "r_0"
    # model = "fixed_high_order"
    # num_agents = 5
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # record_root_path = f"{current_dir}/records/{model}/"
    
    # for config_index in config_list:
    #     print(f"Running configuration: {config_index}")
    #     record_path = record_root_path + config_index
    #     memory = get_records_memory(record_path, num_agents)
    #     plot_graph(memory, record_path)

    # get_multi_initi_value_convergence_time(record_root_path + config_list[0], num_agents)

    plot_compare_graph([ "c_1","c_3",   "c_4"])