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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d plots

mpl.rcParams['figure.dpi'] = 600 
# mpl.rcParams['lines.linewidth'] = 1
current_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(current_dir, 'resource/font/Times New Roman.ttf')
plt.rcParams['mathtext.fontset'] = 'stix'

from matplotlib.font_manager import FontProperties, fontManager

prop = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = prop.get_name()
fontManager.addfont(font_path)


def align_list(irregular_list):
    min_length = min(len(lst) for lst in irregular_list)
    uniform_array = np.array([lst[:min_length] for lst in irregular_list])
    return uniform_array

def get_records_memory(record_path, agent_nums):
    """
    读取指定路径下的 agent 记录，并将其提取整合为 memory 字典。
    
    :param record_path: 记录文件所在的目录路径
    :param agent_nums: agent 的数量 (N)
    :return: 包含所有 agent 数据的 memory 字典
    """
    # 1. 读取记录 (对应原 read_records)
    records = []
    for i in range(agent_nums):
        file_path = f'{record_path}/agent_{i}.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            record_str = f.read()
            records.append(json.loads(record_str))
            
    # 2. 提取并整合记录 (对应原 extract_records)
    memory = defaultdict(list)

    for record in records:
        vector = defaultdict(list)
        for item in record:
            for k, v in item.items():
                vector[k].append(v)
        
        # 将当前 agent 的 vector 归集到全局 memory 中
        for k, v in vector.items():
            memory[k].append(v)

    # 转换为普通 dict 返回（如果您后续严格需要 defaultdict，可以直接返回 memory）
    return dict(memory)

def plot_multi_dimension_status_converge_dynamic_graph(
    time,
    status_vector,
    figure_dir,
    opt_value=None,
    y_title="x",
    label_opt="x",
    label = "x",
    file_name_prefix="dynamic_convergence",
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape

    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for d in range(D):
        for i in range(N):
            y = status_vector[i, :, d]
            color = colors[i % len(colors)]
            label_string = r'$%s_{%d%d}$' % (label, i+1, d+1)
            opt_label_string = r'$%s_{%d%d}^{\star}$' % (label_opt, i+1, d+1)
            plt.plot(time, y, color=color, label=label_string)
            if opt_value is not None:
                plt.plot(time, opt_value[i,:, d], color=color, linestyle='dashed', label=opt_label_string)
                    
        plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
        y_title_string = r'$%s_{i%d}$' % (y_title, d+1)
        plt.ylabel(y_title_string, fontsize=15, fontproperties=prop)

        plt.legend(fontsize=12, loc='upper right', ncol=2)
        plt.xlim(left=0, right=time[-1])
        plt.tight_layout()

        y_min = np.min(status_vector[:, :, d])
        y_max = np.max(status_vector[:, :, d])
        # 增加 30% 的顶部空间
        padding = (y_max - y_min) * 0.4 if y_max != y_min else 1.0
        plt.ylim(y_min - padding * 0.1, y_max + padding)    

        fname = f"{file_name_prefix}_dim{d+1}.png"
        path = os.path.join(figure_dir, fname)
        plt.savefig(path)
        plt.close()
        print(f"Saved figure: {path}")


def plot_error_value_graph(
    time,
    status_vector,
    target_vector,
    figure_dir,
    file_name_prefix='absolute_error',
    ylabel_list=None,
    y_title='||x - x*||',
    xlim=None
):
    os.makedirs(figure_dir, exist_ok=True)
    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape[:3]
    
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(N):
        error_value = np.zeros(T)
        for j in range(T):
            error_value[j] = np.linalg.norm((status_vector[i, j, :] - target_vector[i, j, :]).flatten(), ord=2)

        color = colors[i % len(colors)]
        plt.plot(time, error_value, color=color, label=ylabel_list[i] if ylabel_list is not None else f"Player {i+1}", linewidth=1)
        plt.hlines(
        0, xmin=time[0], xmax=time[-1],
        colors='black', linestyles='dashed', linewidth=0.8)
    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    plt.ylabel(y_title, fontsize=15, fontproperties=prop)

    plt.legend(fontsize=12, loc='upper right')
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(left=0, right=time[-1])
    plt.tight_layout()

    fname = f"{file_name_prefix}.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")



def plot_status_converge_graph(
    time,
    status_vector,
    opt_value,
    figure_dir,
    file_name_prefix=None,
    var_name='x',
    ylabel=None,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape

    colors = list(mcolors.TABLEAU_COLORS.values())

    for i in range(N):
        color = colors[i % len(colors)]
        plt.plot(time, status_vector[i, :, :], color=color, label=f"${var_name}_{i+1}$")
        plt.plot(time, opt_value[i, :, :], color=color, linestyle='dashed', label=f"${var_name}_{i+1}^*$")
        
    
    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=15, fontproperties=prop)
    else:
        plt.ylabel(f"${var_name}_i$", fontsize=15, fontproperties=prop)
    
    plt.legend(fontsize=12, loc='upper right', ncol=2)
    plt.xlim(left=0, right=time[-1])
    plt.tight_layout()
    y_min = np.min(status_vector)
    y_max = np.max(status_vector)
    # 增加 30% 的顶部空间
    padding = (y_max - y_min) * 0.4 if y_max != y_min else 1.0
    plt.ylim(y_min - padding * 0.1, y_max + padding)
    if file_name_prefix:
        fname = f"{file_name_prefix}.png"
    else:
        fname = f"status_convergence.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")

def plot_status_graph(
    time,
    status_vector,
    figure_dir,
    file_name_prefix=None,
    var_name='x',
    ylabel=None,
    x_labels = None,
    xlim=None,
    ylim=None,
    equilibrium_value=None,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i in range(N):
        color = colors[i % len(colors)]
        if x_labels is None:
            plt.plot(time, status_vector[i, :, :], color=color, label=f"${var_name}_{i+1}$")
        else:
            plt.plot(time, status_vector[i, :, :], color=color, label=x_labels[i])
    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=15, fontproperties=prop)
    else:
        plt.ylabel(f"${var_name}_i$", fontsize=15, fontproperties=prop)
    plt.legend(fontsize=12, loc='upper right')
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(left=0, right=time[-1])
    if ylim is not None:
        plt.ylim(ylim)
    if equilibrium_value is not None:
        plt.hlines(
        equilibrium_value, xmin=time[0], xmax=time[-1],
        colors='black', linestyles='dashed', linewidth=0.8)
    plt.tight_layout()
    y_min = np.min(status_vector)
    y_max = np.max(status_vector)
    # 增加 30% 的顶部空间
    padding = (y_max - y_min) * 0.4 if y_max != y_min else 1.0
    plt.ylim(y_min - padding * 0.1, y_max + padding)
    if file_name_prefix:
        fname = f"{file_name_prefix}.png"
    else:
        fname = f"status.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")




def plot_3d_trajectory_global_graph(status_vector, figure_dir, global_target, file_tag="", p_center=None, var_name='x'):
    """
    status_vector: numpy array (N, T, 3), N条轨迹，每条T步，三维坐标
    figure_dir: 保存图片的目录，MATLAB风格绘图
    """
    os.makedirs(figure_dir, exist_ok=True)
    # MATLAB默认颜色序列
    matlab_colors = [
        '#0072BD',  # 蓝色
        '#77AC30',  # 绿色
        '#EDB120',  # 黄色
        '#7E2F8E',  # 紫色
        '#D95319',  # 橙色
        '#77AC30',  # 绿色
        '#4DBEEE',  # 淡蓝
        "#FC0733",  # 红褐色
    ]

    status_vector = np.array(status_vector)
    N = status_vector.shape[0]

    # 创建图形，使用MATLAB默认大小比例，稍微增大以容纳标签
    plt.figure(figsize=(16, 12))
    ax = plt.subplot(111, projection='3d')

    # 设置背景色为白色，MATLAB风格
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.7, color='#D9D9D9')

    # 绘制轨迹，使用MATLAB样式
    for i in range(N):
        x = status_vector[i, :, 0]
        y = status_vector[i, :, 1]
        z = status_vector[i, :, 2]
        # z = np.zeros((status_vector.shape[1]))
        color = matlab_colors[i % len(matlab_colors)]
        
        # MATLAB风格的线条更粗
        ax.plot(x, y, z,
                color=color,
                linestyle='-',
                linewidth=2.0,
                label=f'Player {i+1}')
                
        # 起点和终点标记，更像MATLAB的默认标记大小
        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=80, edgecolor='k', zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='s', s=80, edgecolor='k', zorder=5)
    
    px = global_target[:, 0]
    py = global_target[:, 1]
    pz = global_target[:, 2]
    # pz = np.zeros((status_vector.shape[1]))
    color = matlab_colors[-1]
    ax.plot(px, py, pz, color=color, linestyle='--', linewidth=2.0, label="Global target")

    ax.scatter(px[0], py[0], pz[0], 
            color=color,    # 边缘颜色（或者设为 'k' 黑色边框）
            marker='o',          # 五角星形状
            s=80,               # 尺寸调大一点
            zorder=5)

    # 2. 绘制【实心五角星】（作为终点）
    # 关键设置: color=color 直接填充，edgecolors='k' 加一圈黑边增加立体感
    ax.scatter(px[-1], py[-1], pz[-1], 
            color=color,         # 内部实心填充
            edgecolors='k',      # 黑色描边（与你的圆形/方形风格保持一致）
            marker='*',          # 五角星形状
            s=200,               # 尺寸调大一点
            zorder=5)

    # print("global target:", global_target[-1])
    # print("individual target:", status_vector[:, -1, :])

    ring_x = [status_vector[i, -1, 0] for i in range(N)]
    ring_y = [status_vector[i, -1, 1] for i in range(N)]
    ring_z = [status_vector[i, -1, 2] for i in range(N)]

    # 为了让线条形成一个闭合的环 (Agent 1 -> 2 -> 3 -> 4 -> 1)
    # 我们把第一个智能体的坐标再次加到列表末尾
    ring_x.append(ring_x[0])
    ring_y.append(ring_y[0])
    ring_z.append(ring_z[0])

    # 绘制这层“保护网”
    ax.plot(ring_x, ring_y, ring_z, 
            color='black',        # 使用灰色，高级且不喧宾夺主
            linestyle='--',      # 虚线样式
            linewidth=1.5,       # 线宽稍微细一点，作为辅助线
            alpha=0.8,           # 增加透明度，体现出“虚拟拓扑连接”的质感
            zorder=4)            # 图层放在点(5)下面，主线上面

    # MATLAB风格的轴标签 - 增加labelpad以确保z轴标签可见
    ax.set_xlabel(f"${var_name}_{{i1}}$ (m)", fontsize=16, labelpad=10)
    ax.set_ylabel(f"${var_name}_{{i2}}$ (m)", fontsize=16, labelpad=10)
    ax.set_zlabel(f"${var_name}_{{i3}}$ (m)", fontsize=16, labelpad=15)  # z轴增加更多间距

    # 轴刻度字体大小，MATLAB风格
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12, pad=8)  # z轴刻度标签额外间距

    # 设置轴边框颜色，MATLAB风格
    ax.xaxis.pane.set_edgecolor('#D9D9D9')
    ax.yaxis.pane.set_edgecolor('#D9D9D9')
    ax.zaxis.pane.set_edgecolor('#D9D9D9')

    # 设置坐标面板填充颜色为白色或透明
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False



    # 设置MATLAB默认视角，稍微调整以更好显示z轴标签
    ax.view_init(elev=25, azim=65)
    # MATLAB风格图例

    # 添加MATLAB风格的边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 调整布局以确保标签可见
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.legend(fontsize=18)
    # 保存高分辨率图像 - 使用pad_inches而不是bbox_inches='tight'
    plt.savefig(os.path.join(figure_dir, file_tag+"3d_trajectories.png"), 
                dpi=600, 
                bbox_inches='tight',
                pad_inches=0.2)  # 增加边距以确保标签不被裁剪

    plt.close()
    print(f"Saved MATLAB-style figure: {os.path.join(figure_dir, file_tag+'3d_trajectories.png')}")


def plot_2d_trajectory_graph(status_vector, figure_dir):
    plt.clf()
    colors = list(mcolors.TABLEAU_COLORS.keys())
    status_vector = np.array(status_vector)
    shape = status_vector.shape
    for i in range(shape[0]):
        x = status_vector[i,:,0]
        y = status_vector[i,:,1]
        plt.plot(x, y, 
            color=colors[i],
            linestyle='-',
            linewidth=1,
            alpha=0.7,
            label=f'Player {i+1}')
        
        # 标记起始点和终点
        plt.scatter(x[0], y[0], color=colors[i], marker='o', s=50, edgecolor='black')
        plt.scatter(x[-1], y[-1], color=colors[i], marker='s', s=50, edgecolor='black')
    plt.xlabel("$x_{i1}$(m)", fontsize=15)
    plt.ylabel("$x_{i2}$(m)", fontsize=15)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),  # 留出2%的边距
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        facecolor='white',
        fontsize=11,
        borderpad=0.8,
        borderaxespad=0.5
    )
    # 自动调整坐标轴范围
    margin = 0.1  # 10%的边界留白
    x_min, x_max = np.min(status_vector[:, :, 0]), np.max(status_vector[:, :, 0])
    y_min, y_max = np.min(status_vector[:, :, 1]), np.max(status_vector[:, :, 1])
    plt.xlim(x_min - (x_max - x_min)*margin, x_max + (x_max - x_min)*5*margin)
    plt.ylim(y_min - (y_max - y_min)*margin, y_max + (y_max - y_min)*5*margin)

    # 显示图表
    plt.tight_layout()
    plt.savefig(figure_dir + "/2d_trajectories.png")

def plot_3d_trajectory_graph(status_vector, figure_dir, file_tag="", p_center=None, var_name='x'):
    """
    status_vector: numpy array (N, T, 3), N条轨迹，每条T步，三维坐标
    figure_dir: 保存图片的目录，MATLAB风格绘图
    """
    os.makedirs(figure_dir, exist_ok=True)
    # MATLAB默认颜色序列
    matlab_colors = [
        '#0072BD',  # 蓝色
        '#D95319',  # 橙色
        '#EDB120',  # 黄色
        '#7E2F8E',  # 紫色
        '#77AC30',  # 绿色
        '#4DBEEE',  # 淡蓝
        '#A2142F',  # 红褐色
    ]

    status_vector = np.array(status_vector)
    N = status_vector.shape[0]

    # 创建图形，使用MATLAB默认大小比例，稍微增大以容纳标签
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d')

    # 设置背景色为白色，MATLAB风格
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.7, color='#D9D9D9')

    # 绘制轨迹，使用MATLAB样式
    for i in range(N):
        x = status_vector[i, :, 0]
        y = status_vector[i, :, 1]
        z = status_vector[i, :, 2]
        color = matlab_colors[i % len(matlab_colors)]
        
        # MATLAB风格的线条更粗
        ax.plot(x, y, z,
                color=color,
                linestyle='-',
                linewidth=2.0,
                label=f'Player {i+1}')
                
        # 起点和终点标记，更像MATLAB的默认标记大小
        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=80, edgecolor='k', zorder=5)
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='s', s=80, edgecolor='k', zorder=5)
    
    if p_center is not None:
        ax.scatter(p_center[0], p_center[1], p_center[2], color=matlab_colors[-1], s=80, marker="*", label="Global target")

    # MATLAB风格的轴标签 - 增加labelpad以确保z轴标签可见
    ax.set_xlabel(f"${var_name}_{{i1}}$ (m)", fontsize=14, labelpad=10)
    ax.set_ylabel(f"${var_name}_{{i2}}$ (m)", fontsize=14, labelpad=10)
    ax.set_zlabel(f"${var_name}_{{i3}}$ (m)", fontsize=14, labelpad=15)  # z轴增加更多间距

    # 轴刻度字体大小，MATLAB风格
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='z', which='major', labelsize=12, pad=8)  # z轴刻度标签额外间距

    # 设置轴边框颜色，MATLAB风格
    ax.xaxis.pane.set_edgecolor('#D9D9D9')
    ax.yaxis.pane.set_edgecolor('#D9D9D9')
    ax.zaxis.pane.set_edgecolor('#D9D9D9')

    # 设置坐标面板填充颜色为白色或透明
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


    # 坐标范围等比例+留白，MATLAB通常保持更均匀的空间分布
    def set_equal_3d(ax, X, Y, Z, margin=0.1):
        x_middle = 0.5*(np.max(X)+np.min(X))
        y_middle = 0.5*(np.max(Y)+np.min(Y))
        z_middle = 0.5*(np.max(Z)+np.min(Z))
        max_range = 0.5*max(np.ptp(X), np.ptp(Y), np.ptp(Z)) * (1+margin)
        ax.set_xlim(x_middle - max_range, x_middle + max_range)
        ax.set_ylim(y_middle - max_range, y_middle + max_range)
        ax.set_zlim(0, z_middle + max_range)

    set_equal_3d(
        ax,
        status_vector[:, :, 0].flatten(),
        status_vector[:, :, 1].flatten(),
        status_vector[:, :, 2].flatten(),
        margin=0.15
    )

    # 设置MATLAB默认视角，稍微调整以更好显示z轴标签
    ax.view_init(elev=30, azim=45)

    # MATLAB风格图例
    legend = ax.legend(
        loc='best',
        fontsize=12,
        frameon=True,
        framealpha=1.0,
        edgecolor='k',
        facecolor='white',
        ncol=1
    )

    # 添加MATLAB风格的边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    # 调整布局以确保标签可见
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # 保存高分辨率图像 - 使用pad_inches而不是bbox_inches='tight'
    plt.savefig(os.path.join(figure_dir, file_tag+"3d_trajectories.png"), 
                dpi=600, 
                bbox_inches='tight',
                pad_inches=0.2)  # 增加边距以确保标签不被裁剪
    plt.close()
    print(f"Saved MATLAB-style figure: {os.path.join(figure_dir, file_tag+'3d_trajectories.png')}")


def get_convergence_time(status_vectors, opt_value, time_vector, error=1e-4):
    status_vectors = np.array(status_vectors)
    N,T,D = status_vectors.shape
    convergence_time = -1
    for i in range(T):
        status_error = status_vectors[:,i,:]-opt_value[:,i,:]
        status_error = np.linalg.norm(status_error.flatten(), ord=2)
        if i==T-1:
            print("last_error:", status_error)
        if status_error <= error:
            print("convergence time:", time_vector[i])
            convergence_time = time_vector[i]
            break
    
    return convergence_time


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




def plot_initial_convergence_line__graph(initial_values, convergence_times, xlable, legneds):
    
    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.clf()
    colors = list(mcolors.TABLEAU_COLORS.values())
    N = len(convergence_times)
    marker = ['o', 's', '^', 'D', 'v', '*', 'P', 'X', 'h', '8']
    y_max = -1
    for i in range(N):
        x = initial_values
        y = convergence_times[i]
        plt.plot(x, y, 
                color=colors[i],           # 自定义颜色
                linewidth=1,               # 线宽
                linestyle='-',             # 线型: '-', '--', '-.', ':'
                marker=marker[i],                # 标记点: 'o', 's', '^', 'v', 'D'等
                markersize=8,              # 标记大小
                # markerfacecolor='red',     # 标记填充色
                markeredgecolor=colors[i],   # 标记边缘色
                markeredgewidth=2,         # 标记边缘宽度
                label=legneds[i]  # 图例标签
                )
        if np.max(y) > y_max:
            y_max = np.max(y)
    
    plt.xlabel(xlable, fontsize=15)
    plt.ylabel("Convergence Time(sec)", fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(left=0, right=max(initial_values)*1.1)
    plt.ylim(bottom=0, top=y_max*1.4)
    plt.tight_layout()
    path = "initial_convergence_time.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")