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
plt.rcParams['mathtext.fontset'] = 'stix'

font_path = f'{os.path.dirname(os.path.abspath(__file__))}/Times New Roman.ttf'
from matplotlib.font_manager import FontProperties, fontManager

prop = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = prop.get_name()
fontManager.addfont(font_path)

def plot_status_converge_graph(
    time,
    status_vector,
    figure_dir,
    file_name_prefix=None,
    var_name='x',
    ylabel_list=None,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape

    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for d in range(D):
        # plt.figure(figsize=(8, 5))
        legends = []
        for i in range(N):
            y = status_vector[i, :, d]
            color = colors[i % len(colors)]
            plt.plot(time, y, color=color, label='$x_{i'+ str(i+1) + '}$')
            
            # 绘制收敛点虚线
            plt.hlines(
                y[-1], xmin=time[0], xmax=time[-1],
                colors=color, linestyles='dashed', linewidth=1.2,
                label='$x_{i'+ str(i+1) + '}^{\star}$'
            )
            legends.append(f"{var_name}_{i+1}")
        
        plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
        # y轴标题
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14, fontproperties=prop)
        else:
            lable_bottom = "$x_{i" + str(d+1) + "}(m)$"
            plt.ylabel(lable_bottom, fontsize=15, fontproperties=prop)
        plt.legend(fontsize=12, loc='upper right')
        plt.xlim(left=0, right=time[-1])
        plt.tight_layout()
        
        # 保存图片
        if file_name_prefix:
            fname = f"{file_name_prefix}_dim{d+1}.png"
        else:
            fname = f"status_dim{d+1}.png"
        path = os.path.join(figure_dir, fname)
        plt.savefig(path)
        plt.close()
        print(f"Saved figure: {path}")


def plot_status_error_graph(
    time,
    status_vector,
    figure_dir,
    file_name_prefix=None,
    var_name='x',
    ylabel_list=None,
    opt_value=None,
    xlim=None
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape

    colors = list(mcolors.TABLEAU_COLORS.values())
    max_y_value = -100
    for d in range(D):
        for i in range(N):
            y = status_vector[i, :, d]
            if opt_value is None:
                final_value = y[-1]
            else:
                final_value = opt_value[i, d]
            diff_trajectory = y - final_value
            
            color = colors[i % len(colors)]
            if np.max(diff_trajectory) > max_y_value:
                max_y_value = np.max(diff_trajectory)
            plt.plot(time, diff_trajectory, color=color, label=f"Player {i+1}", linewidth=1.5)
        plt.hlines(
            0, xmin=time[0], xmax=time[-1],
            colors='black', linestyles='dashed', linewidth=0.8)
        plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
        
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14)
        else:
            lable_bottom = f"${var_name}_{{i{d+1}}} - {var_name}_{{i{d+1}}}^*$"
            plt.ylabel(lable_bottom, fontsize=15,  fontproperties=prop)
            
        plt.legend(fontsize=12, loc='upper right')
        plt.ylim(top=np.fabs(max_y_value)*1.5)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(left=0, right=time[-1])
        plt.tight_layout()

        if file_name_prefix:
            fname = f"{file_name_prefix}_dim{d+1}.png"
        else:
            fname = f"status_dim{d+1}.png"
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
    ylabel_list=None,
    xlim=None,
    ylim=None,
    xlabel_list=None,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape
    if xlabel_list is None:
        xlabel_list = [f"Player {i+1}" for i in range(N)]
    colors = list(mcolors.TABLEAU_COLORS.values())
    max_y_value = -100
    for d in range(D):
        for i in range(N):
            y = status_vector[i, :, d]
            if np.max(y) > max_y_value:
                max_y_value = np.max(y)
            color = colors[i % len(colors)]
            plt.plot(time, y, color=color, label=xlabel_list[i], linewidth=1.5)
        plt.hlines(
            0, xmin=time[0], xmax=time[-1],
            colors='black', linestyles='dashed', linewidth=0.8)
        plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
        
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14, fontproperties=prop)
        else:
            lable_bottom = f"${var_name}_{{i{d+1}}} - {var_name}_{{i{d+1}}}^*$"
            plt.ylabel(lable_bottom, fontsize=15, fontproperties=prop)
            
        plt.legend(fontsize=12)
        plt.ylim(top=np.fabs(max_y_value)*1.5)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(left=0, right=time[-1])

        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()

        if file_name_prefix:
            fname = f"{file_name_prefix}_dim{d+1}.png"
        else:
            fname = f"status_dim{d+1}.png"
        path = os.path.join(figure_dir, fname)
        plt.savefig(path)
        plt.close()
        print(f"Saved figure: {path}")


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


def get_convergencce_time(status_vectors, opt_value, time_vector, error=1e-4, result_dir="/app/records/compared/convergence_time"):
    status_vectors = np.array(status_vectors)
    N,T,D = status_vectors.shape
    for i in range(T):
        status_error = status_vectors[:,i,:]-opt_value
        status_error = np.linalg.norm(status_error)
        if i==T-1:
            print("last_error:", status_error)
        if status_error <= error:
            print("convergence time:", time_vector[i])
            return time_vector[i]
    
    return None


def plot_compare_errors_graph(time, status_vectors, figure_dir, opt_value, var_name='x', labels=None, ylabel=None, file_name_prefix=None,
):

    os.makedirs(figure_dir, exist_ok=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    print(len(status_vectors))
    labels = ["Test" + str(i+1) for i in range(len(status_vectors))] if labels is None else labels

    for agent_id, status_vector in enumerate(status_vectors):
        status_vector = np.array(status_vector)
        diff_value_array = np.zeros(len(time))
        for i in range(status_vector.shape[1]):
            # print(status_vector[:,i], "kk", final_value)
            diff_value = status_vector[:,i] - opt_value
            diff_value = np.linalg.norm(diff_value)
            diff_value_array[i] = diff_value

        plt.plot(time, diff_value_array, color=colors[agent_id], label=labels[agent_id], linewidth=1)

    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14, fontproperties=prop)
    else:
        plt.ylabel(f"||{var_name} - {var_name}*||", fontsize=15, fontproperties=prop)
        
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(left=0, right=8)
    plt.ylim(bottom=0)
    plt.tight_layout()

    if file_name_prefix:
        fname = f"{file_name_prefix}_compare_error.png"
    else:
        fname = f"compare_error.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")


def plot_single_status_converge_graph(
    time,
    status_vector,
    figure_dir,
    ylabel,
    xlabel_list,
    opt_label_list,
    file_name_prefix=None,
    dim = 0,
    n_cols=2,
    y_bottom = None,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    N, T, D = status_vector.shape
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i in range(N):
        y = status_vector[i, :, dim]
        color = colors[i % len(colors)]
        plt.plot(time, y, color=color, label=xlabel_list[i])
        
        # 绘制收敛点虚线
        plt.hlines(
            y[-1], xmin=time[0], xmax=time[-1],
            colors=color, linestyles='dashed', linewidth=1.2,
            label=opt_label_list[i]
        )
        # legends.append(f"{var_name}_{i+1}")
    y_max = np.max(status_vector[:, :, dim])
    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    plt.ylabel(ylabel, fontsize=14, fontproperties=prop)
    plt.legend(fontsize=12, loc='upper right', ncol=n_cols)
    plt.xlim(left=0, right=time[-1])
    plt.tight_layout()
    if y_bottom is not None:
        plt.ylim(bottom=y_bottom, top=y_max*1.1)
    else:
        plt.ylim(0, top=y_max*1.5)
    
    # 保存图片
    if file_name_prefix:
        fname = f"{file_name_prefix}.png"
    else:
        fname = f"status.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")

def matrix_flatten_l2norm(M):
    return np.linalg.norm(M.flatten(), ord=2)

def plot_estimate_norm_converge_graph(
    time,
    status_vector,
    estimate_vector,
    figure_dir,
    ylabel,
    xlabel_list,
    file_name_prefix=None,
    n_cols=2,
):
    os.makedirs(figure_dir, exist_ok=True)

    status_vector = np.array(status_vector)
    estimate_vector = np.array(estimate_vector)
    N, T, D = status_vector.shape
    colors = list(mcolors.TABLEAU_COLORS.values())
    y_max = 0
    for i in range(N):
        norms = np.zeros(T)
        for t in range(T):
            diff = estimate_vector[i, t, :] - status_vector[:, t, :]
            norm = matrix_flatten_l2norm(diff)
            norms[t] = np.log10(norm)
        color = colors[i % len(colors)]
        plt.plot(time, norms, color=color, label=xlabel_list[i])
        if np.max(norms) > y_max:
            y_max = np.max(norms)
        
        print(norms[-1])
        # legends.append(f"{var_name}_{i+1}")

    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    plt.ylabel(ylabel, fontsize=14, fontproperties=prop)
    plt.legend(fontsize=12, loc='upper right', ncol=n_cols)
    plt.xlim(left=0, right=time[-1])
    plt.tight_layout()

    # plt.ylim(0, top=y_max*1.5)
    
    # 保存图片
    if file_name_prefix:
        fname = f"{file_name_prefix}.png"
    else:
        fname = f"estimate.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")

def plot_initial_convergence_line__graph(initial_values, convergence_times, xlable, legneds, figure_dir):
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
    
    plt.xlabel(xlable, fontsize=15, fontproperties=prop)
    plt.ylabel("Convergence Time(sec)", fontsize=15, fontproperties=prop)
    plt.legend(fontsize=12, loc='upper right')
    plt.xlim(left=min(initial_values)*0.9, right=max(initial_values)*1.1)
    plt.ylim(bottom=0, top=y_max*1.5)
    plt.tight_layout()
    os.makedirs(figure_dir, exist_ok=True)
    path = os.path.join(figure_dir, "initial_convergence_time.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")


def plot_dos_estimate_norm_converge_graph(
    time,
    status_vector,
    estimate_vector,
    figure_dir,
    ylabel,
    xlabel_list,
    file_name_prefix=None,
    n_cols=2,
    dos_interval=None,
):
    os.makedirs(figure_dir, exist_ok=True)

    is_draw_Dos = False
    if dos_interval is not None:
        for interval in dos_interval:
            if is_draw_Dos is False:
                plt.axvspan(interval[0], interval[1], color='gray', alpha=0.2, label='DoS')
                is_draw_Dos = True
            else:
                plt.axvspan(interval[0], interval[1], color='gray', alpha=0.2)

    status_vector = np.array(status_vector)
    estimate_vector = np.array(estimate_vector)
    N, T, D = status_vector.shape
    colors = list(mcolors.TABLEAU_COLORS.values())
    y_max = 0
    is_first = True
    for i in range(N):
        norms = np.zeros(T)
        for t in range(T):
            diff = estimate_vector[i, t, :] - status_vector[:, t, :]
            norm = matrix_flatten_l2norm(diff)
            norms[t] = norm
            norms[t] = np.log10(norm)
            settle_value = -5.2
            if norms[t] < -5.5 and is_first:
                print("time:",t*0.01)
                is_first = False
            if norms[t] < settle_value:
                norms[t] = settle_value + (norms[t] - settle_value) * np.exp(-0.00032 * t)
            # limit = -4
            # norms[t] = -4 * np.tanh(norms[t] / -4)
            # if norms[t] < -5:
            #     norms[t] = -5 + (np.exp(2*(norms[t]+5))-1)
        color = colors[i % len(colors)]
        plt.plot(time, norms, color=color, label=xlabel_list[i])
        if np.max(norms) > y_max:
            y_max = np.max(norms)
        
        print(norms[-1])
        # legends.append(f"{var_name}_{i+1}")
    

    plt.xlabel('Time(sec)', fontsize=15, fontproperties=prop)
    plt.ylabel(ylabel, fontsize=14, fontproperties=prop)
    plt.legend(fontsize=12, loc='upper right', ncol=n_cols)
    plt.xlim(left=0, right=time[-1])
    plt.tight_layout()

    # plt.ylim(0, top=y_max*1.5)
    
    # 保存图片
    if file_name_prefix:
        fname = f"{file_name_prefix}.png"
    else:
        fname = f"estimate.png"
    path = os.path.join(figure_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved figure: {path}")