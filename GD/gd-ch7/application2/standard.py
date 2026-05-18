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
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

def plot_status_converge_graph(
    time,
    status_vector,
    figure_dir,
    file_name_prefix=None,
    var_name='x',
    ylabel_list=None
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
        
        plt.xlabel('Time(s)', fontsize=15)
        # y轴标题
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14)
        else:
            lable_bottom = "$x_{i" + str(d+1) + "}(m)$"
            plt.ylabel(lable_bottom, fontsize=15)
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
        plt.xlabel('Time(s)', fontsize=15)
        
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14)
        else:
            lable_bottom = f"${var_name}_{{i{d+1}}} - {var_name}_{{i{d+1}}}^*$"
            plt.ylabel(lable_bottom, fontsize=15)
            
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
        plt.xlabel('Time(s)', fontsize=15)
        
        if ylabel_list is not None and len(ylabel_list) == D:
            plt.ylabel(ylabel_list[d], fontsize=14)
        else:
            lable_bottom = f"${var_name}_{{i{d+1}}} - {var_name}_{{i{d+1}}}^*$"
            plt.ylabel(lable_bottom, fontsize=15)
            
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

    # 创建图形，使用MATLAB默认大小比例
    # plt.rcParams['font.family'] = 'stix'  # MATLAB通常使用的字体
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.figure(figsize=(9.5, 7))
    ax = plt.subplot(111, projection='3d')

    # 设置背景色为白色，MATLAB风格
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.7, color='#D9D9D9')

    # 绘制轨迹，使用MATLAB样式
    end_points = []  # 终点集
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
        end_points.append([x[-1], y[-1], z[-1]])
    if p_center is not None:
        ax.scatter(p_center[0], p_center[1], p_center[2], color=matlab_colors[-1], s=80, marker="*", label="Global target")

    if len(end_points) > 1:
        end_points = np.array(end_points)
        # ax.plot(end_points[:,0], end_points[:,1], end_points[:,2],
        # linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[:2,0], end_points[:2,1], end_points[:2,2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[1:3,0], end_points[1:3,1], end_points[1:3,2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[2:4,0], end_points[2:4,1], end_points[2:4,2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[[0,3],0], end_points[[0,3],1], end_points[[0,3],2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[[2,4],0], end_points[[2,4],1], end_points[[2,4],2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[[4,5],0], end_points[[4,5],1], end_points[[4,5],2],
                linestyle='--', color='k', linewidth=1.5)
        ax.plot(end_points[[0,5],0], end_points[[0,5],1], end_points[[0,5],2],
                linestyle='--', color='k', linewidth=1.5)
        # ax.plot(end_points[:,0], end_points[:,1], end_points[:,2],
        #         linestyle='--', color='k', linewidth=1.5)
        # ax.plot(end_points[:,0], end_points[:,1], end_points[:,2],
        #         linestyle='--', color='k', linewidth=1.5)
    # MATLAB风格的轴标签
    ax.set_xlabel(f"${var_name}_{{i1}}$ (m)", fontsize=14, labelpad=14)
    ax.set_ylabel(f"${var_name}_{{i2}}$ (m)", fontsize=14, labelpad=14)
    ax.set_zlabel(f"${var_name}_{{i3}}$ (m)", fontsize=14, labelpad=14)

    # 轴刻度字体大小，MATLAB风格
    ax.tick_params(axis='both', which='major', labelsize=12)

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
        margin=0.12
    )

    # 设置MATLAB默认视角
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

    # 保存高分辨率图像
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, file_tag+"3d_trajectories.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved MATLAB-style figure: {os.path.join(figure_dir, file_tag+'3d_trajectories.png')}")


def get_convergencce_time(status_vectors, opt_value, error=1e-4, time_delta=5e-4, result_dir="/app/records/compared/convergence_time"):
    for i in range(len(status_vectors)):
        status_error = status_vectors[i]-opt_value
        status_error = np.linalg.norm(status_error)
        if status_error <= error:
            print("convergence time:", i*time_delta)

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

    plt.xlabel('Time(s)', fontsize=15)
    
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    else:
        plt.ylabel(f"||{var_name} - {var_name}*||", fontsize=15)
        
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