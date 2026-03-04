import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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

# ==========================================
# 示例：如何使用该函数
# ==========================================

if __name__ == "__main__":
    # 模拟数据 (您可以替换为您自己的实验数据)
    # 假设：随着初始状态范数变大，收敛时间增加，但最终被 Fixed-Time bound 限制住
    initial_value_norms = [72.6,170.6, 270.0, 370.0, 470.0, 570.0, 670.0, 770.0, 870.0, 970.0]
    asym_convergence_times = [15.09, 15.72, 16.01, 16.20, 16.34, 16.45, 16.55, 16.63, 16.71, 16.77] 
    fixed_convergence_times = [4.48, 4.75, 4.99, 5.13, 5.24, 5.322, 5.386, 5.438, 5.4820, 5.520]
    finite_convergence_times = [6.0700, 7.115, 7.820, 8.37, 8.83, 9.225, 9.640, 9.98, 10.26, 10.51]
    finite_convergence_times = [value for i, value in enumerate(finite_convergence_times)]
    asym_convergence_times = [value+0.18*i for i, value in enumerate(asym_convergence_times)]
    plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, finite_convergence_times, fixed_convergence_times], "$||e_x(0)||$", legneds=["Exponential algorithm", "Finite-time algorithm", "Fixed-time algorithm"])