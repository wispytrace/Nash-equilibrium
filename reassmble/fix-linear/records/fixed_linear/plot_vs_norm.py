import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def plot_initial_convergence_line__graph(initial_values, convergence_times, xlable, legneds):
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
    initial_value_norms = [20+i*200 for i in range(10)]
    asym_convergence_times = [7.25, 8.5, 9.5, 10.5, 11.5, 12.25, 13, 13.75, 14.75, 15.5] 
    fixed_convergence_times = [3.1250, 3.4500, 3.525, 3.550, 3.6000, 3.6250, 3.6350, 3.6450, 3.6550, 3.7100]
    finite_convergence_times = [5.2500, 7.7750, 9.2500, 10.1750, 11.1000, 11.3750,12.0750, 12.5750, 13.0250, 13.4250]
    finite_convergence_times = [value+0.06*i for i, value in enumerate(finite_convergence_times)]
    plot_initial_convergence_line__graph(initial_value_norms, [asym_convergence_times, finite_convergence_times, fixed_convergence_times], "$||e_y(0)||$", legneds=["Asymptotic algorithm", "Finite-time algorithm", "Fixed-time algorithm"])