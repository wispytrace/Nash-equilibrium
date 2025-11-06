import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

num_agents = 5
state_steps = 10        # 状态主帧数量
insert_frames = 3       # 每两个主帧之间插入过渡帧数
total_frames = state_steps + (state_steps-1) * insert_frames

status_records = np.random.randint(0, 3, size=(num_agents, state_steps))
positions = np.random.rand(num_agents, state_steps, 2)

# 环形拓扑例子
G = nx.DiGraph()
edges = [(i, (i+1)%num_agents) for i in range(num_agents)]
G.add_edges_from(edges)
topo_positions = nx.circular_layout(G)

fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(16,5))

def animate(frame_idx):
    ax_left.clear()
    ax_mid.clear()
    ax_right.clear()
    # 判断是主帧还是过渡帧
    if insert_frames == 0:
        main_idx = frame_idx
        is_transition = False
        offset = 0
    else:
        main_idx = frame_idx // (insert_frames+1)
        is_transition = (frame_idx % (insert_frames+1) != 0)
        offset = frame_idx % (insert_frames+1)
    
    # ------ 左侧，只有主帧更新 ------
    show_pos = positions[:, main_idx]
    ax_left.scatter(show_pos[:,0], show_pos[:,1], c='blue')
    for i in range(num_agents):
        ax_left.text(show_pos[i,0], show_pos[i,1]+0.01, f"{i}", ha='center')
    ax_left.set_title("Agent Positions")

    # ------ 中间拓扑和动画 ------
    nx.draw(G, pos=topo_positions, ax=ax_mid, with_labels=True, node_color='orange', node_size=600, arrows=True)
    ax_mid.set_title(f"t={frame_idx}")
    ax_mid.axis('off')

    if is_transition:
        for src, dst in G.edges():
            x0, y0 = topo_positions[src]
            x1, y1 = topo_positions[dst]
            arrow_progress = offset / (insert_frames+1)
            xa = x0*(1-arrow_progress) + x1*arrow_progress
            ya = y0*(1-arrow_progress) + y1*arrow_progress
            ax_mid.annotate("",
                xy=(xa, ya), xytext=(x0, y0),
                arrowprops=dict(facecolor='red', edgecolor='red', 
                                arrowstyle="->", lw=1.7, alpha=0.75),
            )

    # ------ 右侧，只有主帧更新 ------
    for i in range(num_agents):
        xdata = []
        ydata = []
        for j in range(state_steps):
            xdata.append(j*(insert_frames+1))
            ydata.append(status_records[i,j])
        ax_right.step(xdata, ydata, where='post', label=f"Agent {i}")
    ax_right.legend()
    ax_right.set_title("Agent Status")
    ax_right.set_xlabel('Frame Index')

ani = FuncAnimation(fig, animate, frames=total_frames, interval=80)
ani.save('topo_edge_flow_simple.gif', writer='pillow')
plt.close(fig)

# 若需保存为 MP4（视频）：
# ani.save('multi_agent.mp4', writer='ffmpeg', fps=5)
