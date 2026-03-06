import numpy as np
import pandas as pd
import plotly.express as px

# ==========================================
# 1. 模拟你的数据 (用你真实的轨迹矩阵替换这里)
# 假设你有 NUM_AGENTS 个智能体，运行了 STEPS 步
# ==========================================
NUM_AGENTS = 5
STEPS = 50

# 构造一个空的列表来装数据
data = []

# 假设纳什均衡点在 (0, 0, 0)
nash_point = np.array([0.0, 0.0, 0.0])

# 随机生成初始位置
positions = np.random.rand(NUM_AGENTS, 3) * 20 - 10

for step in range(STEPS):
    for agent_id in range(NUM_AGENTS):
        # 记录当前帧、智能体ID、X、Y、Z
        data.append({
            "Frame": step,
            "Agent_ID": f"Agent {agent_id}",
            "X": positions[agent_id, 0],
            "Y": positions[agent_id, 1],
            "Z": positions[agent_id, 2],
            "Size": 10 # 控制点的大小
        })
        
        # 模拟向纳什均衡点收敛的过程 (加上一点盘旋的特效)
        direction = nash_point - positions[agent_id]
        positions[agent_id] += direction * 0.1
        # 加点螺旋旋转，让三维轨迹更好看
        positions[agent_id, 0] += positions[agent_id, 1] * 0.1
        positions[agent_id, 1] -= positions[agent_id, 0] * 0.1

# 将数据转换为 Pandas DataFrame，这是 Plotly 最喜欢的数据格式
df = pd.DataFrame(data)

# ==========================================
# 2. 使用 Plotly 一键生成 3D 交互式动画
# ==========================================
print("正在渲染高精度 3D 图表...")

fig = px.scatter_3d(
    df, 
    x="X", 
    y="Y", 
    z="Z", 
    animation_frame="Frame",   # 关键帧，基于 DataFrame 的 Frame 列
    animation_group="Agent_ID",# 告诉软件哪些点属于同一个智能体
    color="Agent_ID",          # 不同智能体用不同颜色
    size="Size",               # 点的大小
    range_x=[-12, 12],         # 固定坐标轴范围，防止播放时乱跳
    range_y=[-12, 12], 
    range_z=[-12, 12],
    title="🌟 分布式纳什均衡 3D 搜索过程 (可鼠标自由拖拽旋转)"
)

# 调整一下连线和视觉样式，增加高级特效感
fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
fig.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor="rgb(20, 24, 82)", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="rgb(20, 24, 82)", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="rgb(20, 24, 82)", gridcolor="white", showbackground=True),
    ),
    paper_bgcolor="black",     # 外框背景设为黑色
    font=dict(color="white"),  # 字体变白
    margin=dict(r=10, l=10, b=10, t=40)
)

# ==========================================
# 3. 导出与展示
# ==========================================
# 导出为一个独立的网页文件
html_filename = "Nash_Equilibrium_Interactive.html"
fig.write_html(html_filename)
print(f"✅ 渲染成功！请双击打开当前目录下的 {html_filename} 在浏览器中观看！")

# 如果你在 Jupyter Notebook 环境中，可以直接运行下面这行在线显示
# fig.show()