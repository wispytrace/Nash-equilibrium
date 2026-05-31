import networkx as nx
import matplotlib.pyplot as plt

def generate_topology_graph():
    # 1. 创建无向图
    G = nx.Graph()
    
    # 2. 定义环形拓扑的边 (1-6在上一排，12-7在下一排，首尾相连)
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),       # 上排横向连接
        (6, 7),                                       # 右侧纵向连接
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),  # 下排横向连接
        (12, 1)                                       # 左侧纵向连接
    ]
    G.add_edges_from(edges)
    
    # 3. 严格定义每个节点的绝对坐标位置 (完美矩形排布)
    pos = {
        1: (0, 1), 2: (1, 1), 3: (2, 1), 4: (3, 1), 5: (4, 1), 6: (5, 1),
        12: (0, 0), 11: (1, 0), 10: (2, 0), 9: (3, 0), 8: (4, 0), 7: (5, 0)
    }
    
    # 4. 设置画布大小 (宽长比例)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    # 5. 绘制连线 (加粗线条使其在论文中更清晰)
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color='black')
    
    # 6. 绘制节点 (白色填充，黑色边框)
    # node_size 控制圆圈大小
    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color='white', edgecolors='black', linewidths=2.5)
    
    # 7. 🌟 绘制数字标签 (极其关键的放大设置)
    # font_size 调大，font_weight 加粗，使其在圆圈内占比极大
    nx.draw_networkx_labels(G, pos, font_size=24, font_family='sans-serif', font_weight='bold')
    
    # 8. 隐藏坐标轴边框
    plt.axis('off')
    
    # 9. 调整边距并保存为高清矢量图 (适合LaTeX直接插入)
    plt.tight_layout()
    plt.savefig('communication_topology.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('communication_topology.png', format='png', bbox_inches='tight', dpi=300)
    
    print("拓扑图生成成功！已保存为 communication_topology.pdf 和 .png")
    plt.show()

if __name__ == '__main__':
    generate_topology_graph()