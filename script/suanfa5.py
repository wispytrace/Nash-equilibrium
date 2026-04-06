import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Simulation:
    """
    模拟车辆/代理系统的演化，包括更新动作、估计以及绘图展示。
    """
    def __init__(self):
        # ----- 系统参数 (严格保留原有设定) -----
        self.agent_count = 3
        self.dt = 1e-3
        self.epochs = 5000

        self.action_count = 6
        
        # 状态、动作初始值 (3 agents * 6 actions = 18 dims)
        self.action = np.array([0.1, 0.0, 0.1,
                                0.0, 0.0, 0.0,
                                0.0, 0.0, 0.1,
                                -0.0, -0.0, 0.0,
                                -0.1, 0.0, 0.1,
                                -0.0, 0.0, 0.0])*10
        
        self.estimate = np.zeros(self.agent_count*self.agent_count*self.action_count) 

        self.p = 0.5      
        self.q = 1.5

        # ----- 矩阵构造：预定义三种 3节点 拓扑用于切换 -----
        # 拓扑 1: 原有的全连接图
        self.adj1 = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        # 拓扑 2: 链式图 (1-2-3)
        self.adj2 = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        # 拓扑 3: 星型/部分断开图 (仅 1-3, 2-3)
        self.adj3 = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 0],
        ])

        # 初始化赋值为拓扑 1
        self.adjacency_matrix = self.adj1
        self.laplacian_matrix = np.diag(self.adjacency_matrix.sum(axis=1)) - self.adjacency_matrix

        self.I = np.eye(self.agent_count)
        self.I2 = np.eye(self.agent_count**2)
        self.I3 = np.eye(self.agent_count*self.action_count)
        self.I4 = np.eye(self.action_count)
        self.digA = np.diag(self.adjacency_matrix.flatten())

        self.desire_position = np.array([3, 2, 1.2, 0, 0, 0,
                                         1.2, 2 , 1.2, 1, 1, 1,
                                         1.2, 4, 1.2, 0, 0, 0])
        
        delta11 = np.array([6, 5, 4])
        self.delta1 = np.kron(np.diag(delta11), self.I3)

        delta12 = np.array([4, 5, 5])
        self.delta2 = np.kron(np.diag(delta12), self.I3)

        delta13 = np.array([10, 8, 10,])
        self.delta3 = np.kron(np.diag(delta13), self.I3)

        delta = np.array([5, 6, 4])
        self.delta = np.kron(np.diag(delta), self.I4)

    @staticmethod
    def my_sign(values, power):
        powered = np.zeros(values.shape)
        for index, value in enumerate(values):
            abs_val = np.abs(value)
            powered[index] = (value / (abs_val + 1e-2)) * (abs_val ** power)
        return powered
    
    def update_topology(self, step):
        """
        每仿真 0.1s 切换一次拓扑图
        """
        cycle_length = int(0.1 / self.dt)  # 0.1s 对应的步数 (100)
        graph_idx = (step // cycle_length) % 3
        
        if graph_idx == 0:
            adj = self.adj1
        elif graph_idx == 1:
            adj = self.adj2
        else:
            adj = self.adj3
            
        # 重新计算依赖于当前拓扑的矩阵
        self.adjacency_matrix = adj
        self.laplacian_matrix = np.diag(adj.sum(axis=1)) - adj
        self.digA = np.diag(adj.flatten())

    def get_gradient(self, estimate, i):
        y_i = estimate[(self.action_count*self.agent_count)*i:(self.action_count*self.agent_count)*(i+1)] 
        x_i = y_i[self.action_count*i:self.action_count*(i+1)]
        graident = np.zeros(x_i.shape)
        p_i = self.desire_position[self.action_count*i:self.action_count*(i+1)]
        graident += (x_i - 2*p_i)
        
        y_sum = np.zeros(x_i.shape)
        # 这里将内层循环变量修改为 j，避免覆盖传入的智能体索引 i
        for j in range(self.agent_count):
            y_sum += y_i[self.action_count*j:self.action_count*(j+1)]
            
        graident += y_sum + x_i
        graident = graident/(self.agent_count)
        return graident

    def update(self, action, estimate, run_time):
        # 构建整体更新矩阵 M，结合拉普拉斯与辅助对角矩阵 (会随拓扑切换自动更新)
        M = np.kron(self.laplacian_matrix, self.I) + self.digA
        estimate_delta = np.kron(M, self.I4) @ (estimate - np.kron(np.ones(self.agent_count).T, action))
        estimate_update = (self.delta1 @ self.my_sign(estimate_delta, self.p) +
                           self.delta2 @ self.my_sign(estimate_delta, self.q) +
                           self.delta3 @ self.my_sign(estimate_delta, 0))

        # 计算梯度信息
        gradients = np.array([])
        for i in range(self.agent_count):
            gradients = np.concatenate((gradients, self.get_gradient(estimate, i)))
            
        action_update = self.delta @ self.my_sign(gradients, 0)
        return -action_update, -estimate_update

    def run(self):
        time_list = []
        action_history = []
        estimate_history = []
        
        for i in range(self.epochs):
            # 每步更新或切换拓扑
            self.update_topology(i)
            
            current_time = i * self.dt
            act_update, est_update = self.update(self.action, self.estimate, current_time)
            self.action = self.action + act_update * self.dt
            self.estimate = self.estimate + est_update * self.dt

            time_list.append(current_time)
            action_history.append(self.action.copy())
            estimate_history.append(self.estimate.copy())

            if i % 100 == 0:
                print(f"epochs {i}/{self.epochs}")
        
        action_history = np.array(action_history)

        self._plot_actions(time_list, action_history)
        self._plot_3d(action_history)

    def _plot_actions(self, time_list, action_history):
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        for agent in range(self.agent_count):
            plt.clf()
            for dim in range(3):
                plt.plot(time_list, action_history[:, agent * self.action_count + dim],
                         label=f'Agent {agent+1} x[{dim}]',
                         color=colors[dim % len(colors)])
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(loc='upper right')
            plt.title(f"Agent {agent+1} evolution")
            plt.savefig(f'action_dimension_{agent+1}.png')

    def _plot_3d(self, action_history):
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for agent in range(self.agent_count):
            x = action_history[:, agent * self.action_count + 0]
            y = action_history[:, agent * self.action_count + 1]
            z = action_history[:, agent * self.action_count + 2]
            
            ax.plot(x, y, z, color=colors[agent], label=f'Agent {agent+1}')
            
            ax.scatter(x[0], y[0], z[0], color=colors[agent], marker='o', s=20)
            ax.text(x[0], y[0], z[0], f'({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})', fontsize=9, color=colors[agent])
            
            ax.scatter(x[-1], y[-1], z[-1], color=colors[agent], marker='X', s=20)
            ax.text(x[-1], y[-1], z[-1], f'({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})', fontsize=9, color=colors[agent])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectories with Start/End Markers and Coordinates')
        ax.legend(loc='upper right')
        plt.savefig('3d_trajectory.png')

if __name__ == "__main__":
    sim = Simulation()
    sim.run()