import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图需要

class Simulation:
    """
    模拟车辆/代理系统的演化，包括更新动作、估计以及绘图展示。
    """
    def __init__(self):
        # ----- 系统参数 -----
        self.agent_count = 3
        self.dt = 1e-3
        self.epochs = 5000

        self.action_count = 6
        
        # 状态、动作初始值
        self.action = np.array([0.2, 0.2, 0.1,
                                0.2, 0.2, 0.2,
                                -0.2, -0.2, 0.2,
                                -0.2, -0.2, 0.2,
                                -0.2, 0.2, 0.2,
                                -0.2, 0.2, 0.1])*10
        
        self.estimate = np.zeros(self.agent_count*self.agent_count*self.action_count)  # 3 agents * 每个agent 9维?

        # 常数常量
        self.p = 0.5      # 策略更新中使用的参数
        self.q = 1.5

        # ----- 矩阵构造 -----
        self.adjacency_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1,],
            [1, 1, 0],
        ])
        self.laplacian_matrix = np.diag(self.adjacency_matrix.sum(axis=1)) - self.adjacency_matrix

        self.I = np.eye(self.agent_count)
        self.I2 = np.eye(self.agent_count**2)
        self.I3 = np.eye(self.agent_count*self.action_count)
        self.I4 = np.eye(self.action_count)
        # digA 为辅助对角矩阵（根据给定序列）
        self.digA = np.diag(self.adjacency_matrix.flatten())

        self.desire_position = np.array([0, 4.25, 10, 0, 0, 0,
                         -2.5, 0 , 10, 1, 1, 1,
                         2.5, 0, 10, 0, 0, 0])
        
        
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
        """
        对向量中的每个元素进行幂次处理，保留符号。
        
        参数：
            values: np.array 数值向量
            power: 幂数
        返回：
            处理后的向量
        """
        powered = np.zeros(values.shape)
        for index, value in enumerate(values):
            abs_val = np.abs(value)
            # 为防止除 0，加一个较小常数
            powered[index] = (value / (abs_val + 1e-2)) * (abs_val ** power)
        return powered
    
    def get_gradient(self, estimate, i):
        y_i = estimate[(self.action_count*self.agent_count)*i:(self.action_count*self.agent_count)*(i+1)] 
        x_i = y_i[self.action_count*i:self.action_count*(i+1)]
        graident = np.zeros(x_i.shape)
        p_i = self.desire_position[self.action_count*i:self.action_count*(i+1)]
        graident += (x_i - 2*p_i)
        y_sum = np.zeros(x_i.shape)
        for i in range(self.agent_count):
            y_sum += y_i[self.action_count*i:self.action_count*(i+1)]
        graident += y_sum + x_i
        graident = graident/(self.agent_count)
        return graident


    def update(self, action, estimate, run_time):
        """
        计算动作和估计的更新量。

        参数：
            action: 当前动作向量
            estimate: 当前估计向量
            run_time: 当前的运行时间（用于计算参考状态）
        返回：
            更新后的动作和估计（注意返回值前乘 -1 实现负反馈更新）
        """
        # 计算参考状态

        # ----- 估计更新 -----
        # 构建整体更新矩阵 M，结合拉普拉斯与辅助对角矩阵
        M = np.kron(self.laplacian_matrix, self.I) + self.digA
        estimate_delta = np.kron(M, self.I4) @ (estimate - np.kron(np.ones(self.agent_count).T, action))
        estimate_update = (self.delta1 @ self.my_sign(estimate_delta, self.p) +
                           self.delta2 @ self.my_sign(estimate_delta, self.q) +
                           self.delta3 @ self.my_sign(estimate_delta, 0))

        # ----- 动作更新 -----
        # 计算梯度信息
        gradients = np.array([])
        for i in range(self.agent_count):
            gradients = np.concatenate((gradients, self.get_gradient(estimate, i)))
        # grad1 = self.get_gradient(estimate[:3], estimate[3:6], self.d12)
        # grad1 = estimate[:3] - estimate[3:6] - self.d12 + self.k * (estimate[:3] - x0t - self.d10)
        # grad2 = (estimate[3:6] - estimate[6:9] - self.d23 +
        #          estimate[3:6] - estimate[0:3] + self.d12 +
        #          self.k * (estimate[3:6] - x0t - self.d20))
        # grad3 = estimate[6:9] - estimate[3:6] + self.d23 + self.k * (estimate[6:9] - x0t - self.d30)
        # gradients = np.concatenate((grad1, grad2, grad3))
        action_update = self.delta @ self.my_sign(gradients, 0)

        # 注意更新公式中返回的是负更新，用于负反馈控制
        return -action_update, -estimate_update

    def run(self):
        """
        计算多个时刻下动作及估计的更新过程，生成并保存折线图（各轴随时间变化）和三维轨迹图。
        """
        time_list = []
        action_history = []
        estimate_history = []
        reference_state_history = []
        
        # 主循环：每个epoch更新动作和估计
        for i in range(self.epochs):
            current_time = i * self.dt
            act_update, est_update = self.update(self.action, self.estimate, current_time)
            self.action = self.action + act_update * self.dt
            self.estimate = self.estimate + est_update * self.dt

            time_list.append(current_time)
            action_history.append(self.action.copy())
            estimate_history.append(self.estimate.copy())

            if i%100  == 0:
                print(f"epochs {i}/{self.epochs}")
        
        
        action_history = np.array(action_history)

        self._plot_actions(time_list, action_history)
        self._plot_3d(action_history)

    def _plot_actions(self, time_list, action_history):
        """
        根据各个代理的动作数据绘制随时间变化折线图，每个维度分别保存一幅图。
        """
        
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        for agent in range(self.agent_count):
            plt.clf()
            for dim in range(3):
                # 每个代理拥有 3 个分量，按序绘制: agent*3+dim
                plt.plot(time_list, action_history[:, agent*3 + dim],
                         label=f'Agent {agent+1} x[{dim}]',
                         color=colors[agent])
            # 同时绘制参考状态
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(loc='upper right')
            plt.title(f"Agent {agent} evolution")
            plt.savefig(f'action_dimension_{agent}.png')

    def _plot_3d(self, action_history):
        """
        绘制三维轨迹图，展示各代理在 3D 空间中的演化，
        同时在轨迹的起点和终点位置做标记，并标注坐标值。
        """
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制各代理轨迹和标记起始点与终点
        for agent in range(self.agent_count):
            # 获取当前代理的三个维度数据
            x = action_history[:, agent * self.action_count + 0]
            y = action_history[:, agent * self.action_count + 1]
            z = action_history[:, agent * self.action_count + 2]
            
            # 绘制轨迹线
            ax.plot(x, y, z, color=colors[agent], label=f'Agent {agent+1}')
            
            # 标记起点，用圆圈表示
            ax.scatter(x[0], y[0], z[0],
                    color=colors[agent],
                    marker='o',
                    s=20)
            # 在起点位置稍微偏移一点，标注坐标文本
            ax.text(x[0], y[0], z[0], 
                    f'({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})', 
                    fontsize=9, color=colors[agent])
            
            # 标记终点，用叉号表示
            ax.scatter(x[-1], y[-1], z[-1],
                    color=colors[agent],
                    marker='X',
                    s=20)
            # 在终点位置稍微偏移一点，标注坐标文本
            ax.text(x[-1], y[-1], z[-1],
                    f'({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})', 
                    fontsize=9, color=colors[agent])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectories with Start/End Markers and Coordinates')
        ax.legend(loc='upper right')
        plt.savefig('3d_trajectory.png')

if __name__ == "__main__":
    sim = Simulation()
    sim.run()