import numpy as np
import copy
from model import Model
from config import config

# 1. 配置参数
config_index = "simu"
num_agents = 3

# 2. 集中式算法框架
class CentralizedModel:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.config = config[config_index]
        self.agent_config = config[config_index]['agent_config']
        self.agntes = self.load_agents()
        self.epochs = self.config['epochs']
        self.adjacency_matrix = np.array(self.config['adjacency_matrix'])
        self.actions = [[] for _ in range(self.num_agents)]
        self.estimates = [[] for _ in range(self.num_agents)]

    def get_model_config(self, id):
        model_config = self.agent_config['model_config']
        for k, v in model_config['share'].items():
            model_config[k] = v
        for k, v in model_config['private'][str(id)].items():
            model_config[k] = v
        model_config['time_delta'] = self.agent_config['time_delta']
        model_config['agent_id'] = id
        return model_config

    def load_agents(self):
        agents = []
        for i in range(self.num_agents):
            model_config = self.get_model_config(i)
            agent = Model(model_config)
            agents.append(agent)
        return agents

    def update(self):
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix)):
                if self.adjacency_matrix[i][j] > 1e-3 and i != j:
                    self.agntes[i].receieve_msg(j, self.agntes[j].memory)

        for i in range(self.num_agents):
            self.agntes[i].update()

    def redcord(self):
        for idx, agent in enumerate(self.agntes):
            self.actions[idx].append(agent.memory['x'])
            self.estimates[idx].append(agent.memory['z'])
        
    def run(self):
        for i in range(self.epochs):
            self.update()
            action_values = []
            # 每个智能体返回的action value 是一个2x2的numpy 数组 如 [[1,2], [3,4]], 其中1和2表示更新后的x坐标和y坐标, 3 和 4 分别代表 更新后的 x轴速度 和y轴速度
            for i in range(num_agents):
                action_values.append(self.agntes[i].get_action_value())
                print(self.agntes[i].get_action_value())

            # if i % 1000 == 0:
            #     self.redcord()
            #     print(f" {i}/{self.epochs} --- Actions: {self.actions}, Estimates: {self.estimates}")


# 3. 运行集中式算法
centralized_system = CentralizedModel(num_agents=num_agents)
centralized_system.run()