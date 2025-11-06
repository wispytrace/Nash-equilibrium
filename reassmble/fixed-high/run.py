import numpy as np
import copy
from model import Model
from config import config
import os
import json
import time
# 1. 配置参数


# 2. 集中式算法框架
class CentralizedModel:
    def __init__(self, num_agents, config_index):
        self.num_agents = num_agents
        self.config_index = config_index
        self.config = config[self.config_index]
        self.agent_config = config[self.config_index]['agent_config']
        self.agntes = self.load_agents()
        self.epochs = self.config['epochs']
        self.adjacency_matrix = np.array(self.config['adjacency_matrix'])
        self.actions = [[] for _ in range(self.num_agents)]
        self.estimates = [[] for _ in range(self.num_agents)]
        self.counts = 0
        self.records = {}

        self.start_time = time.time()
        self.time_estimate = time.time()
        self.PROCESS_BAR_INTERVAL = 2000

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
            
    def memory_to_list(self, memory):
        listed_memory = {}
        for k, v in memory.items():
            listed_memory[k] = v.tolist()
        return listed_memory
    
    def record(self):
        if self.counts % self.agent_config['record_interval'] == 0:
            for i in range(self.num_agents):
                record_dict = self.memory_to_list(self.agntes[i].memory)
                record_dict['time'] = self.counts * self.agent_config['time_delta']
                if i not in self.records.keys():
                    self.records[i] = []
                self.records[i].append(record_dict)
            
    def done(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        for i in range(self.num_agents):
            record_path = current_path+f"/records/{self.config['agent_config']['model']}/" +config_index
            os.makedirs(record_path, exist_ok=True)
            with open(record_path+"/"+f"agent_{i}.txt", "w") as f:
                f.write(json.dumps(self.records[i]))
                f.flush()


    def seconds_to_hms_string(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{round(seconds,2):04}"

    def publish_process_bar(self):
        now_time = time.time()
        interval = now_time - self.time_estimate
        total_time = (self.epochs / self.PROCESS_BAR_INTERVAL) * interval
        used_time = now_time - self.start_time
        total_time = self.seconds_to_hms_string(total_time)
        used_time = self.seconds_to_hms_string(used_time)
        square_num = int(int(self.counts / self.epochs * 100) / 10)
        square = "■" * square_num
        blank = " " * (10 -square_num)
        bar = f"|{square}{blank}|"
        ratio = round(self.counts / self.epochs * 100, 3) 
        message = f"{ratio:5}% {bar} {str(self.counts) + '/' + str(self.epochs):15} [{used_time}<{total_time}, {round(interval,2)}s/{self.PROCESS_BAR_INTERVAL}its]"
        print(message)
        self.time_estimate = now_time        

    def run(self):
        for i in range(self.epochs):
            # for agent in self.agntes:
            #     if i > self.epochs * 0.2:
            #         agent.time_delta = agent.model_config['time_delta']*10
                # agent.memory['time'] = self.counts * self.agent_config['time_delta']
            self.update()
            self.record()
            
            if self.counts % self.PROCESS_BAR_INTERVAL == 0:
                self.publish_process_bar()
                print(f"Agent 0 position: {self.agntes[0].memory['x']}, virtual_status: {self.agntes[0].memory['y']}, ei_sum: {self.agntes[0].memory['ei_sum']}" )

            self.counts += 1

            # action_values = []
            # # 每个智能体返回的action value 是一个2x2的numpy 数组 如 [[1,2], [3,4]], 其中1和2表示更新后的x坐标和y坐标, 3 和 4 分别代表 更新后的 x轴速度 和y轴速度
            # for i in range(num_agents):
            #     action_values.append(self.agntes[i].get_action_value())
        self.done()


if __name__ == "__main__":
    config_list = ["r_0"]
    # config_index = "r_0"
    num_agents = 6

# 3. 运行集中式算法
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        centralized_system = CentralizedModel(num_agents=num_agents, config_index=config_index)
        centralized_system.run()