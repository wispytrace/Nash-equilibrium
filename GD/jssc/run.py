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
    def __init__(self, num_agents, config_index, init_value=None, simu_id=None):
        self.num_agents = num_agents
        self.init_value = init_value
        self.config_index = config_index
        self.simu_id = simu_id
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
        if 'private' in model_config:
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
            if self.init_value is not None:
                for key, value in self.init_value.items():
                    agent.set_init_value(key, value[i])
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
            # 基础路径
            base_record_path = current_path + f"/records/{self.config['agent_config']['model']}/{self.config_index}"
            
            # 如果传入了 simu_id，则在配置目录下增加 simu_x 子目录
            if self.simu_id is not None:
                record_path = os.path.join(base_record_path, f"simu_{self.simu_id}")
            else:
                record_path = base_record_path
                
            os.makedirs(record_path, exist_ok=True)
            with open(record_path + "/" + f"agent_{i}.txt", "w") as f:
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
            self.update()
            self.record()
            if self.counts % self.PROCESS_BAR_INTERVAL == 0:
                self.publish_process_bar()
                print(f"status: {self.agntes[0].memory['x']}" )
                print(0.5*self.counts*self.agent_config['time_delta'])

            self.counts += 1
        self.done()


def run_batch_simulations(config_list, num_agents, init_values_list):
    """
    批量运行相同配置下的不同初始条件
    """
    for config_index in config_list:
        print(f"========== Starting batch runs for configuration: {config_index} ==========")
        
        for simu_id, init_val in enumerate(init_values_list):
            print(f"\n--- Running simulation {simu_id} ---")
            
            # 实例化系统，传入对应的 simu_id
            centralized_system = CentralizedModel(
                num_agents=num_agents, 
                config_index=config_index, 
                init_value=init_val,
                simu_id=simu_id  # 传入 ID 用于创建对应的文件夹
            )
            
            # 运行算法
            centralized_system.run()
            
        print(f"========== Finished batch runs for configuration: {config_index} ==========\n")


def run_single_simulation(config_index, num_agents, init_value):
    print(f"Running single simulation for configuration: {config_index}")
    
    # 实例化系统
    centralized_system = CentralizedModel(
        num_agents=num_agents, 
        config_index=config_index, 
        init_value=init_value
    )
    
    # 运行算法
    centralized_system.run()

if __name__ == "__main__":
    config_list = ["r_0"]
    z = np.array([
        # ================= 智能体 1 =================
        [
            [ 0.0,  5.5,  0.5],  # 0阶状态 (位置，你提供的值)
            [ 1.0, -1.0,  0.0],  # 1阶状态 (例如：速度)
            [ 0.5,  0.5,  0.1],  # 2阶状态 (例如：加速度)
            [-0.1,  0.2,  0.0]   # 3阶状态 (例如：加加速度)
        ],
        
        # ================= 智能体 2 =================
        [
            [-4.0,  1.5,  0.5],  # 0阶状态 (位置，你提供的值)
            [-2.0,  1.0,  0.5],  # 1阶状态
            [-0.5, -0.5, -0.1],  # 2阶状态
            [ 0.2, -0.1,  0.1]   # 3阶状态
        ],
        
        # ================= 智能体 3 =================
        [
            [ 0.0, -3.5,  0.5],  # 0阶状态 (位置，你提供的值)
            [ 1.5,  2.0, -0.5],  # 1阶状态
            [ 0.2, -0.2,  0.2],  # 2阶状态
            [-0.2, -0.2, -0.1]   # 3阶状态
        ],
        
        # ================= 智能体 4 =================
        [
            [ 4.0,  1.5,  0.5],  # 0阶状态 (位置，你提供的值)
            [-1.0, -2.0,  0.2],  # 1阶状态
            [-0.2,  0.2, -0.2],  # 2阶状态
            [ 0.1,  0.1,  0.2]   # 3阶状态
        ]
    ])
    num_agents = 4
    init_value = {"x": np.array([[5, 5, -3], [-5, 5, -3], [-5, -5, -3], [5, -5, -3]], dtype=float), "vr": np.array([[5, 5, -3], [-5, 5, -3], [-5, -5, -3], [5, -5, -3]], dtype=float), "y": np.array([[0,5.5,0.5], [-4, 1.5 ,0.5], [0, -3.5, 0.5], [4, 1.5, 0.5]]),
                  "z": z}
    
    run_single_simulation(config_list[0], num_agents, init_value)

# 设定 8 种不同的初始幅度大小（从小到大，测试算法对极大初始偏差的收敛鲁棒性）
    # amplitudes = [5, 25, 45, 65, 85, 105, 125, 145.0]
    
    # # 设定智能体初始分布的基础方向矩阵（分布在四个象限）
    # base_position = np.array([
    #     [ 1.0,  1.0, -0.5],
    #     [-1.0,  1.0, -0.5],
    #     [-1.0, -1.0, -0.5],
    #     [ 1.0, -1.0, -0.5]
    # ], dtype=float)

    # init_values_list = []
    
    # # 自动生成 8 组不同幅度的初始条件
    # for amp in amplitudes:
    #     # 将基础方向乘以当前的幅度放大倍数
    #     scaled_pos = base_position * amp
        
    #     init_values_list.append({
    #         "x": scaled_pos.copy(),
    #         "vr": scaled_pos.copy(),
    #         # y 的初始值保持不变（如果是静态博弈，这里其实写 np.zeros((4,3)) 也可以）
    #         "y": np.array([[0, 4, 0.5], [-4, 0, 0.5], [0, -4, 0.5], [4, 0, 0.5]], dtype=float)
    #     })
        
    # # 调用批量运行函数
    # run_batch_simulations(config_list, num_agents, init_values_list)
