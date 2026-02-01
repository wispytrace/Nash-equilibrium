import numpy as np
import copy
from model import Model
# from config import config # 假设config就在本文件中定义
import os
import json
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import random
np.random.seed(42)  # 确保每次运行的随机数相同
# 1. 全局配置
config = {
    "f1":
    {
        "simulation_time": 5, # 总仿真时长 (秒)
        "epochs" : 300000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 5e-4,
            "model": "euler_constraint",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,
                "memory" : {"x": np.zeros((2)), "dot_x": np.zeros((2)), "y": np.full((2), 0), "z": np.zeros((5, 2)), "v": np.zeros((5, 2)), "gama": np.zeros((1))},
                
                'share': {
                    "init_value_x": np.array([[-1, 1], [1, -1], [-1, 0], [1, 0], [1.5, 1.5]]),
                    "init_value_dotx": np.array([[-2, -1], [-1, 2], [-2, 3], [1, 3], [2, 2]]),
                    "parameter_matrix": np.array([[1.19, 1.16, 1.13, 1.11, 1],
                                                  [1.41, 1.43, 1.45, 1.47, 1.5],
                                                  [0.31, 0.33, 0.32, 0.34, 0.47],
                                                  [1.78, 1.76, 1.74, 1.72, 1],
                                                  [0.73, 0.76, 0.79, 0.72, 1]]).T,
                    "pos": np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0], [0.5, 0], [0, 0.5]]),
                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.65,
                    "q": 1.35,
                    "gama": 25,
                    "lipsthitz": 1,
                    "scale_dict": {'alpha': 4, 'beta': 4, 'eta': 2, 'h1': 1, 'h2':1},
                    "l": np.array([-2, -2]),
                    "u": np.array([2, 2]),
                    "c": np.array([0.4, 0.4]),
                    "pos_c": np.array([0, 2.5])
                },

                'private': {
                '0': { 'alpha': [10, 10, 10], 'beta':[10, 10, 10], 'eta': [1, 1, 1], 'h1': 2, 'h2': 2},
                '1': { 'alpha': [10, 10, 10], 'beta':[10, 10, 10], 'eta': [1, 1, 1], 'h1': 2, 'h2': 2},
                '2': { 'alpha': [10, 10, 10], 'beta':[10, 10, 10], 'eta': [1, 1, 1], 'h1': 2, 'h2': 2},
                '3': { 'alpha': [10, 10, 10], 'beta':[10, 10, 10], 'eta': [1, 1, 1], 'h1': 2, 'h2': 2},
                '4': { 'alpha': [10, 10, 10], 'beta':[10, 10, 10], 'eta': [1, 1, 1], 'h1': 2, 'h2': 2},
                },
            }
        }
    },
    "f2":
    {
        "simulation_time": 5, # 总仿真时长 (秒)
        "epochs" : 300000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 5e-4,
            "model": "euler_constraint",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,
                "memory" : {"x": np.zeros((2)), "dot_x": np.zeros((2)), "y": np.full((2), 0), "z": np.zeros((5, 2)), "v": np.zeros((5, 2)), "gama": np.zeros((1))},
                
                'share': {
                    "init_value_x": np.array([[-1, 1], [1, -1], [-1, 0], [1, 0], [1.5, 1.5]]),
                    "init_value_dotx": np.array([[-2, -1], [-1, 2], [-2, 3], [1, 3], [2, 2]]),
                    "parameter_matrix": np.array([[1.19, 1.16, 1.13, 1.11, 1],
                                                  [1.41, 1.43, 1.45, 1.47, 1.5],
                                                  [0.31, 0.33, 0.32, 0.34, 0.47],
                                                  [1.78, 1.76, 1.74, 1.72, 1],
                                                  [0.73, 0.76, 0.79, 0.72, 1]]).T,
                    "pos": np.array([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0], [0.5, 0], [0, 0.5]]),
                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.65,
                    "q": 1.35,
                    "gama": 25,
                    "lipsthitz": 1,
                    "scale_dict": {'alpha': 4, 'beta': 4, 'eta': 2, 'h1': 1, 'h2':1},
                    "l": np.array([-2, -2]),
                    "u": np.array([2, 2]),
                    "c": np.array([0.4, 0.4]),
                    "pos_c": np.array([0, 2.5])
                },

                'private': {
                '0': { 'alpha': [10, 0, 10], 'beta':[10, 0, 10], 'eta': [1, 0, 1], 'h1': 2, 'h2': 2},
                '1': { 'alpha': [10, 0, 10], 'beta':[10, 0, 10], 'eta': [1, 0, 1], 'h1': 2, 'h2': 2},
                '2': { 'alpha': [10, 0, 10], 'beta':[10, 0, 10], 'eta': [1, 0, 1], 'h1': 2, 'h2': 2},
                '3': { 'alpha': [10, 0, 10], 'beta':[10, 0, 10], 'eta': [1, 0, 1], 'h1': 2, 'h2': 2},
                '4': { 'alpha': [10, 0, 10], 'beta':[10, 0, 10], 'eta': [1, 0, 1], 'h1': 2, 'h2': 2},
                },
            }
        }
    },
}

config_index = "f2"
num_agents = 5

class CentralizedModel:
    def __init__(self, num_agents, sim_id=0, init_value_override=None, sio_client=None):
        self.num_agents = num_agents
        self.sim_id = sim_id  
        self.sio = sio_client

        self.config = copy.deepcopy(config[config_index]) 
        self.agent_config = self.config['agent_config']
        
        if init_value_override is not None:
            self.agent_config['model_config']['share']['init_value_x'] = init_value_override

        self.agntes = self.load_agents()
        self.adjacency_matrix = np.array(self.config['adjacency_matrix'])
        
        # 时间控制参数
        self.total_sim_time = self.config.get('simulation_time', 5.0)
        self.current_sim_time = 0.0
        
        self.counts = 0
        self.records = {}

        self.start_time = time.time()
        self.time_estimate = time.time()
        self.PROCESS_BAR_INTERVAL = 200 
        

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
        # 1. 通信
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
            current_dt = self.agntes[0].time_delta
            for i in range(self.num_agents):
                record_dict = self.memory_to_list(self.agntes[i].memory)
                record_dict['time'] = self.current_sim_time
                if i not in self.records.keys():
                    self.records[i] = []
                self.records[i].append(record_dict)

    def get_save_path(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        sim_folder = f"sim_{self.sim_id.split('_')[-1]}" if hasattr(self, 'sim_id') else "single_run"
        path = os.path.join(current_path, "records", self.config['agent_config']['model'], config_index, sim_folder)
        os.makedirs(path, exist_ok=True)
        return path

    def save_simulation_data(self):
        if self.num_agents == 0 or 0 not in self.records:
            return None

        centralized_data = {
            "time_steps": [],
            "num_agents": self.num_agents,
            "sim_id": self.sim_id,
            "trajectories": {
                "x": [[] for _ in range(self.num_agents)],
                "z": [[] for _ in range(self.num_agents)],
                "v": [[] for _ in range(self.num_agents)],
            }
        }

        # 提取时间轴
        centralized_data["time_steps"] = [entry['time'] for entry in self.records[0]]
        
        # 提取数据
        for i in range(self.num_agents):
            if i not in self.records: continue
            for entry in self.records[i]:
                centralized_data["trajectories"]["x"][i].append(entry['x'])
                centralized_data["trajectories"]["z"][i].append(entry['z'])
                centralized_data["trajectories"]["v"][i].append(entry['v'])

        save_path = self.get_save_path()
        file_path = os.path.join(save_path, "all_agents_trajectories.json")
        
        with open(file_path, "w") as f:
            json.dump(centralized_data, f)
            
        return centralized_data

    def plot_simulation_result(self, centralized_data=None):
        if centralized_data is None:
            return

        save_path = self.get_save_path()
        sim_id = centralized_data.get('sim_id', 0)
        time_steps = np.array(centralized_data["time_steps"])
        
        # 定义 NE 点
        NE_vector = np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]])

        # 数据处理
        x_trajs_list = centralized_data["trajectories"]["x"]
        try:
            # Shape: (Num_Agents, Time)
            x_matrix = np.array(x_trajs_list)
        except ValueError as e:
            print(np.array(x_trajs_list).shape, e)
            return

        # 计算距离
        error_matrix = x_matrix - NE_vector[:,None,:]
        error_swapped = np.swapaxes(error_matrix, 0, 1)
        
        # 第二步：把 Agents 和 Coords 维度合并（展平）
        # 我们希望每个时间步对应一个长度为 10 的向量 (5个智能体 * 2个坐标)
        # 变换后: (2001, 10)
        error_flattened = error_swapped.reshape(error_swapped.shape[0], -1)
        
        # 第三步：沿着展平后的维度 (axis=1) 计算范数
        # 结果 shape: (2001,) -> 这是一个随时间变化的一维数组
        dist_to_NE = np.linalg.norm(error_flattened, axis=1)
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_steps, np.log10(dist_to_NE), linewidth=1.5, color='blue', label='$\|x(t) - x^*\|$')
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to NE (Log Scale)')
        ax.set_xlim(left=0)
        ax.set_title(f'Convergence (Sim {sim_id}) - Init Norm: {dist_to_NE[0]:.1f}')
        ax.legend()

        img_path = os.path.join(save_path, "distance_to_NE.png")
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    def done_centralized(self):
        data = self.save_simulation_data()
        if data is not None:
            self.plot_simulation_result(data)

    def seconds_to_hms_string(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{round(seconds,2):04}"

    def publish_process_bar(self):
        now_time = time.time()
        
        # 基于时间计算进度
        progress_ratio = self.current_sim_time / self.total_sim_time
        if progress_ratio <= 0: progress_ratio = 1e-9
        
        used_real_time = now_time - self.start_time
        estimated_total_real_time = used_real_time / progress_ratio
        remaining_real_time = estimated_total_real_time - used_real_time
        
        used_str = self.seconds_to_hms_string(used_real_time)
        remain_str = self.seconds_to_hms_string(remaining_real_time)
        
        square_num = int(progress_ratio * 10)
        bar = f"|{'■' * square_num}{' ' * (10 - square_num)}|"
        
        message = (f"[Sim {self.sim_id}] {progress_ratio*100:5.1f}% {bar} "
                   f"T:{self.current_sim_time:.2f}/{self.total_sim_time:.1f}s "
                   f"[{used_str}<{remain_str}]")
        print(message)
        self.time_estimate = now_time        

    def run(self):
        print(f"Start Sim {self.sim_id}, Target Time: {self.total_sim_time}s")

        if self.sio and self.sio.connected:
            self.sio.emit('sim_event', {'type': 'start', 'sim_id': self.sim_id})
        # 基于时间的循环
        while self.current_sim_time < self.total_sim_time:
            # 获取当前这一步的 dt
            current_dt = self.agntes[0].time_delta
            self.update()
            self.record()
            self.current_sim_time += current_dt

            if self.sio:
                self.sio.sleep(0) # 或者 time.sleep(0.0001)

            if self.counts % self.PROCESS_BAR_INTERVAL == 0:
                self.publish_process_bar()

            self.counts += 1

        if self.sio and self.sio.connected:
            self.sio.emit('sim_event', {'type': 'end', 'sim_id': self.sim_id})

        self.done_centralized()
        print(f"Sim {self.sim_id} Done. Steps: {self.counts}")


def run_single_simulation():
    centralized_system = CentralizedModel(num_agents=num_agents, sim_id="single_run_101")
    centralized_system.run()

if __name__ == "__main__":
    # run_single_simulation()

    # 【新增】生成一个进程唯一的 ID 前缀
    # 比如: "P1234_"，这样终端A发的ID是 "P1234_0", 终端B发的ID是 "P5678_0"
    import os
    process_prefix = f"Proc{os.getpid()}_" 
    print(f"Running with Process Prefix: {process_prefix}")

    TOTAL_SIMULATIONS = 10
    magnitudes = [5+i*15 for i in range(TOTAL_SIMULATIONS)]

    for i, magnitude in enumerate(magnitudes):
        # if i != 43:
        #     continue
        # 【修改】组合出全局唯一的 sim_id
        # 前端收到的是字符串，这样就不会冲突了
        unique_sim_id = f"{process_prefix}{i}"
        
        random_vec = [[-1, 1], [1, -1], [-1, 0], [1, 0], [1.5, 1.5]]
        init_value_large = (random_vec / np.linalg.norm(random_vec)) * magnitude
        
        centralized_system = CentralizedModel(
            num_agents=num_agents, 
            sim_id=unique_sim_id,  # <--- 传入这个唯一 ID
            init_value_override=init_value_large,
            sio_client=None
        )
        centralized_system.run()