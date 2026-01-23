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
import socketio  # 新增
import random
np.random.seed(42)  # 确保每次运行的随机数相同
# 1. 全局配置
config = {
    "r_a": {
        "simulation_time": 40, # 总仿真时长 (秒)
        "adjacency_matrix": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config": {
            "time_delta": 1e-4, # 初始步长
            "model": "fixed4",
            "record_interval": 50, # 每多少步记录一次数据
            "record_flag": 1,
            "model_config": {
                "N": 5,
                "memory": {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                    'init_value': np.zeros((5, 1)),
                    'c': 18, 'a': 0.2, 'b': 2.5, 'l': 0, 'u': 8.0,
                    'p': 0.8, 'q': 1.2, 'min_c1': 40, 'min_delta': 2, 'gama': 40,
                },
                'private': {
                '0': { 'c1': 3, 'c2': 3, 'delta': 3, 'varphi':  3, 'sigma':  3,'eta':  3, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  3, 'c2': 3,'delta':  3,'varphi':  3, 'sigma':  3, 'eta':  3, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  3, 'c2': 3,'delta':  3,'varphi':  3, 'sigma':  3, 'eta':  3, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  3, 'c2': 3,'delta':  3,'varphi':  3, 'sigma':  3, 'eta':  3, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  3, 'c2': 3,'delta':  3,'varphi':  3, 'sigma':  3, 'eta':  3, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },
    "r_r": {
        "simulation_time": 50, # 总仿真时长 (秒)
        "adjacency_matrix": [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config": {
            "time_delta": 5e-4, # 初始步长
            "model": "fixed4",
            "record_interval": 50, # 每多少步记录一次数据
            "record_flag": 1,
            "model_config": {
                "N": 5,
                "memory": {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                    'init_value': np.zeros((5, 1)),
                    'c': 18, 'a': 0.2, 'b': 2.5, 'l': 0, 'u': 8.0,
                    'p': 0.8, 'q': 1.2, 'min_c1': 40, 'min_delta': 2, 'gama': 40,
                },
                'private': {
                '0': { 'c1': 2.58, 'c2': 2.58, 'delta': 2.5, 'varphi':  2.58, 'sigma':  2.58,'eta':  2.5, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  2.58, 'c2': 2.58,'delta':  2.5,'varphi':  2.58, 'sigma':  2.58, 'eta':  2.5, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  2.58, 'c2': 2.58,'delta':  2.5,'varphi':  2.58, 'sigma':  2.58, 'eta':  2.5, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  2.58, 'c2': 2.58,'delta':  2.5,'varphi':  2.58, 'sigma':  2.58, 'eta':  2.5, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  2.58, 'c2': 2.58,'delta':  2.5,'varphi':  2.58, 'sigma':  2.58, 'eta':  2.5, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

    "r_r_d": {
        "simulation_time": 40, # 总仿真时长 (秒)
        "adjacency_matrix": [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 1, 1, 0, 0]],
        "agent_config": {
            "time_delta": 1e-3, # 初始步长
            "model": "fixed4",
            "record_interval": 50, # 每多少步记录一次数据
            "record_flag": 1,
            "model_config": {
                "N": 5,
                "memory": {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                    'init_value': np.zeros((5, 1)),
                    'c': 18, 'a': 0.2, 'b': 2.5, 'l': 0, 'u': 8.0,
                    'p': 0.8, 'q': 1.2, 'min_c1': 40, 'min_delta': 2, 'gama': 40,
                },
                'private': {
                '0': { 'c1': 10, 'c2': 10, 'delta': 6, 'varphi':  10, 'sigma':  10,'eta':  6, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

    "r_r_dr": {
        "simulation_time": 40, # 总仿真时长 (秒)
        "adjacency_matrix": [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 1, 1, 0, 0]],
        "agent_config": {
            "time_delta": 1e-4, # 初始步长
            "model": "fixed4",
            "record_interval": 50, # 每多少步记录一次数据
            "record_flag": 1,
            "model_config": {
                "N": 5,
                "memory": {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                    'init_value': np.zeros((5, 1)),
                    'c': 18, 'a': 0.2, 'b': 2.5, 'l': 0, 'u': 8.0,
                    'p': 0.8, 'q': 1.2, 'min_c1': 40, 'min_delta': 2, 'gama': 40,
                },
                'private': {
                '0': { 'c1': 10, 'c2': 10, 'delta': 6, 'varphi':  10, 'sigma':  10,'eta':  6, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  10, 'c2': 10,'delta':  6,'varphi':  10, 'sigma':  10, 'eta':  6, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

   "r_r1":
    {
        "simulation_time" : 40,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-3,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 30,
                },

                'private': {
                '0': { 'c1': 0.5, 'c2': 0.5, 'delta': 2, 'varphi':  0.5, 'sigma':  0.5,'eta':  2, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

}

config_index = "r_r_dr"
num_agents = 5

class CentralizedModel:
    def __init__(self, num_agents, sim_id=0, init_value_override=None, sio_client=None):
        self.num_agents = num_agents
        self.sim_id = sim_id  
        self.sio = sio_client

        self.config = copy.deepcopy(config[config_index]) 
        self.agent_config = self.config['agent_config']
        
        if init_value_override is not None:
            self.agent_config['model_config']['share']['init_value'] = init_value_override

        self.agntes = self.load_agents()
        self.adjacency_matrix = np.array(self.config['adjacency_matrix'])
        
        # 时间控制参数
        self.total_sim_time = self.config.get('simulation_time', 5.0)
        self.current_sim_time = 0.0
        
        self.counts = 0
        self.records = {}

        self.start_time = time.time()
        self.time_estimate = time.time()
        self.PROCESS_BAR_INTERVAL = 2000 
        
        # 自适应步长参数
        self.dt_min = 1e-5
        self.dt_max = 1e-1
        self.k = 0.2

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

        # 2. 更新每个智能体，并获取最大梯度范数
        max_system_norm = 0.0
        for i in range(self.num_agents):
            agent_norm = self.agntes[i].update() # 假设 model.update() 返回范数
            if agent_norm > max_system_norm:
                max_system_norm = agent_norm
        
        # 3. 自适应步长计算 (指数衰减策略)
        decay_factor = np.exp(-self.k * max_system_norm)
        new_dt = self.dt_min + (self.dt_max - self.dt_min) * decay_factor
        new_dt = max(new_dt, self.dt_min) # 确保不低于最小值
        
        # 4. 应用新步长
        # for agent in self.agntes:
        #     agent.time_delta = new_dt

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
            
            if self.sio:
                try:
                    # 检查连接状态 (虽然有时候它反应慢，但还是要查)
                    if self.sio.connected:
                        x_list = [agent.memory['x'] for agent in self.agntes]
                        x_matrix = np.array(x_list).reshape(self.num_agents, 1)
                        NE_vector = np.array([2.06, 31, 2.97, 3.42, 3.88]).reshape(-1, 1)
                        dist = np.linalg.norm(x_matrix - NE_vector)
                        
                        self.sio.emit('sim_data', {
                            'sim_id': self.sim_id,
                            'time': round(self.current_sim_time, 4),
                            'value': float(dist)
                        })
                except Exception as e:
                    # 打印错误但不中断仿真
                    print(f"Visualization warning: {e}")
                    # 可选：如果断开了，尝试重连（或直接忽略，等待下一轮）
                    # self.sio.connect('http://localhost:5000')

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
                "v": [[] for _ in range(self.num_agents)]
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
        NE_vector = np.array([2.06, 2.51, 2.97, 3.42, 3.88]).reshape(-1, 1)

        # 数据处理
        x_trajs_list = centralized_data["trajectories"]["x"]
        try:
            # Shape: (Num_Agents, Time)
            x_matrix = np.array(x_trajs_list).squeeze(-1) 
        except ValueError:
            return

        # 计算距离
        error_matrix = x_matrix - NE_vector
        dist_to_NE = np.linalg.norm(error_matrix, axis=0)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_steps, np.log10(dist_to_NE), linewidth=1.5, color='blue', label='$\|x(t) - x^*\|$')
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance to NE (Log Scale)')
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
            self.record()
            self.update()
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


if __name__ == "__main__":
    # 初始化 Socket
    sio = socketio.Client()
    try:
        sio.connect('http://localhost:5000')
    except:
        sio = None

    # 【新增】生成一个进程唯一的 ID 前缀
    # 比如: "P1234_"，这样终端A发的ID是 "P1234_0", 终端B发的ID是 "P5678_0"
    import os
    process_prefix = f"Proc{os.getpid()}_" 
    print(f"Running with Process Prefix: {process_prefix}")

    TOTAL_SIMULATIONS = 100
    magnitudes = np.logspace(1, 4, TOTAL_SIMULATIONS)

    for i, magnitude in enumerate(magnitudes):
        # if i != 43:
        #     continue
        # 【修改】组合出全局唯一的 sim_id
        # 前端收到的是字符串，这样就不会冲突了
        unique_sim_id = f"{process_prefix}{i}"

        random_vec = np.random.randn(num_agents, 1) 
        init_value_large = (random_vec / np.linalg.norm(random_vec)) * magnitude
        
        centralized_system = CentralizedModel(
            num_agents=num_agents, 
            sim_id=unique_sim_id,  # <--- 传入这个唯一 ID
            init_value_override=init_value_large,
            sio_client=sio
        )
        centralized_system.run()
        
        if sio: time.sleep(0.1)

    if sio: sio.disconnect()