import numpy as np
import copy
import random

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(model_config['time_delta'])
        self.initial_scale = self.model_config.get('initial_scale', 1.0)


        self.is_finite = self.model_config.get('is_finite', False)
        self.time = 0
        self.agent_id = self.model_config['agent_id']
        self.memory['y'] = copy.deepcopy(self.model_config['y0'])
        # self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
        # self.memory['y'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
        print(f"{self.agent_id}: init_value {self.memory['y']}")
        self.reset_memroy_updation()
        self.topology_index = 0
        self.switching_time = 0

        self.load_scaled_config()
        self.init_topology_list()
        self.memory['cost'] = self.cost_function()

        print("Agent ID:", self.agent_id)
        self.A = self.getStateMatrix()
        self.B = self.getInputMatrix()
        self.C = self.getOutputMatrix()
        self.K1, self.K2 = self.getKiMatrix()


    def load_scaled_config(self):
        scale_dict = self.model_config['scale_dict']
        for k, v in scale_dict.items():
            if k in self.model_config.keys():
                if isinstance(self.model_config[k], list):
                    for i in range(len(self.model_config[k])):
                        self.model_config[k][i] = v * self.model_config[k][i]
                else:
                    self.model_config[k] = v * self.model_config[k]
                    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)

        self.memory['partial_cost'] = self.partial_cost()
        self.memory['cost'] = self.cost_function()
    
    def get_estimate_update_value(self, src_memory, dst_memory, power_value, adj_agent_id):
        estimate_update = src_memory['z'] - dst_memory['z']
        estimate_update = self.power(estimate_update, power_value)
        estimate_update[adj_agent_id] += self.power((src_memory['z'][adj_agent_id] - dst_memory['y']),power_value)
        return estimate_update

    def receieve_msg(self, adj_agent_id, memory):
        p = self.model_config['p']
        q = self.model_config['q']
        alpha = self.model_config['alpha']
        if self.topology_list[self.topology_index%len(self.topology_list)][self.agent_id][adj_agent_id] > 1e-2:
            self.memory_updation['z'] -= alpha[0]* self.get_estimate_update_value(self.memory, memory, p, adj_agent_id)
            self.memory_updation['z'] -= alpha[1]* self.get_estimate_update_value(self.memory, memory, q, adj_agent_id)
            # self.memory_updation['z'] -= alpha[2]* self.get_estimate_update_value(self.memory, memory, 2*p-1, adj_agent_id)
            # self.memory_updation['z'] -= alpha[3]* self.get_estimate_update_value(self.memory, memory, 2*q-1, adj_agent_id)
            # self.memory_updation['z'] -= alpha[3]* self.get_estimate_update_value(self.memory, memory, 0, adj_agent_id)

    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 5.1e-6:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.approximate_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 5.1e-6:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.approximate_sign(value[i])
        
        return powered_value
    
    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            if np.fabs(value[i]) < 5.1e-6:
                sign_value[i] = 0
            else:
                sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value

    def approximate_sign(self, value):
        extra = 5e-3
        value = value/(np.fabs(value)+extra)
        return value
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']
        tau = self.model_config.get('tau', self.model_config['beta'])

        partial_value_cost = self.partial_cost()

        # if self.topology_index != 0:
        #     update_value = -tau[0] * self.power(partial_value_cost, p) - tau[1] * self.power(partial_value_cost, q)
        # else:
        #     update_value = - beta[0] * self.power(partial_value_cost, p) - beta[1] * self.power(partial_value_cost, q)
        # print(partial_value_cost)
        update_value = - beta[0] * self.power(partial_value_cost, p) - beta[1] * self.power(partial_value_cost, q)

        self.memory['update_value'] = update_value
        
        return update_value

    def getStateMatrix(self):
        parameters = self.model_config['parameters']
        Twi = parameters[0]
        TRi = parameters[1]
        TGi = parameters[2]
        TWi_hat = Twi*0.5
        TRi_hat = TRi*7.4
        A = np.zeros((3, 3))
        A[0,0] = -1/TWi_hat
        A[0,1] = 1/TWi_hat + (Twi)/(TWi_hat*TRi_hat)
        A[0,2] = -2*(1/TRi_hat-1/(7.6*TGi))
        A[1,1] = -1/TRi_hat
        A[1,2] = -1/(7.6*TGi)
        A[2,2] = -1/(TGi)
        # print("A:", A)
        return np.matrix(A)

    def getInputMatrix(self):
        parameters = self.model_config['parameters']
        Twi = parameters[0]
        TRi = parameters[1]
        TGi = parameters[2]
        TWi_hat = Twi*0.5
        TRi_hat = TRi*7.4
        B = np.zeros((3, 3))
        B[0,0] = -1/(3.8*TGi)
        B[1,1]= 1/(7.6*TGi)
        B[2,2] = 1/(TGi)
        # print("B:", B)
        return np.matrix(B)

    def getOutputMatrix(self):
        C = np.zeros((1, 3))
        C[0,0] = 1
        # print("C:", C)
        return np.matrix(C)

    def getKiMatrix(self):
        K1 = np.array([1/self.B[0,0], 0, 0])
        K2 = np.linalg.inv(self.B)@self.A@self.B@K1
        # print("K1, K2:", K1, K2)
        return np.matrix(K1), np.matrix(K2)

    # def getNormalMatrix(self):
    #     Qs = np.array([self.B, self.A@self.B, self.A@self.A@self.B])
    #     t1 = np.array([0,0,1])@np.linalg.inv(Qs)
    #     T = np.array([t1, t1@self.A, t1@self.A@self.A]).T
    #     return T
        
            
    def cost_function(self):
        # z 保存了所有智能体状态（即发射功率 power profile x）的估计
        # 将其展平为一维数组，方便进行向量化计算，对应图1中的 x = [x1, ..., xN]
        z = self.memory['z'].flatten()
        agent_id = self.agent_id
        
        # --- 1. 获取公式所需的所有参数 ---
        # 对应能源消耗定价系数 c_i 和 带宽权重因子 w_i
        ci = self.model_config['ci']
        wi = self.model_config['wi']
        
        # 对应背景噪声功率 \sigma^2 和 信道增益矩阵 G
        sigma2 = self.model_config['sigma2']
        G = np.array(self.model_config['G_matrix']) 
        
        # --- 2. 提取自身的功率 x_i 和直接信道增益 g_{ii} ---
        x_i = z[agent_id]
        g_ii = G[agent_id, agent_id]
        
        # --- 3. 计算交叉信道干扰 \sum_{j \neq i} g_{ji} x_j ---
        # 【性能优化】使用矩阵点积 np.dot 计算该基站收到的总功率，然后减去自身功率。
        # 这比写 for 循环遍历 j != i 要快几十倍
        total_received_power = np.dot(G[agent_id, :], z)
        interference = total_received_power - g_ii * x_i
        
        # --- 4. 计算信干噪比 (SINR) ---
        # 公式 (1.1): \gamma_i(x) = (g_ii * x_i) / (\sum_{j \neq i} g_ji * x_j + \sigma^2)
        # 添加 np.abs 确保在偏导数微小扰动计算时，如果干扰项极小且出现浮点误差，分母保持正数
        sinr = (g_ii * x_i) / (np.abs(interference) + sigma2)
        
        # --- 5. 计算最终的 Cost ---
        # 公式 (1.2): f_i(x_i, x_{-i}) = c_i * x_i - w_i * ln(1 + \gamma_i(x))
        # 使用 np.maximum(sinr, 0.0) 是一种安全机制，防止在系统仿真剧烈波动的瞬态出现负功率，导致 ln(负数) 抛出 NaN 错误
        cost = ci * x_i - wi * np.log(1 + np.maximum(sinr, 0.0))
        
        return cost

    def partial_cost(self):
        delta = 1e-4
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta


    def estimation_update_function(self):
        return self.memory_updation['z']

    def init_topology_list(self):
        topology_list = []
        
        # Topo 0: 健康状态 (5节点环形拓扑: 0-1-2-3-4-0)
        topology_list.append([[0, 1, 0, 0, 1],
                              [1, 0, 1, 0, 0],
                              [0, 1, 0, 1, 0],
                              [0, 0, 1, 0, 1],
                              [1, 0, 0, 1, 0]])
        
        # Topo 1: 节点 0 被孤立 (剩余链路: 1-2, 2-3, 3-4)
        topology_list.append([[0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 1, 0, 1, 0],
                              [0, 0, 1, 0, 1],
                              [0, 0, 0, 1, 0]])
        
        # Topo 2: 节点 1 被孤立 (剩余链路: 2-3, 3-4, 4-0)
        topology_list.append([[0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 1],
                              [1, 0, 0, 1, 0]])
        
        # Topo 3: 节点 2 被孤立 (剩余链路: 3-4, 4-0, 0-1)
        topology_list.append([[0, 1, 0, 0, 1],
                              [1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [1, 0, 0, 1, 0]])

        # Topo 4: 节点 3 被孤立 (剩余链路: 4-0, 0-1, 1-2)
        topology_list.append([[0, 1, 0, 0, 1],
                              [1, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0]])

        # Topo 5: 节点 4 被孤立 (剩余链路: 0-1, 1-2, 2-3)
        topology_list.append([[0, 1, 0, 0, 0],
                              [1, 0, 1, 0, 0],
                              [0, 1, 0, 1, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0]])

        # Topo 6: 全网瘫痪 (所有节点通信中断，全 0)
        topology_list.append([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]])
                              
        self.topology_list = topology_list

    def switching(self):
        Dos_interval = self.model_config.get('DoS_interval', {})
        is_found = False
        for index, interval_list in Dos_interval.items():
            for interval in interval_list:
                if self.time >= interval[0] and self.time <= interval[1]:
                    self.topology_index = int(index)
                    is_found = True
                    break
        if not is_found:
            self.topology_index = 0
        # duration = 0.1
        # self.switching_time += self.time_delta
        # if self.switching_time >= 0.4:
        #     is_switch = random.random() < self.model_config['epsilon']
        #     if is_switch:
        #         self.topology_index = random.randint(1, len(self.topology_list))
        
        # if self.switching_time >= 0.4+ duration:
        #     self.switching_time = 0
        
        self.memory['topology_index'] = np.array(self.topology_index)

        # if self.switching_time < (1-self.model_config['epsilon'])*duration:
        #     self.topology_index = 0
        # elif self.switching_time < (1-self.model_config['epsilon']/2)*duration:
        #     self.topology_index = 1
        # else:
        #     self.topology_index = 2
        
        # if self.switching_time >= 0.3:
        #     self.switching_time = 0

    
    def update(self):
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                # 【修改这里】使用 np.asarray 安全地处理标量和数组
                self.memory[k] = np.asarray(self.memory[k], dtype=np.float64) + 1e-20
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        self.switching()
        self.reset_memroy_updation()
        
    
    def get_action_value(self):
        return eval(self.model_config['action'])


