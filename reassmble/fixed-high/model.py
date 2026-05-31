
import numpy as np
import copy

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(model_config['time_delta'])
        self.initial_scale = model_config.get('initial_scale', 1.0)
        self.is_finite = model_config.get('is_finite', False)
        self.time = 0
        self.agent_id = model_config['agent_id']
        self.order = model_config['order']
        
        # 【核心修复】强制将初始状态转换为 float64，避免原地加法 += 时的类型冲突
        x0_float = (copy.deepcopy(self.model_config['x0']) * self.initial_scale).astype(np.float64)
        
        self.memory['x'][0,:] = x0_float
        self.memory['y'] = np.copy(x0_float)
        self.memory['ei_sum'] = 0.0  # 也明确声明为浮点数
        
        # 【优化】将每次循环都要计算且永远不变的 gama 提前至初始化阶段计算
        self._init_gamas()
        
        # 【优化】预分配 memory_updation 的内存空间
        self.memory_updation = {}
        for k, v in self.memory.items():
            if isinstance(v, np.ndarray):
                self.memory_updation[k] = np.zeros_like(v, dtype=np.float64) # 强制指明 dtype
            else:
                self.memory_updation[k] = 0.0

        self.reset_memroy_updation()
        self.load_scaled_config()
    def _init_gamas(self):
        """提前计算 status_update 中的 gama 序列，避免每步冗余计算"""
        order = self.order
        gama = self.model_config['gama']
        self.gama_i = np.zeros(order)
        self.gama_i_tilde = np.zeros(order)
        for i in range(order):
            if i==0:
                self.gama_i[order-i-1] = gama[0]
                self.gama_i_tilde[order-i-1] = gama[1]
            elif i==1:
                self.gama_i[order-i-1] = gama[0]/(2-gama[0])
                self.gama_i_tilde[order-i-1] = gama[1]/(2-gama[1])
            else:
                self.gama_i[order-i-1] = self.gama_i[order-i]*self.gama_i[order-i+1]/(2*self.gama_i[order-i+1]-self.gama_i[order-i])
                self.gama_i_tilde[order-i-1] = self.gama_i_tilde[order-i]*self.gama_i_tilde[order-i+1]/(2*self.gama_i_tilde[order-i+1]-self.gama_i_tilde[order-i])

    def load_scaled_config(self):
        scale_dict = self.model_config.get('scale_dict', {})
        for k, v in scale_dict.items():
            if k in self.model_config.keys():
                if isinstance(self.model_config[k], list):
                    for i in range(len(self.model_config[k])):
                        self.model_config[k][i] = v * self.model_config[k][i]
                else:
                    self.model_config[k] = v * self.model_config[k]

    def reset_memroy_updation(self):
        # 【优化】使用 fill(0.0) 原地清空数组，而不是不断 new 新的 np.zeros
        for k in self.memory_updation.keys():
            if isinstance(self.memory_updation[k], np.ndarray):
                self.memory_updation[k].fill(0.0)
            else:
                self.memory_updation[k] = 0.0
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
    def power(self, value, a):
        # 【优化】完全舍弃 for 循环，使用 np.where 进行高性能条件计算
        value_arr = np.asarray(value)
        abs_v = np.abs(value_arr)
        p_val = np.power(abs_v, a)
        return np.where(abs_v < 1e-10, 0.0, p_val * self.approximate_sign(value_arr))

    def sign(self, value):
        # 【优化】矢量化符号函数计算
        value_arr = np.asarray(value)
        return np.where(np.abs(value_arr) < 1e-10, 0.0, self.approximate_sign(value_arr))
    
    def approximate_sign(self, value):
        # 原生支持矢量化
        extra = 10e-3
        return value / (np.abs(value) + extra)
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']
        partial_cost = self.partial_cost()
        
        # 【优化】直接代入向量化计算更新，摒弃原有的 for 循环逐个赋值
        if self.is_finite:
            partial_cost_norm = np.linalg.norm(partial_cost)
            update_value = -1 * (beta[0] + beta[1]) / (partial_cost_norm + 1e-6) * partial_cost
        else:
            update_value = -1 * (beta[0] * self.power(partial_cost, p) + beta[1] * self.power(partial_cost, q))
            
        self.memory['update_value'] = update_value
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        alpha = self.model_config['alpha']
        z_upd = self.memory_updation['z']

        # 【优化】直接通过多维数组相加进行计算并使用 numpy 的原生 clip，代替 for 循环遍历
        if self.is_finite:
            estimation_update = -30 * self.sign(z_upd)
        else:
            estimation_update = -1 * (
                alpha[0] * self.power(z_upd, p) + 
                alpha[1] * self.power(z_upd, q) + 
                alpha[2] * self.power(z_upd, 2*p - 1) + 
                alpha[3] * self.power(z_upd, 2*q - 1)
            )
        
        return np.clip(estimation_update, -10e3, 10e3)

    def status_update_function(self):
        p = self.model_config['p1']
        q = self.model_config['q1']
        eta = self.model_config['eta']
        zeta = self.model_config['zeta']
        order = self.order
        
        x_i = self.memory['x']
        # 【优化】利用数组拷贝与切片代替 for 循环生成 eij 矩阵
        eij = np.copy(x_i)
        eij[0] = x_i[0] - self.memory['y']

        error_sum = np.zeros_like(eij[0])
        for i in range(order):
            error_sum += self.model_config['ki'][i] * self.power(eij[i], self.gama_i[i]) + \
                         self.model_config['k_i_tilde'][i] * self.power(eij[i], self.gama_i_tilde[i])
                         
        si = eij[order-1, :] + self.memory['ei_sum']
        self.memory['ei_sum'] += error_sum * self.time_delta
        
        ui = -1 * (eta * self.power(si, p) + zeta * self.power(si, q)) - error_sum
        
        x_i_update = np.zeros_like(x_i)
        if order > 1:
            x_i_update[:order-1] = x_i[1:order]
        x_i_update[order-1] = ui
        self.memory['ui'] = ui
        
        return x_i_update
            
    def cost_function(self):
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        
        # 【优化】使用 np.sum() 替代慢速的 np.linalg.norm()**2，在数学上完全等价于 Frobenius Norm 范数平方
        cost = 0.5 * np.sum((zi - posi) ** 2)
        
        # 【优化】使用 numpy 内置的矩阵纵列求和 (axis=0) 代替原始的缓慢求和循环，并通过 broadcast_to 保留原本维度的计算特征
        sum_z = np.sum(self.memory['z'], axis=0)
        # status_sum = np.broadcast_to(sum_z, self.memory['x'].shape)
        
        cost += 0.5 * np.sum((sum_z / self.model_config['N'] - self.model_config['pos_c']) ** 2)
        
        if 'cost_scale' in self.model_config:
            cost *= self.model_config['cost_scale']
            
        return cost

    def partial_cost(self):
        delta = 1e-5
        partial_cost_value = np.zeros_like(self.memory['z'][self.agent_id])
        
        for i in range(len(self.memory['z'][self.agent_id])):
            cost = self.cost_function()
            self.memory['z'][self.agent_id][i] += delta
            cost_hat = self.cost_function()
            self.memory['z'][self.agent_id][i] -= delta
            partial_cost_value[i] = (cost_hat - cost) / delta
            
        return partial_cost_value
    
    def update(self):
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'] = self.status_update_function()

        for k in ['y', 'z', 'x']:
            if k in self.memory_updation:
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        self.reset_memroy_updation()
    
    def get_action_value(self):
        return eval(self.model_config['action'])