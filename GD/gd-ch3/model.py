import numpy as np
import copy

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(model_config['time_delta'])
        self.initial_scale = model_config.get('initial_scale', 1.0)
        self.time = 0
        self.agent_id = model_config['agent_id']
        # self.memory['y'] = copy.deepcopy(self.model_config['x0']) * self.initial_scale
        self.reset_memroy_updation()        

    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(np.array(v).shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        
    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 1e-10:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.approximate_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 1e-10:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.approximate_sign(value[i])
        
        return powered_value

    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-10:
                sign_value[i] = 0
            else:
                sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value
    

    def approximate_sign(self, value):
        extra = 1e-10
        value = value/(np.fabs(value)+extra)
        return value

    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_cost = self.partial_cost()
        update_value = -1*(beta[0]*self.power(partial_cost, p) + beta[1]*self.power(partial_cost, q))
        # print(partial_cost, update_value, beta[0]*self.power(partial_cost[i], p), beta[1]*self.power(partial_cost[i], q))
        self.memory['update_value'] = update_value
        
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(
                    value, q) + alpha[2]*self.power(value, 2*p-1) + alpha[3]*self.power(value, 2*q-1))
            estimation_update[i] = np.clip(estimation_update[i], -10e3, 10e3)
        
        return estimation_update

            
    def cost_function(self):
        """
        5个智能体的非合作博弈代价函数
        该函数联合伪梯度满足强单调性 (mu 约等于 2.64) 和 Lipschitz连续 (L 约等于 3.5)
        """
        # 固定的博弈交互矩阵 A (非对称)
        A = np.array([
            [ 3.06,  0.53, -0.27, -0.07,  0.33],
            [-0.47,  2.94,  0.55, -0.13, -0.05],
            [ 0.13, -0.05,  3.00,  0.52, -0.36],
            [-0.07,  0.07, -0.68,  3.02,  0.40],
            [-0.47, -0.05,  0.04, -0.60,  2.91]
        ])

        i = self.agent_id
        # 获取当前智能体的状态
        zi = self.memory['z'][i]
        # 获取当前智能体的局部目标位置 (用作线性偏置项)
        posi = i+1
        
        # 1. 局部凸代价项: 1/2 * A_ii * zi^2
        cost = 0.5 * A[i, i] * (zi ** 2)
        
        # 2. 邻居博弈交互项: zi * sum(A_ij * zj) (j != i)
        interaction_sum = 0.0
        for j in range(len(self.memory['z'])):
            if j != i:
                interaction_sum += A[i, j] * self.memory['z'][j]
        cost += zi * interaction_sum
        
        # 3. 线性偏置项: - posi * zi (确保最小点偏移出原点)
        cost -= posi * zi
        
        # 4. 整体缩放系数
        if 'cost_scale' in self.model_config.keys():
            cost = cost * self.model_config['cost_scale']
            
        return cost


    def partial_cost(self):
        delta = 1e-5
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        partial_cost_value = (cost_hat - cost) / delta
            
        return partial_cost_value

    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()

        for k in self.memory.keys():
            if k in self.memory_updation.keys():
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()
    
    def get_action_value(self):
        return eval(self.model_config['action'])