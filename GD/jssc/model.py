import numpy as np
import copy

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(self.model_config['time_delta'])
        self.time = 0
        self.agent_id = self.model_config['agent_id']
        # self.memory['y'] = copy.deepcopy(self.model_config['x0']) * self.initial_scale
        self.reset_memroy_updation()

    def set_init_value(self, key, init_value):
        self.memory[key] = copy.deepcopy(init_value)
        # 【修改点 3】：当初始化虚拟信号 vr 时，顺便把观测器里关于自己的状态初始化，大幅减小初始误差
        if key == 'vr':
            self.memory['z'][self.agent_id] = copy.deepcopy(init_value)

    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(np.array(v).shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['vr'])
        
        alpha = self.model_config['alpha']
        self.memory_updation['y'] += alpha[2]*self.power(self.memory['y'] - memory['y'], self.model_config['mu']) + alpha[3]*self.power(self.memory['y'] - memory['y'], self.model_config['nu'])
        
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

    def y_update_function(self):
        ddotpi  = np.array([-8*np.cos(2*self.time+self.agent_id*np.pi/2), -8*np.sin(2*self.time+self.agent_id*np.pi/2), 0])
        y_update_value = -1*self.memory_updation['y'] + ddotpi
        return y_update_value


    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_cost = self.partial_cost()
        N = self.memory['z'].shape[0]
        dot_pi = np.array([-4*np.sin(2*self.time+self.agent_id*np.pi/2), 4*np.cos(2*self.time+self.agent_id*np.pi/2), 0.5])
        dot_pg = np.array([-2.5*np.sin(0.5*self.time), 2.5*np.cos(0.5*self.time), 0.5])
        # 
        # if self.time > 1:
        #     print(partial_cost, self.memory['y'], dot_pi, dot_pg)
        update_value = -beta[0]*self.power(partial_cost, p) - beta[1]*self.power(partial_cost, q)+ dot_pi + 1/(N+1)*dot_pg - 1/(N+1)*self.memory['y']
        self.memory['update_value'] = update_value
        
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        gama = self.model_config['gama']
        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(value, q)) - gama*self.sign(value)
            estimation_update[i] = np.clip(estimation_update[i], -10e3, 10e3)
        
        return estimation_update

            
    def cost_function(self):
        # 固定的博弈交互矩阵 A (非对称)
        N = self.memory['z'].shape[0]
        i = self.agent_id
        # 获取当前智能体的状态
        zi = self.memory['z'][i]
        # 获取当前智能体的局部目标位置 (用作线性偏置项)
        pi = [2*np.cos(2*self.time+i*np.pi/2), 2*np.sin(2*self.time+i*np.pi/2), 0.5*self.time]
        pg = [5*np.cos(0.5*self.time), 5*np.sin(0.5*self.time), 0.5*self.time]
        individual_cost = np.power(np.linalg.norm(zi - pi), 2)
        status_sum = np.zeros(3)
        for j in range(self.memory['z'].shape[0]):
            zj = self.memory['z'][j]
            status_sum += zj
        status_sum = status_sum / N
        coupling_cost = np.power(np.linalg.norm(status_sum - pg), 2)
        cost = individual_cost + coupling_cost
            
        return cost


    def partial_cost(self):
        """
        使用解析解 (Analytical Gradient) 直接计算梯度
        彻底消除数值差分带来的浮点数噪声，防止在分数次幂下产生抖振
        """
        N = self.memory['z'].shape[0]
        i = self.agent_id
        
        # 1. 转化为 numpy array
        pi = np.array([2*np.cos(2*self.time+i*np.pi/2), 2*np.sin(2*self.time+i*np.pi/2), 0.5*self.time])
        pg = np.array([5*np.cos(0.5*self.time), 5*np.sin(0.5*self.time), 0.5*self.time])
        # print(pg)
        
        # 2. 获取当前智能体状态与全局均值
        zi = self.memory['z'][i]
        status_mean = np.mean(self.memory['z'], axis=0) 
        
        # 3. 严格按照代价函数求导的精确解析解公式计算梯度
        # f_i(z_i) = ||z_i - p_i||^2 + || (1/N)*sum(z_j) - p_g ||^2
        # 对 z_i 求导 -> 2*(z_i - p_i) + 2/N * ((1/N)*sum(z_j) - p_g)
        grad = 2 * (zi - pi) + (2.0 / N) * (status_mean - pg)
        # if self.agent_id == 0:
        #     print(grad, status_mean, zi, pi ,pg )
        return grad
    
    def update(self):
        
        self.memory_updation['vr'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['y'] = self.y_update_function()

        for k in self.memory.keys():
            if k in self.memory_updation.keys():
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()
