import numpy as np
from scipy.optimize import fsolve
import copy
class Model:
    
    DESC = "Distributed nash equilibrium seeking: Continuous-time control-theoretic approaches"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = model_config['time_delta']
        self.agent_id = model_config['agent_id']
        self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])
        
        self.memory_updation['v'] += (self.memory['v'] - memory['v'])
        self.memory_updation['v'][adj_agent_id] += (self.memory['v'][adj_agent_id] - memory['partial_cost'])

    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 1e-3:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.my_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 1e-3:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.my_sign(value[i])
        
        return powered_value

    def my_sign(self, value):
        eplsilon = 5e-3
        value_fabs = np.fabs(value)
        
        return value/(value_fabs+eplsilon)
    
    def partial_value_estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        c1 = self.model_config['varphi']
        c2 = self.model_config['sigma']
        gama = self.model_config['gama']
        
        estimation_update = np.zeros(self.memory_updation['v'].shape)

        for i, value in enumerate(self.memory_updation['v']):
            
            value_fabs = np.fabs(value)
            sign = self.my_sign(value)

            # if np.fabs(value) < 1e-6:
            #     sign = 0
            # elif value > 0:
            #     sign = 1
            # else:
            #     sign = -1
            
            estimation_update[i] = -1*sign*(c1*np.power(value_fabs, p) + c2*np.power(value_fabs, q)) - gama*sign*1.5
            
        self.memory['estimation_update'] = estimation_update
        return estimation_update

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        c1 = self.model_config['c1']
        c2 = self.model_config['c2']
        gama = self.model_config['gama']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            value_fabs = np.fabs(value)
            sign = self.my_sign(value)

            # if np.fabs(value) < 1e-6:
            #     sign = 0
            # elif value > 0:
            #     sign = 1
            # else:
            #     sign = -1
            
            estimation_update[i] = -1*sign*(c1*np.power(value_fabs, p) + c2*np.power(
                value_fabs, q)) - gama*sign
            
        
        return estimation_update

    def project(self):
        values = np.zeros(self.memory_updation['z'].shape)
        for i in range(len(self.memory['z'])):
            if i == self.agent_id:
                values[i] = self.memory['x']
            else:
                values[i] = self.memory['z'][i]
                
        values = np.array(values)
        values = values - 0.4*self.memory['v']
        
        value_sum = 0
        for value in values:
            value_sum += value
        
        if value_sum > self.model_config['c']:
            lambda_star = fsolve(self.equation, 1, args=(values, ))
            for i in range(len(values)):
                values[i] = values[i] - lambda_star
        
        for i in range(len(values)):
            values[i] = min(max(values[i], self.model_config['l']), self.model_config['u'])
        
        self.memory['values'] = values
        
        
        return values
            
        
    def equation(self, lambda_star, values):
        
        c = self.model_config['c']
        value_sum = 0
        for value in values:
            value_sum += min(max(value - lambda_star, self.model_config['l']), self.model_config['u'])
        return value_sum - c
        
    def status_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        u = self.model_config['u']
        l = self.model_config['l']
        c = self.model_config['c']
        delta = self.model_config['delta']
        eta = self.model_config['eta']
        x_i = self.memory['x']
        
        values = self.project()
        base_value = -1*x_i + values[self.agent_id]

        norm_value = np.linalg.norm(base_value)
        norm_value = min(norm_value, 2*self.model_config['u']*np.sqrt(len(self.memory['z'])))
        update_value = np.zeros(base_value.shape)
        update_value += base_value* eta * np.power(norm_value, q-1)
        if norm_value > 1e-6:
            update_value += base_value* delta / np.power(norm_value, 1-p)

        # update_value = update_value *(delta * np.power(norm_value, 1-p) + eta / (np.power(norm_value, 1-q)))
        self.memory['update_value'] = update_value
        
        return update_value
    
    def cost_function(self):
        a = self.model_config['a']
        b = self.model_config['b']
        xr = self.model_config['r']
        xi = self.memory['z'][self.agent_id]

        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
            
        Pi = a * status_sum + b
        
        cost = (xi - xr)**2 + xi * Pi

        return cost

    def partial_cost(self):
        #以此替换原本的差分计算逻辑
        a = self.model_config['a']
        b = self.model_config['b']
        xr = self.model_config['r']
        xi = self.memory['z'][self.agent_id]

        # 1. 计算总和 Status Sum
        status_sum = np.sum(self.memory['z'])
            
        # 2. 计算当前的价格 Pi (a * sum + b)
        Pi = a * status_sum + b
        
        # 3. 计算解析梯度 (Analytical Gradient)
        # 公式: 2*(xi - xr) + Pi + a*xi
        # 解释: 
        #   2*(xi - xr) 来自 (xi-xr)^2 的导数
        #   Pi          来自 xi * Pi 对 xi 的直接导数
        #   a * xi      来自 xi * Pi 对内部求和项中 xi 的链式法则导数
        grad = 2 * (xi - xr) + Pi + a * xi

        return grad

    # def partial_cost(self):
    #     delta = 1e-10
    #     cost = self.cost_function()
    #     self.memory['z'][self.agent_id] += delta
    #     cost_hat = self.cost_function()
    #     self.memory['z'][self.agent_id] -= delta
    #     return (cost_hat - cost) / delta

    def update(self):
        self.memory_updation['x'] = self.status_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['v'] = self.partial_value_estimation_update_function()

        all_gradients = np.concatenate([
            self.memory_updation['x'].flatten(),
            self.memory_updation['z'].flatten(),
            self.memory_updation['v'].flatten()
        ])
        
        # 计算 L2 范数
        update_norm = np.linalg.norm(all_gradients)

        # 3. 执行更新 (Apply Update)
        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                # 注意：这里使用的是上一时刻确定的 self.time_delta
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.reset_memroy_updation()
        
        # 4. 返回范数供外部调整步长使用
        return update_norm
