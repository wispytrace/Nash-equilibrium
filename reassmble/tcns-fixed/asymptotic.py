import numpy as np
import copy

class Model:
    
    DESC = "Continuous-time distributed Nash equilibrium seeking algorithms for non-cooperative constrained games"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(self.model_config['time_delta'])
        self.agent_id = model_config['agent_id']
        self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'][0])

        self.memory_updation['lambda'] += (self.memory['lambda'] - memory['lambda'])

    def estimation_update_function(self):
        e = self.model_config['e']
        estimation_update = np.zeros(self.memory_updation['z'].shape)
        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1 * value * e
        
        return estimation_update
    
    def status_update_function(self):
        x_i = self.memory['x']
        u = self.model_config['u']
        l = self.model_config['l']
        partial_value = self.partial_cost()
        gama = self.model_config['gama']
        p_omega = x_i - partial_value - self.memory['lambda']
        if p_omega < l:
            p_omega = l
        if p_omega > u:
            p_omega = u
        
        update_value = -1*x_i + p_omega 
        return update_value * gama

    def lambda_update_function(self):
        lamda_i = self.memory['lambda']
        alpha = self.model_config['alpha']
        c = self.model_config['c']

        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
            
        p_epsilon = lamda_i + status_sum - c
        if p_epsilon <= 0:
            p_epsilon = 0
        if p_epsilon >= alpha:
            p_epsilon = alpha
        
        update_value = -1*lamda_i + p_epsilon
        
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
        delta = 1e-10
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta

    def update(self):
        self.memory_updation['x'] = self.status_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['lambda'] = self.lambda_update_function()
        all_gradients = np.concatenate([
            self.memory_updation['x'].flatten(),
            self.memory_updation['z'].flatten(),
        ])
        
        # 计算 L2 范数
        update_norm = np.linalg.norm(all_gradients)
        for k, v in self.memory.items():
            self.memory[k] += self.memory_updation[k] * self.time_delta
        self.memory['z'][self.agent_id] = self.memory['x'][0]

        self.reset_memroy_updation()

        return update_norm
 