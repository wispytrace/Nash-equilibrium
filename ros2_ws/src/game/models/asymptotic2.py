import numpy as np
import copy
class Model:
    
    DESC = "Distributed Generalized Nash Equilibrium Seeking for Noncooperative Games of Second-Order Multi-agent Systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = self.model_config['memory']
        self.time_delta = model_config['time_delta']
        self.agent_id = model_config['agent_id']
        self.count = 0
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.count += 1
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'][0])
        
    def estimation_update_function(self):
        e = self.model_config['e']
        estimation_update = np.zeros(self.memory_updation['z'].shape)
        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1 * value * e
        
        return estimation_update
    
    def status_update_function(self):
        u = self.model_config['u']
        l = self.model_config['l']
        partial_value = self.partial_cost()
        k2 = self.model_config['k2']
        x_i = self.memory['x']
        c = self.model_config['c']
        
        status_sum = -c
        for status in self.memory['z']:
            status_sum += status
        p_omega = x_i - (partial_value + max(0, status_sum+self.memory['lambda']))
        
        p_omega = min(u, max(l, p_omega))

        # print(self.memory['z'], self.memory['lambda'], 10/(self.count * self.time_delta + 1), partial_value, status_sum+self.memory['lambda'])
        alpha_t = 4/(self.count * self.time_delta + 1)
        
        update_value = -alpha_t * x_i + alpha_t*p_omega
        
        return update_value 

    def lambda_update_function(self):
        
        c = self.model_config['c']

        status_sum = -c
        for status in self.memory['z']:
            status_sum += status
        
        alpha_t = 4/(self.count * self.time_delta + 1)
        update_value = -1*alpha_t/2*self.memory['lambda'] + alpha_t/2*(max(0, self.memory['lambda'] + status_sum))

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
        
        for k, v in self.memory.items():
            self.memory[k] += self.memory_updation[k] * self.time_delta

        self.reset_memroy_updation()
 