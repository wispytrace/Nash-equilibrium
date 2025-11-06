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
        # p = self.model_config['p']
        # q = self.model_config['q']
        # eta = self.model_config['eta']
        update_value = self.memory['x'] - 1*self.partial_cost()
        k1 = 0
        k2 = 0
        k1 -= self.memory['lambda'][0]*-1
        k1 -= self.memory['lambda'][1]*1
        k2 -= self.memory['lambda'][2]*-1
        k2 -= self.memory['lambda'][3]*1


        update_value += np.array([k1, k2])

        # update_value[0] = min(max(-2, update_value[0]), 2)
        # update_value[1] = min(max(-2, update_value[1]), 2)

        # update_value = update_value *(eta[0] / np.power(norm_value, 1-p) + eta[1] / (np.power(norm_value, 1-q)) + eta[2])
        # self.memory['update_value'] = update_value
        
        return 1*(-self.memory['x'] + update_value)
    

    def lambda_update_function(self):
        lamda_value = self.memory['lambda']
        alpha = self.model_config['alpha']
        u = self.model_config['u']
        l = self.model_config['l']
        status_sum = np.zeros(self.memory['x'].shape)
        for status in self.memory['z']:
            status_sum += status
        
        g_value = np.zeros(self.memory['lambda'].shape)
        g_value[0] = -1*status_sum[0] - l
        g_value[1] = status_sum[0] - u
        g_value[2] = -1*self.memory['z'][self.agent_id]
        g_value[3] = self.memory['z'][self.agent_id] - 5.5
            
        
        lambda_update_value = np.zeros(self.memory_updation['lambda'].shape)

        for i, value in enumerate(self.memory_updation['lambda']):
            lambda_update_value[i] = lamda_value[i]+g_value[i] - value
            if lambda_update_value[i] < 0:
                lambda_update_value[i] = 0
            lambda_update_value[i] -= lamda_value[i]
    
        self.memory['lambda_update_value'] = lambda_update_value
        return lambda_update_value

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
 