import numpy as np
import copy
class Model:
    
    DESC = "Distributed nash equilibrium seeking: Continuous-time control-theoretic approaches"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = self.model_config['memory']
        self.time_delta = model_config['time_delta']
        self.agent_id = model_config['agent_id']
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        c1 = self.model_config['c1']
        c2 = self.model_config['c2']
        gama = self.model_config['gama']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            value_fabs = np.fabs(value)
            sign = None

            if np.fabs(value) < 1e-6:
                sign = 0
            elif value > 0:
                sign = 1
            else:
                sign = -1
            
            estimation_update[i] = -1*sign*(c1*np.power(value_fabs, p) + c2*np.power(
                value_fabs, q))
        
        return estimation_update

    def status_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        delta = self.model_config['delta']
        eta = self.model_config['eta']
        epsilon = self.model_config['epsilon']
        epsilon = 0

        partial_value = self.partial_cost() + self.model_config['alpha'] * self.l1_constrained_cost()

        sign = None
        if partial_value > 0:
            sign = 1
        else:
            sign = -1

        partial_value_fabs = np.fabs(partial_value)

        update_value = -(sign*delta*np.power(partial_value_fabs, p) + sign *
                         eta*np.power(partial_value_fabs, q))

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

    def l1_constrained_cost(self):
        u = self.model_config['u']
        l = self.model_config['l']
        c = self.model_config['c']
        xi = self.memory['z'][self.agent_id]
        
        constrained_cost = 0
        
        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
            
        if status_sum > c:
            constrained_cost += 1

        if xi > u:
            constrained_cost += 1
        
        if xi < l:
            constrained_cost += -1
            
        return constrained_cost

    def update(self):
        self.memory_updation['x'] = self.status_update_function()
        self.memory_updation['z'] = self.estimation_update_function()

        for k, v in self.memory.items():
            self.memory[k] += self.memory_updation[k] * self.time_delta
        self.reset_memroy_updation()
