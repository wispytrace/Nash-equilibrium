import numpy as np
import copy
from scipy.optimize import fsolve

class Model:
    
    DESC = "Mi(qi)¨qi +Ci(qi, ˙qi) ˙qi +Gi(qi) = ui=> || Distributed Nash Equilibrium Seeking for Uncertain Euler-Lagrange Systems over Jointly Strongly Connected Networks"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = self.model_config['memory']
        self.time_delta = model_config['time_delta']
        self.time = 0
        self.switching_time = 0
        self.topology_index = 0
        self.agent_id = model_config['agent_id']
        self.reset_memroy_updation()
        self.init_topology_list()
        self.memory['cost'] = self.cost_function()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.memory['partial_cost'] = self.partial_cost()
        
    def receieve_msg(self, adj_agent_id, memory):
        p = self.model_config['mu']
        q = self.model_config['nu']
        alpha = self.model_config['alpha']
        beta = self.model_config['alpha']
        if self.topology_list[self.topology_index%len(self.topology_list)][self.agent_id][adj_agent_id] > 0:
            self.memory_updation['z'] += -1*(alpha* self.power((self.memory['z'] - memory['z']), p) + beta* self.power((self.memory['z'] - memory['z']), q))

    def power(self, value, a):
        powered_value = np.power(np.fabs(value),a) * np.sign(value)
        return powered_value
    
    def sign(self, value):
        sign_value = value/(np.fabs(value)+0.01)

        return sign_value
    
    
    def status_update_function(self):
        p = self.model_config['mu']
        q = self.model_config['nu']
        delta = self.model_config['delta']
        eta = self.model_config['eta']

        partial_value_cost = self.partial_cost()
        ppartial_value_cost = 1+1/6
        
        update_value = np.zeros(self.memory['x'].shape)
        update_value = -1/ppartial_value_cost*(delta*self.power(partial_value_cost, p) + eta*self.power(partial_value_cost, q))
        
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
        delta = 1e-10
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta


    def estimation_update_function(self):
        return self.memory_updation['z']

    def init_topology_list(self):
        topology_list = []
        topology_list.append([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]])
        topology_list.append([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        topology_list.append([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]])
        self.topology_list = topology_list


    def switching(self):
        duration = 0.3
        self.switching_time += self.time_delta

        if self.switching_time < (1-self.model_config['epsilon'])*duration:
            self.topology_index = 0
        elif self.switching_time < (1-self.model_config['epsilon']/2)*duration:
            self.topology_index = 1
        else:
            self.topology_index = 2
        
        if self.switching_time >= 0.3:
            self.switching_time = 0


    def update(self):
        self.memory['cost'] = self.cost_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['z'][self.agent_id] += self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.memory['x'] = self.memory['z'][self.agent_id]
        
        self.time += self.time_delta
        self.switching()
        
        self.reset_memroy_updation()


