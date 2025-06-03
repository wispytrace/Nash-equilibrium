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
        
    
    def receieve_msg(self, adj_agent_id, memory):
        if self.topology_list[self.topology_index%len(self.topology_list)][self.agent_id][adj_agent_id] > 0:
            self.memory_updation['z'] += (self.memory['z'] - memory['z'])
            self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])
        

    def power(self, value, a):
        powered_value = np.power(np.fabs(value),a) * np.sign(value)
        
        return powered_value
    
    def sign(self, value):
        sign_value = value/(np.fabs(value)+0.01)
        # sign_value = np.zeros(value.shape)
        # for i in range(len(value)):
        #     sign_value[i] = np.sign(value[i])
        
        return sign_value
    
    
    def status_update_function(self):

        delta = self.model_config['delta']
        partial_cost_value = self.partial_cost()
        self.memory['partial'] = partial_cost_value
        update_value = -1*self.sign(partial_cost_value) * delta

        return update_value
    
            
    def cost_function(self):
        

        s = 0.5
        f = 20
        pi = self.model_config['p']
        qi = self.model_config['q']
        xi = self.memory['z'][self.agent_id]
        cost = 0
        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
        
        cost = pi*((xi-qi)**2) + (s*status_sum+f)*xi

        return cost

    def partial_cost(self):
        delta = 1e-10
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta


    def estimation_update_function(self):
        mu = self.model_config['mu']
        nu = self.model_config['nu']

        alpha = self.model_config['alpha']
        gama = self.model_config['gama']
        beta = self.model_config['beta']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            value_fabs = np.fabs(value)
            sign = self.sign(value)
            
            estimation_update[i] = -1*sign*(alpha*np.power(value_fabs, mu) + beta*np.power(
                value_fabs, nu)) - gama*sign
                    
        return estimation_update

    def init_topology_list(self):
        topology_list = []
        topology_list.append([[0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]])
        topology_list.append([[0, 1, 0, 0, 1, 0],
                              [1, 0, 1, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 1, 0],
                              [1, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 1, 0]])
        topology_list.append([[0, 1, 0, 0, 0, 0],
                              [1, 0, 1, 1, 1, 1],
                              [0, 1, 0, 1, 0, 0],
                              [0, 1, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1],
                              [0, 1, 0, 0, 1, 0]])
        self.topology_list = topology_list


    def switching(self):
        if self.time - self.switching_time > self.model_config['epsilon']:
            print("switch!", self.time)
            print(self.topology_list[self.topology_index%len(self.topology_list)])
            self.topology_index += 1
            self.switching_time = self.time

    def update(self):
        self.memory['cost'] = self.cost_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        self.switching()
        
        self.reset_memroy_updation()


