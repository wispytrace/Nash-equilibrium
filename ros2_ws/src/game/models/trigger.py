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
        self.agent_id = model_config['agent_id']
        # self.memory['x'] = self.model_config['init_value_x'][self.agent_id]
        # self.memory['z_event'] = self.model_config['z_event']
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])
        

    def power(self, value, a):
        powered_value = np.power(np.fabs(value),a) * np.sign(value)
        
        return powered_value
    
    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            sign_value[i] = np.sign(value[i])
        
        return sign_value
    
    
    def status_update_function(self):

        delta = self.model_config['delta']
        partial_cost_value = self.partial_cost()
        update_value = -1*np.sign(partial_cost_value) * delta

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
        
        estimation_update = np.zeros(self.memory['z_event'].shape)

        for i, value in enumerate(self.memory['z_event']):
            value_fabs = np.fabs(value)
            sign = None
            if value_fabs < 1e-4:
                sign = 0
            elif value > 0:
                sign = 1
            else:
                sign = -1
            
            estimation_update[i] = -1*sign*(alpha*np.power(value_fabs, mu) + beta*np.power(
                value_fabs, nu)) - gama*sign
                    
        return estimation_update

    def evet_trigger(self):
        mu = self.model_config['mu']
        nu = self.model_config['nu']
        alpha = self.model_config['alpha']
        gama = self.model_config['gama']
        beta = self.model_config['beta']
        epsilon = self.model_config['epsilon']

        # print("z")
        # print(self.memory_updation['z'])
        # print("z_event")
        # print(self.memory['z_event'])
    
        for i, value in enumerate(self.memory_updation['z']):
            v = alpha*self.power(self.memory['z_event'][i], mu) + beta*self.power(self.memory['z_event'][i], nu) + gama*np.sign(self.memory['z_event'][i])
            v = v- (alpha*self.power(self.memory_updation['z'][i], mu) + beta*self.power(self.memory_updation['z'][i], nu) + gama*np.sign(self.memory_updation['z'][i]))
            v = np.fabs(v)
            value_fabs = np.fabs(self.memory_updation['z'][i])
            trigger_value = epsilon*(alpha*np.power(value_fabs, mu) + beta*np.power(value_fabs, nu) + gama)
            if v >= trigger_value:
                self.memory['z_event'][i] = self.memory_updation['z'][i]
                self.memory['is_trigger'][i] = 1
        #     print(v-trigger_value, v)

        # print("trigger")
        # print(self.memory['is_trigger'])
        

    def update(self):
        self.memory['is_trigger'] = np.zeros(self.memory['is_trigger'].shape)
        self.evet_trigger()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()


