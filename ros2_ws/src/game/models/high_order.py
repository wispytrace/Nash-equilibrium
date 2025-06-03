import numpy as np
import copy
from scipy.optimize import fsolve

class Model:
    
    DESC = "Mi(qi)¨qi +Ci(qi, ˙qi) ˙qi +Gi(qi) = ui=> || Distributed Nash Equilibrium Seeking for Uncertain Euler-Lagrange Systems over Jointly Strongly Connected Networks"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = {}
        self.time_delta = model_config['time_delta']
        self.time = 0
        self.count = 0
        self.is_switch = False
        self.agent_id = model_config['agent_id']
        self.delay = model_config['delay']
        self.max_delay = model_config['max_delay']
        self.buffer = {}
        self.load_scaled_config()
        self.init_states()
        
        self.reset_memroy_updation()
    
    def init_states(self):
        
        self.memory['y'] = self.model_config['init_value_x'][self.agent_id]
        self.memory['v'] = self.model_config['init_value_v'][self.agent_id]
        self.memory['z'] = self.model_config['init_value_z'][self.agent_id]
        self.memory['uz'] = self.model_config['init_value_uz'][self.agent_id]
        self.memory['uv'] = self.model_config['init_value_uv'][self.agent_id]
        self.memory['z_hat'] = self.memory['z']
        self.memory['v_hat'] = self.memory['v']
        self.memory['y_hat'] = self.memory['y']
        self.memory['partial_cost_hat'] = self.partial_cost(self.memory['z_hat'])
        self.memory['gama'] =  np.array([self.model_config['d0']])

        self.buffer['y'] = []
        self.buffer['v'] = []
        self.buffer['z'] = []
        self.buffer['uz'] = []
        self.buffer['uv'] = []
        for i in range(self.max_delay):
            self.buffer['y'].append(self.memory['y'])
            self.buffer['v'].append(self.memory['v'])
            self.buffer['z'].append(self.memory['z'])
            self.buffer['uz'].append(self.memory['uz'])
            self.buffer['uv'].append(self.memory['uv'])

    
    def load_scaled_config(self):
        scale_dict = self.model_config['scale_dict']
        for k, v in scale_dict.items():
            if k in self.model_config.keys():
                if isinstance(self.model_config[k], list):
                    for i in range(len(self.model_config[k])):
                        self.model_config[k][i] = v * self.model_config[k][i]
                else:
                    self.model_config[k] = v * self.model_config[k]

    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z_hat'] - memory['z_hat'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y_hat'])
        
        self.memory_updation['v'] += (self.memory['v_hat'] - memory['v_hat'])
        self.memory_updation['v'][adj_agent_id] += (self.memory['v_hat'][adj_agent_id] - memory['partial_cost_hat'])
        
    def approximate_sign(self, value):
        extra = 0.1
        value = value/(np.fabs(value)+extra)
        return value

    def power(self, value, a):
        if isinstance(value, float):
            if np.fabs(value) < 1e-3:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.approximate_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 1e-3:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.approximate_sign(value[i])
        
        return powered_value
    
    def sign(self, value):
        if isinstance(value, float):
            return self.approximate_sign(value)
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value
    
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        eta = self.model_config['eta']

        si = np.zeros(self.memory['v'].shape)
        all_cost = []
        for i in range(len(si)):
            si[i] = min(np.fabs(self.memory['v'][i]), self.model_config['di']) * np.sign(self.memory['v'][i])
            all_cost.append(si[i])
        all_cost = np.array(all_cost)
        cost_norm = np.linalg.norm(all_cost)


        update_value = -1*(eta[0]*si[self.agent_id]* self.power(cost_norm, p-1) + eta[1]*si[self.agent_id]*self.power(cost_norm, q-1) + eta[2]*si[self.agent_id])
                
        return update_value
        
            
    def cost_function(self, states):
        a = self.model_config['a']
        b = self.model_config['b']
        xr = self.model_config['r']
        xi = states[self.agent_id]

        status_sum = 0
        for status in states:
            status_sum += status
            
        Pi = a * status_sum + b
        
        cost = (xi - xr)**2 + xi * Pi

        return cost


    def partial_cost(self, states=None):
        delta = 1e-6
        states = states if states is not None else self.memory['z']

        cost = self.cost_function(states)
        states[self.agent_id] += delta
        cost_hat = self.cost_function(states)
        states[self.agent_id] -= delta
        partial_cost_value = (cost_hat - cost) / delta
            
        return partial_cost_value


    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        alpha = self.model_config['alpha']
        gama = self.memory['gama'][0]
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(
                value, q) + alpha[2]*self.power(value, 1) + gama*self.sign(value))
        
        self.memory['uz'] = estimation_update
                    
        return estimation_update


    def partial_value_estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        beta = self.model_config['beta']
        gama = self.memory['gama'][0]
        lipsthitz = self.model_config['lipsthitz']
        N = self.model_config['N']
        
        estimation_update = np.zeros(self.memory_updation['v'].shape)

        for i, value in enumerate(self.memory_updation['v']):
            estimation_update[i] = -1*(beta[0]*self.power(value, p) + beta[1]*self.power(value, q) + beta[2]*self.power(value,1) + lipsthitz*np.sqrt(N)*gama*self.sign(value))

        self.memory['uv'] = estimation_update            
        
        return estimation_update


    def store_buffer(self):

        self.buffer['v'].append(self.memory['v'])
        self.buffer['z'].append(self.memory['z'])
        self.buffer['uz'].append(self.memory['uz'])
        self.buffer['uv'].append(self.memory['uv'])
        self.buffer['y'].append(self.memory['y'])
        self.buffer['v'].pop(0)
        self.buffer['z'].pop(0)
        self.buffer['uz'].pop(0)
        self.buffer['uv'].pop(0)
        self.buffer['y'].pop(0)
        self.memory['y_hat'] = self.buffer['y'][-self.max_delay]
        self.memory['z_hat'] = self.buffer['z'][-self.max_delay]
        self.memory['v_hat'] = self.buffer['v'][-self.max_delay]
        self.memory['partial_cost_hat'] = self.partial_cost(self.memory['z_hat'])

        for i in range(self.max_delay):
            self.memory['z_hat'] += self.time_delta * self.buffer['uz'][-self.max_delay + i]
            self.memory['v_hat'] += self.time_delta * self.buffer['uv'][-self.max_delay + i]
        
    
    def update(self):
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['v'] = self.partial_value_estimation_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.store_buffer()
        
        vi_max = -1
        for i in range(len(self.memory['v'])):
            fabs_value = np.fabs(self.memory['v'][i])
            if fabs_value > vi_max:
                vi_max = fabs_value

        di_max = self.model_config['d0'] + vi_max
        eta_max = self.model_config['eta_max']

        di = self.model_config['di']
        self.memory['gama'] = np.array([eta_max[0]*self.power(di, self.model_config['p']) + eta_max[1]*self.power(di, self.model_config['q']) + eta_max[2]*di])


        if self.time >= self.model_config['tau']:
            if not self.is_switch:
                self.model_config['di'] = self.model_config['d0'] + di_max
                self.is_switch = True
        
        self.time += self.time_delta
        self.count += 1
        
        self.reset_memroy_updation()


