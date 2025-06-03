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
        self.is_switch = False
        self.agent_id = model_config['agent_id']
        self.memory['x'] = self.model_config['init_value_x'][self.agent_id]
        self.memory['y'] = self.model_config['init_value_x'][self.agent_id]
        self.memory['dot_x'] = self.model_config['init_value_dotx'][self.agent_id]
        self.load_scaled_config()
        
        self.reset_memroy_updation()
    
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
        p = self.model_config['p']
        q = self.model_config['q']
        alpha = self.model_config['alpha']
        self.memory_updation['z'] += -1*(alpha[0]* self.power((self.memory['z'] - memory['z']), p) + alpha[1]* self.power((self.memory['z'] - memory['z']), q))
        
        
    def approximate_sign(self, value):
        extra = 0.01
        value = value/(np.fabs(value)+extra)
        return value

    def power(self, value, a):
        if len(value.shape) == 0:
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
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value
    
    
    def get_Matrix(self):
        i = self.agent_id
        a = self.model_config['parameter_matrix']
        x = self.memory['x']
        dot_x = self.memory['dot_x']
        Mi = [[a[i,0]+a[i,1]+2*a[i,2]*np.cos(x[1]), a[i, 1]+a[i, 2]*np.cos(x[1])],
              [a[i, 1]+a[i, 2]*np.cos(x[1]), a[i, 1]]]
        
        Ci = [[-a[i,2]*dot_x[1]*np.sin(x[1]), -a[i,2]*(dot_x[0]+dot_x[1])*np.sin(x[1])],
              [a[i, 2]*dot_x[0]*np.sin(x[1]), 0]]
        
        Gi = [a[i,3]*9.8*np.cos(x[0])+ a[i, 4]*9.8*np.cos(x[0]+x[1]), a[i, 4]*9.8*np.cos(x[0]+x[1])]
        
        return np.array(Mi), np.array(Ci), np.array(Gi)
    
    
    def status_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']

        x = self.memory['x']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        
        track_error = x - self.memory['y']
        sign_track_error = np.zeros(track_error.shape)
        for i in range(len(track_error)):
            sign_track_error[i] = self.approximate_sign(track_error[i])
        dot_track_error = np.multiply(dot_x - dot_y, sign_track_error)
        
        Mi, Ci, Gi = self.get_Matrix()
        
        oi = dot_x + h1*(self.power(track_error,p) + self.power(track_error, q) + track_error)
        ui = Gi + Ci@dot_x - h2*Mi@(self.power(oi, p)+self.power(oi, q)) - h1*Mi@(p*np.multiply(self.power(track_error, p-1),dot_track_error)+ q*np.multiply(
            self.power(track_error, q-1), dot_track_error)+ dot_track_error)
        self.memory['ui'] = ui

        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x-Gi)
        
         
        return dot_x, ddot_x 
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        eta = self.model_config['eta']

        partial_value_cost = self.partial_cost()
        ppartial_value_cost = 1+1/6
        
        update_value = np.zeros(self.memory['x'].shape)
        update_value = -1/ppartial_value_cost*(eta[0]*self.power(partial_value_cost, p) + eta[1]*self.power(partial_value_cost, 1))
        
        self.memory['update_value'] = update_value
        
        return update_value
        
            
    def cost_function(self):
        cost = 0
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        cost += 1/2*np.linalg.norm(zi-posi)
        status_sum = np.zeros(self.memory['x'].shape)
        pos_sum = np.zeros(self.memory['x'].shape)
        for i in range(len(self.memory['z'])):
            status_sum += self.memory['z'][i]
            pos_sum += self.model_config['pos'][i]
        status_sum = status_sum/5
        pos_sum = pos_sum/5
        cost += 1/2*np.linalg.norm(status_sum-pos_sum)
        return cost

    def partial_cost(self):
        delta = 1e-6
        partial_cost_value = np.zeros(self.memory['z'][self.agent_id].shape)
        for i in range(len(self.memory['z'][self.agent_id])):
            cost = self.cost_function()
            self.memory['z'][self.agent_id][i] += delta
            cost_hat = self.cost_function()
            self.memory['z'][self.agent_id][i] -= delta
            partial_cost_value[i] = (cost_hat - cost) / delta
            
        return partial_cost_value


    def estimation_update_function(self):
                    
        return self.memory_updation['z']
    
    
    def update(self):
        yi_update = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'], self.memory_updation['dot_x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta

        self.memory['z'][self.agent_id] += yi_update
        self.memory['y'] = self.memory['z'][self.agent_id]
    
        self.time += self.time_delta
        
        self.reset_memroy_updation()


