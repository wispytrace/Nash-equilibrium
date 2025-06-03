import numpy as np
import copy
from scipy.optimize import fsolve

class Model:
    
    DESC = "Mi(qi)¨qi +Ci(qi, ˙qi) ˙qi +Gi(qi) = ui=> || Distributed Nash Equilibrium Seeking Algorithms with fixe-time convergence for Euler-Lagrange Systems over Jointly Strongly Connected Networks"
    
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
        self.memory['gama'] =  self.model_config['d0']
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        self.memory_updation['v'] += (self.memory['v'] - memory['v'])
        self.memory_updation['v'][adj_agent_id] += (self.memory['v'][adj_agent_id] - memory['partial_cost'])
        

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
    
    def approximate_sign(self, value):
        extra = 0.1
        value = value/(np.fabs(value)+extra)
        return value

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
        
        disturbance = [2*np.cos(10*self.time), 2*np.sin(10*self.time)]

        oi = dot_x + h1*(self.power(track_error,p) + self.power(track_error, q) + track_error)
        theta = 5*(h1*(p*np.multiply(self.power(track_error, p-1),dot_track_error)+ q*np.multiply(self.power(track_error, q-1), dot_track_error)+ dot_track_error) + 10 + (np.power(np.linalg.norm(dot_x),2)) + 3)
        theta = np.linalg.norm(theta)
        ui = -h2*Mi@(self.power(oi, p)+self.power(oi, q) + self.power(oi, 1)) - theta*self.approximate_sign(oi)

        self.memory['ui'] = ui

        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x-Gi+disturbance)
        
         
        return dot_x, ddot_x 
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        eta = self.model_config['eta']

        si = np.zeros(self.memory['v'].shape)
        all_cost = []
        for i in range(len(si)):
            for j in range(len(si[i])):
                si[i][j] =  min(np.fabs(self.memory['v'][i][j]), self.model_config['di'][j]) * self.approximate_sign(self.memory['v'][i][j])
                all_cost.append(si[i][j])
        
        all_cost = np.array(all_cost)
        cost_norm = np.linalg.norm(all_cost)
        # print(si, self.memory['v']


        update_value = np.zeros(self.model_config['di'].shape)
        for i in range(len(update_value)):
            update_value[i] = -1*(eta[0]*si[self.agent_id][i]* self.power(cost_norm, p-1) + eta[1]*si[self.agent_id][i]*self.power(cost_norm, q-1) + eta[2]*si[self.agent_id][i])

        # update_value = -1*(eta[0]*si[self.agent_id]/(max(1e-4,np.power(np.linalg.norm(si,axis=0), 1-p))))
        # partial_cost = self.partial_cost()
        # si = np.zeros(partial_cost.shape)
        # for i in range(len(partial_cost)):
        #     si[i] = min(np.fabs(partial_cost[i]), self.model_config['di'][i]) * np.sign(partial_cost[i])

        # update_value = -1*(eta[0]*self.power(si, p) +  eta[1]*self.power(si, q) + eta[2]*si)
        
        self.memory['update_value'] = update_value
        
        return update_value
        
            
    def cost_function(self):
        cost = 0
        cost_matrix = np.array(self.model_config['cost'])
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if cost_matrix[i][j] > 0:
                    cost += cost_matrix[i, j]*np.dot(self.memory['z'][i], self.memory['z'][j])
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
        p = self.model_config['p']
        q = self.model_config['q']

        alpha = self.model_config['alpha']
        gama = self.memory['gama']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(
                value, q) + alpha[2]*self.power(value, 1) + gama*self.sign(value))
                    
        return estimation_update


    def partial_value_estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        beta = self.model_config['beta']
        gama = self.memory['gama']
        lipsthitz = self.model_config['lipsthitz']
        N = self.model_config['N']
        
        estimation_update = np.zeros(self.memory_updation['v'].shape)

        for i, value in enumerate(self.memory_updation['v']):
            estimation_update[i] = -1*(beta[0]*self.power(value, p) + beta[1]*self.power(value, q) + beta[2]*self.power(value,1) + lipsthitz*np.sqrt(N)*gama*self.sign(value))
                    
        return estimation_update
    
    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['v'] = self.partial_value_estimation_update_function()
        self.memory_updation['x'], self.memory_updation['dot_x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        di_max = self.model_config['d0'] + np.fabs(np.max(self.memory['v'],axis=0))
        # self.memory['gama'] = self.model_config['delta_max']*self.power(self.model_config['d0'], self.model_config['p']) + self.model_config['eta_max']*self.power(self.model_config['d0'], self.model_config['q'])
        eta_max = self.model_config['eta_max']
        self.memory['gama'] = eta_max[0]*self.power(di_max, self.model_config['p']) + eta_max[1]*self.power(di_max, self.model_config['q']) + eta_max[2]*di_max

        if self.time >= self.model_config['tau']:
            if not self.is_switch:
                self.model_config['di'] = self.model_config['d0'] + np.fabs(np.max(self.memory['v'], axis=0))
                self.is_switch = True
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()


