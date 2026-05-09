import numpy as np
import copy
from scipy.optimize import fsolve

class Model:
    
    DESC = "Mi(qi)¨qi +Ci(qi, ˙qi) ˙qi +Gi(qi) = ui=> || Distributed Nash Equilibrium Seeking for Uncertain Euler-Lagrange Systems over Jointly Strongly Connected Networks"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = self.model_config['memory']
        self.time_delta = self.model_config['time_delta']
        self.agent_id = self.model_config['agent_id']
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
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        self.memory_updation['v'] += (self.memory['v'] - memory['v'])
        self.memory_updation['v'][adj_agent_id] += (self.memory['v'][adj_agent_id] - memory['partial_cost'])
        
    def approximate_sign(self, value):
        extra = 1e-3
        value = value/(np.fabs(value)+extra)
        return value

    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 1e-9:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.approximate_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 1e-9:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.approximate_sign(value[i])
        
        return powered_value
    
    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-9:
                sign_value[i] = 0
            else:
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
        self.memory['oi'] = oi
        ui = Gi + Ci@dot_x - h2*Mi@(self.power(oi, p)+self.power(oi, q)) - h1*Mi@(p*np.multiply(self.power(track_error, p-1),dot_track_error)+ q*np.multiply(
            self.power(track_error, q-1), dot_track_error)+ dot_track_error)
        self.memory['ui'] = ui
        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x-Gi)
        
         
        return dot_x, ddot_x 
    
    def virtual_signal_update_function(self):

        p = self.model_config['p']
        q = self.model_config['q']
        eta = self.model_config['eta']
        x_i = self.memory['y']
        
        values = self.project()
        update_value = -1*x_i + values[self.agent_id]

               
        norm_value = np.linalg.norm(values.flatten())
        norm_value = min(max(norm_value, 1e-4), 2*max(self.model_config['u'])*np.sqrt(len(self.memory['z'].flatten())))

        update_value = update_value *(eta[0] / np.power(norm_value, 1-p) + eta[1] / (np.power(norm_value, 1-q)) + eta[2])
        # self.memory['update_value'] = update_value
        
        return update_value


    def project(self):
        values = np.zeros(self.memory_updation['z'].shape)
        for i in range(len(self.memory['z'])):
            if i == self.agent_id:
                values[i] = self.memory['y']
            else:
                values[i] = self.memory['z'][i]
                
        # values = np.array(values)
        values = values - 1.4*self.memory['v']

        # self.memory['before_values'] = np.copy(values)

        value_sum = np.zeros(self.memory['y'].shape)
        for value in values:
            value_sum += value

        # self.memory['before_values_sum'] = value_sum

        projected_values = copy.deepcopy(values)
        project_minus = np.zeros(self.memory['y'].shape)

        for i in range(len(value_sum)):
            l  = self.model_config['l'][i]
            u = self.model_config['u'][i]
            c = self.model_config['c'][i]
            if value_sum[i] > c:
                calculate_value = np.copy(values[:,i])
                project_minus[i] = fsolve(self.equation, 0, args=(calculate_value, c , l ,u, ))


        self.memory['lmabda'] = project_minus

        for i in range(len(projected_values)):
            for j in range(len(projected_values[i])):
                projected_values[i, j] = projected_values[i,j] - project_minus[j]
                projected_values[i,j] = min(max(projected_values[i, j], self.model_config['l'][j]), self.model_config['u'][j])
        
        self.memory['values'] = projected_values
        
        return projected_values


    def equation(self, lambda_star, values, c, l ,u):
        
        value_sum = 0
        for value in values:
            value_sum += min(max(value - lambda_star, l), u)
        return value_sum - c
            
    def cost_function(self):
        cost = 0
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        cost += 1/2*0.5*(np.linalg.norm(zi-posi)**2)
        status_sum = np.zeros(self.memory['x'].shape)
        # pos_sum = np.zeros(self.memory['x'].shape)
        for i in range(len(self.memory['z'])):
            status_sum += self.memory['z'][i]
            # pos_sum += self.model_config['pos'][i]
        # status_sum = status_sum
        # pos_sum = pos_sum/5
        cost += 1/2*0.01*(np.linalg.norm(status_sum-5*self.model_config['pos_c'])**2)
        # self.memory['status_sum'] = status_sum
        # self.memory['zi-posi'] = zi-posi
        # self.memory['caulcate'] = zi-posi + status_sum - 5*self.model_config['pos_c']
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
        gama = self.model_config['gama']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(
                value, q) + alpha[2]*self.power(value, 1) + gama*self.sign(value))

                    
        return estimation_update


    def partial_value_estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        beta = self.model_config['beta']
        gama = self.model_config['gama']
        lipsthitz = self.model_config['lipsthitz']
        N = self.model_config['N']
        
        estimation_update = np.zeros(self.memory_updation['v'].shape)
        for i, value in enumerate(self.memory_updation['v']):
            estimation_update[i] = -1*(beta[0]*self.power(value, p) + beta[1]*self.power(value, q) + beta[2]*self.power(value,1) + lipsthitz*gama/2*self.sign(value))
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
        
        self.reset_memroy_updation()


