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
        self.memory['x'] = self.model_config['init_value_x'][self.agent_id]
        self.memory['y'] = self.model_config['init_value_x'][self.agent_id]
        self.memory['dot_x'] = self.model_config['init_value_dotx'][self.agent_id]
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        self.memory_updation['lambda'] += (self.memory['lambda'] - memory['lambda'])

    def power(self, value, a):
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            powered_value[i] = np.power(np.fabs(value[i]),a) * np.sign(value[i])
        
        return powered_value
    
    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            sign_value[i] = np.sign(value[i])
        
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
    
    def get_Yi(self):
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']
        alpha = self.model_config['alpha']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        dot_qri = h1*dot_y-h2*(x-y)
        # ddot_qri = 4*self.estimation_update_function()[self.agent_id] -alpha*(dot_x-dot_y)
        if 'pre_dot_y' in self.memory.keys():
            ddot_y = (dot_y - self.memory['pre_dot_y'])/self.time_delta
        else:
            ddot_y = 0

        ddot_qri = h1*ddot_y -h2*(dot_x-dot_y)
        Yi = np.array([[ddot_qri[0], ddot_qri[0]+ddot_qri[1], np.cos(x[1])*(2*ddot_qri[0]+ddot_qri[1])-np.sin(x[1])*dot_x[1]*dot_qri[0]- np.sin(x[1])*(dot_x[0]+dot_x[1])*dot_qri[1],
                      9.8*np.cos(x[0]), 9.8*np.cos(x[0]+x[1])],
                      [0, ddot_qri[0]+ddot_qri[1], np.cos(x[1])*ddot_qri[0]+np.sin(x[1])*dot_x[0]*dot_qri[0], 0, 9.8*np.cos(x[0]+x[1])]])
        return Yi

    
    def get_ui(self):
        # alpha = self.model_config['alpha']
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        
        si = h1*dot_x-dot_y + h2*(x-y)

        ui = -1*si + self.get_Yi()@self.memory['theta']
        return ui

    
    def status_update_function(self):

        dot_x = self.memory['dot_x']        
        Mi, Ci, Gi = self.get_Matrix()
        # dot_qri = [2, 3]
        # ddot_qri = [2, 4]
        # Yi = np.array([[ddot_qri[0], ddot_qri[0]+ddot_qri[1], np.cos(x[1])*(2*ddot_qri[0]+ddot_qri[1])-np.sin(x[1])*dot_x[1]*dot_qri[0]- np.sin(x[1])*(dot_x[0]+dot_x[1])*dot_qri[1],
        #               9.8*np.cos(x[0]), 9.8*np.cos(x[0]+x[1])],
        #               [0, ddot_qri[0]+ddot_qri[1], np.cos(x[1])*ddot_qri[0]+np.sin(x[1])*dot_x[0]*dot_qri[0], 0, 9.8*np.cos(x[0]+x[1])]])
        # estimate = self.get_Yi()@self.memory['theta']
        # exact = Mi@ddot_qri + Ci@dot_qri + Gi
        # print(estimate, exact, estimate-exact)
        # print(self.memory['theta'])
        
        ddot_x = np.linalg.inv(Mi)@(-Ci@dot_x-Gi+self.get_ui())
         
        return dot_x, ddot_x 

    def theta_update_function(self):
        # alpha = self.model_config['alpha']
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        oi = h1*(dot_x-dot_y) + h2*(x-y)

        update_value = -1*self.get_Yi().T@oi
        return update_value
    
    def virtual_signal_update_function(self):

        # p = self.model_config['p']
        # q = self.model_config['q']
        # eta = self.model_config['eta']
        update_value = self.memory['y'] - 1.4*self.partial_cost()
        k1 = 0
        k2 = 0
        k1 -= self.memory['lambda'][0]*-1
        k1 -= self.memory['lambda'][1]*1
        k2 -= self.memory['lambda'][2]*-1
        k2 -= self.memory['lambda'][3]*1


        update_value += np.array([k1, k2])

        update_value[0] = min(max(-2, update_value[0]), 2)
        update_value[1] = min(max(-2, update_value[1]), 2)

        # update_value = update_value *(eta[0] / np.power(norm_value, 1-p) + eta[1] / (np.power(norm_value, 1-q)) + eta[2])
        # self.memory['update_value'] = update_value
        
        return 1*(-self.memory['y'] + update_value)
            
            
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

    def lambda_update_function(self):
        lamda_value = self.memory['lambda']
        alpha = self.model_config['alpha']
        c = self.model_config['c']
        status_sum = np.zeros(self.memory['y'].shape)
        for status in self.memory['z']:
            status_sum += status
        
        g_value = np.zeros(self.memory['lambda'].shape)
        g_value[0] = -1*status_sum[0] - 0.4
        g_value[1] = status_sum[0] - 0.4
        g_value[2] = -1*status_sum[1] - 0.4
        g_value[3] = status_sum[1] - 0.4
            
        
        lambda_update_value = np.zeros(self.memory_updation['lambda'].shape)

        for i, value in enumerate(self.memory_updation['lambda']):
            lambda_update_value[i] = lamda_value[i]+g_value[i] - value
            if lambda_update_value[i] < 0:
                lambda_update_value[i] = 0
            lambda_update_value[i] -= lamda_value[i]
    
        self.memory['lambda_update_value'] = lambda_update_value
        return lambda_update_value
    

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

        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            estimation_update[i] = -1*0.2*(alpha[0]+alpha[1]+alpha[2])*value
            # print(alpha[0]*self.power(value, p) + alpha[1]*self.power(
            #     value, q) + alpha[2]*self.power(value, 1), gama*self.sign(value))
                    
        return estimation_update
    
    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'], self.memory_updation['dot_x'] = self.status_update_function()
        self.memory_updation['theta'] = self.theta_update_function()
        self.memory_updation['lambda'] = self.lambda_update_function()
        self.memory['pre_dot_y'] = self.virtual_signal_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()


