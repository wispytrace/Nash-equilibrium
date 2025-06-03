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
        alpha = self.model_config['alpha']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        dot_qri = dot_y-alpha*(x-y)
        # ddot_qri = 4*self.estimation_update_function()[self.agent_id] -alpha*(dot_x-dot_y)

        ddot_qri = 4*self.estimation_update_function()[self.agent_id]*0 -alpha*(dot_x-dot_y)
        Yi = np.array([[ddot_qri[0], ddot_qri[0]+ddot_qri[1], np.cos(x[1])*(2*ddot_qri[0]+ddot_qri[1])-np.sin(x[1])*dot_x[1]*dot_qri[0]- np.sin(x[1])*(dot_x[0]+dot_x[1])*dot_qri[1],
                      9.8*np.cos(x[0]), 9.8*np.cos(x[0]+x[1])],
                      [0, ddot_qri[0]+ddot_qri[1], np.cos(x[1])*ddot_qri[0]+np.sin(x[1])*dot_x[0]*dot_qri[0], 0, 9.8*np.cos(x[0]+x[1])]])
        return Yi

    
    def get_ui(self):
        alpha = self.model_config['alpha']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        
        si = dot_x-dot_y + alpha*(x-y)

        ui = -1*si + self.get_Yi()@self.memory['theta']
        return ui

    
    def status_update_function(self):

        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        
        Mi, Ci, Gi = self.get_Matrix()

        alpha = self.model_config['alpha']
        dot_qri = dot_y-alpha*(x-y)
        ddot_qri = 4*self.estimation_update_function()[self.agent_id]-alpha*(dot_x-dot_y)
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
        alpha = self.model_config['alpha']
        x = self.memory['x']
        y = self.memory['y']
        dot_x = self.memory['dot_x']
        dot_y = self.virtual_signal_update_function()
        oi = dot_x-dot_y + alpha*(x-y)

        update_value = -1*self.get_Yi().T@oi
        return update_value
    
    def virtual_signal_update_function(self):
        delta = self.model_config['delta']
        partial_cost = self.partial_cost()
        # si = np.zeros(partial_cost.shape)
        # for i in range(len(partial_cost)):
        #     si[i] = partial_cost[i]

        update_value = -1*delta*partial_cost
        # update_value = 0
        # print(update_value)
        
        # self.memory['update_value'] = update_value
        
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

        alpha = self.model_config['alpha']
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            
            estimation_update[i] = -1*alpha*self.power(value, 1)
        
        return estimation_update
    
    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'], self.memory_updation['dot_x'] = self.status_update_function()
        self.memory_updation['theta'] = self.theta_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float)
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()


