import numpy as np
import copy

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = model_config
        self.memory = self.model_config['memory']
        self.time_delta = model_config['time_delta']
        self.time = 0
        self.agent_id = model_config['agent_id']
        self.memory['x'][0,:] = self.model_config['x0']
        self.memory['y'] = self.model_config['x0']
        self.memory['ei_sum'] = 0
        self.reset_memroy_updation()

        self.load_scaled_config()
            
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
            self.memory_updation[k] = np.zeros(np.array(v).shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        
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
    

    def approximate_sign(self, value):
        extra = 1e-2
        value = value/(np.fabs(value)+extra)
        return value

    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_cost = self.partial_cost()
        
        update_value = np.zeros(partial_cost.shape)
        for i in range(len(update_value)):
            update_value[i] = -1*(beta[0]*self.power(partial_cost[i], p-1) + beta[1]*self.power(partial_cost[i], q-1))
        self.memory['update_value'] = update_value
        
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']

        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(
                value, q) + alpha[2]*self.power(value, 2*p-1) + alpha[3]*self.power(value, 2*q-1))
                    
        return estimation_update

    
    def status_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        eta = self.model_config['eta']
        zeta = self.model_config['zeta']
        order = self.model_config['order']
        gama = self.model_config['gama']
        gama_i = np.zeros(order)
        gama_i_tilde = np.zeros(order)
        k_i = self.model_config['ki']
        k_i_tilde = self.model_config['k_i_tilde']
        for i in range(order):
            if i==0:
                gama_i[order-i-1] = gama[0]
                gama_i_tilde[order-i-1] = gama[1]
            elif i==1:
                gama_i[order-i-1] = gama[0]/(2-gama[0])
                gama_i_tilde[order-i-1] = gama[1]/(2-gama[1])
            else:
                gama_i[order-i-1] = gama_i[order-i]*gama_i[order-i+1]/(2*gama_i[order-i+1]-gama_i[order-i])
                gama_i_tilde[order-i-1] = gama_i_tilde[order-i]*gama_i_tilde[order-i+1]/(2*gama_i_tilde[order-i+1]-gama_i_tilde[order-i])
        x_i = self.memory['x']
        eij = np.zeros(x_i.shape)
        for i in range(order):
            if i == 0:
                eij[i] = x_i[i]-self.memory['y']
            else:
                eij[i] = x_i[i]

        ei_sum = self.memory['ei_sum']
        for i in range(order):
            ei_sum += k_i[i]*self.power(eij[i], gama_i[i])*self.time_delta + k_i_tilde[i]*self.power(eij[i], gama_i_tilde[i])*self.time_delta
        self.memory['ei_sum'] = ei_sum

        si = x_i[-1,:] + ei_sum
        ui = -1*(eta*self.power(si, p) + zeta*self.power(si, q))
        for i in range(order):
            ui +=  -1*(k_i[i]*self.power(eij[i], gama_i[i]) + k_i_tilde[i]*self.power(eij[i], gama_i_tilde[i]))
        
        x_i_update = np.zeros(x_i.shape)
        for i in range(order):
            if i == order-1:
                x_i_update[i] = ui
            else:
                x_i_update[i] = x_i[i+1]

        return x_i_update

    def cost_function_follower(self):
        cost = 0
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        cost += 1/2*(np.linalg.norm(zi-posi)**2)
        status_sum = 0
        for i in range(len(self.memory['z'])):
            status_sum += 1/2*(np.linalg.norm(zi[:2]-self.memory['z'][i][:2]))
        cost += status_sum

        return cost

    def cost_function_leader(self):
        cost = 0
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        cost += 1/2*(np.linalg.norm(zi-posi)**2)
        status_sum = 0
        for i in range(len(self.memory['z'])):
            status_sum += 1/2*(np.linalg.norm(zi[:2]-self.memory['z'][i][:2]))
            if i==4 or i==5:
                status_sum += 1/2*(np.linalg.norm(zi[-1]-self.memory['z'][i][-1]))
        # pos_sum = np.zeros(self.memory['x'].shape)
        # for i in range(len(self.memory['z'])):
        #     status_sum += self.memory['z'][i]
            # pos_sum += self.model_config['pos'][i]
        # status_sum = status_sum
        # pos_sum = pos_sum/5
        cost += status_sum
        return cost
        
            
    def cost_function(self):
        cost = 0
        zi = self.memory['z'][self.agent_id]
        posi = self.model_config['pos'][self.agent_id]
        cost += 1/2*(np.linalg.norm(zi-posi)**2)
        status_sum = np.zeros(self.memory['x'].shape)
        # pos_sum = np.zeros(self.memory['x'].shape)
        for i in range(len(self.memory['z'])):
            status_sum += self.memory['z'][i]
            # pos_sum += self.model_config['pos'][i]
        # status_sum = status_sum
        # pos_sum = pos_sum/5
        cost += 1/2*(np.linalg.norm(status_sum/6-self.model_config['pos_c'])**2)
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

    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        # print(self.memory_updation['y'])
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float) + 1e-20
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()
    
    def get_action_value(self):
        return eval(self.model_config['action'])


