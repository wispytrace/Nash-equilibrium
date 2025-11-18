import numpy as np
import copy
import random

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(model_config['time_delta'])
        self.initial_scale = self.model_config.get('initial_scale', 1.0)
        self.is_finite = self.model_config.get('is_finite', False)
        self.time = 0
        self.agent_id = self.model_config['agent_id']
        self.memory['x'] = copy.deepcopy(self.model_config['x0']) * self.initial_scale
        self.memory['y'] = copy.deepcopy(self.model_config['y0']) * self.initial_scale
        
        self.reset_memroy_updation()
        self.topology_index = 0
        self.switching_time = 0

        self.load_scaled_config()
        self.init_topology_list()
        self.memory['cost'] = self.cost_function()

        print("Agent ID:", self.agent_id)
        self.A = self.getStateMatrix()
        self.B = self.getInputMatrix()
        self.C = self.getOutputMatrix()
        self.K1, self.K2 = self.getKiMatrix()


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
    
    def get_estimate_update_value(self, src_memory, dst_memory, power_value, adj_agent_id):
        estimate_update = src_memory['z'] - dst_memory['z']
        estimate_update = self.power(estimate_update, power_value)
        estimate_update[adj_agent_id] += self.power((src_memory['z'][adj_agent_id] - dst_memory['y']),power_value)
        return estimate_update

    def receieve_msg(self, adj_agent_id, memory):
        p = self.model_config['p']
        q = self.model_config['q']
        alpha = self.model_config['alpha']
        if self.topology_list[self.topology_index%len(self.topology_list)][self.agent_id][adj_agent_id] > 1e-2:
            self.memory_updation['z'] -= alpha[0]* self.get_estimate_update_value(self.memory, memory, p, adj_agent_id)
            self.memory_updation['z'] -= alpha[1]* self.get_estimate_update_value(self.memory, memory, q, adj_agent_id)
            self.memory_updation['z'] -= alpha[2]* self.get_estimate_update_value(self.memory, memory, 2*p-1, adj_agent_id)
            self.memory_updation['z'] -= alpha[3]* self.get_estimate_update_value(self.memory, memory, 2*q-1, adj_agent_id)

    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 1e-6:
                return 0
            else:
                return np.power(np.fabs(value), a) * self.approximate_sign(value)
    
        powered_value = np.zeros(value.shape)
        for i in range(len(value)):
            fabs_value = np.fabs(value[i])
            if fabs_value < 1e-10:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]),a) * self.approximate_sign(value[i])
        
        return powered_value
    
    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-10:
                sign_value[i] = 0
            else:
                sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value

    def approximate_sign(self, value):
        extra = 5e-3
        value = value/(np.fabs(value)+extra)
        return value
    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_value_cost = self.partial_cost()

        update_value = - beta[0] * self.power(partial_value_cost, p) - beta[1] * self.power(partial_value_cost, q)
        # print(partial_value_cost)

        self.memory['update_value'] = update_value
        
        return update_value

    def getStateMatrix(self):
        parameters = self.model_config['parameters']
        Twi = parameters[0]
        TRi = parameters[1]
        TGi = parameters[2]
        TWi_hat = Twi*0.5
        TRi_hat = TRi*7.4
        A = np.zeros((3, 3))
        A[0,0] = -1/TWi_hat
        A[0,1] = 1/TWi_hat + (Twi)/(TWi_hat*TRi_hat)
        A[0,2] = -2*(1/TRi_hat-1/(7.6*TGi))
        A[1,1] = -1/TRi_hat
        A[1,2] = -1/(7.6*TGi)
        A[2,2] = -1/(TGi)
        print("A:", A)
        return np.matrix(A)

    def getInputMatrix(self):
        parameters = self.model_config['parameters']
        Twi = parameters[0]
        TRi = parameters[1]
        TGi = parameters[2]
        TWi_hat = Twi*0.5
        TRi_hat = TRi*7.4
        B = np.zeros((3, 3))
        B[0,0] = -1/(3.8*TGi)
        B[1,1]= 1/(7.6*TGi)
        B[2,2] = 1/(TGi)
        print("B:", B)
        return np.matrix(B)

    def getOutputMatrix(self):
        C = np.zeros((1, 3))
        C[0,0] = 1
        print("C:", C)
        return np.matrix(C)

    def getKiMatrix(self):
        K1 = np.array([1/self.B[0,0], 0, 0])
        K2 = np.linalg.inv(self.B)@self.A@self.B@K1
        print("K1, K2:", K1, K2)
        return np.matrix(K1), np.matrix(K2)

    # def getNormalMatrix(self):
    #     Qs = np.array([self.B, self.A@self.B, self.A@self.A@self.B])
    #     t1 = np.array([0,0,1])@np.linalg.inv(Qs)
    #     T = np.array([t1, t1@self.A, t1@self.A@self.A]).T
    #     return T
        

    def status_update_function(self):
        ki = self.model_config['ki']
        xi = self.memory['x']
        gama = self.model_config['gama']
        yi = np.matrix(self.memory['y']).T
        dot_yi = np.matrix(self.memory['update_value']).T
        epsilon_i = xi - np.array(self.B@self.K1.T@yi).flatten()
        Omega_i = -1*(ki[0] * self.power(epsilon_i, gama[0]) + ki[1]*self.power(epsilon_i, gama[1]))
        self.memory['Omega_i'] = Omega_i
        ui_hat =  np.linalg.inv(self.B)@(np.matrix(Omega_i).T-self.A@np.matrix(epsilon_i).T)
        ui = -self.K2.T@yi+ self.K1.T@dot_yi + ui_hat
        self.memory['ui'] = np.array(ui).flatten()
        
        status_update = self.A @ np.matrix(xi).T + (self.B @ ui)

        status_update = np.array(status_update).flatten()

        return status_update

            
    def cost_function(self):
        a = self.model_config['a']
        po = self.model_config['po']
        xi = self.model_config['xi']

        action = self.memory['z'][self.agent_id]

        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
        
        price =  status_sum*a + po

        cost = (action - xi)**2 + price*action


        return cost*0.1

    def partial_cost(self):
        delta = 1e-4
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta


    def estimation_update_function(self):
        return self.memory_updation['z']

    def init_topology_list(self):
        topology_list = []
        topology_list.append([[0, 1, 0, 1],
                              [1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [1, 0, 1, 0],])
        
        topology_list.append([[0, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1],
                              [1, 0, 1, 0],])
        
        topology_list.append([[0, 1, 0, 0],
                              [1, 0, 1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0],])
        
        topology_list.append([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],])
        self.topology_list = topology_list


    def switching(self):
        Dos_interval = self.model_config.get('DoS_interval', {})
        is_found = False
        for index, interval_list in Dos_interval.items():
            for interval in interval_list:
                if self.time >= interval[0] and self.time <= interval[1]:
                    self.topology_index = int(index)
                    is_found = True
                    break
        if not is_found:
            self.topology_index = 0
        # duration = 0.1
        # self.switching_time += self.time_delta
        # if self.switching_time >= 0.4:
        #     is_switch = random.random() < self.model_config['epsilon']
        #     if is_switch:
        #         self.topology_index = random.randint(1, len(self.topology_list))
        
        # if self.switching_time >= 0.4+ duration:
        #     self.switching_time = 0
        
        self.memory['topology_index'] = np.array(self.topology_index)

        # if self.switching_time < (1-self.model_config['epsilon'])*duration:
        #     self.topology_index = 0
        # elif self.switching_time < (1-self.model_config['epsilon']/2)*duration:
        #     self.topology_index = 1
        # else:
        #     self.topology_index = 2
        
        # if self.switching_time >= 0.3:
        #     self.switching_time = 0

    
    def update(self):
        
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['x'] = self.status_update_function()

        for k, v in self.memory.items():
            if k in self.memory_updation.keys():
                self.memory[k] = self.memory[k].astype(float) + 1e-20
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        self.switching()
        self.reset_memroy_updation()
    
    def get_action_value(self):
        return eval(self.model_config['action'])


