import numpy as np
import copy

class Model:
    
    DESC = "Continuous-time distributed Nash equilibrium seeking algorithms for non-cooperative constrained games"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(self.model_config['time_delta'])
        self.agent_id = model_config['agent_id']
        self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
        self.reset_memroy_updation()
    
    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(v.shape)
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'][0])

        # self.memory_updation['lambda'] += (self.memory['lambda'] - memory['lambda'])

    def estimation_update_function(self):
        e = self.model_config['e']
        estimation_update = np.zeros(self.memory_updation['z'].shape)
        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1 * value * e
        
        return estimation_update
    
    def status_update_function(self):
        x_i = self.memory['x']
        u = self.model_config['u']
        l = self.model_config['l']
        partial_value = self.partial_cost()
        gama = self.model_config['gama']
        p_omega = x_i - partial_value - self.memory['lambda']

        if p_omega < l:
            p_omega = l
        if p_omega > u:
            p_omega = u
        
        update_value = -1*x_i + p_omega 
        return update_value * gama

    def lambda_update_function(self):
        lamda_i = self.memory['lambda']
        alpha = self.model_config['alpha']
        c = self.model_config['c']

        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
            
        p_epsilon = lamda_i + status_sum - c
        if p_epsilon <= 0:
            p_epsilon = 0
        if p_epsilon >= alpha:
            p_epsilon = alpha
        
        update_value = -1*lamda_i + p_epsilon
        
        return update_value

    def cost_function(self):
        a = self.model_config['a']
        b = self.model_config['b']
        xr = self.model_config['r']
        xi = self.memory['z'][self.agent_id]

        status_sum = 0
        for status in self.memory['z']:
            status_sum += status
            
        Pi = a * status_sum + b
        
        cost = (xi - xr)**2 + xi * Pi

        return cost

    def partial_cost(self):
        delta = 1e-10
        cost = self.cost_function()
        self.memory['z'][self.agent_id] += delta
        cost_hat = self.cost_function()
        self.memory['z'][self.agent_id] -= delta
        return (cost_hat - cost) / delta

    def update(self):
        self.memory_updation['x'] = self.status_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['lambda'] = self.lambda_update_function()
        all_gradients = np.concatenate([
            self.memory_updation['x'].flatten(),
            self.memory_updation['z'].flatten(),
        ])
        
        # 计算 L2 范数
        update_norm = np.linalg.norm(all_gradients)
        for k, v in self.memory.items():
            self.memory[k] += self.memory_updation[k] * self.time_delta
        self.memory['z'][self.agent_id] = self.memory['x'][0]

        self.reset_memroy_updation()

        return update_norm


# import numpy as np
# import copy
# class Model:
    
#     def __init__(self, model_config) -> None:
#         self.model_config = copy.deepcopy(model_config)
#         self.memory = copy.deepcopy(self.model_config['memory'])
#         self.time_delta = copy.deepcopy(self.model_config['time_delta'])
#         self.agent_id = model_config['agent_id']
#         self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
#         self.count = 0
#         self.reset_memroy_updation()
    
#     def reset_memroy_updation(self):
#         self.memory_updation = {}
#         for k, v in self.memory.items():
#             self.memory_updation[k] = np.zeros(v.shape)
#         self.count += 1
    
#     def receieve_msg(self, adj_agent_id, memory):
#         self.memory_updation['z'] += (self.memory['z'] - memory['z'])
#         self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'][0])
        
#     def estimation_update_function(self):
#         e = self.model_config['e']
#         estimation_update = np.zeros(self.memory_updation['z'].shape)
#         for i, value in enumerate(self.memory_updation['z']):
#             estimation_update[i] = -1 * value * e
        
#         return estimation_update
    
#     def status_update_function(self):
#         # p = self.model_config['p']
#         # q = self.model_config['q']
#         # eta = self.model_config['eta']
#         update_value = self.memory['x'] - 1*self.partial_cost()
#         k1 = 0
#         k2 = 0
#         k1 -= self.memory['lambda'][0]*-1
#         k1 -= self.memory['lambda'][1]*1
#         k2 -= self.memory['lambda'][2]*-1
#         k2 -= self.memory['lambda'][3]*1


#         update_value += np.array([k1, k2])

#         # update_value[0] = min(max(-2, update_value[0]), 2)
#         # update_value[1] = min(max(-2, update_value[1]), 2)

#         # update_value = update_value *(eta[0] / np.power(norm_value, 1-p) + eta[1] / (np.power(norm_value, 1-q)) + eta[2])
#         # self.memory['update_value'] = update_value
        
#         return 1*(-self.memory['x'] + update_value)
    

#     def lambda_update_function(self):
#         lamda_value = self.memory['lambda']
#         alpha = self.model_config['alpha']
#         u = self.model_config['u']
#         l = self.model_config['l']
#         status_sum = np.zeros(self.memory['x'].shape)
#         for status in self.memory['z']:
#             status_sum += status
        
#         g_value = np.zeros(self.memory['lambda'].shape)
#         g_value[0] = -1*status_sum[0] - l
#         g_value[1] = status_sum[0] - u
#         g_value[2] = -1*self.memory['z'][self.agent_id]
#         g_value[3] = self.memory['z'][self.agent_id] - 5.5
            
        
#         lambda_update_value = np.zeros(self.memory_updation['lambda'].shape)

#         for i, value in enumerate(self.memory_updation['lambda']):
#             lambda_update_value[i] = lamda_value[i]+g_value[i] - value
#             if lambda_update_value[i] < 0:
#                 lambda_update_value[i] = 0
#             lambda_update_value[i] -= lamda_value[i]
    
#         self.memory['lambda_update_value'] = lambda_update_value
#         return lambda_update_value

#     def cost_function(self):
#         a = self.model_config['a']
#         b = self.model_config['b']
#         xr = self.model_config['r']
#         xi = self.memory['z'][self.agent_id]

#         status_sum = 0
#         for status in self.memory['z']:
#             status_sum += status
            
#         Pi = a * status_sum + b
        
#         cost = (xi - xr)**2 + xi * Pi

#         return cost

#     def partial_cost(self):
#         delta = 1e-10
#         cost = self.cost_function()
#         self.memory['z'][self.agent_id] += delta
#         cost_hat = self.cost_function()
#         self.memory['z'][self.agent_id] -= delta
#         return (cost_hat - cost) / delta

#     def update(self):
#         self.memory_updation['x'] = self.status_update_function()
#         self.memory_updation['z'] = self.estimation_update_function()
#         self.memory_updation['lambda'] = self.lambda_update_function()
        
#         for k, v in self.memory.items():
#             self.memory[k] += self.memory_updation[k] * self.time_delta

#         self.reset_memroy_updation()
 

# import numpy as np
# import copy
# class Model:

#     def __init__(self, model_config) -> None:
#         self.model_config = copy.deepcopy(model_config)
#         self.memory = copy.deepcopy(self.model_config['memory'])
#         self.time_delta = copy.deepcopy(self.model_config['time_delta'])
#         self.agent_id = model_config['agent_id']
#         self.memory['x'] = copy.deepcopy(self.model_config['init_value'][self.agent_id])
#         self.reset_memroy_updation()
    
#     def reset_memroy_updation(self):
#         self.memory_updation = {}
#         for k, v in self.memory.items():
#             self.memory_updation[k] = np.zeros(v.shape)
    
#     def receieve_msg(self, adj_agent_id, memory):
#         self.memory_updation['z'] += (self.memory['z'] - memory['z'])
#         self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])

#     def estimation_update_function(self):
#         p = self.model_config['p']
#         q = self.model_config['q']

#         c1 = self.model_config['c1']
#         c2 = self.model_config['c2']
#         gama = self.model_config['gama']
        
#         estimation_update = np.zeros(self.memory_updation['z'].shape)

#         for i, value in enumerate(self.memory_updation['z']):
            
#             value_fabs = np.fabs(value)
#             sign = None

#             if np.fabs(value) < 1e-6:
#                 sign = 0
#             elif value > 0:
#                 sign = 1
#             else:
#                 sign = -1
            
#             estimation_update[i] = -1*sign*(c1*np.power(value_fabs, p) + c2*np.power(
#                 value_fabs, q))
        
#         return estimation_update

#     def status_update_function(self):
#         p = self.model_config['p']
#         q = self.model_config['q']

#         delta = self.model_config['delta']
#         eta = self.model_config['eta']
#         epsilon = self.model_config['epsilon']
#         epsilon = 0

#         partial_value = self.partial_cost() + self.model_config['alpha'] * self.l1_constrained_cost()

#         sign = None
#         if partial_value > 0:
#             sign = 1
#         else:
#             sign = -1

#         partial_value_fabs = np.fabs(partial_value)

#         update_value = -(sign*delta*np.power(partial_value_fabs, p) + sign *
#                          eta*np.power(partial_value_fabs, q))

#         return update_value

#     def cost_function(self):
#         a = self.model_config['a']
#         b = self.model_config['b']
#         xr = self.model_config['r']
#         xi = self.memory['z'][self.agent_id]

#         status_sum = 0
#         for status in self.memory['z']:
#             status_sum += status
            
#         Pi = a * status_sum + b
        
#         cost = (xi - xr)**2 + xi * Pi

#         return cost

#     def partial_cost(self):
#         delta = 1e-10
#         cost = self.cost_function()
#         self.memory['z'][self.agent_id] += delta
#         cost_hat = self.cost_function()
#         self.memory['z'][self.agent_id] -= delta
#         return (cost_hat - cost) / delta

#     def l1_constrained_cost(self):
#         u = self.model_config['u']
#         l = self.model_config['l']
#         c = self.model_config['c']
#         xi = self.memory['z'][self.agent_id]
        
#         constrained_cost = 0
        
#         status_sum = 0
#         for status in self.memory['z']:
#             status_sum += status
            
#         if status_sum > c:
#             constrained_cost += 1

#         if xi > u:
#             constrained_cost += 1
        
#         if xi < l:
#             constrained_cost += -1
            
#         return constrained_cost

#     def update(self):
#         self.memory_updation['x'] = self.status_update_function()
#         self.memory_updation['z'] = self.estimation_update_function()

#         for k, v in self.memory.items():
#             self.memory[k] += self.memory_updation[k] * self.time_delta
#         self.reset_memroy_updation()
