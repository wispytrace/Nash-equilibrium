import numpy as np
import copy

class Model:
    
    DESC = "Hetegeous systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(model_config['time_delta'])
        self.initial_scale = model_config.get('initial_scale', 1.0)
        self.time = 0
        self.agent_id = model_config['agent_id']
        self.init_x()
        # self.memory['y'] = copy.deepcopy(self.model_config['x0']) * self.initial_scale
        self.reset_memroy_updation()        

    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(np.array(v).shape)
        self.memory['partial_cost'] = self.partial_cost()
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['omega'])
        

    def set_init_value(self, key, init_value):
        self.memory[key] = copy.deepcopy(init_value)

    def power(self, value, a):
        if len(value.shape) == 0:
            if np.fabs(value) < 1e-10:
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
        extra = 1e-10
        value = value/(np.fabs(value)+extra)
        return value

    
    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_cost = self.partial_cost()
        update_value = -1*(beta[0]*self.power(partial_cost, p) + beta[1]*self.power(partial_cost, q))
        # print(partial_cost, update_value, beta[0]*self.power(partial_cost[i], p), beta[1]*self.power(partial_cost[i], q))
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
            estimation_update[i] = np.clip(estimation_update[i], -10e3, 10e3)
        
        return estimation_update

            
    def cost_function(self):
        """
        5个智能体的非合作博弈代价函数
        该函数联合伪梯度满足强单调性 (mu 约等于 2.64) 和 Lipschitz连续 (L 约等于 3.5)
        """
        # 提取参数
        pg = self.model_config['pg']
        N = self.model_config['N']
        zi = self.memory['z'][self.agent_id]
        
        # 修复逻辑BUG：Python的 % 可以直接处理负数，无需复杂的if-else
        pre_agent = (self.agent_id - 1) % N
        next_agent = (self.agent_id + 1) % N
        
        # 修复数学BUG：必须使用平方L2范数 (** 2)，否则不可导且不满足强单调性
        cost = 0.5 * (np.linalg.norm(zi - pg) ** 2)
        cost += 0.5 * (np.linalg.norm(zi - self.memory['z'][pre_agent]) ** 2) 
        cost += 0.5 * (np.linalg.norm(zi - self.memory['z'][next_agent]) ** 2)
        
        return cost


    def partial_cost(self):
        delta = 1e-5
        partial_cost_value = np.zeros(self.memory['z'][self.agent_id].shape)
        for i in range(len(self.memory['z'][self.agent_id])):
            cost = self.cost_function()
            self.memory['z'][self.agent_id][i] += delta
            cost_hat = self.cost_function()
            self.memory['z'][self.agent_id][i] -= delta
            partial_cost_value[i] = (cost_hat - cost) / delta
            
        return partial_cost_value


# 下面是高阶系统特有的函数
    def hl_status_update_function(self):
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
        x_i = self.memory['x_hl']
        eij = np.zeros(x_i.shape)
        for i in range(order):
            if i == 0:
                eij[i] = x_i[i]-self.memory['omega']
            else:
                eij[i] = x_i[i]

        error_sum = np.zeros(2)
        for i in range(order):
            error_sum += k_i[i]*self.power(eij[i], gama_i[i]) + k_i_tilde[i]*self.power(eij[i], gama_i_tilde[i])
        si = eij[order-1,:] + self.memory['ei_sum']
        self.memory['ei_sum'] += error_sum * self.time_delta
        ui = -1*(eta*self.power(si, p) + zeta*self.power(si, q)) - error_sum
        self.memory['ui_hl'] = ui
        
        x_i_update = np.zeros(x_i.shape)
        for i in range(2):
            if i == order-1:
                x_i_update[i] = ui
            elif i < order:
                x_i_update[i] = x_i[i+1]
            else:
                x_i_update[i] = 0
        
        return x_i_update


# 下面是线性系统特有的函数
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
        # print("A:", A)
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
        # print("B:", B)
        return np.matrix(B)

    def getOutputMatrix(self):
        C = np.zeros((1, 3))
        C[0,0] = 1
        # print("C:", C)
        return np.matrix(C)

    def getKiMatrix(self):
        K1 = np.array([1/self.B[0,0], 0, 0])
        K2 = np.linalg.inv(self.B)@self.A@self.B@K1
        return np.matrix(K1), np.matrix(K2)

    def linear_status_update_function(self):
        self.A = self.getStateMatrix()
        self.B = self.getInputMatrix()
        self.C = self.getOutputMatrix()
        self.K1, self.K2 = self.getKiMatrix()
        status_update = np.zeros_like(self.memory['x_li'])
        # 假设这里的 self.memory['x'] 形状是 (3, 2)
        for i in range(self.memory['x_li'].shape[0]):
            ki = self.model_config['ki']
            xi = self.memory['x_li'][i]
            yi = np.matrix(self.memory['omega'][i]).T
            gama = self.model_config['gama']
            dot_yi = np.matrix(self.memory['update_value'][i]).T
            epsilon_i = xi - np.array(self.B@self.K1.T@yi).flatten()
            Omega_i = -1*(ki[0] * self.power(epsilon_i, gama[0]) + ki[1]*self.power(epsilon_i, gama[1]))
            ui_hat =  np.linalg.inv(self.B)@(np.matrix(Omega_i).T-self.A@np.matrix(epsilon_i).T)
            ui = -self.K2.T@yi+ self.K1.T@dot_yi + ui_hat
            self.memory['ui_li'][i] = np.array(ui).flatten()
            
            status_update[i] = (self.A @ np.matrix(xi).T + (self.B @ ui)).squeeze()

            status_update[i] = np.array(status_update[i]).flatten()

        return status_update


# 下面是欧拉拉格朗日系统特有的函数

    def get_Matrix(self):
        i = self.agent_id-1
        a = self.model_config['parameter_matrix']
        x = self.memory['x_el'][0]
        dot_x = self.memory['x_el'][1]
        Mi = [[a[i,0]+a[i,1]+2*a[i,2]*np.cos(x[1]), a[i, 1]+a[i, 2]*np.cos(x[1])],
              [a[i, 1]+a[i, 2]*np.cos(x[1]), a[i, 1]]]
        
        Ci = [[-a[i,2]*dot_x[1]*np.sin(x[1]), -a[i,2]*(dot_x[0]+dot_x[1])*np.sin(x[1])],
              [a[i, 2]*dot_x[0]*np.sin(x[1]), 0]]
        
        Gi = [a[i,3]*9.8*np.cos(x[0])+ a[i, 4]*9.8*np.cos(x[0]+x[1]), a[i, 4]*9.8*np.cos(x[0]+x[1])]
        
        return np.array(Mi), np.array(Ci), np.array(Gi)
    
    
    def euler_status_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']

        x = self.memory['x_el'][0]
        dot_x = self.memory['x_el'][1]
        dot_y = self.virtual_signal_update_function()
        
        track_error = x - self.memory['omega']
        sign_track_error = np.zeros(track_error.shape)
        for i in range(len(track_error)):
            sign_track_error[i] = self.approximate_sign(track_error[i])
        dot_track_error = np.multiply(dot_x - dot_y, sign_track_error)
        
        Mi, Ci, Gi = self.get_Matrix()
        
        oi = dot_x + h1*(self.power(track_error,p) + self.power(track_error, q) + track_error)
        ui = Gi + Ci@dot_x - h2*Mi@(self.power(oi, p)+self.power(oi, q)) - h1*Mi@(p*np.multiply(self.power(track_error, p-1),dot_track_error)+ q*np.multiply(
            self.power(track_error, q-1), dot_track_error)+ dot_track_error)
        self.memory['ui_el'] = ui
        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x-Gi)
        
        update_value = np.zeros(self.memory['x_el'].shape)
        update_value[0] = dot_x
        update_value[1] = ddot_x
         
        return update_value

    def action_update(self):
        hight_agents = [0, 1]
        linear_agents = [2, 3]
        euler_agents = [4, 5]

        if self.agent_id in hight_agents:
            self.memory_updation['x_hl'] = self.hl_status_update_function()
            self.memory['x'] = self.memory['x_hl'][0]
        
        elif self.agent_id in linear_agents:
            self.memory_updation['x_li'] = self.linear_status_update_function()
            self.memory['x'] = self.memory['x_li'][:,0]
        
        elif self.agent_id in euler_agents:
            self.memory_updation['x_el'] = self.euler_status_update_function()
            self.memory['x'] = self.memory['x_el'][0]

    def init_x(self, scale_factor=1.0):
        """
        初始化智能体状态
        :param scale_factor: 初始值放缩倍数，默认为1.0（不改变原值）
        """
        hight_agents = [0, 1]
        linear_agents = [2, 3]
        euler_agents = [4, 5]

        # 提前计算好翻倍/缩放后的初始值，使代码更简洁
        # 注意：如果 init_value 是 Python 原生列表(list)，直接相乘会变成列表拼接。
        # 建议确保 init_value 是 numpy 数组或者标量数字。
        scaled_init_value = self.model_config['init_value'] * scale_factor

        if self.agent_id in hight_agents:
            self.memory['x_hl'] = scaled_init_value
        
        elif self.agent_id in linear_agents:
            self.memory['x_li'] = scaled_init_value
        
        elif self.agent_id in euler_agents:
            self.memory['x_el'] = scaled_init_value
        
        if 'init_omega' in self.model_config:
            self.memory['omega'] = self.model_config['init_omega']

    def update(self):
        
        self.memory_updation['omega'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.action_update()


        for k in self.memory.keys():
            if k in self.memory_updation.keys():
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()
    
    def get_action_value(self):
        return eval(self.model_config['action'])