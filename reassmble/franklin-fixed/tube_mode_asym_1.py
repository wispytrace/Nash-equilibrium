import numpy as np
import copy
from scipy.optimize import fsolve, root_scalar

class Model:
    
    DESC = "Mi*ddot_qi + Ci*dot_qi + Gi(qi) = tau_i => Generator System Standard EL Form"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = self.model_config['memory']
        self.time_delta = self.model_config['time_delta']
        self.agent_id = self.model_config['agent_id']
        
        # 针对单自由度系统，初始化状态维度通常为 (1,)
        print(self.agent_id, self.model_config['init_value_x'][self.agent_id], self.model_config['init_value_dotx'][self.agent_id])
        self.memory['x'] = self.model_config['init_value_x'][self.agent_id]
        # self.memory['y'] = self.model_config['init_value_y'][self.agent_id]
        self.memory['dot_x'] = self.model_config['init_value_dotx'][self.agent_id]
        
        self.reset_memory_updation()
    
    def reset_memory_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            if isinstance(v, np.ndarray):
                self.memory_updation[k] = np.zeros_like(v)
        
        self.sum_z_diff = 0
        self.sum_y_diff = 0
        self.sum_dotq_diff = 0
        self.memory['partial_cost'] = self.partial_cost()
    
    def approximate_sign(self, value):
        extra = 1e-3
        return value / (np.fabs(value) + extra)

    def sign(self, value):
        sign_value = np.zeros(value.shape)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-9:
                sign_value[i] = 0
            else:
                sign_value[i] = self.approximate_sign(value[i])
        
        return sign_value



    def power(self, value, a):
        # 兼容标量和数组的幂次运算结构
        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
            if np.fabs(value) < 1e-9:
                return np.zeros_like(value)
            return np.power(np.fabs(value), a) * self.approximate_sign(value)
            
        powered_value = np.zeros_like(value)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-9:
                powered_value[i] = 0
            else:
                powered_value[i] = np.power(np.fabs(value[i]), a) * self.approximate_sign(value[i])
        return powered_value

    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['x'])

        self.sum_z_diff += (self.memory['z_aux'] - memory['z_aux'])
        self.sum_y_diff += (self.memory['y_aux'] - memory['y_aux'])
        self.sum_dotq_diff += (self.memory['dot_x'] - memory['dot_x'])

    def auxiliary_update_function(self):
        e = self.model_config['h1']
        k = self.model_config['h2']
        bi = self.model_config['c']/6 # 对应公式里的 b_i
        
        # 公式 (7b) dot_yi
        dot_y = e * k * (self.sum_z_diff - bi + self.memory['x']) \
                + 2 * k * self.memory['dot_x'] \
                - e * k * self.sum_y_diff
                
        # 公式 (7c) dot_zi
        dot_z = -k * (e * self.sum_y_diff + self.sum_dotq_diff)
        
        return dot_y, dot_z

    def get_Matrix(self):
        """
        根据图片公式构建新的 M, C, G 矩阵。
        由于 qi 是单变量，这里返回 1x1 的 numpy array 以兼容后续的矩阵乘法 (@)
        """
        i = self.agent_id
        
        # 获取当前智能体的系统参数
        Tm = self.model_config['T_m'][i]
        Te = self.model_config['T_e'][i]
        Km = self.model_config['K_m'][i]
        
        qi = self.memory['x'] # 取出标量位置状态
        
        # M_i = (T_mi * T_ei) / K_mi
        Mi = np.array([[ (Tm * Te) / Km ]])
        
        # C_i = (T_mi + T_ei) / K_mi
        Ci = np.array([[ (Tm + Te) / Km ]])
        
        # G_i = 1 / K_mi * q_i
        Gi = np.array([[ (1.0 / Km) * qi ]])
        
        return Mi, Ci, Gi
    
    def status_update_function(self):
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']
        # 【维度对齐】确保读取的状态变量是纯一维数组，防止循环和矩阵乘法变维
        x = self.memory['x'].flatten()
        dot_x = self.memory['dot_x'].flatten()
        
        Mi, Ci, Gi = self.get_Matrix()
        
        # 【维度对齐】get_Matrix 返回的 Gi 可能是 (1,1) 矩阵。
        # 如果不 flatten，Gi + Ci@dot_x 会变成 (1,1) + (1,)，导致结果被意外广播成二维矩阵
        Gi_flat = Gi.flatten()
        
        # 完全保留你的控制律公式，仅将 Gi 替换为 Gi_flat
        ui = Gi_flat + Ci@dot_x - Mi@(h1*h2*dot_x+h1**2*self.partial_cost()+h1**2*self.memory['y_aux'])
        self.memory['ui'] = ui
        
        # 同理，计算 ddot_x 时使用 Gi_flat
        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x - Gi_flat)
        
        # 【维度对齐】强制返回一维数组，完美对接外部的 += updation 操作
        return dot_x.flatten(), ddot_x.flatten()
        

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

        current_x = action # 假设单维度
    
            
        return cost/4 

    def update(self):
        """
        单步更新逻辑
        """
        # 注意：此处省略了原代码中的分布式协议和搜索算法更新(estimation_update等)
        # 直接更新物理状态
        dot_x, ddot_x = self.status_update_function()
        
        self.memory_updation['x'] = dot_x
        self.memory_updation['dot_x'] = ddot_x
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['y_aux'], self.memory_updation['z_aux'] = self.auxiliary_update_function()

        print(f"Agent {self.agent_id} - Pre-Update State: x={self.memory['x']}, dot_x={self.memory['dot_x']}, z={self.memory['z']}")
        for k in self.memory_updation.keys():
            self.memory[k] = self.memory[k].astype(float)
            self.memory[k] += self.memory_updation[k] * self.time_delta
                
        # lower_bound = 0.0
        # upper_bound = 8.0
        
        # # 将 x 强制限制在 [0, 8] 之间
        # import numpy as np # 确保文件头部导入了 numpy
        # self.memory['x'] = np.clip(self.memory['x'], lower_bound, upper_bound)

        self.reset_memory_updation()

