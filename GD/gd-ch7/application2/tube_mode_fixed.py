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
        if 'init_value_y' in self.model_config.keys():
            self.memory['y'] = self.model_config['init_value_y'][self.agent_id]
        self.memory['dot_x'] = self.model_config['init_value_dotx'][self.agent_id]
        
        self.reset_memory_updation()
    
    def reset_memory_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            if isinstance(v, np.ndarray):
                self.memory_updation[k] = np.zeros_like(v)
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
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['y'])
        
        self.memory_updation['v'] += (self.memory['v'] - memory['v'])
        self.memory_updation['v'][adj_agent_id] += (self.memory['v'][adj_agent_id] - memory['partial_cost'])

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
        p = self.model_config['p']
        q = self.model_config['q']
        h1 = self.model_config['h1']
        h2 = self.model_config['h2']

        # 【维度对齐】确保读取的状态变量是纯一维数组，防止循环和矩阵乘法变维
        x = self.memory['x'].flatten()
        dot_x = self.memory['dot_x'].flatten()
        dot_y = self.virtual_signal_update_function().flatten()
        
        # 【维度对齐】确保 memory['y'] 也是一维
        track_error = x - self.memory['y'].flatten()
        sign_track_error = np.zeros(track_error.shape)
        for i in range(len(track_error)):
            sign_track_error[i] = self.approximate_sign(track_error[i])
        dot_track_error = np.multiply(dot_x - dot_y, sign_track_error)
        
        Mi, Ci, Gi = self.get_Matrix()
        
        # 【维度对齐】get_Matrix 返回的 Gi 可能是 (1,1) 矩阵。
        # 如果不 flatten，Gi + Ci@dot_x 会变成 (1,1) + (1,)，导致结果被意外广播成二维矩阵
        Gi_flat = Gi.flatten()
        
        oi = dot_x + h1*(self.power(track_error,p) + self.power(track_error, q) + track_error)
        self.memory['oi'] = oi
        
        # 完全保留你的控制律公式，仅将 Gi 替换为 Gi_flat
        ui = Gi_flat + Ci@dot_x - h2*Mi@(self.power(oi, p)+self.power(oi, q)) - h1*Mi@(p*np.multiply(self.power(track_error, p-1),dot_track_error)+ q*np.multiply(
            self.power(track_error, q-1), dot_track_error)+ dot_track_error)
        self.memory['ui'] = ui
        
        # 同理，计算 ddot_x 时使用 Gi_flat
        ddot_x = np.linalg.inv(Mi)@(ui - Ci@dot_x - Gi_flat)
        
        # 【维度对齐】强制返回一维数组，完美对接外部的 += updation 操作
        return dot_x.flatten(), ddot_x.flatten()
    
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

        # self.memory['before_values_sum'] = value_sum

        if self.model_config['c'] == 0:
            projected_values = np.zeros_like(values)
            projected_values = np.clip(values, self.model_config['l'], self.model_config['u'])
        else:
            projected_values, _ =  self.project_to_box_and_hyperplane(values.flatten(), self.model_config['c'], self.model_config['l'], self.model_config['u'])
        
        return projected_values

    def project_to_box_and_hyperplane(self, v, c, l, u):
        """
        将向量 v 投影到满足箱式约束 (l <= x <= u) 和等式约束 (sum(x) = c) 的可行域上。
        
        参数:
        v : array_like, 目标向量 (例如你要投影前的状态)
        c : float, 约束总和常量 (目标等式的值)
        l : float 或 array_like, 各维度的下界 (可以是一个统一的数字，也可以是数组)
        u : float 或 array_like, 各维度的上界 (可以是一个统一的数字，也可以是数组)
        
        返回:
        x_proj : ndarray, 投影后的合法向量
        lam_star : float, 求得的最优拉格朗日乘子
        """
        v = np.asarray(v, dtype=float)
        l = np.broadcast_to(l, v.shape)
        u = np.broadcast_to(u, v.shape)

        # 1. 可行性检查：如果连所有上限加起来都不够 c，或者下限加起来超过了 c，说明无解
        sum_l = np.sum(l)
        sum_u = np.sum(u)
        if sum_l > c + 1e-7 or sum_u < c - 1e-7:
            raise ValueError(f"约束不可行: 目标总和 {c} 不在可能的最大总和 {sum_u} 和最小总和 {sum_l} 之间。")

        # 2. 构造拉格朗日关于总和误差的单调递减函数 g(lambda)
        def g(lam):
            # 截断投影：x_i(lam) = clip(v_i - lam, l_i, u_i)
            x = np.clip(v - lam, l, u)
            return np.sum(x) - c

        # 3. 确定 lambda 的绝对搜索边界 (Bracket)
        # 因为 x_i = v_i - lambda, 且 l_i <= x_i <= u_i
        # 所以 v_i - u_i <= lambda <= v_i - l_i
        lam_min = np.min(v - u)
        lam_max = np.max(v - l)

        # 如果边界刚好使得函数值为0，直接返回（避免浮点数精度导致的微小越界）
        g_min = g(lam_min)
        g_max = g(lam_max)
        if np.abs(g_min) < 1e-9:
            lam_star = lam_min
        elif np.abs(g_max) < 1e-9:
            lam_star = lam_max
        else:
            # 4. 使用 Brent 法求解 (比 fsolve 更快、更稳，100% 收敛)
            res = root_scalar(g, bracket=[lam_min, lam_max], method='brentq')
            if not res.converged:
                raise RuntimeError("求根算法未能收敛，请检查输入数据。")
            lam_star = res.root

        # 5. 根据找到的最优 lambda* 计算最终投影
        x_proj = np.clip(v - lam_star, l, u)

        return x_proj, lam_star


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
        
        price =  po - status_sum*a 

        cost = (action - xi)**2 - price*action


        return cost

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
        """
        单步更新逻辑
        """
        # 注意：此处省略了原代码中的分布式协议和搜索算法更新(estimation_update等)
        # 直接更新物理状态
        dot_x, ddot_x = self.status_update_function()
        
        self.memory_updation['x'] = dot_x
        self.memory_updation['dot_x'] = ddot_x
        self.memory_updation['y'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['v'] = self.partial_value_estimation_update_function()

        for k in self.memory_updation.keys():
            # if k == 'x' and self.memory[k] >= self.model_config['u']:
            #     continue
            self.memory[k] = self.memory[k].astype(float)
            self.memory[k] += self.memory_updation[k] * self.time_delta
        
        # print(sum(self.memory['z']), self.memory['y'])
        
        # print(f"Agent {self.agent_id} - Updated State: x={self.memory['x']}, dot_x={self.memory['dot_x']}, y={self.memory['y']}, z={self.memory['z']}, v={self.memory['v']}")
        
        self.reset_memory_updation()

