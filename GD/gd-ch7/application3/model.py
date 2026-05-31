import numpy as np
import copy

import numpy as np
import copy


def get_pg(time):

    pos = np.array([0.5*time, 0, 0])

    return pos

def pg_dot(time):
    """虚拟中心的一阶导数 (速度)"""
    return np.array([0.5, 0.0, 0.0])

def pg_ddot(time):
    """虚拟中心的二阶导数 (加速度)"""
    return np.array([0.0, 0.0, 0.0])

def get_pi(time, i ):
    radius = 2 if i <4 else 4
    if i == 0:
        pi = [0.5*time+radius, radius, 0]
    elif i == 1:
        pi = [0.5*time - 1, 1, 0]
    elif i == 2:
        pi = [0.5*time-1, -1-radius*(1+np.sin(time)), 0]
    elif i == 3:
        pi = [0.5*time+radius, -radius-radius*(1+np.sin(time)), 0]
    elif i == 4:
        pi = [0.5*time+radius, radius, 0]
    elif i == 5:
        pi = [0.5*time - 3, 3, 0]
    elif i == 6:
        pi = [0.5*time-3, -3-radius*(1+np.sin(time)), 0]
    elif i == 7:
        pi = [0.5*time+radius, -radius-radius*(1+np.sin(time)), 0]
    
    # 统一转换成 numpy array 格式返回
    return np.array(pi)

def pi_dot(time, i):
    """时变目标 pi 的一阶导数 (速度)"""
    radius = 2 if i < 4 else 4
    
    if i in [0, 1, 4, 5]:
        # 直线轨迹组：Y 轴均为常数，求导后 Y 轴速度为 0
        pi_d = [0.5, 0.0, 0.0]
    elif i in [2, 3, 6, 7]:
        # 波动轨迹组：Y 轴对 time 求导
        # d/dt (-C - radius*(1 + sin(t))) = -radius * cos(t)
        pi_d = [0.5, -radius * np.cos(time), 0.0]
        
    return np.array(pi_d)

def pi_ddot(time, i):
    """时变目标 pi 的二阶导数 (加速度)"""
    radius = 2 if i < 4 else 4
    
    if i in [0, 1, 4, 5]:
        # 直线轨迹组：速度已经是全常数，再次求导全为 0
        pi_dd = [0.0, 0.0, 0.0]
    elif i in [2, 3, 6, 7]:
        # 波动轨迹组：Y 轴速度对 time 再次求导
        # d/dt (-radius * cos(t)) = radius * sin(t)
        pi_dd = [0.0, radius * np.sin(time), 0.0]
        
    return np.array(pi_dd)

def get_agent_ne(time, i, N=8):
    """
    计算第 i 个智能体在指定时间步的理论纳什均衡点 x_i^*
    
    参数:
        time: 当前时间步 (float)
        i: 当前智能体索引 (0 到 N-1)
        N: 智能体总数 (默认 8 个，根据你之前的 pi 函数设置)
    返回:
        np.array: 形如 [x, y, z] 的 3D 纳什均衡点向量
    """
    # 1. 获取当前智能体自己的时变目标 p_i (调用你之前的 pi 函数)
    p_i = get_pi(time, i)
    
    # 2. 获取虚拟中心轨迹 p_g (调用你之前的 pg 函数)
    p_g = get_pg(time)
    
    # 3. 计算所有智能体时变目标的加和: \sum_{j=1}^N p_j
    # 注意: 这里的 j 从 0 到 N-1 遍历
    sum_p_j = np.zeros(3)
    for j in range(N):
        sum_p_j += get_pi(time, j)
        
    # 4. 代入单体拆解公式计算 x_i^*
    term1 = p_i
    term2 = (1.0 / (N + 1)) * p_g
    term3 = (1.0 / (N * (N + 1))) * sum_p_j
    
    x_i_star = term1 + term2 - term3
    
    return x_i_star

def get_agent_dot_ne(time, i, N=8):
    """
    计算第 i 个智能体在指定时间步的理论纳什均衡点的一阶导数 \dot{x}_i^*
    
    参数:
        time: 当前时间步 (float)
        i: 当前智能体索引 (0 到 N-1)
        N: 智能体总数 (默认 8 个)
    返回:
        np.array: 形如 [vx, vy, vz] 的 3D 速度向量
    """
    # 1. 获取当前智能体目标的一阶导数 (调用 pi_dot)
    dot_p_i = pi_dot(time, i)
    
    # 2. 获取虚拟中心轨迹的一阶导数 (调用 pg_dot)
    dot_p_g = pg_dot(time)
    
    # 3. 计算所有智能体目标一阶导数的加和: \sum_{j=1}^N \dot{p}_j
    sum_dot_p_j = np.zeros(3)
    for j in range(N):
        sum_dot_p_j += pi_dot(time, j)
        
    # 4. 代入公式计算
    term1 = dot_p_i
    term2 = (1.0 / (N + 1)) * dot_p_g
    term3 = (1.0 / (N * (N + 1))) * sum_dot_p_j
    
    dot_x_i_star = term1 + term2 - term3
    
    return dot_x_i_star

def get_agent_ne_ddot(time, i, N=8):
    """
    计算第 i 个智能体在指定时间步的理论纳什均衡点的二阶导数 (加速度) ddot_x_i^*
    
    参数:
        time: 当前时间步 (float)
        i: 当前智能体索引 (0 到 N-1)
        N: 智能体总数 (默认 8 个)
    返回:
        np.array: 形如 [ddot_x, ddot_y, ddot_z] 的 3D 加速度向量
    """
    # 1. 获取当前智能体自己的时变目标加速度 ddot_p_i
    ddot_p_i = pi_ddot(time, i)
    
    # 2. 获取虚拟中心轨迹的加速度 ddot_p_g
    ddot_p_g = pg_ddot(time)
    
    # 3. 计算所有智能体时变目标加速度的加和: \sum_{j=1}^N ddot_p_j
    sum_ddot_p_j = np.zeros(3)
    for j in range(N):
        sum_ddot_p_j += pi_ddot(time, j)
        
    # 4. 代入单体拆解公式计算纳什均衡点加速度 ddot_x_i^*
    term1 = ddot_p_i
    term2 = (1.0 / (N + 1)) * ddot_p_g
    term3 = (1.0 / (N * (N + 1))) * sum_ddot_p_j
    
    ddot_x_i_star = term1 + term2 - term3
    
    return ddot_x_i_star

class Model:
    
    DESC = "High-order systems"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(self.model_config['time_delta'])
        self.time = 0
        self.agent_id = self.model_config['agent_id']
        self.memory['vr'] =  np.array(self.model_config['share']['init_x'][self.agent_id]) 
        self.memory['x'] = np.array(self.model_config['share']['init_x'][self.agent_id]) 
        self.memory['y'] = pi_dot(0, self.agent_id)
        self.memory['v'] = np.zeros(3)
        self.reset_memroy_updation()

    def set_init_value(self, key, init_value):
        self.memory[key] = copy.deepcopy(init_value)
        # 【修改点 3】：当初始化虚拟信号 vr 时，顺便把观测器里关于自己的状态初始化，大幅减小初始误差
        # if key == 'vr':
        #     self.memory['z'][self.agent_id] = copy.deepcopy(init_value)

    def reset_memroy_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros(np.array(v).shape)
        self.memory['partial_cost'] = self.partial_cost()
        self.memory['NE'] = get_agent_ne(self.time, self.agent_id, 8)
        self.memory['pg'] = get_pg(self.time)
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['vr'])
        
        alpha = self.model_config['alpha']
        self.memory_updation['y'] += alpha[2]*self.power(self.memory['y'] - memory['y'], self.model_config['mu']) + alpha[3]*self.power(self.memory['y'] - memory['y'], self.model_config['nu'])
        
    def power(self, value, a):
        # 处理标量
        if len(np.shape(value)) == 0:
            if np.fabs(value) < 1e-10:
                return 0.0
            return np.power(np.fabs(value), a) * np.sign(value)

        # 处理数组
        powered_value = np.zeros_like(value)
        for i in range(len(value)):
            if np.fabs(value[i]) < 1e-10:
                powered_value[i] = 0.0
            else:
                powered_value[i] = np.power(np.fabs(value[i]), a) * np.sign(value[i])
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
        extra = 1e-5
        value = value/(np.fabs(value)+extra)
        return value

    def y_update_function(self):
        # 完整的二阶导数：全局目标的加速度 + 相对旋转的加速度
        ddotpi = pi_ddot(self.time, self.agent_id)
        y_update_value = -1 * self.memory_updation['y'] + ddotpi
        return y_update_value


    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']

        partial_cost = self.partial_cost()
        N = self.memory['z'].shape[0]
        # 目标轨迹的导数 (常数偏移量 +4 求导后消失)

        dot_pg = pg_dot(self.time)
        dot_pi = pi_dot(self.time, self.agent_id)
        # 
        # if self.time > 1:
        #     print(partial_cost, self.memory['y'], dot_pi, dot_pg)
        update_value = -beta[0]*self.power(partial_cost, p) - beta[1]*self.power(partial_cost, q)+ dot_pi + 1/(N+1)*dot_pg - 1/(N+1)*self.memory['y']
        # dot_x_i_star = get_agent_dot_ne(self.time, self.agent_id)
        # update_value = -beta[0]*self.power(partial_cost, p) - beta[1]*self.power(partial_cost, q)+ dot_x_i_star
        self.memory['update_value'] = update_value
        
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        gama = self.model_config['gama']
        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(value, q)) - gama*self.sign(value)
            # estimation_update[i] = np.clip(estimation_update[i], -10e3, 10e3)
        
        return estimation_update


    def partial_cost(self):
        """
        使用解析解 (Analytical Gradient) 直接计算梯度
        彻底消除数值差分带来的浮点数噪声，防止在分数次幂下产生抖振
        """
        N = self.memory['z'].shape[0]
        i = self.agent_id
        
        # 1. 转化为 numpy array
        # 修改后的全局目标 pg (向正方向偏移 4)
        pg = get_pg(self.time)
        pi = get_pi(self.time, self.agent_id)
        
        # 2. 获取当前智能体状态与全局均值
        zi = self.memory['z'][i]
        status_sum = np.sum(self.memory['z'], axis=0)
        status_mean = status_sum / N 
        
        # 3. 严格按照代价函数求导的精确解析解公式计算梯度
        # f_i(z_i) = ||z_i - p_i||^2 + || (1/N)*sum(z_j) - p_g ||^2
        # 对 z_i 求导 -> 2*(z_i - p_i) + 2/N * ((1/N)*sum(z_j) - p_g)
        grad = 2 * (zi - pi) + (2.0 / N) * (status_mean - pg)
        grad[2] = 0
        # if self.agent_id == 0:
        #     print(grad, status_mean, zi, pi ,pg )
        return grad
    
    def status_updation(self):
        """
        异构动力学控制器 (带加速度前馈，消除时变轨迹滞后)：
        - UAV (0-3): 串级控制 [外环: FxT固定时间滑模算加速度] -> [内环: 真实欠驱动姿态解算]
        - UGV (4-7): 二阶固定时间积分滑模控制 (Integral Sliding Mode)
        """
        # ==========================================
        # 0. 防 NaN 绝对安全幂函数 (极其重要)
        # 替代容易出隐患的 self.power，确保绝对不会产生复数或 NaN
        # ==========================================
        def safe_power(x, a):
            return np.sign(x) * (np.abs(x) ** a)

        # ==========================================
        # 1. 状态读取与维度对齐
        # ==========================================
        xi = np.array(self.memory['x'], dtype=float).flatten()
        vi = np.array(self.memory['v'], dtype=float).flatten()
        vr = np.array(self.memory['vr'], dtype=float).flatten()
        dot_vr = self.memory_updation['vr']

        if 'ddot_vr' in self.memory:
            ddot_vr = np.array(self.memory['ddot_vr'], dtype=float).flatten()
        else:
            if 'last_dot_vr' not in self.memory:
                self.memory['last_dot_vr'] = dot_vr.copy()
            # 数值求导近似目标的加速度
            ddot_vr = (dot_vr - self.memory['last_dot_vr']) / self.time_delta
            self.memory['last_dot_vr'] = dot_vr.copy()

        # 🚨 修正 ID 判断逻辑：UAV 通常是 0, 1, 2, 3 (即 < 4)
        if self.agent_id >= 4:
            # ===================================================================
            # 【飞行器 (UAV): 串级控制架构】
            # ===================================================================
            p = self.model_config.get('p', 1.2)
            q = self.model_config.get('q', 0.8)
            h1 = self.model_config.get('h1', 1.0)
            h2 = self.model_config.get('h2', 1.0)
            
            # 1. 计算核心误差
            track_error = xi - vr
            dot_e = vi - dot_vr  # 真实的速度误差
            
            # 🌟 使用 safe_power 构建滑模面 oi
            oi = dot_e + h1 * (safe_power(track_error, p) + safe_power(track_error, q) + track_error)
            self.memory['oi'] = oi
            
            # 🌟 引入 eps 防止 q-1 (负指数) 在误差为 0 时导致 inf * 0 = NaN 崩溃
            eps = 1e-3
            
            
            term_p_deriv = p * ((np.abs(track_error) + eps) ** (p - 1)) * dot_e
            term_q_deriv = q * ((np.abs(track_error) + eps) ** (q - 1)) * dot_e
            
            # 组合非线性求导项
            term3_inner = term_p_deriv + term_q_deriv + dot_e
            
            # 计算期望加速度 u_des (完美前馈 + FxT反馈)
            u_des = ddot_vr - h2 * (safe_power(oi, p) + safe_power(oi, q)) - h1 * term3_inner
            
            # ---------------------------------------------------------
            # 内环真实四旋翼物理姿态解算与响应
            # ---------------------------------------------------------
            g = 9.81
            try:
                # 按照正常逻辑，如果 agent_id < 4，直接用 agent_id 查字典
                uav_params = self.model_config['uav_physics'][str(self.agent_id-4)]
            except KeyError:
                uav_params = self.model_config['uav_physics'][self.agent_id-4]
                
            m_i = uav_params['m']
            d_i = np.array(uav_params['d']) 

            u_ned = np.array([u_des[0], -u_des[1], -u_des[2]])
            v_ned = np.array([vi[0], -vi[1], -vi[2]])
            ux, uy, uz = u_ned[0], u_ned[1], u_ned[2]

            # 加上 1e-8 防止除零 NaN
            T_m = np.sqrt(ux**2 + uy**2 + (g - uz)**2)
            T = m_i * T_m
            
            phi_d = np.arcsin(np.clip(uy / (T_m + 1e-8), -1.0, 1.0))
            theta_d = np.arctan2(-ux, g - uz)
            psi_d = 0.0 

            ddot_x_ned = -(T/m_i) * (np.cos(psi_d)*np.sin(theta_d)*np.cos(phi_d) + np.sin(psi_d)*np.sin(phi_d)) - (d_i[0]/m_i) * v_ned[0]
            ddot_y_ned = -(T/m_i) * (np.sin(psi_d)*np.sin(theta_d)*np.cos(phi_d) - np.cos(psi_d)*np.sin(phi_d)) - (d_i[1]/m_i) * v_ned[1]
            ddot_z_ned = g - (T/m_i) * np.cos(phi_d)*np.cos(theta_d) - (d_i[2]/m_i) * v_ned[2]

            ddot_x = np.array([ddot_x_ned, -ddot_y_ned, -ddot_z_ned])
            self.memory['update_value'] = np.array([T, phi_d, theta_d]) 
            # print(ddot_x)

        else:
            # ===================================================================
            # 【无人车 (UGV): 离散安全版 二阶固定时间积分滑模控制器 (ISMC)】
            # ===================================================================
            p = self.model_config.get('p', 1.35)
            q = self.model_config.get('q', 0.65)
            eta = self.model_config.get('eta', 4.0)
            zeta = self.model_config.get('zeta', 5.0)
            
            gama = self.model_config.get('gama_hl', [1.3, 0.7])  
            k_i = self.model_config.get('ki', [5.0, 6.0])
            k_i_tilde = self.model_config.get('k_i_tilde', [7.0, 8.0])
            
            gama_1, gama_1_tilde = gama[0], gama[1]
            gama_0 = gama[0] / (2.0 - gama[0])
            gama_0_tilde = gama[1] / (2.0 - gama[1])

            e0 = xi - vr          
            e1 = vi - dot_vr      

            e0[2] = 0.0  # 强制消除 Z 轴位置误差
            e1[2] = 0.0  # 强制消除 Z 轴速度误差
            
            # 使用 safe_power，不管里面指数是大于1还是小于1，都不会爆 NaN
            term_e0_p = k_i[0] * safe_power(e0, gama_0)
            term_e0_q = k_i_tilde[0] * safe_power(e0, gama_0_tilde)
            
            term_e1_p = k_i[1] * safe_power(e1, gama_1)
            term_e1_q = k_i_tilde[1] * safe_power(e1, gama_1_tilde)
            
            error_sum = term_e0_p + term_e0_q + term_e1_p + term_e1_q

            # 消除积分饱和
            if 'ei_sum' not in self.memory:
                self.memory['ei_sum'] = -e1.copy() 
            
            # leak_rate = 0.5 
            
            # 每一帧不仅累加新误差，还让历史积累量按比例衰减
            # self.memory['ei_sum'] = self.memory['ei_sum'] * (1.0 - leak_rate * self.time_delta) + error_sum * self.time_delta
            self.memory['ei_sum'] += error_sum * self.time_delta

            si = e1 + self.memory['ei_sum']
            # if self.agent_id == 0:
            #     print("si", si)

            # 计算最终的 ui (加入前馈 ddot_vr 和 tanh 抗扰项)
            ui = ddot_vr - eta * safe_power(si, p) - zeta * safe_power(si, q) - error_sum
            
            self.memory['ui_hl'] = ui

            ddot_x = ui.copy()
            ddot_x[2] = 0.0 # UGV 强制贴地
            
            self.memory['update_value'] = ddot_x 
            
        return vi.flatten(), ddot_x.flatten()

    def status_updation_asymptotic(self):
        """
        异构动力学控制器 (标准线性/渐近收敛 Baseline 版本)：
        - UAV (0-3): 传统串级渐近滑模控制 (ASMC) -> [内环: 真实欠驱动姿态解算]
        - UGV (4-7): 标准二阶渐近积分滑模控制 (AISM)
        """
        # ==========================================
        # 1. 状态读取与维度对齐
        # ==========================================
        xi = np.array(self.memory['x'], dtype=float).flatten()
        vi = np.array(self.memory['v'], dtype=float).flatten()
        vr = np.array(self.memory['vr'], dtype=float).flatten()
        
        # 解析解高精度速度与加速度前馈
        dot_vr = get_agent_dot_ne(self.time, self.agent_id)
        ddot_vr = get_agent_ne_ddot(self.time, self.agent_id)

        if self.agent_id < 4:
            # ===================================================================
            # 【飞行器 (UAV): 传统渐近滑模控制 (无非线性分数幂)】
            # ===================================================================
            # 线性增益参数 (对应传统 PD 型滑模，可自由微调)
            c1 = 2.0  # 滑模面误差权重
            c2 = 4.0  # 趋近律增益
            
            track_error = xi - vr
            dot_e = vi - dot_vr  
            
            # 🌟 渐近滑模面：退化为纯线性组合 s = \dot{e} + c1 * e
            oi = dot_e + c1 * track_error
            self.memory['oi'] = oi
            
            # 🌟 线性求导项：对应的导数也是纯线性的
            term3_inner = c1 * dot_e
            
            # 🌟 渐近控制律：等效前馈 - 线性趋近项 - 偏导补偿 - 传统鲁棒项
            # 去除了所有 safe_power 运算，天然杜绝了 NaN 风险
            u_des = ddot_vr - c2 * oi - term3_inner - 2.0 * np.tanh(oi / 0.1)
            
            # 输入限幅保护
            u_des = np.clip(u_des, -15.0, 15.0)
            
            # ---------------------------------------------------------
            # 内环真实四旋翼物理姿态解算与响应 (保持刚体动力学一致)
            # ---------------------------------------------------------
            g = 9.81
            try:
                uav_params = self.model_config['uav_physics'][str(self.agent_id)]
            except KeyError:
                uav_params = self.model_config['uav_physics'][self.agent_id]
                
            m_i = uav_params['m']
            d_i = np.array(uav_params['d']) 

            u_ned = np.array([u_des[0], -u_des[1], -u_des[2]])
            v_ned = np.array([vi[0], -vi[1], -vi[2]])
            ux, uy, uz = u_ned[0], u_ned[1], u_ned[2]

            T_m = np.sqrt(ux**2 + uy**2 + (g - uz)**2)
            T = m_i * T_m
            
            phi_d = np.arcsin(np.clip(uy / (T_m + 1e-8), -1.0, 1.0))
            theta_d = np.arctan2(-ux, g - uz)
            psi_d = 0.0 

            ddot_x_ned = -(T/m_i) * (np.cos(psi_d)*np.sin(theta_d)*np.cos(phi_d) + np.sin(psi_d)*np.sin(phi_d)) - (d_i[0]/m_i) * v_ned[0]
            ddot_y_ned = -(T/m_i) * (np.sin(psi_d)*np.sin(theta_d)*np.cos(phi_d) - np.cos(psi_d)*np.sin(phi_d)) - (d_i[1]/m_i) * v_ned[1]
            ddot_z_ned = g - (T/m_i) * np.cos(phi_d)*np.cos(theta_d) - (d_i[2]/m_i) * v_ned[2]

            ddot_x = np.array([ddot_x_ned, -ddot_y_ned, -ddot_z_ned])
            self.memory['update_value'] = np.array([T, phi_d, theta_d]) 

        else:
            # ===================================================================
            # 【无人车 (UGV): 标准标准渐近积分滑模控制器 (AISM)】
            # ===================================================================
            # 传统线性滑模参数
            c_u = 3.0
            eta_u = 4.0
            
            e0 = xi - vr          
            e1 = vi - dot_vr      

            e0[2] = 0.0  # 强制消除 Z 轴位置误差
            e1[2] = 0.0  # 强制消除 Z 轴速度误差
            
            # 🌟 线性综合误差项：无分数阶
            error_sum = c_u * e1 + (c_u ** 2 / 4.0) * e0

            # 积分器累加
            if 'ei_sum' not in self.memory:
                self.memory['ei_sum'] = -e1.copy() 
                
            self.memory['ei_sum'] += error_sum * self.time_delta

            si = e1 + self.memory['ei_sum']

            # 🌟 线性控制律 ui：完全移除固定时间非线性幂次
            ui = ddot_vr - eta_u * si - error_sum - 2.0 * np.tanh(si / 0.1)
            
            self.memory['ui_hl'] = ui

            ddot_x = ui.copy()
            ddot_x[2] = 0.0 # UGV 强制贴地
            
            self.memory['update_value'] = ddot_x 
            
        return vi.flatten(), ddot_x.flatten()
    
    def update(self):
        
        self.memory_updation['vr'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['y'] = self.y_update_function()
        self.memory_updation['x'], self.memory_updation['v']  = self.status_updation_asymptotic()

        for k in self.memory.keys():
            if k in self.memory_updation.keys():
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        
        self.reset_memroy_updation()
