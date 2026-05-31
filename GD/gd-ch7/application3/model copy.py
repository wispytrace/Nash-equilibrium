import numpy as np
import copy

class Model:
    
    DESC = "Air-Ground Cooperative Target Protection (Heterogeneous EL & 2nd Order Integrator)"
    
    def __init__(self, model_config) -> None:
        self.model_config = copy.deepcopy(model_config)
        self.memory = copy.deepcopy(self.model_config['memory'])
        self.time_delta = copy.deepcopy(self.model_config['time_delta'])
        self.time = 0.0
        self.agent_id = self.model_config['agent_id']
        
        # 赋予错落的初始位置
        init_pos = self.model_config['share']['init_x'][self.agent_id]
        self.memory['x'] = np.array(init_pos, dtype=float)
        # self.memory['y'] = np.array(init_pos, dtype=float)
        # self.memory['z'][self.agent_id] = np.array(init_pos, dtype=float)
        # self.memory['vr'] = np.array(init_pos, dtype=float)
        self.memory['y'], _ = self.get_target_derivatives()  # 初始化 y 为目标轨迹的初始导数，减少初始误差 
        
        self.reset_memroy_updation()

    def get_target_derivatives(self):
        """
        计算个体目标 p_i(t) 的一阶导数 (dot_p_i) 和二阶导数 (ddot_p_i)
        """
        t = self.time
        i = self.agent_id
        is_uav = i >= 4
        
        # --- 2. 设定智能体在目标周围的理想相位位置 (Formation Control) ---
        # ID 0-3: UGV, ID 4-7: UAV
        
        radius = 3.0/5 if not is_uav else 5.0/5      # UAV 半径更大
        height = 0.0 if not is_uav else 5.0/5      # UAV 高度更高
        
        # 计算围绕目标的相位: 4个智能体平均分布在 0, pi/2, pi, 3pi/2
        phase = (i % 4) * (2 * np.pi / 4) + (0.5 * self.time) # 加上时间项让队形转动
        
        # ==========================================
        # 1. 全局目标 p_g(t) 的导数计算
        # 原方程: p_g = [10*sin(0.2*t) + 1.0*t, 5*sin(0.4*t), 0]
        # ==========================================
        dot_p_g = np.array([
            -2.5 * np.sin(0.5 * t)/5,   # 10 * 0.2 * cos(0.2t) + 1.0
            2.5 * np.cos(0.5 * t)/5,         # 5 * 0.4 * cos(0.4t)
            0.0
        ])
        
        ddot_p_g = np.array([
            -1.25 * np.cos(0.5 * t)/5,        # -2.0 * 0.2 * sin(0.2t)
            -1.25 * np.sin(0.5 * t)/5,        # -2.0 * 0.4 * sin(0.4t)
            0.0
        ])
        
        # ==========================================
        # 2. 局部相对队形 p_rel(t) 的导数计算
        # ==========================================
        group_idx = i % 4
        phase = group_idx * (np.pi / 2) + (0.5 * t)
        
        # 高度 H_i(t) 的导数计算
        dot_h = 0.0
        ddot_h = 0.0

        # 水平相位的导数 (角速度为 0.5)
        dot_p_rel = np.array([
            -0.5 * radius * np.sin(phase),
            0.5 * radius * np.cos(phase),
            dot_h
        ])
        
        # 水平相位的二阶导数 (角加速度影响，0.5 * 0.5 = 0.25)
        ddot_p_rel = np.array([
            -0.25 * radius * np.cos(phase),
            -0.25 * radius * np.sin(phase),
            ddot_h
        ])
        
        # ==========================================
        # 3. 组合最终导数
        # 由于静态偏差 bias_vector 是常数，其导数为 0，因此可以直接相加
        # ==========================================
        dot_p_i = dot_p_g + dot_p_rel
        ddot_p_i = ddot_p_g + ddot_p_rel
        
        return dot_p_i, ddot_p_i

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
    
    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['vr'])
        
        alpha = self.model_config['alpha']
        self.memory_updation['y'] += alpha[2]*self.power(self.memory['y'] - memory['y'], self.model_config['mu']) + alpha[3]*self.power(self.memory['y'] - memory['y'], self.model_config['nu'])
        
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

    def y_update_function(self):
        # 完整的二阶导数：全局目标的加速度 + 相对旋转的加速度
        ddotpi = self.get_target_derivatives()[1]  # 直接调用方法获取 ddot_p_i
        y_update_value = -1 * self.memory_updation['y'] + ddotpi
        return y_update_value


    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']
        partial_cost = self.partial_cost()
        dotpi = self.get_target_derivatives()[0]
        dot_p_g = np.array([
            -2.5 * np.sin(0.5 * self.time)/5 ,   # 10 * 0.2 * cos(0.2t) + 1.0
            2.5 * np.cos(0.5 * self.time)/5,         # 5 * 0.4 * cos(0.4t)
            0.0
        ])
        update_value = -beta[0]*self.power(partial_cost, p) - beta[1]*self.power(partial_cost, q) + dotpi + 1/(8+1) * dot_p_g - 1/(8+1)*self.memory['y']
        return update_value

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        gama = self.model_config['gama']
        alpha = self.model_config['alpha']
        
        estimation_update = np.zeros(self.memory_updation['z'].shape)

        for i, value in enumerate(self.memory_updation['z']):
            estimation_update[i] = -1*(alpha[0]*self.power(value, p) + alpha[1]*self.power(value, q)) - gama*self.sign(value)
            estimation_update[i] = np.clip(estimation_update[i], -10e3, 10e3)
        
        return estimation_update

    def cost_function(self):
        N = self.memory['z'].shape[0]  # 总数 8
        i = self.agent_id
        zi = self.memory['z'][i]       # 当前智能体位置 [x, y, z]

        is_uav = i >= 4

        # --- 1. 设定保护目标中心 (Target trajectory) ---
        # 假设目标沿着螺旋线运动
        if is_uav:
            target_pos = np.array([5*np.cos(0.5*self.time)/5, 5*np.sin(0.5*self.time)/5, 2])
        else:
            target_pos = np.array([5*np.cos(0.5*self.time)/5, 5*np.sin(0.5*self.time)/5, 0])
        
        # --- 2. 设定智能体在目标周围的理想相位位置 (Formation Control) ---
        # ID 0-3: UGV, ID 4-7: UAV
        
        radius = 3.0/5 if not is_uav else 5.0/5      # UAV 半径更大
        height = 0.0 if not is_uav else 5.0/5      # UAV 高度更高
        
        # 计算围绕目标的相位: 4个智能体平均分布在 0, pi/2, pi, 3pi/2
        phase = (i % 4) * (2 * np.pi / 4) + (0.5 * self.time) # 加上时间项让队形转动
        
        # 理想位置 pi
        pi = target_pos-[0.5/5, 0.5/5, 0] + np.array([
            radius * np.cos(phase), 
            radius * np.sin(phase), 
            height
        ])
        
        # --- 3. 代价计算 ---
        # 个体追踪代价：试图保持在目标周围的预定相位位置
        individual_cost = np.power(np.linalg.norm(zi[:3] - pi), 2)
        
        # 群体协同代价：确保整体质心紧跟目标中心 (Coupling)
        # 这样整个队形不会因为个体波动而偏离目标太远
        status_sum = np.zeros(3)
        for j in range(N):
            if not is_uav:
                status_sum += np.array([self.memory['z'][j,0], self.memory['z'][j,1], 0])
            else:
                status_sum += self.memory['z'][j]
        
        status_sum /= N  # 计算质心位置

        coupling_cost = np.power(np.linalg.norm(status_sum - target_pos), 2)
        

        return 0.5*individual_cost + 0.5*coupling_cost 

    def partial_cost(self):
        delta = 1e-5
        partial_cost_value = np.zeros_like(self.memory['z'][self.agent_id])
        
        for i in range(len(self.memory['z'][self.agent_id])):
            cost = self.cost_function()
            self.memory['z'][self.agent_id][i] += delta
            cost_hat = self.cost_function()
            self.memory['z'][self.agent_id][i] -= delta
            partial_cost_value[i] = (cost_hat - cost) / delta
            
        return partial_cost_value

    def status_updation(self):
        """
        异构动力学控制器：完全嵌入真实的 6-DOF 四旋翼 Newton-Euler 方程
        """
        xi = self.memory['x']
        vi = self.memory['v']
        vr = self.memory['vr']
        
        # 1. 协同外环追踪误差
        ei = xi - vr
        dot_ei = vi - np.zeros(3) 
        
        mu = self.model_config['mu']
        nu = self.model_config['nu']
        k1, k2, k3, k4 = 5.0, 5.0, 3.0, 3.0
        
        # 期望的三维空间加速度 (基于 ENU 东-北-上 坐标系)
        u_des = -k1 * self.power(ei, mu) - k2 * self.power(ei, nu) - k3 * ei - k4 * dot_ei
        
        if self.agent_id < 4:
            # ===================================================================
            # 【真实四旋翼无人机动力学 - 严格采用您提供的公式】
            # ===================================================================
            g = 9.81
            uav_params = self.model_config['uav_physics'][str(self.agent_id)]
            m_i = uav_params['m']
            d_i = np.array(uav_params['d']) # 异构空气阻力系数 [dx, dy, dz]

            # 步骤 A: 坐标系映射 (ENU -> NED)
            # 您的公式是在 Z轴向下 的 NED 坐标系中推导的，为保持右手系，我们作如下翻转
            u_ned = np.array([u_des[0], -u_des[1], -u_des[2]])
            v_ned = np.array([vi[0], -vi[1], -vi[2]])

            # 步骤 B: 底层飞控姿态解算 (提取 T, phi, theta)
            ux, uy, uz = u_ned[0], u_ned[1], u_ned[2]
            
            # 计算所需总推力 T (推导自您公式的平方和)
            T_m = np.sqrt(ux**2 + uy**2 + (g - uz)**2)
            T = m_i * T_m
            
            # 计算期望的滚转角(phi)、俯仰角(theta)，偏航角(psi) 锁定为 0
            phi_d = np.arcsin(np.clip(uy / T_m, -1.0, 1.0))
            theta_d = np.arctan2(-ux, g - uz)
            psi_d = 0.0 

            # 步骤 C: 物理世界的真实响应 (严格代入您图片中的方程)
            # 在此基础上减去真实的异构空气阻尼 (d_i/m_i * v_ned)，防止高速飞行时发散
            ddot_x_ned = -(T/m_i) * (np.cos(psi_d)*np.sin(theta_d)*np.cos(phi_d) + np.sin(psi_d)*np.sin(phi_d)) - (d_i[0]/m_i) * v_ned[0]
            ddot_y_ned = -(T/m_i) * (np.sin(psi_d)*np.sin(theta_d)*np.cos(phi_d) - np.cos(psi_d)*np.sin(phi_d)) - (d_i[1]/m_i) * v_ned[1]
            ddot_z_ned = g - (T/m_i) * np.cos(phi_d)*np.cos(theta_d) - (d_i[2]/m_i) * v_ned[2]

            # 步骤 D: 将真实的加速度映射回 ENU 坐标系，供仿真引擎更新 3D 位置
            ddot_x = np.array([ddot_x_ned, -ddot_y_ned, -ddot_z_ned])
            self.memory['update_value'] = np.array([T, phi_d, theta_d]) # 记录飞控输出供分析
            
        else:
            # ===================================================================
            # 【真实 UGV 无人车动力学 (贴地二阶积分)】
            # ===================================================================
            ddot_x = u_des 
            ddot_x[2] = 0.0 # 强制 Z 轴加速度为 0
            vi[2] = 0.0     # 强制 Z 轴速度为 0
            self.memory['update_value'] = np.array(u_des) # UGV 无飞控输出
            
        return vi, ddot_x

    def update(self):
        self.memory_updation['vr'] = self.virtual_signal_update_function()
        self.memory_updation['z'] = self.estimation_update_function()
        self.memory_updation['y'] = self.y_update_function()
        
        dot_x, ddot_x = self.status_updation()
        
        self.memory_updation['x'] = dot_x
        self.memory_updation['v'] = ddot_x

        for k in self.memory.keys():
            if k in self.memory_updation.keys():
                self.memory[k] += self.memory_updation[k] * self.time_delta
        
        self.time += self.time_delta
        self.reset_memroy_updation()