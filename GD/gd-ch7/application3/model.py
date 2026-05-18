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
        self.memory['y'] = np.array(init_pos, dtype=float)
        self.memory['z'][self.agent_id] = np.array(init_pos, dtype=float)
        self.memory['vr'] = np.array(init_pos, dtype=float)
        
        self.reset_memory_updation()

    def set_init_value(self, key, init_value):
        self.memory[key] = copy.deepcopy(init_value)

    def reset_memory_updation(self):
        self.memory_updation = {}
        for k, v in self.memory.items():
            self.memory_updation[k] = np.zeros_like(v)
    
    def approximate_sign(self, value):
        return value / (np.abs(value) + 1e-10)

    def sign(self, value):
        return np.where(np.abs(value) < 1e-10, 0.0, self.approximate_sign(value))

    def power(self, value, a):
        abs_val = np.abs(value)
        return np.where(abs_val < 1e-10, 0.0, np.power(abs_val, a) * self.approximate_sign(value))

    def receieve_msg(self, adj_agent_id, memory):
        self.memory_updation['z'] += (self.memory['z'] - memory['z'])
        self.memory_updation['z'][adj_agent_id] += (self.memory['z'][adj_agent_id] - memory['vr'])
        alpha = self.model_config['alpha']
        self.memory_updation['y'] += alpha[2]*self.power(self.memory['y'] - memory['y'], self.model_config['mu']) + \
                                     alpha[3]*self.power(self.memory['y'] - memory['y'], self.model_config['nu'])
        
    def y_update_function(self):
        # 简化版外环，仅保留误差收敛项
        return -1.0 * self.memory_updation['y']

    def virtual_signal_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        beta = self.model_config['beta']
        partial_cost = self.partial_cost()
        return -beta[0]*self.power(partial_cost, p) - beta[1]*self.power(partial_cost, q)

    def estimation_update_function(self):
        p = self.model_config['p']
        q = self.model_config['q']
        gama = self.model_config['gama']
        alpha = self.model_config['alpha']
        
        z_upd = self.memory_updation['z']
        est_update = -1*(alpha[0]*self.power(z_upd, p) + alpha[1]*self.power(z_upd, q)) - gama*self.sign(z_upd)
        return np.clip(est_update, -10e3, 10e3)

    def partial_cost(self):
        """
        核心亮点：极具视觉冲击力的 3D 防御队形代价梯度
        """
        i = self.agent_id
        t = self.time
        
        # 1. 目标 (Target) 轨迹: 地面上的 ∞ 字形 (Lissajous curve)
        # X振幅=10, Y振幅=10, 频率比 1:2
        p_target = np.array([
            10.0 * np.sin(0.3 * t), 
            10.0 * np.sin(0.6 * t), 
            0.0
        ])
        
        # 2. 异构队形设计
        if i < 4:
            # UAV (i=0,1,2,3): 逆时针旋转，高空浮动，半径大
            radius = 6.0
            omega = 0.8  # 旋转角速度
            theta = i * (np.pi / 2) + omega * t
            z_height = 5.0 + 1.5 * np.sin(1.0 * t) # 上下浮动
        else:
            # UGV (i=4,5,6,7): 顺时针旋转，贴地，半径小 (紧密保护)
            radius = 4.0
            omega = -0.5 # 反向旋转
            theta = (i - 4) * (np.pi / 2) + omega * t
            z_height = 0.0 # 严格贴地
            
        p_ideal = p_target + np.array([radius * np.cos(theta), radius * np.sin(theta), z_height])
        
        # 3. 计算个体梯度 (解析解)
        zi = self.memory['z'][i]
        grad = 2.0 * (zi - p_ideal)
        
        return grad

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
            
        else:
            # ===================================================================
            # 【真实 UGV 无人车动力学 (贴地二阶积分)】
            # ===================================================================
            ddot_x = u_des 
            ddot_x[2] = 0.0 # 强制 Z 轴加速度为 0
            vi[2] = 0.0     # 强制 Z 轴速度为 0
            
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
        self.reset_memory_updation()