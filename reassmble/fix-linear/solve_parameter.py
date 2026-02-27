import numpy as np
from scipy.optimize import minimize

def calculate_sigmas_with_intermediates(rho, params):
    """
    根据基础参数计算中间变量 c1~c4，并计算 sigma1~sigma4。
    
    参数:
    rho: 包含 rho_1(t) 到 rho_5(t) 的列表或数组
    params: 包含 N, n, mu, nu, theta, phi, h_m 的字典
    
    返回:
    sigmas: 包含 [sigma1, sigma2, sigma3, sigma4] 的 numpy 数组
    c_vars: 包含 [c1, c2, c3, c4] 的字典，方便调试和验证
    """
    # 1. 解构基础参数
    N = params['N']
    n = params['n']
    mu = params['mu']
    nu = params['nu']
    theta = params['theta']
    phi = params['phi']  # 对应公式中的 \varphi
    h_m = params['h_m']
    
    sqrt_N = np.sqrt(N)
    
    # 2. 计算中间变量 c1, c2, c3, c4
    # c1 = N^{1 - \mu/2} * n^{1 - \mu} * 2^{1 - \mu} * \theta^\mu
    c1 = (N**(1 - mu/2)) * (n**(1 - mu)) * (2**(1 - mu)) * (theta**mu)
    
    # c2 = N^{1/2} * (2^{\nu - 2} + 2) * \nu * \theta^\nu
    c2 = sqrt_N * (2**(nu - 2) + 2) * nu * (theta**nu)
    
    # max_val = \max\{1, n^{1 - \nu/2}\}
    # 使用 np.maximum 保证即使传入数组也能安全计算
    max_val = np.maximum(1, n**(1 - nu/2)) 
    
    # c3 = N^{1/2} * (2^{\nu - 2} + 2) * \theta * \max\{1, n^{1 - \nu/2}\}
    c3 = sqrt_N * (2**(nu - 2) + 2) * theta * max_val
    
    # c4 = N^{1/2} * n^{\nu - 1} * (2^{\nu - 2} + 2) * \theta * (\nu - 1) * \max\{1, n^{1 - \nu/2}\}
    c4 = sqrt_N * (n**(nu - 1)) * (2**(nu - 2) + 2) * theta * (nu - 1) * max_val
    
    # 保存中间变量，方便返回查看
    c_vars = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}

    # 3. 提取当前的 rho_i(t) 状态值
    r1, r2, r3, r4, r5 = rho[0], rho[1], rho[2], rho[3], rho[4]

    # --- 公式 (18): \sigma_1(\rho(t)) ---
    sigma1 = (c1 + 
              (sqrt_N * (r2**(-mu))) / (mu + 1) + 
              (h_m * c1 * (r4**(-1/mu)) * mu) / (sqrt_N * (mu + 1)))

    # --- 公式 (19): \sigma_2(\rho(t)) ---
    term2_base = c2 + c3 * (r1**(1 - nu))
    term2_mid  = ((c4 * r1 + sqrt_N) * (r3**(-nu))) / (nu + 1)
    term2_last = (h_m * term2_base * (r5**(-1/nu)) * nu) / (sqrt_N * (nu + 1))
    sigma2 = term2_base + term2_mid + term2_last

    # --- 公式 (20): \sigma_3(\rho(t)) ---
    sigma3 = (phi - 
              (sqrt_N * r2 * (n**((1 - mu) / 2)) * mu) / (mu + 1) - 
              (h_m * c1 * r4) / (sqrt_N * (mu + 1)))

    # --- 公式 (21): \sigma_4(\rho(t)) ---
    term4_mid   = ((c4 * r1 + sqrt_N) * r3 * nu) / (nu + 1)
    term4_last1 = (h_m * term2_base * r5) / (sqrt_N * (nu + 1))
    term4_last2 = (h_m * c4 * r1) / sqrt_N
    sigma4 = phi - term4_mid - term4_last1 - term4_last2

    sigmas = np.array([sigma1, sigma2, sigma3, sigma4])
    
    return sigmas, c_vars

def objective_function(rho, params, target_sigma3, target_sigma4):
    """
    目标函数（误差函数/损失函数）。
    我们要让这个函数的返回值尽可能接近于 0。
    """
    # 算出现有 rho 对应的 sigmas
    sigmas,_ = calculate_sigmas_with_intermediates(rho, params)
    
    # 提取我们关心的 sigma3 和 sigma4
    sigma3_calc = sigmas[2]
    sigma4_calc = sigmas[3]
    
    # 计算均方误差 (MSE)
    # 也就是： (计算值 - 目标值)^2 的总和
    error = (sigma3_calc - target_sigma3)**2 + (sigma4_calc - target_sigma4)**2
    
    return error


def calculate_delta(params):
    target_val = params['kappa'] * params['phi']
    TARGET_SIGMA_3 = target_val
    TARGET_SIGMA_4 = target_val
    
    # 3. 提供一个初始的猜测值 (rho1, rho2, rho3, rho4, rho5)
    # 算法会从这个点开始搜索
    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    # 4. 设置边界条件！非常重要！
    # 因为公式里有 rho^(-mu) 这种负指数，rho 必须大于 0，否则会报错。
    # 这里我们设定每个 rho 的范围是 [0.001, 无穷大]
    bnds = ((0.001, None), (0.001, None), (0.001, None), (0.001, None), (0.001, None))
    
    print("开始搜索满足条件的 rho 值...")
    
    # 5. 调用优化器进行搜索求解
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(params, TARGET_SIGMA_3, TARGET_SIGMA_4),
        method='L-BFGS-B',  # 这种算法支持边界条件 (Bounds)
        bounds=bnds
    )
    
    # 6. 输出结果
    if result.success:
        print("\n✅ 求解成功！找到了一组满足条件的 rho:")
        optimal_rho = result.x
        print(f"rho_1 = {optimal_rho[0]:.4f}")
        print(f"rho_2 = {optimal_rho[1]:.4f}")
        print(f"rho_3 = {optimal_rho[2]:.4f}")
        print(f"rho_4 = {optimal_rho[3]:.4f}")
        print(f"rho_5 = {optimal_rho[4]:.4f}")
        
        # 验证一下找到的 rho 算出来的 sigma 到底准不准
        final_sigmas,_ = calculate_sigmas_with_intermediates(optimal_rho, params)
        print("\n🔍 验证结果:")
        print(f"目标 σ3 = {TARGET_SIGMA_3}, 实际算出 σ3 = {final_sigmas[2]:.6f}")
        print(f"目标 σ4 = {TARGET_SIGMA_4}, 实际算出 σ4 = {final_sigmas[3]:.6f}")
        print(f"(附带的 σ1 = {final_sigmas[0]:.4f}, σ2 = {final_sigmas[1]:.4f})")
    else:
        print(f"\n❌ 求解失败，原因: {result.message}")
    
    return final_sigmas, optimal_rho

def objective_function_eta(rho, params):
    """
    目标函数（误差函数/损失函数）。
    我们要让这个函数的返回值尽可能接近于 0。
    """
    # 算出现有 rho 对应的 sigmas
    sigmas,_ = calculate_sigmas_with_intermediates(rho, params)
    
    # 提取我们关心的 sigma3 和 sigma4
    sigma1_calc = sigmas[0]
    sigma2_calc = sigmas[1]
    sigma3_calc = sigmas[2]
    sigma4_calc = sigmas[3]
    
    # 计算均方误差 (MSE)
    # 也就是： (计算值 - 目标值)^2 的总和
    error = (sigma1_calc - np.fabs(sigma3_calc))**2 + (sigma2_calc - np.fabs(sigma4_calc))**2
    
    return error

def calculate_eta(params):    
    # 3. 提供一个初始的猜测值 (rho1, rho2, rho3, rho4, rho5)
    # 算法会从这个点开始搜索
    initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
    
    # 4. 设置边界条件！非常重要！
    # 因为公式里有 rho^(-mu) 这种负指数，rho 必须大于 0，否则会报错。
    # 这里我们设定每个 rho 的范围是 [0.001, 无穷大]
    bnds = ((0.001, None), (0.001, None), (0.001, None), (0.001, None), (0.001, None))
    
    print("开始搜索满足条件的 rho 值...")
    
    # 5. 调用优化器进行搜索求解
    result = minimize(
        objective_function_eta, 
        initial_guess, 
        args=(params),
        method='L-BFGS-B',  # 这种算法支持边界条件 (Bounds)
        bounds=bnds
    )
    
    # 6. 输出结果
    if result.success:
        print("\n✅ 求解成功！找到了一组满足条件的 rho:")
        optimal_rho = result.x
        print(f"rho_1 = {optimal_rho[0]:.4f}")
        print(f"rho_2 = {optimal_rho[1]:.4f}")
        print(f"rho_3 = {optimal_rho[2]:.4f}")
        print(f"rho_4 = {optimal_rho[3]:.4f}")
        print(f"rho_5 = {optimal_rho[4]:.4f}")
        
        # 验证一下找到的 rho 算出来的 sigma 到底准不准
        final_sigmas,_ = calculate_sigmas_with_intermediates(optimal_rho, params)
        print("\n🔍 验证结果:")
        print(f"最终计算出的 sigmas = {final_sigmas}")
    else:
        print(f"\n❌ 求解失败，原因: {result.message}")
    
    return final_sigmas, optimal_rho

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 设定系统参数
    base_parameters = {
        'N':  4, 
        'n':  4,    
        'mu': 0.85, 
        'nu': 1.15,
        'theta': 2.200000/7, # 补充了公式下方出现的 theta
        'phi': 2.040000/7,   # \varphi
        'h_m': 2.081/7,
        'kappa': 0.92,
    }

    final_sigmas_1, optimal_rho_1 = calculate_delta(base_parameters)
    final_sigmas_2, optimal_rho_2 = calculate_eta(base_parameters)
    optimal = base_parameters['kappa'] * base_parameters['phi']

    alpha1 = (final_sigmas_1[0] + optimal)*2/((2*0.267949)**(0.5+base_parameters['mu']/2)) # 这里的 0.267949 是 (sqrt(3)-1)/2 的近似值
    b = (base_parameters['N']**2*(base_parameters['n']+1))**(1/2-base_parameters['nu']/2) * 2**((2*(1-base_parameters['nu']))/((base_parameters['nu']+1)**2))
    alpha2 = (final_sigmas_1[1] + optimal)*2/((2*0.267949)**(0.5+base_parameters['nu']/2))/b
    print(f"计算得到的 alpha1 = {alpha1:.4f}")
    print(f"计算得到的 alpha2 = {alpha2:.4f}")

    # rhod = optimal/ (optimal + final_sigmas_2[0]*(2**(0.5-base_parameters['mu']/2)) +final_sigmas_2[1]) # 假设 rhod 是一个简单的函数，比如 (kappa * phi) / (kappa * phi + 1)
    rhod = optimal/ (optimal + max(final_sigmas_2[0]*(2**(0.5-base_parameters['mu']/2)), (2**(base_parameters['nu']/2-0.5))*final_sigmas_2[1])) # 假设 rhod 是一个简单的函数，比如 (kappa * phi) / (kappa * phi + 1)
    print(f"\n计算得到的 rhod = {rhod:.4f}")
    
    # 2. 设定你要达到的目标值 (假设你图片里 phi * varphi 算出来是某个具体的值，比如 2.5)
    
