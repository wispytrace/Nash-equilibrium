import numpy as np



def compute_eigenvalues(matrix):
    """
    计算给定矩阵的最小和最大特征值
    
    参数:
        matrix: numpy数组，一个方阵
        
    返回:
        min_eigenvalue: 最小特征值
        max_eigenvalue: 最大特征值
    """
    # 检查输入是否为方阵
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("输入必须是方阵")
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(matrix)
    
    # 对于对称矩阵，特征值一定是实数
    # 对于非对称矩阵，我们取模长作为特征值的大小
    eigenvalues_abs = np.abs(eigenvalues)
    
    # 获取最小和最大特征值
    min_eigenvalue = np.min(eigenvalues_abs)
    max_eigenvalue = np.max(eigenvalues_abs)
    
    return min_eigenvalue, max_eigenvalue

def parameter_calculate():
    """
    Calculate the parameters for the given index.
    """
    N = 5
    scale = 0.10
    adjacency_matrix = np.array([[0, 1, 0, 0, 1],
                  [1, 0, 1, 0,0],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1],
                  [1, 0, 0, 1, 0]])
    
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    digA = np.diag(adjacency_matrix.flatten())
    I = np.eye(N)
    M = np.kron(L, I) + digA
    min_eig_M, max_eig_M = compute_eigenvalues(M)
    print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
    print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")


def calculate_chi(beta1, beta2, p, q, N):
    """
    根据给定的公式计算 chi1 和 chi2
    
    参数:
    beta1, beta2: 相关参数 (对应公式中的 β1, β2)
    p: 参数 p
    q: 参数 q
    N: 参数 N (仅用于 chi2)
    
    返回:
    (chi1, chi2) 的元组
    """
    
    # 预先计算公共部分，提高效率并保持代码整洁
    # 对应公式：min{β1^2, β2^2}
    min_beta_sq = min(beta1**2, beta2**2)
    
    # 对应公式：max{β1, β2}
    max_beta = max(beta1, beta2)
    
    # 对应公式括号内的部分：(1+p) / max{β1, β2}
    base_term = (1 + p) / max_beta
    
    # ---------------------------------------------------------
    # 计算 Chi 1
    # ---------------------------------------------------------
    # 指数部分：(p+1) / (1+q)
    exp_chi1 = (p + 1) / (1 + q)
    
    # chi1 = 1/2 * min_beta_sq * (base_term ^ exp_chi1)
    chi1 = 0.5 * min_beta_sq * (base_term ** exp_chi1)
    
    # ---------------------------------------------------------
    # 计算 Chi 2
    # ---------------------------------------------------------
    # 指数部分 1 (N的指数)：(1-q) / (1+q)
    exp_N = (1 - q) / (1 + q)
    
    # 指数部分 2 (base_term的指数)：2q / (q+1)
    exp_chi2 = (2 * q) / (q + 1)
    
    # chi2 = 1/2 * (2N)^exp_N * min_beta_sq * (base_term ^ exp_chi2)
    chi2 = 0.5 * ((2 * N) ** exp_N) * min_beta_sq * (base_term ** exp_chi2)
    
    return chi1, chi2

import numpy as np

def calculate_complex_parameters(params):
    """
    根据提供的图片公式计算 C3, C4, O3, O4, theta2
    
    参数 params 字典需包含:
    mu, nu : 指数参数 (μ, ν)
    N      : 系统维度或节点数
    sigma_min, sigma_max : σ 的极值
    phi_min, phi_max     : φ (phi) 的极值
    xi_max               : ξ (xi) 的最大值
    lambda_min_S_ST      : 矩阵 (S + S^T) 的最小特征值
    """
    
    # 提取变量以简化代码书写
    mu = params['mu']
    nu = params['nu']
    N = params['N']
    sig_min = params['sigma_min']
    sig_max = params['sigma_max']
    phi_min = params['phi_min']
    phi_max = params['phi_max']
    xi_max = params['xi_max']
    lam_min = params['lambda_min_S_ST'] # λ_min(S + S^T)

    # ==========================================
    # 1. 计算 Theta 2 (θ2)
    # ==========================================
    # 公式: max{ phi_max/phi_min, sigma_max/sigma_min }
    theta2 = max(phi_max / phi_min, sig_max / sig_min)

    # ==========================================
    # 2. 计算 O4 (最复杂的部分)
    # ==========================================
    # 为了防止出错，我们将 O4 拆分为 4 个项 (Term 1 到 Term 4)
    # 辅助指数定义
    exp_mu = (2 * mu) / (mu + 1)  # 2μ / (μ+1)
    exp_nu = (2 * nu) / (nu + 1)  # 2ν / (ν+1)
    coeff_2_nu = 2 ** ((nu - 1) / (nu + 1)) # 2^((ν-1)/(ν+1))

    # Term 1: (xi_max * phi_min / (mu + 1)) ^ exp_mu
    term1 = (xi_max * phi_min / (mu + 1)) ** exp_mu

    # Term 2: (xi_max * sigma_min / (nu + 1)) ^ exp_mu  <-- 注意图片中这里指数是 μ 相关
    term2 = (xi_max * sig_min / (nu + 1)) ** exp_mu

    # Term 3: 2_coeff * (xi_max * phi_min / (mu + 1)) ^ exp_nu <-- 交叉项
    term3 = coeff_2_nu * ((xi_max * phi_min / (mu + 1)) ** exp_nu)

    # Term 4: 2_coeff * (xi_max * sigma_min / (nu + 1)) ^ exp_nu
    term4 = coeff_2_nu * ((xi_max * sig_min / (nu + 1)) ** exp_nu)

    O4 = term1 + term2 + term3 + term4

    # ==========================================
    # 3. 计算 O3
    # ==========================================
    # 项 A: (sigma_min^2 * N^(2-4nu)) / O4
    item_a = (sig_min**2 * (N ** (2 - 4 * nu))) / O4
    
    # 项 B: (phi_min^2 * N^(2-4mu)) / O4
    item_b = (phi_min**2 * (N ** (2 - 4 * mu))) / O4

    O3 = min(item_a, item_b)

    # ==========================================
    # 4. 计算 C3 和 C4
    # ==========================================
    # C3 = 0.5 * lambda * O3 * theta2^(-2mu/(mu+1))
    C3 = 0.5 * lam_min * O3 * (theta2 ** (-exp_mu))

    # C4 = 0.5 * lambda * O3 * theta2^(-2nu/(nu+1))
    # 注意：图片中 C4 也是乘以 O3
    C4 = 0.5 * lam_min * O3 * (theta2 ** (-exp_nu))

    return {
        "C3": C3,
        "C4": C4,
        "O3": O3,
        "O4": O4,
        "theta2": theta2
    }

# ==========================================
# 用户输入区域 (请修改这里的数值)
# ==========================================




c = 0.726
p = 0.85
q = 1.15
phic = 0.174

delta = 8
eta = 8

C5 = delta*(1-c)/((1+c)**(1-p))
C6 = eta*(1-c)/((1-c)**(1-q)) 

print(C5, C6, 1/((1+c)**(1-p)), c/((1-c)**(1-p)))
T = 2/(C5*(1-p)) + 2/(C6*(q-1))
print("Upper bound on the convergence rate:", T)
parameter_calculate()

c1 = 20
c2 = 20
lambda_min = 0.24
G1 = 2**((1+p)/2) * c1 / (6**((1+p)/2))
G2 = 2**((1+q)/2) * c2 / (6**((1+q)/2)) * (5*5)**((1-q)/2)

T2 = 2/(G1*(1-p)) + 2/(G2*(q-1))
print("Another upper bound on the convergence rate:", T2)

chi1, chi2 = calculate_chi(20, 20, 0.85, 1.15, N=25)
chi1 = chi1 * (1/0.258)**((1-p)/(1+q))
chi2 = chi2 * (1/0.258)**((2*q)/(1+q))
print(f"Chi1: {chi1}, Chi2: {chi2}")
T = 1/(chi1*0.04*(1-(1-p)/(1+q))) + 1/(chi2*0.04*((2*q)/(1+q)-1))
print(T)


# 假设的输入值
input_params = {
    'mu': 1.15,            # μ > 1
    'nu': 0.85,            # 0 < ν < 1
    'N': 5,              # 节点数量
    'sigma_min': 2.58,     # σ_min
    'sigma_max': 2.58,     # σ_max
    'phi_min': 2.58,       # φ_min
    'phi_max': 2.58,       # φ_max
    'xi_max': 1,        # ξ_max
    'lambda_min_S_ST': 1 # S+S^T 的最小特征值
}

results = calculate_complex_parameters(input_params)

print("-" * 40)
print("计算结果:")
print("-" * 40)
print(f"Theta 2 (θ2) = {results['theta2']:.6f}")
print(f"O4           = {results['O4']:.6f}")
print(f"O3           = {results['O3']:.6f}")
print("-" * 40)
print(f"C3           = {results['C3']:.6f}")
print(f"C4           = {results['C4']:.6f}")
print("-" * 40)
