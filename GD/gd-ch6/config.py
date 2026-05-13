import numpy as np
import copy

config = {
    "r_0":
    {
        "epochs" : 60000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 1 ],
                              [1, 0, 1, 0, 0, 0],
                              [0, 1, 0, 1, 0, 0],
                              [0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 1, 0, 1],
                              [1, 0, 0, 0, 1, 0]], 
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed_high_order",
            "record_interval": 50,
            "record_flag": 1,

            "model_config": 
            {
                "N": 6,
                "memory" : {"omega": np.zeros(2), "z": np.zeros((6,2)), "x": np.zeros(2), "x_hl": np.zeros((2, 2)), "x_li": np.zeros((2,3)), "x_el": np.zeros((2, 2)), "ei_sum": np.zeros((2)),
                            'ui_hl': np.zeros((2)), 'ui_li': np.zeros((2,3)), 'ui_el': np.zeros((2))},
                'share': {
                    "p": 0.8,
                    "q": 1.2,
                    'alpha': [150, 250, 520, 600],
                    'beta':[2, 2],
                },
                'private':{
                    '0': {
                        'zeta': 5,
                        'eta': 5,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 4, 5],
                        'order': 2,
                        'pg': np.array([2,2]),
                        "init_value": np.array([[4, 4], [-2, 2]], dtype=float)
                        },

                    '1': {
                        'zeta': 5,
                        'eta': 5,
                        'gama': [0.7, 1.3],
                        'ki': [3, 4, 5],
                        'k_i_tilde': [5, 6, 5],
                        'order': 2,
                        'pg': np.array([4,0]),
                        "init_value": np.array([[6, -1], [2, -2]], dtype=float)
                    },
                    '2': { 
                            'parameters': np.array([1, 5, 0.2]),
                            'gama': [0.7, 1.3],
                            'ki': [2, 3],
                            'pg': np.array([2,-2]),
                            "init_value": np.array([[4, 0,1], [-4, 2, 0]], dtype=float)
                        },

                    '3': {
                            'parameters': np.array([1.2, 5, 0.2]),
                            'gama': [0.85, 1.3],
                            'ki': [4, 3],
                            'pg': np.array([-2,-2]),
                            "init_value": np.array([[-4, 1,0], [-4, 1, 1]], dtype=float)
                            },
                    '4': {
                        "parameter_matrix": np.array([[1.19, 1.16, 1.13, 1.11, 1],
                                                        [1.41, 1.43, 1.45, 1.47, 1.5],
                                                        [0.31, 0.33, 0.32, 0.34, 0.47],
                                                        [1.78, 1.76, 1.74, 1.72, 1],
                                                        [0.73, 0.76, 0.79, 0.72, 1]]).T,
                        'pg': np.array([-4,0]),
                        "h1": 2,
                        "h2": 2,
                        "init_value": np.array([[-6, 1], [1, -1]], dtype=float)
                    },
                    '5': {
                        "parameter_matrix": np.array([[1.19, 1.16, 1.13, 1.11, 1],
                                                        [1.41, 1.43, 1.45, 1.47, 1.5],
                                                        [0.31, 0.33, 0.32, 0.34, 0.47],
                                                        [1.78, 1.76, 1.74, 1.72, 1],
                                                        [0.73, 0.76, 0.79, 0.72, 1]]).T,
                        'pg': np.array([-2,2]),
                        "h1": 2,
                        "h2": 2,
                        "init_value": np.array([[-4, 4], [2, 1]], dtype=float)
                        }
                    }                
                }
            },
            
        }
}


def set_by_path(dic, path, value):
    """
    dic: 要操作的字典
    path: 字符串路径，例 "agent_config.model_config.private.1.order"
    value: 新值
    """
    keys = path.split('.')
    d = dic
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

def batch_modify_config(base_config, path_list, value_list):
    new_config = copy.deepcopy(base_config)
    for path, value in zip(path_list, value_list):
        set_by_path(new_config, path, value)
    return new_config

# 用法示例

# config['finite_1'] = batch_modify_config(
#     config["r_0"], 
#     ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta"], 
#     [True, 10, 1e-4]
# )


# print(config['fixed_1'])




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


def phi(x,y):
    return max(1, x**(1-y))

def phi_m(x, y):
    return min(1, x**(1-y))
    

def parameter_calculate(index):
    """
    Calculate the parameters for the given index.
    """
    N = 5
    m = 5
    sclae = 0.5
    A =  np.array([
        [ 3.06,  0.53, -0.27, -0.07,  0.33],
        [-0.47,  2.94,  0.55, -0.13, -0.05],
        [ 0.13, -0.05,  3.00,  0.52, -0.36],
        [-0.07,  0.07, -0.68,  3.02,  0.40],
        [-0.47, -0.05,  0.04, -0.60,  2.91]
    ])

    A = A*sclae
    min_eig, max_eig = compute_eigenvalues(A)
    print(f"最小特征值: {min_eig:.6f}")
    print(f"最大特征值: {max_eig:.6f}")
    m_hat = min_eig
    h_m = max_eig
    l_hat = max_eig
    for i in range(A.shape[0]):
        if l_hat< np.linalg.norm(A[i,:]):
            l_hat = np.linalg.norm(A[i,:])
    # h_m = 3.9317
    print(f"m_hat: {m_hat:.6f}, l_hat: {l_hat:.6f}, h_m: {h_m:.6f}")
    adjacency_matrix = np.array( [[0, 1, 0, 0, 1 ],
                              [1, 0, 1, 0, 0],
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
    
    rho1 = 0.006
    rho2 = 0.01
    rho3 = 0.01
    rho4 = 0.05

    beta1 = 1
    beta2 = 0.5
    alpha1 = 150
    alpha2 = 250
    alpha3 = 520
    alpha4 = 800

    p = 0.8
    q = 1.2

    # rho1 = 0.006
    # rho2 = 0.005
    # rho3 = 0.01
    # rho4 = 0.02

    c1 = beta1* 2 **(1-p) * (l_hat**p) * (min_eig_M**(-p)) * (N**(1-p/2)) * (m**(1/2-p/2)) 
    print(f"c1: {c1:.6f}")
    c2 = np.sqrt(N)*beta2 * (min_eig_M**(-q))* (q*l_hat**q + rho1**(1-q)*l_hat* (m**(1-q/2)))
    print(f"c2: {c2:.6f}")
    c3 = np.sqrt(N)*rho1*l_hat*(q-1)* m**(q/2-1/2)* (m**(1-q/2))
    print(f"c3: {c3:.6f}")

    sigma1 = m_hat - h_m*(c1*rho3 + c2*rho4 + c3)/(2*np.sqrt(N)) - rho2*(c3+np.sqrt(N))/2
    print(f"sigma1: {sigma1:.6f}")
    sigma2 = alpha1 - c1 - (c3+np.sqrt(N))/(4*rho2)
    print(f"sigma2: {sigma2:.6f}")
    sigma3 = alpha2 * (m*N)**(1/2 - q/2) - c2 - (c3+np.sqrt(N))/(4*rho3)
    print(f"sigma3: {sigma3:.6f}")
    sigma4 = alpha3 - c1*h_m/(2*rho3*np.sqrt(N)) - (c3+np.sqrt(N))/(4*rho2)
    print(f"sigma4: {sigma4:.6f}")
    sigma5 = alpha4 * (m*N)**(1-q) - c2*h_m/(2*rho4*np.sqrt(N)) - (c3+np.sqrt(N))/(4*rho2)
    print(f"sigma5: {sigma5:.6f}")

    # w1 = (2**(1-p))*(l_hat**(p))*(min_eig_M**(-p))*((m*N)**(1-p))
    # print(f"w1: {w1:.6f}")

    # w2 = (min_eig_M**(-q))*(2**(q-2)+2)*(q*l_hat**q*phi(m*N, q)+rho1**(1-q)*l_hat*phi(m, q-1))
    # print(f"w2: {w2:.6f}", (min_eig_M**(-q)), q*l_hat**q*phi(m*N, q), phi(m, q-1))

    # w3 = (2**(q-2)+2)*rho1*l_hat*(q-1)*phi(m,q-1)
    # print(f"w3: {w3:.6f}")

    # w4 = m**(1-p)
    # print(f"w4: {w4:.6f}")

    # w5 = w3 + 1
    # print(f"w5: {w5:.6f}")

    # g1 = h_m* beta1**2 * w1 * m**(1-p)
    # print(f"g1: {g1:.6f}")
    # g2 = h_m * beta1 * beta2 * m**(1-p) * w2
    # print(f"g2: {g2:.6f}")
    # g3 = h_m * beta1 * beta2 * w1
    # print(f"g3: {g3:.6f}")

    # g4 = h_m * beta2**2 * w2
    # print(f"g4: {g4:.6f}")

    # g5 = h_m * beta2**2 * w3
    # print(f"g5: {g5:.6f}")

    # delta1 = m_hat - varepsilon
    # print(f"delta1: {delta1:.6f}")
    
    # delta2 = (varepsilon * beta1**2 - (g1 * rho4)/2 - (rho2 * p * beta1 * N**(1/2) * w4)/(1+p) - (rho3 * q * beta2 * N**(1/2) * w5)/(1+q))
    # print(f"delta2: {delta2:.6f}", (rho2 * p * beta1 * N**(1/2) * w4)/(1+p) + (rho3 * q * beta2 * N**(1/2) * w5)/(1+q), (g1 * rho4)/2)

    # delta3 = varepsilon * beta2**2 * m**(1-q) - (g4*rho5)/2 - (rho2 * p * beta1 * N**(1/2) * w4)/(1+p) - (rho3 * q * beta2 * N**(1/2) * w5)/(1+q) - g5
    # print(f"delta3: {delta3:.6f}", (g4*rho5)/2, (rho2 * p * beta1 * N**(1/2) * w4)/(1+p))

    # delta4 = 2*varepsilon* beta1 * beta2 * phi_m(m, (p+q)/2) - (g2 * rho6 * p)/(p+1) - (g3 * rho7 * q)/(p+q)
    # print(f"delta4: {delta4:.6f}")

    # delta5 = alpha1 - w1 * beta1 * N**(1/2) - (rho2**(-1/p)*beta1* N**(1/2) * w4)/(1+p) 
    # print(f"delta5: {delta5:.6f}", (rho2**(-1/p)*beta1* N**(1/2) * w4)/(1+p) )

    # delta6 = alpha2*((m*N)**(1/2-q/2)) - w2 * beta2 * N**(1/2) - (rho3**(-1/q)*beta2* N**(1/2) * w5)/(1+q)
    # print(f"delta6: {delta6:.6f}")

    # delta7 = alpha3 - (g1 * 1/rho4)/2 - (g2 * q * (rho6**(-1*p/q)) )/(p+q) - (g4 * p * (rho7**(-1*q/p)) )/(p+q)
    # print(f"delta7: {delta7:.6f}")

    # delta8 = alpha4 * (m *N)**(1-q) - (g3 * 1/rho5)/2 - (g2 * q * rho6**(-1*p/q))/(p+q) - (g4 * p * rho7**(-1*q/p))/(p+q)
    # print(f"delta8: {delta8:.6f}")
    
if __name__ == "__main__":
    # Example usage
    index = "fixed_1"
    try:
        result = parameter_calculate(index)
        print(f"Parameters for {index}: {result}")
    except ValueError as e:
        print(e)  # Handle the case where the index is not found


    import numpy as np

    # 构造无规律但对角占优的对称矩阵
    Q =  np.array([
        [ 3.06,  0.53, -0.27, -0.07,  0.33],
        [-0.47,  2.94,  0.55, -0.13, -0.05],
        [ 0.13, -0.05,  3.00,  0.52, -0.36],
        [-0.07,  0.07, -0.68,  3.02,  0.40],
        [-0.47, -0.05,  0.04, -0.60,  2.91]
    ])

    # 检查对称性 (确保特征值为实数)
    # assert np.allclose(Q, Q.T), "矩阵必须是对称的"

    # 计算特征值
    eigenvalues = np.linalg.eigvalsh(Q)

    # 提取强凸系数(mu)和李普希兹常数(L)
    mu = np.min(eigenvalues)
    L = np.max(eigenvalues)

    print("矩阵的特征值:", np.round(eigenvalues, 4))
    print(f"强凸系数 (mu) = {mu:.4f} (> 0, 满足强凸)")
    print(f"李普希兹常数 (L) = {L:.4f} (满足梯度 Lipschitz 连续)")
    print(f"条件数 (Kappa = L/mu) = {L/mu:.4f}")