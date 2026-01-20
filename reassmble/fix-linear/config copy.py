import numpy as np




config = {
    "r_1":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[1, 1, 1, 1 ],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1 ],
                                [1, 1, 1, 1 ]], 
        "agent_config":
        {  
            "time_delta": 2e-4,
            "model": "fixed_linear",
            "record_interval": 50,
            "record_flag": 1,
            
            "model_config": 
            {
                "action": "np.array(self.C@np.matrix(self.memory['x']).T).flatten()",
                "scale_dict": {},
                "N": 4,
                "memory" : {"x": np.zeros(3), "y": np.zeros(1), "z": np.zeros((4, 1))},
                # "cost_scale": 0.1,
                # "epsilon": 0.1,
                "DoS_interval":{
                    "1":[[1, 1.5], [3.5, 4]],
                    "2":[[2.5, 2.7], [4.2, 4.3]],
                    "3":[[5, 5.5], [7, 7.5]],
                },

                'share': {
                    "p": 0.8,
                    "q": 1.2,
                    'alpha': [250, 250, 250, 250],
                    'beta':[1, 1],
                    'po': 5,
                    "a": 0.04
                },

                'private': {
                '0': { 
                        'xi': 10,
                        'parameters': np.array([1, 5, 0.2]),
                        'gama': [0.7, 1.3],
                        'ki': [2, 3],
                        'x0': np.array([3, 10, 6]),
                        'y0': np.array([3])
                        },
                '1': {
                        'xi': 20,
                        'parameters': np.array([1.2, 5, 0.2]),
                        'gama': [0.85, 1.3],
                        'ki': [4, 3],
                        'x0': np.array([5, 8, 7]),
                        'y0': np.array([5])
                        },
                '2': { 
                        'xi': 30,
                        'parameters':np.array([1.2, 4.8, 0.2]),
                        'gama': [0.8, 1.3],
                        'ki': [5, 4],
                        'x0': np.array([7, 7, 2]),
                        'y0': np.array([7])
                        },
                '3': { 
                        'xi': 40,
                        'parameters': np.array([1, 4.8, 0.21]),
                        'gama': [0.75, 1.25],
                        'ki': [3, 3],
                        'x0': np.array([9, 6, 8]),
                        'y0': np.array([9])
                        },
                },
            }
        }
    },

}


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
    return max(1, x**(1-y/2))

def phi_m(x, y):
    return min(1, x**(1-y/2))


def parameter_calculate(index):
    """
    Calculate the parameters for the given index.
    """
    N = 4
    m = 1
    n = 4
    scale = 0.5
    A = np.array([[2.08, 0.04, 0.04, 0.04], 
                [0.04, 2.08, 0.04, 0.04], 
                [0.04, 0.04, 2.08, 0.04],
                [0.04, 0.04, 0.04, 2.08]])
    print(A.shape)

    A = A * scale
    min_eig, max_eig = compute_eigenvalues(A)
    print(f"最小特征值: {min_eig:.6f}")
    print(f"最大特征值: {max_eig:.6f}")
    m_hat = min_eig
    l_hat = np.sqrt(2.08**2+0.04**2*3)*scale
    h_m = max_eig
    print("h_m:", h_m) 

    adjacency_matrix = np.array([[0, 1, 0, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0],])
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    digA = np.diag(adjacency_matrix.flatten())
    I = np.eye(N)
    M = np.kron(L, I) + digA
    min_eig_M, max_eig_M = compute_eigenvalues(M)
    print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
    print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")


    beta1 = 1
    beta2 = 0.5
    alpha1 = 600
    alpha2 = 2200


    mu = 0.85
    nu = 1.15


    rho1 = 0.02
    rho2 = 0.06
    rho3 = 0.05
    rho4 = 0.04
    rho5 = 0.025

    Nsqrt = N**(1/2)

    c1 = beta1 * N**(1-mu/2)* n**(1-mu) * 2**(1-mu) * l_hat**mu
    print(f"c1: {c1:.6f}")

    c2 = beta2 * N**(1/2) * (2**(nu-2)+2) * (nu) * l_hat**nu
    print(f"c2: {c2:.6f}")

    c3 = beta2 * N**(1/2) * (2**(nu-2)+2) * l_hat * n**(1-nu/2)
    print(f"c3: {c3:.6f}")

    c4 = beta2 * N**(1/2) * n**(nu-1) * (2**(nu-2)+2) * l_hat * (nu-1) * n**(1-nu/2)
    print(f"c4: {c4:.6f}")

    b1 = (N**2*(n+1))**(1/2-nu/2) * 2**((2*(1-nu))/((nu+1)**2))
    print(f"b1: {b1:.6f}")

    b2 = (N**2*(n+1))**(1-nu) * 2**((1-nu)/(nu**2))
    print(f"b2: {b2:.6f}")

    g1 = c1 + beta1*Nsqrt*rho2**(-mu)/(mu+1) + h_m*c1*rho4**(-1/mu)*mu/(Nsqrt*(mu+1))
    print(f"g1: {g1:.6f}")

    g2 = c2 + c3*rho1*(1-nu) + (c4*rho1+beta2*Nsqrt)*rho3**(-nu)/(1+nu) + h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu
    print(f"g2: {g2:.6f}", h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu)

    delta1 = (alpha1* (2*min_eig_M)**((1+mu)/2))/2 - g1
    print(f"delta1: {delta1:.6f}")

    delta2 = alpha2*b1*(2*min_eig_M)**((1+nu)/2)/2 - g2
    print(f"delta2: {delta2:.6f}")

    delta3 = (2*m_hat-h_m)*beta1 - Nsqrt*beta1*rho2*n**(1/2-mu/2)*mu/(mu+1) - h_m*c1*rho4/(Nsqrt*(mu+1))
    print(f"delta3: {delta3:.6f}")

    delta4 = (2*m_hat-h_m)*beta2 - (c4*rho1+Nsqrt*beta2)*rho3*nu/(1+nu) - h_m*(c2+c3*rho1**(1-nu))*rho5/(Nsqrt*(nu+1)) - h_m*c4*rho1/Nsqrt
    print(f"delta4: {delta4:.6f}", h_m*(c2+c3*rho1**(1-nu))*rho5/(Nsqrt*(nu+1)), (c4*rho1+Nsqrt*beta2)*rho3*nu/(1+nu), h_m*c4*rho1/Nsqrt)

    d1 = min(delta3, 2**(1/2-nu/2)*delta4)
    print(f"d1: {d1:.6f}")

    
    rho1 = 2
    rho2 = 1
    rho3 = 0.6
    rho4 = 0.5
    rho5 = 3.2

    beta1 = 0.005
    beta2 = 0.005
    c1 = beta1 * N**(1-mu/2)* n**(1-mu) * 2**(1-mu) * l_hat**mu
    print(f"c1: {c1:.6f}")

    c2 = beta2 * N**(1/2) * (2**(nu-2)+2) * (nu) * l_hat**nu
    print(f"c2: {c2:.6f}")

    c3 = beta2 * N**(1/2) * (2**(nu-2)+2) * l_hat * n**(1-nu/2)
    print(f"c3: {c3:.6f}")

    c4 = beta2 * N**(1/2) * n**(nu-1) * (2**(nu-2)+2) * l_hat * (nu-1) * n**(1-nu/2)
    print(f"c4: {c4:.6f}")

    b1 = (N**2*(n+1))**(1/2-nu/2) * 2**((2*(1-nu))/((nu+1)**2))
    print(f"b1: {b1:.6f}")

    b2 = (N**2*(n+1))**(1-nu) * 2**((1-nu)/(nu**2))
    print(f"b2: {b2:.6f}")

    g1 = c1 + beta1*Nsqrt*rho2**(-mu)/(mu+1) + h_m*c1*rho4**(-1/mu)*mu/(Nsqrt*(mu+1))
    print(f"g1: {g1:.6f}")

    g2 = c2 + c3*rho1*(1-nu) + (c4*rho1+beta2*Nsqrt)*rho3**(-nu)/(1+nu) + h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu
    print(f"g2: {g2:.6f}", h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu)

    g1 = c1 + beta1*Nsqrt*rho2**(-mu)/(mu+1) + h_m*c1*rho4**(-1/mu)*mu/(Nsqrt*(mu+1))
    print(f"g1: {g1:.6f}")

    g2 = c2 + c3*rho1*(1-nu) + (c4*rho1+beta2*Nsqrt)*rho3**(-nu)/(1+nu) + h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu
    print(f"g2: {g2:.6f}", h_m*(c2+c3*rho1**(1-nu))*rho5**(-1/nu)*nu)

    delta3 = (2*m_hat-h_m)*beta1 - Nsqrt*beta1*rho2*n**(1/2-mu/2)*mu/(mu+1) - h_m*c1*rho4/(Nsqrt*(mu+1))
    print(f"delta3: {delta3:.6f}")

    delta4 = (2*m_hat-h_m)*beta2 - (c4*rho1+Nsqrt*beta2)*rho3*nu/(1+nu) - h_m*(c2+c3*rho1**(1-nu))*rho5/(Nsqrt*(nu+1)) - h_m*c4*rho1/Nsqrt
    print(f"delta4: {delta4:.6f}", h_m*(c2+c3*rho1**(1-nu))*rho5/(Nsqrt*(nu+1)), h_m*c2+c3*rho1**(1-nu))


    d2 = max(g1*2**(1/2-mu/2), g2, np.fabs(delta3)*2**(1/2-mu/2), np.fabs(delta4))
    print(f"d2: {d2:.6f}")

    v = d1 / (d1+d2)

    print(f"v: {v:.6f}")
 

def seek_optimal_dos(g1, epsilon, g2):
    epochs = 1000
    min_rd = 0
    min_td = 0
    max_value = 1e6
    for i in range(epochs):
        rd = (i+1) * 1
        for j in range(epochs):
            td = (j+1) * 1
            value = 4*rd*td*g1*epsilon - 4*td*(g1*epsilon + g2*epsilon*(epsilon+2)) - 3.14*rd*(2+epsilon)
            if value > 0 and value < max_value:
                max_value = value
                min_rd = rd
                min_td = td
    print(f"Optimal rd: {min_rd}, Optimal td: {min_td}, Max Value: {max_value}")


    
if __name__ == "__main__":
    # Example usage
    index = "fixed_1"
    try:
        result = parameter_calculate(index)
        print(f"Parameters for {index}: {result}")
    except ValueError as e:
        print(e)  # Handle the case where the index is not found