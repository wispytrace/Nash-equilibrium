import numpy as np
import copy


config = {
    "r_r":
    {
        "epochs" : 100000,
        "adjacency_matrix" : [[1, 1, 1, 1 ],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1 ],
                                [1, 1, 1, 1 ]], 
        "agent_config":
        {  
            "time_delta": 5e-3,
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
                    "1":[[0.23, 0.63], [2, 2.5]],
                    "2":[[5, 5.5], [11, 11.5]],
                    "3":[[20, 20.7], [30, 30.5], [40, 40.5]],
                },

                'share': {
                    "p": 0.75,
                    "q": 1.25,
                    'alpha': [200, 500, 0, 0],
                    'beta':[1, 0.5],
                    'po': 5,
                    "a": 0.04
                },


                'private': {
                '0': { 
                        'xi': 10,
                        'parameters': np.array([1, 5, 0.2]),
                        'gama': [0.7, 1.3],
                        'ki': [2, 3],
                        'x0': np.array([12, 10, 6]),
                        'y0': np.array([6])
                        },
                '1': {
                        'xi': 20,
                        'parameters': np.array([1.2, 5, 0.2]),
                        'gama': [0.85, 1.3],
                        'ki': [4, 3],
                        'x0': np.array([7, 8, 7]),
                        'y0': np.array([5])
                        },
                '2': { 
                        'xi': 30,
                        'parameters':np.array([1.2, 4.8, 0.2]),
                        'gama': [0.8, 1.3],
                        'ki': [5, 4],
                        'x0': np.array([6, 7, 2]),
                        'y0': np.array([2])
                        },
                '3': { 
                        'xi': 40,
                        'parameters': np.array([1, 4.8, 0.21]),
                        'gama': [0.75, 1.25],
                        'ki': [3, 3],
                        'x0': np.array([10, 6, 8]),
                        'y0': np.array([7])
                        },
                },
            }
        }
    },
    "r_1":
    {
        "epochs" : 100000,
        "simulation_time": 50, # 总仿真时长 (秒)
        "adjacency_matrix" : [[1, 1, 1, 1 ],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1 ],
                                [1, 1, 1, 1 ]], 
        "agent_config":
        {  
            "time_delta": 5e-4,
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
                    "1":[[0.2, 0.3], [2, 2.3], [25.5, 25.6]],
                    "2":[[0.3, 0.4], [10, 10.2], [25.8, 26.0]],
                    "3":[[0.4, 0.5], [2.3, 2.5], [26.2, 26.3]],
                    "4":[[0.5, 0.6], [10, 10.3], [26.5, 26.6]],
                    "5":[[0.6, 0.7], [15, 15.5], [40, 40.5]]
                },

                'share': {
                    "p": 0.75,
                    "q": 1.25,
                    'alpha': [100, 100, 0, 0],
                    'beta':[1, 0.5],
                    'tau': [1, 0.5],
                    'po': 5,
                    "a": 0.04
                },

                'private': {
                '0': { 
                        'xi': 10,
                        'parameters': np.array([1, 5, 0.2]),
                        'gama': [0.7, 1.3],
                        'ki': [2, 3],
                        'x0': np.array([20, 10, 6]),
                        'y0': np.array([10])
                        },
                '1': {
                        'xi': 20,
                        'parameters': np.array([1.2, 5, 0.2]),
                        'gama': [0.85, 1.3],
                        'ki': [4, 3],
                        'x0': np.array([10, 8, 7]),
                        'y0': np.array([12])
                        },
                '2': { 
                        'xi': 30,
                        'parameters':np.array([1.2, 4.8, 0.2]),
                        'gama': [0.8, 1.3],
                        'ki': [5, 4],
                        'x0': np.array([12, 7, 2]),
                        'y0': np.array([6])
                        },
                '3': { 
                        'xi': 40,
                        'parameters': np.array([1, 4.8, 0.21]),
                        'gama': [0.75, 1.25],
                        'ki': [3, 3],
                        'x0': np.array([3, 6, 8]),
                        'y0': np.array([8])
                        },
                },
            }
        }
    },
    "r_5":
    {
        "epochs" : 100000,
        "adjacency_matrix" : [[1, 1, 1, 1 ],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1 ],
                                [1, 1, 1, 1 ]], 
        "agent_config":
        {  
            "time_delta": 5e-4,
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
                    "1":[[1, 1.5], [15.5, 16], [24.5, 25]],
                    "2":[[5.5, 5.7], [11, 11.3], [30, 30.5]],
                    "3":[[10, 10.5], [20, 20.5], [35, 35.5]],
                },

                'share': {
                    "p": 0.75,
                    "q": 1.25,
                    'alpha': [200, 500, 0, 0],
                    'beta':[1, 0.5],
                    'po': 5,
                    "a": 0.04
                },

                'private': {
                '0': { 
                        'xi': 10,
                        'parameters': np.array([1, 5, 0.2]),
                        'gama': [0.7, 1.3],
                        'ki': [2, 3],
                        'x0': np.array([12, 10, 6]),
                        'y0': np.array([6])
                        },
                '1': {
                        'xi': 20,
                        'parameters': np.array([1.2, 5, 0.2]),
                        'gama': [0.85, 1.3],
                        'ki': [4, 3],
                        'x0': np.array([7, 8, 7]),
                        'y0': np.array([5])
                        },
                '2': { 
                        'xi': 30,
                        'parameters':np.array([1.2, 4.8, 0.2]),
                        'gama': [0.8, 1.3],
                        'ki': [5, 4],
                        'x0': np.array([6, 7, 2]),
                        'y0': np.array([2])
                        },
                '3': { 
                        'xi': 40,
                        'parameters': np.array([1, 4.8, 0.21]),
                        'gama': [0.75, 1.25],
                        'ki': [3, 3],
                        'x0': np.array([10, 6, 8]),
                        'y0': np.array([7])
                        },
                },
            }
        }
    },

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

def get_scaled_dos(dos_intrvals, scale=0.5):
    scaled_dos = {}
    for k,v in dos_intrvals.items():
        scaled_dos[k] = []
        for interval in v:
            scaled_dos[k].append([interval[0], interval[0]+ (interval[1]-interval[0])*scale])
    return scaled_dos

config["r_2"] = batch_modify_config(config["r_1"],
    ["agent_config.model_config.DoS_interval"],
    [
    get_scaled_dos({
                    "1":[[0.2, 0.3], [2, 2.3], [25.5, 25.6]],
                    "2":[[0.3, 0.4], [10, 10.2], [25.8, 26.0]],
                    "3":[[0.4, 0.5], [2.3, 2.5], [26.2, 26.3]],
                    "4":[[0.5, 0.6], [10, 10.3], [26.5, 26.6]],
                    "5":[[0.6, 0.7], [15, 15.5], [40, 40.5]]
                })
    ]
)

config["r_3"] = batch_modify_config(config["r_1"],
    ["agent_config.model_config.share.alpha", "agent_config.model_config.share.beta"],
    [[100, 0,0,0], [1, 0]]
)

config["r_4"] = batch_modify_config(config["r_1"],
    ["agent_config.model_config.share.alpha", "agent_config.model_config.share.beta", "agent_config.model_config.share.p", "agent_config.model_config.share.q"],
    [[100, 0,0,0], [0.7, 0], 1, 1]
)


config["r_5"] = batch_modify_config(config["r_1"],
    ["agent_config.model_config.DoS_interval"],
    [
    {
                    "1":[],
                    "2":[],
                    "3":[],
                    "4":[],
                    "5":[]
        }
    ]
)

dos1 = [[0.2+i, i+1] for i in range(10)]
dos2 = [[10.2+i, 11+i] for i in range(10)]
dos3 = [[20.2+i, 21+i] for i in range(10)]
dos4 = [[30.2+i, 31+i] for i in range(10)]
dos5 = [[40.2+i, 41+i] for i in range(10)]

config["r_6"] = batch_modify_config(config["r_1"],
    ["agent_config.model_config.share.tau", "agent_config.model_config.DoS_interval"],
    [[0.005,0.005],
    {
        "1":dos1,
        "2":dos2,
        "3":dos3,
        "4":dos4,
        "5":dos5
    }
    ]
)

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
    # scale = 0.1
    A = np.array([[2.08, 0.04, 0.04, 0.04], 
                [0.04, 2.08, 0.04, 0.04], 
                [0.04, 0.04, 2.08, 0.04],
                [0.04, 0.04, 0.04, 2.08]])
    print(A.shape)

    # A = A * scale
    min_eig, max_eig = compute_eigenvalues(A)
    print(f"最小特征值: {min_eig:.6f}")
    print(f"最大特征值: {max_eig:.6f}")
    m_hat = min_eig
    l_hat = np.sqrt((2.08**2) + 3*(0.04**2))
    h_m = l_hat
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


    beta1 = 0.01
    beta2 = 0.01
    alpha1 = 200
    alpha2 = 200
    alpha3 = 200
    alpha4 = 200

    mu = 0.8
    nu = 1.2


    rho1 = 0.05
    rho2 = 0.06
    rho3 = 0.02
    rho4 = 0.04

    c1 = beta1 * N**(1.5-mu)* n**(1-mu) * 2**(1-mu) * l_hat**mu
    print(f"c1: {c1:.6f}")

    c2 = beta2 * N**(1/2) * (2**(nu-2)+2) * (nu) * l_hat**nu
    print(f"c2: {c2:.6f}")

    c3 = beta2 * N**(1/2) * (2**(nu-2)+2) * l_hat * n**(2-nu)
    print(f"c3: {c3:.6f}")

    c4 = N**(1/2) * n**(nu-1) * (2**(nu-2)+2) * l_hat * (nu-1) * n**(2-nu)
    print(f"c4: {c4:.6f}")

    b1 = (n*N**2*(N+1))**(1/2-nu/2) * 2**((2*(1-nu))/((nu+1)**2))
    print(f"b1: {b1:.6f}")

    b2 = (n*N**2*(N+1))**(1-nu) * 2**((1-nu)/(nu**2))
    print(f"b2: {b2:.6f}")

    delta1 = (alpha1* (2*min_eig_M)**((1+mu)/2))/2 - c1 - (c4*rho1+N**(1/2))/(2*rho2)
    print(f"delta1: {delta1:.6f}")

    delta2 = alpha2*b1*(2*min_eig_M)**((1+nu)/2)/2 - c2 - (c4*rho1 + N**(1/2))/(2*rho2)
    print(f"delta2: {delta2:.6f}")

    delta3 = (alpha3 * (2*min_eig_M)**(mu))/2 - c1/(2*rho3*N**(1/2))
    print(f"delta3: {delta3:.6f}")

    delta4 = alpha4*b2*(2*min_eig_M)**(nu)/2 - (c2+c3*(rho1**(1-nu)))*h_m/(2*rho4*N**(1/2))
    print(f"delta4: {delta4:.6f}")

    delta5 = m_hat - (h_m/(2*N**(1/2)))*(c1*rho3+(c2+c3*(rho1**(1-nu)))*rho4 + c4*rho1) - (rho2*(c4*rho1+N**(1/2)))/2
    print(f"delta5: {delta5:.6f}")

    beta_min = np.min([beta1, beta2])
    beta_max = np.max([beta1, beta2])
    
    ch1  = 1/2 * beta_min * ((1+mu)/(beta_max))**((mu+1)/(1+nu))
    print(f"ch1: {ch1:.6f}")

    ch2 = 1/2 * (2*N)**((1-nu)/(1+nu)) * beta_min * ((1+mu)/(beta_max))**((2*nu)/(1+nu))
    print(f"ch2: {ch2:.6f}")

    delta_min_1 = np.min([delta1, delta2])
    delta_min_2 = np.min([delta3, delta4])
    g1 = np.min([delta5*ch1, 2**((1-nu)/(1+nu))* delta5*ch2, 2**((-2*nu)/(1+nu))*(delta_min_1+delta_min_2)])
    print(f"g1: {g1:.6f}")
    
    g2 = c1 + (c4*rho1 + N**(1/2))/(2*rho2) + c2 + (c4*rho1 + N**(1/2))/(2*rho2) + np.max([c1*h_m/(2*rho3*N**(1/2)), (c2 + c3*(rho1**(1-nu)))*h_m/(2*rho4*N**(1/2))])
    print(f"g2: {g2:.6f}")

    print("g2/g1:", g2/g1)

    epsilon = 0.2
    rd = 500
    td = 25
    seek_optimal_dos(g1, epsilon, g2)
    value = 4*rd*td*g1*epsilon - 4*td*(g1*epsilon + g2*epsilon*(epsilon+2)) - 3.14*rd*(2+epsilon)
    print( 4*rd*td*g1*epsilon, 4*td*(g1*epsilon + g2*epsilon*(epsilon+2)), 3.14*(2+epsilon))
    print(f"value: {value:.6f}")

 

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
    # try:
    #     result = parameter_calculate(index)
    #     print(f"Parameters for {index}: {result}")
    # except ValueError as e:
    #     print(e)  # Handle the case where the index is not found