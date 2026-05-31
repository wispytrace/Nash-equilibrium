import numpy as np
import copy

config = {
    "r_0": {
        "epochs": 80000,
        # 8节点全连通图，保证分布式协同信息交换畅通
        "adjacency_matrix": np.ones((8, 8)) - np.eye(8),
        "agent_config": {
            "time_delta": 2e-4,
            "model": "air_ground_protection",
            "record_interval": 100, # 每100步记录一次，避免画图数据量过大卡顿
            "record_flag": 1,
            "model_config": {
                "N": 8,
                "memory": {
                    "x": np.zeros(3),
                    "v": np.zeros(3),
                    "y": np.zeros(3),
                    "z": np.zeros((8, 3)),
                    "vr": np.zeros(3),
                    'NE': np.zeros(3),
                    'pg': np.zeros(3),
                },
                "uav_physics": {
                    '0': {'m': 1.5, 'd': [0.15, 0.15, 0.25]},
                    '1': {'m': 1.2, 'd': [0.10, 0.10, 0.18]},
                    '2': {'m': 1.8, 'd': [0.20, 0.20, 0.35]},
                    '3': {'m': 1.4, 'd': [0.12, 0.12, 0.20]},
                },
                'share': {
                    "p": 0.7,
                    "q": 1.2,
                    'alpha': [50, 50, 40, 40],
                    'beta': [2.5, 1.5],
                    'mu': 0.7,
                    'nu': 1.5,
                    'gama': 35,
                    # 给定错落有致的初始位置，方便观察在空中的收敛过程
                    "init_x": [
                        [  6.25,   6.84,  0.0],
                        [ -6.41,   6.52,  0.0],
                        [ -6.44,  -6.22,  0.0],
                        [  6.95,  -6.88,  0.0],
                        [  5.12,   5.76,  3.0],
                        [ -5.88,   5.56,  3.0],
                        [ -5.61,  -4.99,  3.0],
                        [  4.77,  -5.42,  3.0]
                    ]
                },
            },
        }
    },

"r_1": {
        "epochs": 80000,
        # 8节点全连通图，保证分布式协同信息交换畅通
        "adjacency_matrix": np.ones((8, 8)) - np.eye(8),
        "agent_config": {
            "time_delta": 2e-4,
            "model": "air_ground_protection",
            "record_interval": 100, # 每100步记录一次，避免画图数据量过大卡顿
            "record_flag": 1,
            "model_config": {
                "N": 8,
                "memory": {
                    "x": np.zeros(3),
                    "v": np.zeros(3),
                    "y": np.zeros(3),
                    "z": np.zeros((8, 3)),
                    "vr": np.zeros(3),
                    'NE': np.zeros(3),
                    'pg': np.zeros(3),
                },
                "uav_physics": {
                    '0': {'m': 1.5, 'd': [0.15, 0.15, 0.25]},
                    '1': {'m': 1.2, 'd': [0.10, 0.10, 0.18]},
                    '2': {'m': 1.8, 'd': [0.20, 0.20, 0.35]},
                    '3': {'m': 1.4, 'd': [0.12, 0.12, 0.20]},
                },
                'share': {
                    "p": 0.7,
                    "q": 1.2,
                    'alpha': [50, 50, 40, 40],
                    'beta': [2.5, 1.5],
                    'mu': 0.7,
                    'nu': 1.5,
                    'gama': 35,
                    # 给定错落有致的初始位置，方便观察在空中的收敛过程
                    "init_x": [
                        [  6.25,   6.84,  0.0],
                        [ -6.41,   6.52,  0.0],
                        [ -6.44,  -6.22,  0.0],
                        [  6.95,  -6.88,  0.0],
                        [  5.12,   5.76,  3.0],
                        [ -5.88,   5.56,  3.0],
                        [ -5.61,  -4.99,  3.0],
                        [  4.77,  -5.42,  3.0]
                    ]
                },
            },
        }
    },

"r_2": {
        "epochs": 80000,
        # 8节点全连通图，保证分布式协同信息交换畅通
        "adjacency_matrix": np.ones((8, 8)) - np.eye(8),
        "agent_config": {
            "time_delta": 2e-4,
            "model": "air_ground_protection",
            "record_interval": 100, # 每100步记录一次，避免画图数据量过大卡顿
            "record_flag": 1,
            "model_config": {
                "N": 8,
                "memory": {
                    "x": np.zeros(3),
                    "v": np.zeros(3),
                    "y": np.zeros(3),
                    "z": np.zeros((8, 3)),
                    "vr": np.zeros(3),
                    'NE': np.zeros(3),
                    'pg': np.zeros(3),
                },
                "uav_physics": {
                    '0': {'m': 1.5, 'd': [0.15, 0.15, 0.25]},
                    '1': {'m': 1.2, 'd': [0.10, 0.10, 0.18]},
                    '2': {'m': 1.8, 'd': [0.20, 0.20, 0.35]},
                    '3': {'m': 1.4, 'd': [0.12, 0.12, 0.20]},
                },
                'share': {
                    "p": 1,
                    "q": 1,
                    'alpha': [50, 0, 50, 0],
                    'beta': [1, 0],
                    'mu': 1,
                    'nu': 1,
                    'gama': 25,
                    # 给定错落有致的初始位置，方便观察在空中的收敛过程
                    "init_x": [
                        [  6.25,   6.84,  0.0],
                        [ -6.41,   6.52,  0.0],
                        [ -6.44,  -6.22,  0.0],
                        [  6.95,  -6.88,  0.0],
                        [  5.12,   5.76,  3.0],
                        [ -5.88,   5.56,  3.0],
                        [ -5.61,  -4.99,  3.0],
                        [  4.77,  -5.42,  3.0]
                    ]
                },
            },
        }
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
    N = 4
    m = 3
    min_eig = 2
    max_eig = 2 + 2/N
    print(f"最小特征值: {min_eig:.6f}")
    print(f"最大特征值: {max_eig:.6f}")
    adjacency_matrix = np.array( [[0, 1, 0, 1 ],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1 ],
                                [1, 0, 1, 0 ]])
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    digA = np.diag(adjacency_matrix.flatten())
    I = np.eye(N)
    M = np.kron(L, I) + digA
    min_eig_M, max_eig_M = compute_eigenvalues(M)
    print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
    print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")
    
    # rho1 = 0.006
    # rho2 = 0.01
    # rho3 = 0.01
    # rho4 = 0.05

    # beta1 = 1
    # beta2 = 0.5
    # alpha1 = 150
    # alpha2 = 250
    alpha3 = 520
    alpha4 = 800

    xi_min = min_eig_M  

    rho1 = 0.006
    rho2 = 0.005
    rho3 = 0.005
    rho4 = 0.02

    beta1 = 1
    beta2 = 0.35
    alpha1 = 1500
    alpha2 = 2000
    # alpha3 = 520
    # alpha4 = 800

    mu = 0.75
    nu = 1.20

    # ==========================================
    # 计算 c1 - c4
    # ==========================================
    c1 = beta1 * (N**(1.5 - mu)) * (m**((1 - mu) / 2)) * (2**(1 - mu)) * ((2 + 2/N)**mu) * (xi_min**(-mu))
    c2 = beta2 * (N**0.5) * nu * ((2 + 2/N)**nu) * (xi_min**(-nu))
    c3 = beta2 * (N**(1.5 - nu/2)) * (m**(1 - nu/2)) * (2 + 2/N) * (xi_min**(-nu))
    c4 = beta2 * (N**(1.5 - nu/2)) * (m**(1 - nu/2)) * (2 + 2/N) * (nu - 1)

    print("\n--- 常数 c1~c4 计算结果 ---")
    print(f"c1 = {c1:.6f}")
    print(f"c2 = {c2:.6f}")
    print(f"c3 = {c3:.6f}")
    print(f"c4 = {c4:.6f}")

    # ==========================================
    # 计算 sigma1 - sigma4
    # ==========================================
    # 式 (17)
    sigma1 = alpha1 - (beta1 * (N**0.5) * (rho2**(-1/mu))) / (mu + 1) - c1 - (mu * (rho4**(-mu)) * c1) / (mu + 1)/2

    # 式 (18)
    # ⚠️ 勘误提示：论文原图中式(18)最后一项的分子印刷为 \mu \rho_4^{-\nu}。
    # 根据系统对称性(对比式17)与数学逻辑，这里的 \mu 极大概率是 \nu 的笔误。
    # 但为保证对原图的绝对忠实，这里严格按照图片打出的 \mu 编写。如果算法发散，可以尝试把这里的 mu 换成 nu。
    term18_1 = alpha2 * ((m * (N**2))**((1 - nu) / 2))
    term18_2 = (beta2 * (N**0.5) * (rho2**(-1/nu))) / (nu + 1)
    term18_3 = c2
    term18_4 = c3 * (rho1**(1 - nu))
    term18_5 = (c4 * rho1 * (rho3**(-nu))) / (nu + 1)
    term18_6 = (nu * (rho4**(-nu)) * (c2 + c3 * (rho1**(1 - nu)))) / (nu + 1)/2
    sigma2 = term18_1 - term18_2 - term18_3 - term18_4 - term18_5 - term18_6

    # 式 (19)
    sigma3 = beta1 - (c1 * rho4) / (mu + 1)/2 - (beta1 * (N**0.5) * mu * rho2 * ((m * N)**((1 - mu) / 2))) / (mu + 1)

    # 式 (20)
    term20_1 = beta2 * ((m * N)**((1 - nu) / 2))
    term20_2 = ((c2 + c3 * (rho1**(1 - nu))) * rho4) / (nu + 1)/2
    term20_3 = c4 * rho1/2
    term20_4 = (beta2 * (N**0.5) * nu * rho2) / (nu + 1)
    term20_5 = (nu * c4 * rho1 * rho3) / (nu + 1)
    sigma4 = term20_1 - term20_2 - term20_3 - term20_4 - term20_5

    print("\n--- 参数 sigma1~sigma4 计算结果 ---")
    print(f"sigma1 = {sigma1:.6f}")
    print(f"sigma2 = {sigma2:.6f}")
    print(f"sigma3 = {sigma3:.6f}")
    print(f"sigma4 = {sigma4:.6f}")

    # 验证收敛条件
    print("\n--- 收敛条件验证 ---")
    print(f"sigma1 > 0: {sigma1 > 0}")
    print(f"sigma2 > 0: {sigma2 > 0}")
    print(f"sigma3 > 0: {sigma3 > 0}")
    print(f"sigma4 > 0: {sigma4 > 0}")
    
if __name__ == "__main__":
    # Example usage
    index = "fixed_1"
    try:
        result = parameter_calculate(index)
        print(f"Parameters for {index}: {result}")
    except ValueError as e:
        print(e)  # Handle the case where the index is not found
