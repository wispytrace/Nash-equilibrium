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
    adjacency_matrix = np.array([[1, 1, 1, 1, 1],
                  [1, 1, 1, 1,1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1]])
    
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    digA = np.diag(adjacency_matrix.flatten())
    I = np.eye(N)
    M = np.kron(L, I) + digA
    min_eig_M, max_eig_M = compute_eigenvalues(M)
    print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
    print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")



c = 0.726
p = 0.85
q = 1.15
phic = 0.174

delta = 2*5
eta = 2*5

C5 = delta/((1+c)**(1-p))- delta*c/((1-c)**(1-p))
C6 = eta/((1-c)**(1-q)) + eta*c/((1+c)**(1-q))

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