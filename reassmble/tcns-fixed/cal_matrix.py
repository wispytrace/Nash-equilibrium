import numpy as np
from scipy.linalg import null_space

def calculate_S_min_eigenvalue_full_M(adj_matrix):
    """
    计算 S + S^T 的最小特征值。
    定义 M 为由邻接矩阵所有 N^2 个元素构成的对角矩阵。
    
    参数:
    adj_matrix (np.array): 图的邻接矩阵 (NxN)
    
    返回:
    min_eig (float): S + S^T 的最小特征值
    debug_info (dict): 包含中间计算结果
    """
    N = adj_matrix.shape[0]
    
    # 1. 构建拉普拉斯矩阵 L
    # L = D - A (标准定义，行和为0)
    # 计算度矩阵 D (Row sum)
    in_degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(in_degrees)
    L = D - adj_matrix
    
    # 2. 计算左特征向量 xi (对应特征值 0)
    # xi.T @ L = 0
    eigenvalues, eigenvectors = np.linalg.eig(L.T)
    idx = np.argmin(np.abs(eigenvalues))
    xi = np.real(eigenvectors[:, idx])
    
    # 归一化 xi (保证 sum=1 且为正)
    xi = np.abs(xi) / np.sum(np.abs(xi))
    Xi = np.diag(xi)
    
    # 3. 构建 M 矩阵 (N^2 x N^2)
    # 将邻接矩阵展平，形成对角线元素
    # flatten() 默认是行优先 (Row-major): [row1, row2, ...]
    m_values = adj_matrix.flatten()
    M = np.diag(m_values)
    
    # 4. 构建 Kronecker 积定义的 S
    # S = (Xi L) ⊗ I_N + (Xi ⊗ I_N) M
    I_N = np.eye(N)
    
    # 第一项: (Xi L) ⊗ I_N
    Xi_L = Xi @ L
    Term1 = np.kron(Xi_L, I_N)
    
    # 第二项: (Xi ⊗ I_N) M
    # Xi ⊗ I_N 是一个 block diagonal 矩阵
    Xi_kron_I = np.kron(Xi, I_N)
    Term2 = Xi_kron_I @ M
    
    # S = Term1 + Term2
    S = Term1 + Term2
    
    # 5. 计算 S + S^T 及其最小特征值
    S_plus_ST = S + S.T
    
    # 计算特征值 (使用 eigh 或 eigvalsh 因为是实对称矩阵)
    final_eigs = np.linalg.eigvalsh(S_plus_ST)
    min_eig = np.min(final_eigs)
    
    return min_eig, {
        "xi": xi,
        "L": L,
        "S_dim": S.shape,
        "all_eigenvalues": final_eigs
    }
# ==========================================
# 示例运行 (您可以修改这里的 A)
# ==========================================
if __name__ == "__main__":
    # 示例：一个简单的 3 节点强连通循环图
    # 1->2, 2->3, 3->1
    A_example = np.array([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]])
    
    # 假设每个节点的参数 mu (对应 M)
    mu_params = np.diag(A_example)
    print(mu_params)
    try:
        min_lambda, info = calculate_S_min_eigenvalue_full_M(A_example)
        
        print("计算结果:")
        print("-" * 30)
        print(f"输入邻接矩阵 A:\n{A_example}")
        print(f"计算得到的左特征向量 xi: {info['xi']}")
        print("-" * 30)
        print(f"矩阵 (S + S^T) 的最小特征值: {min_lambda:.6f}")
        print("-" * 30)
        
        # 验证引理：Xi L + L^T Xi 是否半正定 (特征值应 >= 0)
        Q = info['xi'] * info['L'] + info['L'].T * np.diag(info['xi']) # 简写逻辑
        Q_eigs = np.linalg.eigvalsh(np.diag(info['xi']) @ info['L'] + info['L'].T @ np.diag(info['xi']))
        print(f"验证 Lemma 4 (Xi L + L^T Xi 的特征值): {Q_eigs}")
        
    except Exception as e:
        print(f"发生错误: {e}")