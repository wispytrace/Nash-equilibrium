import numpy as np
import copy

config = {
    "0":
    {
        "epochs" : 200000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 1],
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
                "scale_dict": {'ki': 2, 'k_i_tilde': 2, 'eta': 2, 'zeta': 2},
                "N": 6,
                "memory" : {"x": np.zeros((3, 3)), "y": np.zeros(3), "z": np.zeros((6, 3))},
                
                'share': {
                    "pos": np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 3],[-2, 0, 3]]),
                    "p": 0.6,
                    "q": 1.4,
                    "pos_c": np.array([0, 0.5, 2])
                },

                'private': {
                '0': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 3, 5],
                       'order': 1,
                       'x0': np.array([-2, 2, 0])
                     },
                '1': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [4, 3, 5],
                       'k_i_tilde': [5, 4, 5],
                       'order': 1,
                       'x0': np.array([-2, -2, 0])
                     },
                '2': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 4, 5],
                       'order': 2,
                       'x0': np.array([2, -2, 0])
                     },
                '3': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [3, 4, 5]*3,
                       'k_i_tilde': [5, 6, 5],
                       'order': 2,
                       'x0': np.array([2, 2, 0])
                     },
                '4': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [2, 4, 6],
                       'k_i_tilde': [3, 5, 7],
                       'order': 3,
                       'x0': np.array([3, -1, 0])
                     },
                '5': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 5,
                       'gama': [0.7, 1.3],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [4, 6, 7],
                       'order': 3,
                       'x0': np.array([-3, 1, 0])
                     },
                },
            }
        }
    },
    "1":
    {
        "epochs" : 200000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 1],
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
                "scale_dict": {'ki': 2, 'k_i_tilde': 2, 'eta': 2, 'zeta': 2},
                "N": 6,
                "memory" : {"x": np.zeros((3, 3)), "y": np.zeros(3), "z": np.zeros((6, 3))},
                
                'share': {
                    "pos": np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 3],[-2, 0, 3]]),
                    "p": 0.6,
                    "q": 1.4,
                    "pos_c": np.array([0, 0.5, 2])
                },

                'private': {
                '0': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 6,
                       'gama': [0.8, 1.2],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 3, 5],
                       'order': 1,
                       'x0': np.array([-2, 2, 0])
                     },
                '1': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 7,
                       'eta': 8,
                       'gama': [0.85, 1.15],
                       'ki': [4, 3, 5],
                       'k_i_tilde': [5, 4, 5],
                       'order': 1,
                       'x0': np.array([-2, -2, 0])
                     },
                '2': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 9,
                       'eta': 10,
                       'gama': [0.8, 1.3],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 4, 5],
                       'order': 2,
                       'x0': np.array([2, -2, 0])
                     },
                '3': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 8,
                       'eta': 9,
                       'gama': [0.75, 1.25],
                       'ki': [3, 4, 5],
                       'k_i_tilde': [5, 6, 5],
                       'order': 2,
                       'x0': np.array([2, 2, 0])
                     },
                '4': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 6,
                       'eta': 7,
                       'gama': [0.65, 1.35],
                       'ki': [2, 4, 6],
                       'k_i_tilde': [3, 5, 7],
                       'order': 3,
                       'x0': np.array([3, -1, 0])
                     },
                '5': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 10,
                       'eta': 10,
                       'gama': [0.6, 1.4],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [4, 6, 7],
                       'order': 3,
                       'x0': np.array([-3, 1, 0])
                     },
                },
            }
        }
    },
    "simu":
    {
        "epochs" : 30000,
        "adjacency_matrix" : [[0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0],],
        "agent_config":
        {  
            "time_delta": 1e-3,
            "model": "fixed_high_order",
            "record_interval": 50,
            "record_flag": 1,
            
            "model_config": 
            {
                "action": "self.memory['x'][:2,:2]",
                "scale_dict": {'ki': 2, 'k_i_tilde': 2, 'eta': 0.2, 'zeta': 0.2, 'alpha': 0.1, 'beta': 0.1},
                "N": 6,
                "memory" : {"x": np.zeros((3, 3)), "y": np.zeros(3), "z": np.zeros((3, 3))},
                
                'share': {
                    "pos": np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0]]),
                    "p": 0.6,
                    "q": 1.4,
                    "pos_c": np.array([0, 0.5, 0])
                },

                'private': {
                '0': { 'alpha': [100, 120, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 5,
                       'eta': 6,
                       'gama': [0.8, 1.2],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 3, 5],
                       'order': 2,
                       'x0': np.array([-2, 2, 0])
                     },
                '1': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 7,
                       'eta': 8,
                       'gama': [0.85, 1.15],
                       'ki': [4, 3, 5],
                       'k_i_tilde': [5, 4, 5],
                       'order': 2,
                       'x0': np.array([-2, -2, 0])
                     },
                '2': { 'alpha': [100, 125, 110, 130],
                       'beta':[1.8, 1.2],
                       'zeta': 9,
                       'eta': 10,
                       'gama': [0.8, 1.3],
                       'ki': [2, 3, 5],
                       'k_i_tilde': [3, 4, 5],
                       'order': 2,
                       'x0': np.array([2, -2, 0])
                     },
            }
        }
    },
},
    "r_0":
    {
        "epochs" : 200000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 1],
                                [1, 0, 1, 0, 0, 0],
                                [0, 1, 0, 1, 0, 0],
                                [0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 1, 0, 1],
                                [1, 0, 0, 0, 1, 0]], 
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed_high_order",
            "record_interval": 100,
            "record_flag": 1,

            "model_config": 
            {
                "scale_dict": {'ki': 1, 'k_i_tilde': 1, 'eta': 1, 'zeta': 1},
                "N": 6,
                "memory" : {"x": np.zeros((3, 3)), "y": np.zeros(3), "z": np.zeros((6, 3))},
                "cost_scale": 0.1,
                "is_finite": False,
                "initial_scale": 1,
                
                'share': {
                    "pos": np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 3],[-2, 0, 3]]),
                    "p": 0.5,
                    "q": 1.2,
                    "pos_c": np.array([0, 0.5, 2]),
                    "p1": 0.5,
                    "q1": 1.5
                },

                'private': {
                '0': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 5,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 3, 5],
                        'order': 1,
                        'x0': np.array([-2, 2, 0])
                        },
                '1': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 8,
                        'gama': [0.7, 1.3],
                        'ki': [4, 3, 5],
                        'k_i_tilde': [5, 4, 5],
                        'order': 1,
                        'x0': np.array([-2, -2, 0])
                        },
                '2': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 4, 5],
                        'order': 2,
                        'x0': np.array([2, -2, 0])
                        },
                '3': { 'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 10,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [3, 4, 5],
                        'k_i_tilde': [5, 6, 5],
                        'order': 2,
                        'x0': np.array([2, 2, 0])
                        },
                '4': { 'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 4,
                        'eta': 5,
                        'gama': [0.7, 1.3],
                        'ki': [2, 4, 6],
                        'k_i_tilde': [3, 5, 7],
                        'order': 3,
                        'x0': np.array([3, -1, 0])
                        },
                '5': { 'alpha':  [220, 250, 1100, 1100],
                        'beta': [3.5, 0.5],
                        'zeta': 8,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [4, 6, 7],
                        'order': 3,
                        'x0': np.array([-3, 1, 0])
                        },
                },
            }
        }
    },
"r_1":
    {
        "epochs" : 300000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], 
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed_high_order",
            "record_interval": 100,
            "record_flag": 1,

            "model_config": 
            {
                "scale_dict": {'ki': 1, 'k_i_tilde': 1, 'eta': 1, 'zeta': 1},
                "N": 12,
                "memory" : {"x": np.zeros((3, 3)), "y": np.zeros(3), "z": np.zeros((12, 3))},
                "cost_scale": 0.1,
                "is_finite": False,
                "initial_scale": 1,
                
                'share': {
                    "pos": np.array([[-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [2, 0, 3], [-2, 0, 3],
                                     [-3, 0, 0], [0, -3, 0], [3, 0, 0], [0, 3, 0], [0, 2, 3], [0, -2, 3]]),
                    "p": 0.5,
                    "q": 1.2,
                    "pos_c": np.array([0, 0.5, 2]),
                    "p1": 0.5,
                    "q1": 1.5
                },

                'private': {
                '0': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 5,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 3, 5],
                        'order': 1,
                        'x0': np.array([-2, 2, 0])
                        },
                '1': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 8,
                        'gama': [0.7, 1.3],
                        'ki': [4, 3, 5],
                        'k_i_tilde': [5, 4, 5],
                        'order': 1,
                        'x0': np.array([-2, -2, 0])
                        },
                '2': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 4, 5],
                        'order': 2,
                        'x0': np.array([2, -2, 0])
                        },
                '3': { 'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 10,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [3, 4, 5],
                        'k_i_tilde': [5, 6, 5],
                        'order': 2,
                        'x0': np.array([2, 2, 0])
                        },
                '4': { 'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 4,
                        'eta': 5,
                        'gama': [0.7, 1.3],
                        'ki': [2, 4, 6],
                        'k_i_tilde': [3, 5, 7],
                        'order': 3,
                        'x0': np.array([3, -1, 0])
                        },
                '5': { 'alpha':  [220, 250, 1100, 1100],
                        'beta': [3.5, 0.5],
                        'zeta': 8,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [4, 6, 7],
                        'order': 3,
                        'x0': np.array([-3, 1, 0])
                        },
                # --------- 新增的 6-11 节点 ---------
                '6': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 5,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 3, 5],
                        'order': 1,
                        'x0': np.array([-4, 4, 0])  # 与 0 节点不同
                        },
                '7': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 8,
                        'gama': [0.7, 1.3],
                        'ki': [4, 3, 5],
                        'k_i_tilde': [5, 4, 5],
                        'order': 1,
                        'x0': np.array([-4, -4, 0]) # 与 1 节点不同
                        },
                '8': { 'alpha': [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 7,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [3, 4, 5],
                        'order': 2,
                        'x0': np.array([4, -4, 0])  # 与 2 节点不同
                        },
                '9': { 'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 10,
                        'eta': 6,
                        'gama': [0.7, 1.3],
                        'ki': [3, 4, 5],
                        'k_i_tilde': [5, 6, 5],
                        'order': 2,
                        'x0': np.array([4, 4, 0])   # 与 3 节点不同
                        },
                '10': {'alpha':  [220, 250, 1100, 1100],
                        'beta':[3.5, 0.5],
                        'zeta': 4,
                        'eta': 5,
                        'gama': [0.7, 1.3],
                        'ki': [2, 4, 6],
                        'k_i_tilde': [3, 5, 7],
                        'order': 3,
                        'x0': np.array([5, -2, 0])  # 与 4 节点不同
                        },
                '11': {'alpha':  [220, 250, 1100, 1100],
                        'beta': [3.5, 0.5],
                        'zeta': 8,
                        'eta': 9,
                        'gama': [0.7, 1.3],
                        'ki': [2, 3, 5],
                        'k_i_tilde': [4, 6, 7],
                        'order': 3,
                        'x0': np.array([-5, 2, 0])  # 与 5 节点不同
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

# 用法示例

config['finite_1'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta"], 
    [True, 10, 1e-4]
)

config['finite_2'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta"], 
    [True, 20, 3e-4]
)

config['finite_3'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta"], 
    [True, 30, 3e-4]
)

config['finite_4'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [True, 40, 3e-4, 400000]
)

config['finite_5'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [True, 50, 4e-4, 400000]
)

config['finite_6'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [True, 60, 5e-4, 400000]
)

config['fixed_1'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale"], 
    [False, 10]
)

config['fixed_2'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale"], 
    [False, 20]
)

config['fixed_3'] = batch_modify_config(
    config["r_1"], 

    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale"], 
    [False, 30]
)

config['fixed_4'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [False, 40, 1e-4, 200000]
)

config['fixed_5'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [False, 50, 1e-4, 300000]
)

config['fixed_6'] = batch_modify_config(
    config["r_1"], 
    ["agent_config.model_config.is_finite", "agent_config.model_config.initial_scale", "agent_config.time_delta", "epochs"], 
    [False, 60, 1e-4, 300000]
)


print(config['fixed_1'])




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
    

# def parameter_calculate(index):
#     """
#     Calculate the parameters for the given index.
#     """
#     N = 12
#     m = 36
#     scale = 0.10
#     A = np.array([[7/6, 1/6, 1/6, 1/6,  1/6, 1/6], 
#                   [1/6, 7/6, 1/6, 1/6, 1/6, 1/6], 
#                   [1/6, 1/6, 7/6, 1/6, 1/6, 1/6],
#                   [1/6, 1/6, 1/6, 7/6, 1/6, 1/6], 
#                   [1/6, 1/6, 1/6, 1/6, 7/6, 1/6],
#                   [1/6, 1/6, 1/6, 1/6, 1/6, 7/6]])

#     A = A * scale
#     min_eig, max_eig = compute_eigenvalues(A)
#     print(f"最小特征值: {min_eig:.6f}")
#     print(f"最大特征值: {max_eig:.6f}")
#     m_hat = min_eig
#     l_hat = np.sqrt((5/36+49/36)*scale**2)
#     h_m = max_eig
#     print(f"m_hat: {m_hat:.6f}, l_hat: {l_hat:.6f}, h_m: {h_m:.6f}")
#     adjacency_matrix = np.array([[0, 1, 0, 0, 0, 1],
#                   [1, 0, 1, 0, 0, 0],
#                   [0, 1, 0, 1, 0, 0],
#                   [0, 0, 1, 0, 1, 0],
#                   [0, 0, 0, 1, 0, 1],
#                   [1, 0, 0, 0, 1, 0]])
#     D = np.diag(np.sum(adjacency_matrix, axis=1))
#     L = D - adjacency_matrix
#     digA = np.diag(adjacency_matrix.flatten())
#     I = np.eye(N)
#     M = np.kron(L, I) + digA
#     min_eig_M, max_eig_M = compute_eigenvalues(M)
#     print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
#     print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")
    
#     rho1 = 0.006
#     rho2 = 0.004
#     rho3 = 0.008
#     rho4 = 0.01

#     varepsilon = m_hat / 2
#     beta1 = 3.5
#     beta2 = 0.5
#     alpha1 = 200
#     alpha2 = 250
#     alpha3 = 320
#     alpha4 = 1200

#     p = 3/5
#     q = 9/7

#     # rho1 = 0.006
#     # rho2 = 0.005
#     # rho3 = 0.01
#     # rho4 = 0.02

#     c1 = beta1* 2 **(1-p) * (l_hat**p) * (min_eig_M**(-p)) * (N**(1-p/2)) * (m**(1/2-p/2)) 
#     print(f"c1: {c1:.6f}")
#     c2 = np.sqrt(N)*beta2 * (min_eig_M**(-q)) * (2**(q-2)+2) * (q*l_hat**q + rho1**(1-q)*l_hat* (m**(1-q/2)))
#     print(f"c2: {c2:.6f}")
#     c3 = np.sqrt(N)*(2**(q-2)+2)*rho1*l_hat*(q-1)* m**(q/2-1/2)* (m**(1-q/2))
#     print(f"c3: {c3:.6f}")

#     sigma1 = m_hat - h_m*(c1*rho3 + c2*rho4 + c3)/(2*np.sqrt(N)) - rho2*(c3+np.sqrt(N))/2
#     print(f"sigma1: {sigma1:.6f}")
#     sigma2 = alpha1 - c1 - (c3+np.sqrt(N))/(4*rho2)
#     print(f"sigma2: {sigma2:.6f}")
#     sigma3 = alpha2 * (m*N)**(1/2 - q/2) - c2 - (c3+np.sqrt(N))/(4*rho3)
#     print(f"sigma3: {sigma3:.6f}")
#     sigma4 = alpha3 - c1*h_m/(2*rho3*np.sqrt(N)) - (c3+np.sqrt(N))/(4*rho2)
#     print(f"sigma4: {sigma4:.6f}")
#     sigma5 = alpha4 * (m*N)**(1-q) - c2*h_m/(2*rho4*np.sqrt(N)) - (c3+np.sqrt(N))/(4*rho2)
#     print(f"sigma5: {sigma5:.6f}")
def compute_eigenvalues(matrix):
    """
    计算矩阵的特征值并返回实部的最小值和最大值
    （补充原代码缺失的辅助函数）
    """
    eigs = np.linalg.eigvals(matrix)
    # 取实部以避免复数比较引发警告或错误
    real_eigs = np.real(eigs)
    print(real_eigs)
    return np.min(real_eigs), np.max(real_eigs)

def parameter_calculate(index):
    """
    Calculate the parameters for the given index (Modified for 12 nodes).
    """
    N = 12
    m = N * N  # 144 (原代码中 6x6 矩阵时 m=36)
    scale = 0.05
    
    # 1. 扩展系统矩阵 A 为 12x12 (对角线 7/6，其余 1/6)
    A = np.ones((N, N)) * (1/12)
    np.fill_diagonal(A, 13/12)
    A = A * scale 
    
    min_eig, max_eig = compute_eigenvalues(A)
    print(f"最小特征值: {min_eig:.6f}")
    print(f"最大特征值: {max_eig:.6f}")
    
    m_hat = min_eig
    # 2. 修改 l_hat: 12个节点时每行包含 1个(7/6) 和 11个(1/6)
    l_hat = np.sqrt((11/36 + 49/36) * scale**2)
    h_m = max_eig
    print(f"m_hat: {m_hat:.6f}, l_hat: {l_hat:.6f}, h_m: {h_m:.6f}")
    
    # 3. 扩展邻接矩阵为 12x12 的无向环形图 (Ring/Cycle Graph)
    adjacency_matrix = np.zeros((N, N))
    for i in range(N):
        adjacency_matrix[i, (i - 1) % N] = 1
        adjacency_matrix[i, (i + 1) % N] = 1
        
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    digA = np.diag(adjacency_matrix.flatten())
    I = np.eye(N)
    
    # 现在 L 是 12x12，I 是 12x12，kron后为 144x144。digA 也是 144x144，维度完美匹配。
    compute_eigenvalues(L)  # 输出 L 的特征值以验证
    M = np.kron(L, I) + digA
    min_eig_M, max_eig_M = compute_eigenvalues(M)
    print(f"矩阵 M 的最小特征值: {min_eig_M:.6f}")
    print(f"矩阵 M 的最大特征值: {max_eig_M:.6f}")
    
    rho1 = 0.006
    rho2 = 0.004
    rho3 = 0.008
    rho4 = 0.01

    varepsilon = m_hat / 2
    beta1 = 3.5
    beta2 = 0.5
    alpha1 = 300
    alpha2 = 500
    alpha3 = 450
    alpha4 = 1600

    p = 3/5
    q = 1.2

    # 计算常数 c1, c2, c3
    c1 = beta1 * 2**(1-p) * (l_hat**p) * (min_eig_M**(-p)) * (N**(1-p/2)) * (m**(1/2-p/2)) 
    print(f"c1: {c1:.6f}")
    c2 = np.sqrt(N) * beta2 * (min_eig_M**(-q)) * (q*l_hat**q + rho1**(1-q) * l_hat * (m**(1-q/2)))
    print(f"c2: {c2:.6f}")
    c3 = np.sqrt(N)  * rho1 * l_hat * (q-1) * m**(q/2-1/2) * (m**(1-q/2))
    print(f"c3: {c3:.6f}")

    # 计算收敛/控制参数 sigma
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

if __name__ == "__main__":
    # Example usage
    index = "fixed_1"
    try:
        result = parameter_calculate(index)
        print(f"Parameters for {index}: {result}")
    except ValueError as e:
        print(e)  # Handle the case where the index is not found