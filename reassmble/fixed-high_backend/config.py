import numpy as np

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
}
}