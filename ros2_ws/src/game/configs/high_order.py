import numpy as np

config = {
    "0":
    {
        "epochs" : 150000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "high_order",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,                
                'share': {
                    "init_value_x": np.array([2, 0, 8 ,6 , 4]),
                    "init_value_v": np.zeros((5, 5)),
                    "init_value_z": np.zeros((5, 5)),
                    "init_value_uz": np.zeros((5, 5)),
                    "init_value_uv": np.zeros((5, 5)),


                    'a': 0.02,
                    'b': 2.5,

                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.5,
                    "q": 1.5,
                    "lipsthitz": 5,
                    "delay": 2,
                    "max_delay": 100,
                    "tau": 3.5,
                    "d0": 0.5,
                    "di": 0.5,
                    "scale_dict": {'alpha': 1, 'beta': 1, 'eta': 1, 'h1': 1, 'h2':1, "tau": 1, "eta_max": 1}
                },

                'private': {
                '0': { 'alpha': [100, 125, 110], 'beta':[125, 105, 100], 'eta': [2, 2, 1.5], 'h1': 1.6, 'h2': 2.6,
                      "r":1.5 },
                '1': {'alpha': [110, 120, 125], 'beta':[115, 100, 110], 'eta': [2.1, 2.2, 1.6], 'h1': 1.7, 'h2': 2.1,
                      "r":2.5},
                '2': {'alpha': [115, 115, 105], 'beta':[110, 110, 125], 'eta': [2.5, 2.6, 1.7], 'h1': 1.8, 'h2': 2.2,
                      "r":3.5},
                '3': {'alpha': [120, 110, 100], 'beta':[100, 120, 120], 'eta': [2.3, 2.4, 1.8], 'h1': 1.9, 'h2': 2.5,
                      "r":4.5},
                '4': {'alpha': [125, 100, 115], 'beta':[105, 115, 105], 'eta': [2.7, 2.8, 1.9], 'h1': 2.0, 'h2': 2.3,
                      "r":5.5},
                },
            }
        }
    },
    "1":
    {
        "epochs" : 150000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "high_order",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,                
                'share': {
                    "init_value_x": np.array([2, 0, 8 ,6 , 4]),
                    "init_value_v": np.zeros((5, 5)),
                    "init_value_z": np.zeros((5, 5)),
                    "init_value_uz": np.zeros((5, 5)),
                    "init_value_uv": np.zeros((5, 5)),


                    'a': 0.02,
                    'b': 2.5,

                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.5,
                    "q": 1.5,
                    "lipsthitz": 5,
                    "delay": 2,
                    "max_delay": 1000,
                    "tau": 3.5,
                    "d0": 0.5,
                    "di": 0.5,
                    "scale_dict": {'alpha': 1, 'beta': 1, 'eta': 1, 'h1': 1, 'h2':1, "tau": 1, "eta_max": 1}
                },

                'private': {
                '0': { 'alpha': [100, 125, 110], 'beta':[125, 105, 100], 'eta': [2, 2, 1.5], 'h1': 1.6, 'h2': 2.6,
                      "r":1.5 },
                '1': {'alpha': [110, 120, 125], 'beta':[115, 100, 110], 'eta': [2.1, 2.2, 1.6], 'h1': 1.7, 'h2': 2.1,
                      "r":2.5},
                '2': {'alpha': [115, 115, 105], 'beta':[110, 110, 125], 'eta': [2.5, 2.6, 1.7], 'h1': 1.8, 'h2': 2.2,
                      "r":3.5},
                '3': {'alpha': [120, 110, 100], 'beta':[100, 120, 120], 'eta': [2.3, 2.4, 1.8], 'h1': 1.9, 'h2': 2.5,
                      "r":4.5},
                '4': {'alpha': [125, 100, 115], 'beta':[105, 115, 105], 'eta': [2.7, 2.8, 1.9], 'h1': 2.0, 'h2': 2.3,
                      "r":5.5},
                },
            }
        }
    },
     "2":
    {
        "epochs" : 200000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "high_order",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,                
                'share': {
                    "init_value_x": np.array([2, 0, 8 ,6 , 4]),
                    "init_value_v": np.zeros((5, 5)),
                    "init_value_z": np.zeros((5, 5)),
                    "init_value_uz": np.zeros((5, 5)),
                    "init_value_uv": np.zeros((5, 5)),


                    'a': 0.02,
                    'b': 2.5,

                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.5,
                    "q": 1.5,
                    "lipsthitz": 5,
                    "delay": 2,
                    "max_delay": 5000,
                    "tau": 4.5,
                    "d0": 0.5,
                    "di": 0.5,
                    "scale_dict": {'alpha': 1, 'beta': 1, 'eta': 1, 'h1': 1, 'h2':1, "tau": 1, "eta_max": 1}
                },

                'private': {
                '0': { 'alpha': [100, 125, 110], 'beta':[125, 105, 100], 'eta': [2, 2, 1.5], 'h1': 1.6, 'h2': 2.6,
                      "r":1.5 },
                '1': {'alpha': [110, 120, 125], 'beta':[115, 100, 110], 'eta': [2.1, 2.2, 1.6], 'h1': 1.7, 'h2': 2.1,
                      "r":2.5},
                '2': {'alpha': [115, 115, 105], 'beta':[110, 110, 125], 'eta': [2.5, 2.6, 1.7], 'h1': 1.8, 'h2': 2.2,
                      "r":3.5},
                '3': {'alpha': [120, 110, 100], 'beta':[100, 120, 120], 'eta': [2.3, 2.4, 1.8], 'h1': 1.9, 'h2': 2.5,
                      "r":4.5},
                '4': {'alpha': [125, 100, 115], 'beta':[105, 115, 105], 'eta': [2.7, 2.8, 1.9], 'h1': 2.0, 'h2': 2.3,
                      "r":5.5},
                },
            }
        }
    },
     "3":
    {
        "epochs" : 150000,
        "adjacency_matrix" : [[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "high_order",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
               
                "N": 5,                
                'share': {
                    "init_value_x": np.array([2, 0, 8 ,6 , 4]),
                    "init_value_v": np.zeros((5, 5)),
                    "init_value_z": np.zeros((5, 5)),
                    "init_value_uz": np.zeros((5, 5)),
                    "init_value_uv": np.zeros((5, 5)),


                    'a': 0.02,
                    'b': 2.5,

                    "eta_max": [2.7, 2.8, 1.9],
                    "p": 0.5,
                    "q": 1.5,
                    "lipsthitz": 5,
                    "delay": 2,
                    "max_delay": 1,
                    "tau": 3.5,
                    "d0": 0.5,
                    "di": 0.5,
                    "scale_dict": {'alpha': 1, 'beta': 1, 'eta': 1, 'h1': 1, 'h2':1, "tau": 1, "eta_max": 1}
                },

                'private': {
                '0': { 'alpha': [100, 125, 110], 'beta':[125, 105, 100], 'eta': [2, 2, 1.5], 'h1': 1.6, 'h2': 2.6,
                      "r":1.5 },
                '1': {'alpha': [110, 120, 125], 'beta':[115, 100, 110], 'eta': [2.1, 2.2, 1.6], 'h1': 1.7, 'h2': 2.1,
                      "r":2.5},
                '2': {'alpha': [115, 115, 105], 'beta':[110, 110, 125], 'eta': [2.5, 2.6, 1.7], 'h1': 1.8, 'h2': 2.2,
                      "r":3.5},
                '3': {'alpha': [120, 110, 100], 'beta':[100, 120, 120], 'eta': [2.3, 2.4, 1.8], 'h1': 1.9, 'h2': 2.5,
                      "r":4.5},
                '4': {'alpha': [125, 100, 115], 'beta':[105, 115, 105], 'eta': [2.7, 2.8, 1.9], 'h1': 2.0, 'h2': 2.3,
                      "r":5.5},
                },
            }
        }
    },
}