import numpy as np

config = {
    "0":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.001,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
    "1":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.01,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
    "2":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.1,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
    "3":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.3,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
        "4":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.5,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
        "5":
    {
        "epochs" : 250000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 4e-5,
            "model": "fixed_switch",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'mu' : 0.6,
                'nu' : 1.4,
                'epsilon': 0.7,
                'a': 0.01,
                'b': 0.125,
                },

                'private': {
                '0': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.5},
                '1': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.55},
                '2': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.60},
                '3': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.65},
                '4': { 'alpha': 100, 'beta':50, 'delta': 10, 'eta': 2, 'r': 0.70},
                },
            }
        }
    },
}