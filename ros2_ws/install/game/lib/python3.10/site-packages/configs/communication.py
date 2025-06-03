import numpy as np

config = {
    "0":
    {
        "epochs" : 30000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "trigger",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 6,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((6)), "z_event": np.zeros((6)), "is_trigger": np.zeros((6))},
                'share': {
                'mu' : 0.5,
                'nu' : 1.5,
                'epsilon': 0.1
                },

                'private': {
                '0': { 'alpha': 6, 'beta':4, 'gama': 5, 'delta': 5, 'p':1, 'q':4},
                '1': { 'alpha': 5, 'beta':5, 'gama': 4, 'delta': 6, 'p':1, 'q':3},
                '2': { 'alpha': 4, 'beta':5, 'gama': 5, 'delta': 4, 'p':1, 'q':4.5},
                '3': { 'alpha': 6, 'beta':3, 'gama': 6, 'delta': 3, 'p':1, 'q':5},
                '4': { 'alpha': 5, 'beta':6, 'gama': 7, 'delta': 3, 'p':1, 'q':6},
                '5': { 'alpha': 6, 'beta':3, 'gama': 8, 'delta': 3, 'p':3, 'q':7},
                },
            }
        }
    },
    "1":
    {
        "epochs" : 20000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "switching",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 6,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((6))},
                'share': {
                'mu' : 0.5,
                'nu' : 1.5,
                'epsilon': 1
                },

                'private': {
                '0': { 'alpha': 6, 'beta':4, 'gama': 5, 'delta': 5, 'p':1, 'q':14},
                '1': { 'alpha': 5, 'beta':5, 'gama': 4, 'delta': 6, 'p':1, 'q':13},
                '2': { 'alpha': 4, 'beta':5, 'gama': 5, 'delta': 4, 'p':1, 'q':14.5},
                '3': { 'alpha': 6, 'beta':3, 'gama': 6, 'delta': 3, 'p':1, 'q':15},
                '4': { 'alpha': 5, 'beta':6, 'gama': 7, 'delta': 3, 'p':1, 'q':16},
                '5': { 'alpha': 6, 'beta':3, 'gama': 8, 'delta': 3, 'p':1, 'q':17},
                },
            }
        }
    },
    "2":
    {
        "epochs" : 20000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "switching",
            "record_interval": 1,
            "record_flag": 1,
            "model_config": 
            {
                "N": 6,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((6))},
                'share': {
                'mu' : 0.5,
                'nu' : 1.5,
                'epsilon': 0.5
                },

                'private': {
                '0': { 'alpha': 6, 'beta':4, 'gama': 5, 'delta': 5, 'p':1, 'q':14},
                '1': { 'alpha': 5, 'beta':5, 'gama': 4, 'delta': 6, 'p':1, 'q':13},
                '2': { 'alpha': 4, 'beta':5, 'gama': 5, 'delta': 4, 'p':1, 'q':14.5},
                '3': { 'alpha': 6, 'beta':3, 'gama': 6, 'delta': 3, 'p':1, 'q':15},
                '4': { 'alpha': 5, 'beta':6, 'gama': 7, 'delta': 3, 'p':1, 'q':16},
                '5': { 'alpha': 6, 'beta':3, 'gama': 8, 'delta': 3, 'p':1, 'q':17},
                },
            }
        }
    },
}