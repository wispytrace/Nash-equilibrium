import numpy as np

config = {
    "1":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 0.5, 'varphi': 4.5, 'sigma': 4,'eta': 0.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 1,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 1.5,'varphi': 3, 'sigma': 1, 'eta': 4, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 2,'varphi': 6, 'sigma': 5, 'eta': 2.4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 2.5,'varphi': 7.5, 'sigma': 2, 'eta': 3.2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "2":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },
                'private': {
                '0': { 'c1': 4, 'c2':6, 'delta': 0.5, 'varphi': 4.5, 'sigma': 4,'eta': 0.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 8, 'c2':10,'delta': 1,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 12, 'c2':8,'delta': 1.5,'varphi': 3, 'sigma': 1, 'eta': 4, 'epsilon': 0, 'r': 60},
                '3': {'c1': 16, 'c2':4,'delta': 2,'varphi': 6, 'sigma': 5, 'eta': 2.4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 20, 'c2':2,'delta': 2.5,'varphi': 7.5, 'sigma': 2, 'eta': 3.2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "2_h":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },
                'private': {
                '0': { 'c1': 8, 'c2':12, 'delta': 0.5, 'varphi': 4.5, 'sigma': 4,'eta': 0.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 16, 'c2':20,'delta': 1,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 24, 'c2':32,'delta': 1.5,'varphi': 3, 'sigma': 1, 'eta': 4, 'epsilon': 0, 'r': 60},
                '3': {'c1': 32, 'c2':8,'delta': 2,'varphi': 6, 'sigma': 5, 'eta': 2.4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 40, 'c2':4,'delta': 2.5,'varphi': 7.5, 'sigma': 2, 'eta': 3.2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "3":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 1, 'varphi': 4.5, 'sigma': 4,'eta': 1.6, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 2,'varphi': 1.5, 'sigma': 3, 'eta': 3.2, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 3,'varphi': 3, 'sigma': 1, 'eta': 8, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 4,'varphi': 6, 'sigma': 5, 'eta': 4.8, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 5,'varphi': 7.5, 'sigma': 2, 'eta': 6.4, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "4":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 0.5, 'varphi': 9, 'sigma': 8,'eta': 0.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 1,'varphi': 3, 'sigma': 6, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 1.5,'varphi': 6, 'sigma': 2, 'eta': 4, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 2,'varphi': 12, 'sigma': 10, 'eta': 2.4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 2.5,'varphi': 15, 'sigma': 4, 'eta': 3.2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "5":
    {
        "epochs" : 25000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 1.5, 'varphi': 4.5, 'sigma': 4,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 1.6,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 1.7,'varphi': 3, 'sigma': 1, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 1.8,'varphi': 6, 'sigma': 5, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 1.9,'varphi': 7.5, 'sigma': 2, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "6":
    {
        "epochs" : 25000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.array([np.random.randint(3, 42)+0.1]), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },
                'private': {
                '0': { 'c1': 4, 'c2':6, 'delta': 1.5, 'varphi': 4.5, 'sigma': 4,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 8, 'c2':10,'delta': 1.6,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 12, 'c2':8,'delta': 1.7,'varphi': 3, 'sigma': 1, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 16, 'c2':4,'delta': 1.8,'varphi': 6, 'sigma': 5, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 20, 'c2':2,'delta': 1.9,'varphi': 7.5, 'sigma': 2, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "6_h":
    {
        "epochs" : 25000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },
                'private': {
                '0': { 'c1': 8, 'c2':12, 'delta': 1.5, 'varphi': 4.5, 'sigma': 4,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 16, 'c2':20,'delta': 1.6,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 24, 'c2':32,'delta': 1.7,'varphi': 3, 'sigma': 1, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 32, 'c2':8,'delta': 1.8,'varphi': 6, 'sigma': 5, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 40, 'c2':4,'delta': 1.9,'varphi': 7.5, 'sigma': 2, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "7":
    {
        "epochs" : 25000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 3, 'varphi': 4.5, 'sigma': 4,'eta': 3.6, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 3.2,'varphi': 1.5, 'sigma': 3, 'eta': 3.2, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 3.4,'varphi': 3, 'sigma': 1, 'eta': 3.8, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 3.6,'varphi': 6, 'sigma': 5, 'eta': 3.4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 3.8,'varphi': 7.5, 'sigma': 2, 'eta': 3.0, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "8":
    {
        "epochs" : 25000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 1.5, 'varphi': 9, 'sigma': 8,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 1.6,'varphi': 3, 'sigma': 6, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 1.7,'varphi': 6, 'sigma': 2, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 1.8,'varphi': 12, 'sigma': 10, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 1.9,'varphi': 15, 'sigma': 4, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "8_h":
    {
        "epochs" : 25000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                'init_value': np.array([[45.0], [35.0], [25.0], [15.0], [10.0]]),
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 4, 'c2':6, 'delta': 1.5, 'varphi': 13.5, 'sigma': 12,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 8, 'c2':10,'delta': 1.6,'varphi': 4.5, 'sigma': 9, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 12, 'c2':8,'delta': 1.7,'varphi': 9, 'sigma': 3, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 16, 'c2':4,'delta': 1.8,'varphi': 18, 'sigma': 15, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 20, 'c2':2,'delta': 1.9,'varphi': 22.5, 'sigma': 6, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "fixed_1":
    {
        "epochs" : 60000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 5,
                'min_delta': 1,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 5, 'c2':5, 'delta': 1, 'eta': 1, 'epsilon': 0, 'r':50},
                '1': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 55},
                '2': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 60},
                '3': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 65},
                '4': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "9":
    {
        "epochs" : 250000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-5,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]),
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'l': 0,
                'u': 55,
                'alpha': 80,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 1000,
                },

                'private': {
                '0': { 'c1': 2, 'c2':3, 'delta': 1.5, 'varphi': 4.5, 'sigma': 4,'eta': 1.8, 'epsilon': 0, 'r':50},
                '1': {'c1': 4, 'c2':5,'delta': 1.6,'varphi': 1.5, 'sigma': 3, 'eta': 1.6, 'epsilon': 0, 'r': 55},
                '2': {'c1': 6, 'c2':4,'delta': 1.7,'varphi': 3, 'sigma': 1, 'eta': 1.9, 'epsilon': 0, 'r': 60},
                '3': {'c1': 8, 'c2':2,'delta': 1.8,'varphi': 6, 'sigma': 5, 'eta': 1.7, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':1,'delta': 1.9,'varphi': 7.5, 'sigma': 2, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "fixed_2":
    {
        "epochs" : 60000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'l': 0,
                'u': 55,
                'p' : 0.7,
                'q' : 1.3,
                'min_c1': 5,
                'min_delta': 1,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 5, 'c2':5, 'delta': 1, 'eta': 1, 'varphi': 1, 'sigma': 1, 'epsilon': 0, 'r':50},
                '1': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'varphi': 1, 'sigma': 1, 'epsilon': 0, 'r': 55},
                '2': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'varphi': 1, 'sigma': 1, 'epsilon': 0, 'r': 60},
                '3': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'varphi': 1, 'sigma': 1, 'epsilon': 0, 'r': 65},
                '4': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'varphi': 1, 'sigma': 1, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
 
    "exp_3":
    {
        "epochs" : 60000 ,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 1,
                'q' : 1,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 100,
                },

                'private': {
                '0': { 'c1': 5, 'c2':5, 'delta': 1, 'eta': 1, 'epsilon': 0, 'r':50},
                '1': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 55},
                '2': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 60},
                '3': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 65},
                '4': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },

    "exp_2":
    {
        "epochs" : 60000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "asym",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.array([0.0]), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share':
                {
                    'c': 18,
                    'a': 0.2,
                    'b': 2.5,
                    'alpha': 100,
                    'l': 0,
                    'u': 8,
                    'p' : 1,
                    'q' : 1,
                    'gama': 4,
                    'e': 1,
                    'min_c1': 5,
                    'min_delta': 1,
                },
                'private': 
                {
                    '0': { 'c1': 5, 'c2':5, 'delta': 1, 'eta': 1, 'epsilon': 0, 'r':5.0},
                    '1': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 5.5},
                    '2': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 6.0},
                    '3': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 6.5},
                    '4': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 7.0},
                }
            },
        }
    },
    "asym_4":
    {
        "epochs" : 120000 ,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "asym2",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.ones((1))},
                'share': {
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'min_c1': 5,
                'min_delta': 1,
                'gama': 1,
                'k1': 10,
                'k2': 2,
                'e': 5
                },

                'private': {
                '0': { 'c1': 5, 'c2':5, 'delta': 1, 'eta': 1, 'epsilon': 0, 'r':5},
                '1': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 6},
                '3': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 5, 'c2':5,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 7},
                },
            }
        }
    },
    "compared":
    {
        'box': [0.45, 0.82],
        'timescale': [0, 3.0],
        'font_size' : 8,
        'labels' : {"fixed4@r_1": "Fixed-time algorithm", "asym@exp_2": "Expoint algorithm", "asym2@asym_4": "asym algorithm"},
    },
    "compared2":
    {
        'box': [0.85, 0.68],
        'timescale': [0, 2.5],
        'font_size' : 8,
        'labels' : {"fixed3@4": "A",
                    "fixed3@5": "B",
                    "fixed3@6": "C",
                    "fixed3@7": "D",
                    "fixed3@8": "E"},
    },
   "r_r":
    {
        "epochs" : 60000,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-4,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 40,
                },

                'private': {
                '0': { 'c1': 0.5, 'c2': 0.5, 'delta': 2, 'varphi':  0.5, 'sigma':  0.5,'eta':  2, 'epsilon': 0, 'r':5.0},
                '1': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 5.5},
                '2': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 6.0},
                '3': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 6.5},
                '4': {'c1':  0.5, 'c2': 0.5,'delta':  2,'varphi':  0.5, 'sigma':  0.5, 'eta':  2, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

    "r_0":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 5e-5,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[4.5], [3.5], [2.5], [1.5], [1.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.75,
                'q' : 1.25,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 45,
                },

                'private': {
                '0': { 'c1': 1, 'c2':1.8, 'delta': 1.5, 'varphi': 1, 'sigma': 1.8,'eta': 1.8, 'epsilon': 0, 'r':5.0},
                '1': {'c1': 1.2, 'c2':1.6,'delta': 1.6,'varphi': 1.2, 'sigma': 1.6, 'eta': 1.6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 1.4, 'c2':1.4,'delta': 1.7,'varphi': 1.4, 'sigma': 1.4, 'eta': 1.9, 'epsilon': 0, 'r': 6.0},
                '3': {'c1': 1.6, 'c2':1.2,'delta': 1.8,'varphi': 1.6, 'sigma': 1.2, 'eta': 1.7, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 1.8, 'c2':1.0,'delta': 1.9,'varphi': 1.8, 'sigma': 1.0, 'eta': 1.5, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

    "r_1":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 5e-5,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[4.5], [3.5], [2.5], [1.5], [1.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.75,
                'q' : 1.25,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 45,
                },

                'private': {
                '0': { 'c1': 10, 'c2':18, 'delta': 1.5, 'varphi': 10, 'sigma': 18,'eta': 1.8, 'epsilon': 0, 'r':5.0},
                '1': {'c1': 12, 'c2':16,'delta': 1.6,'varphi': 12, 'sigma': 16, 'eta': 1.6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 14, 'c2':14,'delta': 1.7,'varphi': 14, 'sigma': 14, 'eta': 1.9, 'epsilon': 0, 'r': 6.0},
                '3': {'c1': 16, 'c2':12,'delta': 1.8,'varphi': 16, 'sigma': 12, 'eta': 1.7, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 18, 'c2':10,'delta': 1.9,'varphi': 18, 'sigma': 10, 'eta': 1.5, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },
    "r_2":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 5e-5,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[4.5], [3.5], [2.5], [1.5], [1.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.75,
                'q' : 1.25,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 45,
                },

                'private': {
                '0': { 'c1': 40, 'c2':72, 'delta': 1.5, 'varphi': 10, 'sigma': 18,'eta': 1.8, 'epsilon': 0, 'r':5.0},
                '1': {'c1': 48, 'c2':64,'delta': 1.6,'varphi': 12, 'sigma': 16, 'eta': 1.6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 56, 'c2':56,'delta': 1.7,'varphi': 14, 'sigma': 14, 'eta': 1.9, 'epsilon': 0, 'r': 6.0},
                '3': {'c1': 64, 'c2':48,'delta': 1.8,'varphi': 16, 'sigma': 12, 'eta': 1.7, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 72, 'c2':40,'delta': 1.9,'varphi': 18, 'sigma': 10, 'eta': 1.5, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },
    "r_3":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 5e-5,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[4.5], [3.5], [2.5], [1.5], [1.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.75,
                'q' : 1.25,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 45,
                },

                'private': {
                '0': { 'c1': 10, 'c2':18, 'delta': 1.5, 'varphi': 40, 'sigma': 72,'eta': 1.8, 'epsilon': 0, 'r':5.0},
                '1': {'c1': 12, 'c2': 16,'delta': 1.6,'varphi': 48, 'sigma': 64, 'eta': 1.6, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 14, 'c2':14,'delta': 1.7,'varphi': 56, 'sigma': 56, 'eta': 1.9, 'epsilon': 0, 'r': 6.0},
                '3': {'c1': 16, 'c2':12,'delta': 1.8,'varphi': 64, 'sigma': 48, 'eta': 1.7, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 18, 'c2':10,'delta': 1.9,'varphi': 72, 'sigma': 40, 'eta': 1.5, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },
    "r_4":
    {
        "epochs" : 50000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 5e-5,
            "model": "fixed4",
            "record_interval": 200,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "v": np.zeros((5))},
                'share': {
                'init_value': np.array([[4.5], [3.5], [2.5], [1.5], [1.0]]),
                'c': 18,
                'a': 0.2,
                'b': 2.5,
                'l': 0,
                'u': 8.0,
                'p' : 0.75,
                'q' : 1.25,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 45,
                },

                'private': {
                '0': { 'c1': 10, 'c2':18, 'delta': 3, 'varphi': 10, 'sigma': 18,'eta': 3.6, 'epsilon': 0, 'r':5.0},
                '1': {'c1': 12, 'c2':16,'delta': 3.2,'varphi': 12, 'sigma': 16, 'eta': 3.2, 'epsilon': 0, 'r': 5.5},
                '2': {'c1': 14, 'c2':14,'delta': 3.4,'varphi': 14, 'sigma': 14, 'eta': 3.8, 'epsilon': 0, 'r': 6.0},
                '3': {'c1': 16, 'c2':12,'delta': 3.6,'varphi': 16, 'sigma': 12, 'eta': 3.4, 'epsilon': 0, 'r': 6.5},
                '4': {'c1': 18, 'c2':10,'delta': 3.8,'varphi': 18, 'sigma': 10, 'eta': 3, 'epsilon': 0, 'r': 7.0},
                },
            }
        }
    },

}