import numpy as np

config = {
    "DESC": "Distributed nash equilibrium seeking: Continuous-time control-theoretic approaches",
    "0":
    {
        "epochs" : 7200000,
        "adjacency_matrix" : [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-6,
            "model": "fixed",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.5,
                'min_c1': 30,
                'min_delta': 2,
                'gama': 200
                },

                'private': {
                '0': { 'c1': 30, 'c2':30, 'delta': 1.5, 'eta': 1.5, 'epsilon': 0, 'r':50},
                '1': {'c1': 30, 'c2':30,'delta': 1.5, 'eta': 1.5, 'epsilon': 0, 'r': 55},
                '2': {'c1': 30, 'c2':30,'delta': 1.5, 'eta': 1.5, 'epsilon': 0, 'r': 60},
                '3': {'c1': 30, 'c2':30,'delta': 1.5, 'eta': 1.5, 'epsilon': 0, 'r': 65},
                '4': {'c1': 30, 'c2':30,'delta': 1.5, 'eta': 1.5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "1":
    {
        "epochs" : 150000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 2e-5,
            "model": "fixed",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.array([0.0]), "z": np.zeros((5))},
                'share':
                {
                    'c': 180,
                    'a': 0.02,
                    'b': 2.5,
                    'alpha': 80,
                    'l': 0,
                    'u': 80,
                    'p' : 1,
                    'q' : 1,
                    'gama': 23
                },
                'private': 
                {
                    '0': { 'c1': 2.5, 'c2':2.5, 'delta': 0.9, 'eta': 0, 'epsilon': 0, 'r':50},
                    '1': {'c1': 2.5, 'c2':2.5,'delta': 0.9, 'eta': 0, 'epsilon': 0, 'r': 55},
                    '2': {'c1': 2.5, 'c2':2.5,'delta': 0.9, 'eta': 0, 'epsilon': 0, 'r': 60},
                    '3': {'c1': 2.5, 'c2':2.5,'delta': 0.9, 'eta': 0, 'epsilon': 0, 'r': 65},
                    '4': {'c1': 2.5, 'c2':2.5,'delta': 0.9, 'eta': 0, 'epsilon': 0, 'r': 70},
                }
            },
        }
    },
    "2":
    {
        "epochs" : 600000,
        "adjacency_matrix" : [[1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-6,
            "model": "fixed",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
                'share': {
                'c': 180,
                'a': 0.02,
                'b': 2.5,
                'alpha': 80,
                'l': 0,
                'u': 80,
                'p' : 0.5,
                'q' : 1.3,
                'min_c1': 40,
                'min_delta': 2,
                'gama': 200
                },

                'private': {
                '0': { 'c1': 30, 'c2':30, 'delta': 2, 'eta': 2, 'epsilon': 0, 'r':50},
                '1': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 55},
                '2': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 60},
                '3': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 65},
                '4': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "3":
    {
        "epochs" : 10000000,
        "adjacency_matrix" : [[1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 0, 1, 1]],
        "agent_config":
        {  
            "time_delta": 1e-7,
            "model": "fixed2",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5))},
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
                'gama': 50,
                },

                'private': {
                '0': { 'c1': 30, 'c2':30, 'delta': 2, 'eta': 2, 'epsilon': 0, 'r':50},
                '1': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 55},
                '2': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 60},
                '3': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 65},
                '4': {'c1': 30, 'c2':30,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "4":
    {
        "epochs" : 500000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 6e-6,
            "model": "fixed3",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.array(0.0)},
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
                'gama': 400,
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
    "5":
    {
        "epochs" : 500000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 6e-6,
            "model": "fixed3",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share': {
                'init_value': np.array([[45], [35], [25], [15], [10]]),
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
                'gama': 1000,
                },

                'private': {
                '0': { 'c1': 10, 'c2':10, 'delta': 2, 'eta': 2, 'epsilon': 0, 'r':50},
                '1': {'c1': 10, 'c2':10,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 55},
                '2': {'c1': 10, 'c2':10,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 60},
                '3': {'c1': 10, 'c2':10,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':10,'delta': 2, 'eta': 2, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "6":
    {
        "epochs" : 1500000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 2e-7,
            "model": "fixed3",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share': {
                'init_value': np.array([[45], [35], [25], [15], [10]]),
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
                'gama': 7200,
                },

                'private': {
                '0': { 'c1': 15, 'c2':15, 'delta': 3, 'eta': 3, 'epsilon': 0, 'r':50},
                '1': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 55},
                '2': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 60},
                '3': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 65},
                '4': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "7":
    {
        "epochs" : 1500000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 2e-7,
            "model": "fixed3",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share': {
                'init_value': np.array([[45], [35], [25], [15], [10]]),
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
                'gama': 9600,
                },

                'private': {
                '0': { 'c1': 20, 'c2':20, 'delta': 4, 'eta': 4, 'epsilon': 0, 'r':50},
                '1': {'c1': 20, 'c2':20,'delta': 4, 'eta': 4, 'epsilon': 0, 'r': 55},
                '2': {'c1': 20, 'c2':20,'delta': 4, 'eta': 4, 'epsilon': 0, 'r': 60},
                '3': {'c1': 20, 'c2':20,'delta': 4, 'eta': 4, 'epsilon': 0, 'r': 65},
                '4': {'c1': 20, 'c2':20,'delta': 4, 'eta': 4, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "8":
    {
        "epochs" : 1500000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 2e-6,
            "model": "fixed3",
            "record_interval": 100,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.zeros((1)), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share': {
                'init_value': np.array([[45], [35], [25], [15], [10]]),
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
                'gama': 12000,
                },

                'private': {
                '0': { 'c1': 25, 'c2':25, 'delta': 5, 'eta': 5, 'epsilon': 0, 'r':50},
                '1': {'c1': 25, 'c2':25,'delta': 5, 'eta': 5, 'epsilon': 0, 'r': 55},
                '2': {'c1': 25, 'c2':25,'delta': 5, 'eta': 5, 'epsilon': 0, 'r': 60},
                '3': {'c1': 25, 'c2':25,'delta': 5, 'eta': 5, 'epsilon': 0, 'r': 65},
                '4': {'c1': 25, 'c2':25,'delta': 5, 'eta': 5, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
    "asy_0":
    {
        "epochs" : 150000,
        "adjacency_matrix" : [[0, 1, 0, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 1, 0]],
        "agent_config":
        {  
            "time_delta": 2e-5,
            "model": "asym",
            "record_interval": 50,
            "record_flag": 1,
            "model_config": 
            {
                "N": 5,
                "memory" : {"x": np.array([0.0]), "z": np.zeros((5)), "lambda": np.array(0.0)},
                'share':
                {
                    'c': 180,
                    'a': 0.02,
                    'b': 2.5,
                    'alpha': 100,
                    'l': 0,
                    'u': 80,
                    'p' : 1,
                    'q' : 1,
                    'gama': 2,
                    'e': 10,
                    'min_c1': 5,
                    'min_delta': 1,
                },
                'private': 
                {
                    '0': { 'c1': 15, 'c2':15, 'delta': 3, 'eta': 3, 'epsilon': 0, 'r':50},
                    '1': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 55},
                    '2': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 60},
                    '3': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 65},
                    '4': {'c1': 15, 'c2':15,'delta': 3, 'eta': 3, 'epsilon': 0, 'r': 70},
                }
            },
        }
    },
    "compared":
    {
        'box': [0.45, 0.82],
        'timescale': [0, 3.0],
        'font_size' : 8,
        'labels' : {"fixed3@4": "Fixed-time Projection operator algorithm", "fixed@1": "Penalty-based algorithm", "asym@asy_0": "Projection operator algorithm"},
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
    "9":
    {
        "epochs" : 150000 ,
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
                'gama': 1000,
                },

                'private': {
                '0': { 'c1': 10, 'c2':10, 'delta': 1, 'eta': 1, 'epsilon': 0, 'r':50},
                '1': {'c1': 10, 'c2':10,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 55},
                '2': {'c1': 10, 'c2':10,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 60},
                '3': {'c1': 10, 'c2':10,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 65},
                '4': {'c1': 10, 'c2':10,'delta': 1, 'eta': 1, 'epsilon': 0, 'r': 70},
                },
            }
        }
    },
}