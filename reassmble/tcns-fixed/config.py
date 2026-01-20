import numpy as np
config = {
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
}