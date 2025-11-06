import json
import os
from models import Res
import time
import numpy as np
class Agent:
    def __init__(self, agent_conifg) -> None:
        
        self.agent_config = agent_conifg
        self.agent_id = self.agent_config['node_id']
        self.model_config = self.get_model_config(self.agent_id)
        self.model = Res[self.agent_config['model']](self.model_config)
        
        self.records = []
        self.record_count = 0
    
    def update(self):
        if self.agent_config['record_flag'] == 1:
            self.record()
        self.model.update()
    
    def receieve_msg(self, adj_agent_id, msg):
        memroy = json.loads(msg)
        self.model.receieve_msg(adj_agent_id, memroy)
    
    def memory_to_list(self, memory):
        listed_memory = {}
        for k, v in memory.items():
            if isinstance(v, np.ndarray):
                listed_memory[k] = v.tolist()
            else:
                listed_memory[k] = v
        return listed_memory
        
    def send_msg(self):
        memory = json.dumps(self.memory_to_list(self.model.memory))
        return memory
    
    def record(self):
        if self.record_count % self.agent_config['record_interval'] == 0:
            record_dict = self.memory_to_list(self.model.memory)
            record_dict['time'] =  self.record_count*self.agent_config['time_delta']
            self.records.append(record_dict)
            
        self.record_count += 1

    def get_model_config(self, id):
        model_config = self.agent_config['model_config']
        for k, v in model_config['share'].items():
            model_config[k] = v
        print(model_config['private'])
        for k, v in model_config['private'][str(id)].items():
            model_config[k] = v
        model_config['time_delta'] = self.agent_config['time_delta']
        model_config['agent_id'] = self.agent_id
        
        return model_config

    def done(self):
        record_path = "/app/records/"+self.agent_config['model']+"/"+self.agent_config["config_index"]
        os.makedirs(record_path, exist_ok=True)
        with open(record_path+"/"+f"agent_{self.agent_id}.txt", "w") as f:
            f.write(json.dumps(self.records))
            f.flush()
        
        time.sleep(1)