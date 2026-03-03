import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import copy
import json
import scipy.special as sp
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from standard import *

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

class DumpRecords:
    charater = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    
    def __init__(self, config, index) -> None:
        self.config = config
        self.index = index
        self.agent_nums = self.config['agent_config']['model_config']['N']
    
    def read_records(self,record_path):
        # record_path = f"/app/records/{model}/{index}"
        records = []
        for i in range(self.agent_nums):
            file_path = f'{record_path}/agent_{i}.txt'
            with open(file_path, 'r') as f:
                record = f.read()
                records.append(json.loads(record))
        return records
    
    def align_list(self, irregular_list):
        min_length = min(len(lst) for lst in irregular_list)
        uniform_array = np.array([lst[:min_length] for lst in irregular_list])
        return uniform_array


    def extract_records(self, records):
        
        memory = defaultdict()

        for id, record in enumerate(records):
            vector = defaultdict(list)
            for item in record:
                for k, v in item.items():
                    vector[k].append(v)
            
            for k,v in vector.items():
                if k  not in memory.keys():
                    memory[k] = []
                memory[k].append(vector[k])

        return memory

    def get_opt_value(self):
        r = [10, 20, 30, 40]
        sum = 2*np.sum(r)
        p = (sum - 20)/2.2
        opt_value = []
        for ri in r:
            yi = (2*ri - 5 - 0.04*p)/2.04
            opt_value.append(yi)
        print(opt_value)
        return np.array(opt_value)


    def cal_partial_cost(self, status):
        r = [10, 20, 30, 40]
        partial_value = []
        for i in range(len(status)):
            partial_value_i = 2*(status[i]-r[i]) + 0.04*sum(status)+5 + 0.04*status[i]
            partial_value_i = partial_value_i/4
            partial_value.append(partial_value_i)
        
        return np.array(partial_value)

    def plot_graph(self, record_path=None):
        if record_path is None:
            record_path = f"/app/records/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(record_path)
        memory = self.extract_records(records)
        
        
        status_vector = self.align_list(memory['x'])
        virtual_vector = self.align_list(memory['y'])
        estiamte_vector = self.align_list(memory['z'])
        partial_cost = self.align_list(memory['partial_cost'])
        valid_status_vector = []
        valid_speed_vector = []
        valid_acc_vector = []


        # valid_status_vector = np.array(valid_status_vector)
        # valid_speed_vector = np.array(valid_speed_vector)
        # valid_acc_vector = np.array(valid_acc_vector)
        time = np.array(memory['time'][-1][:len(virtual_vector[0])])
        # ui = self.align_list(memory['ui'])
        DoS_interval = self.config['agent_config']['model_config']['DoS_interval']
        dos_interval = []
        for key, intervals in DoS_interval.items():
            dos_interval.extend(intervals)
            

        plot_single_status_converge_graph(time, virtual_vector, figure_dir, file_name_prefix="virtual_state", ylabel="$\omega_{i}$",xlabel_list=["$\omega_1$", "$\omega_2$", "$\omega_3$", "$\omega_4$"], opt_label_list=["$y_1^*$", "$y_2^*$", "$y_3^*$", "$y_4^*$"])
        plot_single_status_converge_graph(time, status_vector, figure_dir, file_name_prefix="state", ylabel="$y_i$",xlabel_list=["$y_1$", "$y_2$", "$y_3$", "$y_4$"], opt_label_list=["$y_1^*$", "$y_2^*$", "$y_3^*$", "$y_4^*$"])
        plot_dos_estimate_norm_converge_graph(time, virtual_vector,  estiamte_vector, figure_dir, file_name_prefix="estimate_norm", ylabel="$log_10(||z_i$-$\omega||)$", xlabel_list=["Player 1", "Player 2", "Player 3", "Player 4"], dos_interval=dos_interval)
        
        partial_cost = np.zeros(virtual_vector.shape)
        print(partial_cost.shape, virtual_vector.shape)
        for i in range(virtual_vector.shape[0]):
            for j in range(virtual_vector.shape[1]):
                partial_cost_value = self.cal_partial_cost(virtual_vector[:,j,:].flatten())
                # print(partial_cost_value)
                partial_cost[:,j,0] = partial_cost_value
        plot_dos_status_norm_converge_graph(time, partial_cost, figure_dir, file_name_prefix="partial_cost", ylabel="$log_10(||\\nabla_i\ f_i(\omega)||)$",xlabel_list=["Player 1", "Player 2", "Player 3", "Player 4"], dos_interval=dos_interval)
        get_convergencce_time(status_vector[:,:,0:1], opt_value=self.get_opt_value(), time_vector=time)

if __name__ == "__main__":
    from config import config
    index = "c_1"
    
    dumpRecords = DumpRecords(config[index], index)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_path = f"{current_dir}/records/{config[index]['agent_config']['model']}/{index}"
    dumpRecords.plot_graph(record_path=record_path)
