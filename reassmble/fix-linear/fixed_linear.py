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


    # def plot_compared_graph(self, config_index_list):
    #     compared_dir = f"/app/records/compared/"
    #     folder_path = ""
    #     for index in config_index_list:
    #         folder_path += '@'+index
    #     figure_dir = compared_dir+folder_path
    #     os.makedirs(figure_dir, exist_ok=True)
    #     status_vectors = []
    #     virtual_vectors = []
    #     estimate_vectors = []
    #     cost_estimates = []
    #     partial_costs = []
    #     opt_values = []
    #     times = []
    #     ois = []
    #     zero_opt_values = []
    #     # labels = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
    #     for index in config_index_list:
    #         print(f"extracting {index}.......")
    #         record_path = f"/app/records/{self.config['agent_config']['model']}/{index}"
    #         records = self.read_records(self.config['agent_config']['model'], index)
    #         memory = self.extract_records(records)
    #         status_vector = self.align_list(memory['x'])
    #         virtual_vector = self.align_list(memory['y'])
    #         estimate_vector = self.align_list(memory['z'])
    #         cost_estimate = self.align_list(memory['v'])
    #         partial_cost = self.align_list(memory['partial_cost'])
    #         if "oi" in memory.keys():
    #             oi = self.align_list(memory['oi'])
    #             zero_opt_value = np.zeros(oi.shape)
    #             ois.append(oi)
    #             zero_opt_values.append(zero_opt_value)
    #         opt_value_signle = status_vector[:,-1,:]
    #         opt_value = np.zeros(status_vector.shape)
    #         for j in range(status_vector.shape[1]):
    #             opt_value[:, j, :] = opt_value_signle
    #         time = np.array(memory['time'][-1][:len(status_vector[0])])

    #         status_vectors.append(status_vector)
    #         virtual_vectors.append(virtual_vector)
    #         estimate_vectors.append(estimate_vector)
    #         cost_estimates.append(cost_estimate)
    #         partial_costs.append(partial_cost)
    #         opt_values.append(opt_value)
    #         times.append(time)
        

    #     self.plot_assemble_estimation_graph(times, estimate_vectors, virtual_vectors, figure_dir, "virtual_status_estimate", "||z-$1_N \otimes y$||", right=0.2)
    #     self.plot_assemble_estimation_graph(times, cost_estimates, partial_costs, figure_dir, "partial_cost_estimate", "||v-$1_N \otimes F(y)$||", right=0.2)
    #     self.plot_assemble_estimation_graph(times, virtual_vectors, opt_values, figure_dir, "virtual_status_opt", "||y-$x^{\star}$||", right=3)
    #     self.plot_assemble_estimation_graph(times, status_vectors, virtual_vectors, figure_dir, "status_virtual", "||x-y||", right=3)
    #     self.plot_assemble_estimation_graph(times, status_vectors, opt_values, figure_dir, "status_opt", "||x-$x^{\star}$||", right=3)
    #     self.get_convergencce_time(status_vectors, opt_value=np.array([[-0.5, -0.32], [0.5, -0.32], [-0.5, 0.18], [0.5, 0.18], [0, 0.68]]))

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
            

        # oi = self.align_list(memory['oi'])
        # track_error = self.align_list(memory['track_error'])
        # dot_x = self.align_list(memory['dot_x'])
        # dot_y = self.align_list(memory['doty'])
        # dot_track_error = self.align_list(memory['dot_track_error'])
        # partial_cost = self.align_list(memory['partial_cost'])
                
        # config_index_list = ["0","0_5", "0_3", "0_4"]
        # self.plot_compared_graph(config_index_list,figure_dir)
        # opt_value = np.array([[-0.9999385195244029, 0.16660663117309651, 0.3333955560508185], [4.8152365222943084e-05, -0.8334287020540452, 0.3334118895597235], [1.0001074180788625, 0.16654159249067055, 0.33345503463020215], [4.815265801524303e-05, 1.166571297885446, 0.33341188945528427], [2.0000614803817807, 0.16660663112190888, 3.333395556015464], [-1.999953791549649, 0.16663322749224166, 3.3333734276470937]])
        # plot_status_error_graph(time, virtual_vector, figure_dir, ylabel_list=["$\omega_{i1} - y_{i1}^*$", "$\omega_{i2} - y_{i2}^*$", "$\omega_{i3} - y_{i3}^*$"], opt_value=opt_value, xlim=[0, 2])
        # plot_status_error_graph(time, valid_status_vector, figure_dir, var_name='
        # y', file_name_prefix='actual', opt_value=opt_value)
        # plot_3d_trajectory_graph(valid_status_vector, figure_dir, "status", p_center=np.array([0, 0.5, 2]), var_name='y')
        # plot_3d_trajectory_graph(virtual_vector, figure_dir, "virtual_status")
        plot_single_status_converge_graph(time, virtual_vector, figure_dir, file_name_prefix="virtual_state", ylabel="$\omega_{i}$",xlabel_list=["$\omega_1$", "$\omega_2$", "$\omega_3$", "$\omega_4$"], opt_label_list=["$y_1^*$", "$y_2^*$", "$y_3^*$", "$y_4^*$"])
        plot_single_status_converge_graph(time, status_vector, figure_dir, file_name_prefix="state", ylabel="$y_i$",xlabel_list=["$y_1$", "$y_2$", "$y_3$", "$y_4$"], opt_label_list=["$y_1^*$", "$y_2^*$", "$y_3^*$", "$y_4^*$"])
        plot_dos_estimate_norm_converge_graph(time, virtual_vector,  estiamte_vector, figure_dir, file_name_prefix="estimate_norm", ylabel="$lg(||z_i$-$\omega||)$", xlabel_list=["Player 1", "Player 2", "Player 3", "Player 4"], dos_interval=dos_interval)
        # plot_status_graph(time, valid_acc_vector[4:, :], figure_dir, file_name_prefix="acc", ylabel_list=["$x_{i31}$", "$x_{i32}$", "$x_{i33}$"],xlabel_list=["Player 5", "Player 6"])

        # self.plot_compared_graph(["3", "3_14", "3_15", "3_11", "3_12", "3_13"])
        # self.plot_status_graph(time, virtual_vector, virtual_vector,figure_dir, "virtual_status", 'y')
        # time = np.array(memory['time'][-1][:len(ui[0])])
        # self.plot_status_graph(time, ui, ui,figure_dir, "ui", "u", "Control torque vector")
        # self.plot_trajectory_graph(status_vector, figure_dir)
        # print(self.index)
        # self.plot_compared_graph(['3', '3_1', '3_2', '3_3'], )
        # self.plot_status_graph(time, oi, oi,figure_dir, "oi", "oi", "Control torque vector")
        # self.plot_status_graph(time, track_error, track_error,figure_dir, "track_error", "track_error", "Control torque vector")
        # self.plot_status_graph(time, dot_track_error, dot_track_error,figure_dir, "dot_track_error", "dot_track_error", "Control torque vector")
        # self.plot_status_graph(time, dot_y, dot_y,figure_dir, "dot_y", "dot_y", "Control torque vector")

        # time = np.array(memory['time'][-1][:len(dot_x[0])])
        # self.plot_status_graph(time, dot_x, dot_x,figure_dir, "dot_x", "dot_x", "Control torque vector")

        # self.plot_status_graph(time, partial_cost, partial_cost,figure_dir, "partial_cost", "partial_cost", "Control torque vector")



        # self.plot_assemble_estimation_graph(time, [estimate_vector], [virtual_vector], figure_dir, "virtual_status_estimate")
    

if __name__ == "__main__":
    from config import config
    index = "r_r"

    dumpRecords = DumpRecords(config[index], index)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_path = f"{current_dir}/records/{config[index]['agent_config']['model']}/{index}"
    dumpRecords.plot_graph(record_path=record_path)
