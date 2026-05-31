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
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

class DumpRecords:
    charater = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    
    def __init__(self, config, index, record_root_path='/app/records') -> None:
        self.config = config
        self.index = index
        self.agent_nums = self.config['agent_config']['model_config']['N']
        self.record_root_path = record_root_path
    
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


    def plot_compared_graph(self, config_index_list):
        compared_dir = self.record_root_path + f"/compared/"
        folder_path = ""
        for index in config_index_list:
            folder_path += '@'+index
        figure_dir = compared_dir+folder_path
        os.makedirs(figure_dir, exist_ok=True)
        
        status_vectors = []
        virtual_vectors = []
        estimate_vectors = []
        cost_estimates = []
        partial_costs = []
        opt_values = []
        times = []
        ois = []
        zero_opt_values = []
        
        fixed_convergence_times = []
        finite_convergence_times = []
        initial_value_norms = []
        
        for index in config_index_list:
            print(f"extracting {index}.......")
            record_path = self.record_root_path + f"/{self.config['agent_config']['model']}/{index}"
            records = self.read_records(record_path)
            memory = self.extract_records(records)
            status_vector = self.align_list(memory['x'])
            virtual_vector = self.align_list(memory['y'])
            estimate_vector = self.align_list(memory['z'])
            
            # 推测 status_vector 形状: (Agents, Time, Dim)
            opt_value_signle = virtual_vector[:, -1, :] 
            opt_value = np.zeros(virtual_vector.shape)
            for j in range(status_vector.shape[1]):
                opt_value[:, j, :] = opt_value_signle
            time_vector = np.array(memory['time'][-1][:len(status_vector[0])])

            status_vectors.append(status_vector)
            virtual_vectors.append(virtual_vector)
            estimate_vectors.append(estimate_vector)
            opt_values.append(opt_value)
            times.append(time_vector)
            
            # --- 修复 1：将计算逻辑移入 for 循环内 ---
            convergence_time = get_convergencce_time(virtual_vector, opt_value, time_vector, 0.065)
            
            if "fixed" in index:
                fixed_convergence_times.append(convergence_time)
                # --- 修复 2 & 3：正确提取 t=0 时刻的初始误差 ||x(0) - x*|| ---
                initial_error = virtual_vector[:, 0,:] - opt_value[:, 0, :]
                print(virtual_vector[:,0,:], opt_value[:, 0, :])
                initial_value_norms.append(np.linalg.norm(initial_error))
            elif "finite" in index:
                finite_convergence_times.append(convergence_time)
        
        # 注意：这里假设你的 config_index_list 中 fixed 和 finite 的实验是成对出现且初始值一致的，
        # 所以 initial_value_norms 仅在 "fixed" 时被 append 了一次，作为两者的共有 X 轴。
        # 修复了传参中的 legneds 和 algorithmn 拼写（如果你的画图函数定义里写死了 legneds，请同步修改那里）
        plot_initial_convergence_line__graph(
            initial_value_norms, 
            [finite_convergence_times, fixed_convergence_times], 
            "$||e_{\omega}(0)||$", 
            legneds=["Finite-time algorithm", "Fixed-time algorithm"], 
            figure_dir=figure_dir
        )


    def plot_graph(self):
        record_path = self.record_root_path + f"/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(record_path)
        memory = self.extract_records(records)
        
        
        status_vector = self.align_list(memory['x'])
        virtual_vector = self.align_list(memory['y'])
        ui = self.align_list(memory['ui'])
        valid_status_vector = []
        valid_speed_vector = []
        valid_acc_vector = []
        for status in status_vector:
            valid_status_vector.append(status[:, 0])
            valid_speed_vector.append(status[:,1])
            valid_acc_vector.append(status[:,2])

        valid_status_vector = np.array(valid_status_vector)
        valid_speed_vector = np.array(valid_speed_vector)
        valid_acc_vector = np.array(valid_acc_vector)
        time = np.array(memory['time'][-1][:len(virtual_vector[0])])
        # ui = self.align_list(memory['ui'])


        # oi = self.align_list(memory['oi'])
        # track_error = self.align_list(memory['track_error'])
        # dot_x = self.align_list(memory['dot_x'])
        # dot_y = self.align_list(memory['doty'])
        # dot_track_error = self.align_list(memory['dot_track_error'])
        # partial_cost = self.align_list(memory['partial_cost'])
                
        # config_index_list = ["0","0_5", "0_3", "0_4"]
        # self.plot_compared_graph(config_index_list,figure_dir)
        # print(status_vector[:,-1,0:])
        print(status_vector.shape)
        opt_value = np.zeros((status_vector.shape[0], 3))
        for i in range(status_vector.shape[0]):
            opt_value[i] = status_vector[i,-1,0]
        print(opt_value)
        # opt_value = np.array([[-0.9999385195244029, 0.16660663117309651, 0.3333955560508185], [4.8152365222943084e-05, -0.8334287020540452, 0.3334118895597235], [1.0001074180788625, 0.16654159249067055, 0.33345503463020215], [4.815265801524303e-05, 1.166571297885446, 0.33341188945528427], [2.0000614803817807, 0.16660663112190888, 3.333395556015464], [-1.999953791549649, 0.16663322749224166, 3.3333734276470937]])
        plot_status_error_graph(time, virtual_vector, figure_dir, ylabel_list=["$\omega_{i1} - y_{i1}^*$", "$\omega_{i2} - y_{i2}^*$", "$\omega_{i3} - y_{i3}^*$"], opt_value=opt_value)
        plot_status_error_graph(time, valid_status_vector, figure_dir, var_name='y', file_name_prefix='actual', opt_value=opt_value)
#         plot_3d_trajectory_graph(valid_status_vector, figure_dir, "status", p_center=np.array([0, 0.5, 2]), var_name='y', topology_edges= [
#     # 第一组：底层小菱形 (z=0 平面，半径 1)
#     # 包含点: 0[-1,0,0], 1[0,-1,0], 2[1,0,0], 3[0,1,0]
#     (0, 1), (1, 2), (2, 3), (3, 0),

#     # 第二组：底层大菱形 (z=0 平面，半径 3)
#     # 包含点: 6[-3,0,0], 7[0,-3,0], 8[3,0,0], 9[0,3,0]
#     (6, 7), (7, 8), (8, 9), (9, 6),

#     # 第三组：上层中菱形 (z=3 平面，半径 2)
#     # 包含点: 4[2,0,3], 10[0,2,3], 5[-2,0,3], 11[0,-2,3]
#     # 注意：这里顺序是 4->10->5->11，这样连出来的线才不会交叉！
#     (4, 10), (10, 5), (5, 11), (11, 4)
# ])
#         plot_3d_trajectory_graph(virtual_vector, figure_dir, "virtual_status")
        x_label_list = [ f"Player {i+1}" for i in range(valid_speed_vector.shape[0])]
        plot_status_graph(time, ui, figure_dir, file_name_prefix="control", ylabel_list=["$u_{i1}$", "$u_{i2}$", "$u_{i3}$"], xlabel_list=x_label_list)

        # plot_status_graph(time, valid_speed_vector[2:, :], figure_dir, file_name_prefix="speed", ylabel_list=["$x_{i21}$", "$x_{i22}$", "$x_{i23}$"],xlabel_list=["Player 3", "Player 4", "Player 5", "Player 6"])
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

def plot_compare_response(record_path_list, figure_dir):
    status_vectors = []
    dumpRecords = DumpRecords(config['fixed_1'], 'fixed_1', "./")
    opt_value = np.array([
    [-1.,  1/26,  1/13],
    [ 0., -25/26, 1/13],
    [ 1.,  1/26,  1/13],
    [ 0.,  27/26, 1/13],
    [ 2.,  1/26, 40/13],
    [-2.,  1/26, 40/13],
    [-3.,  1/26,  1/13],
    [ 0., -77/26, 1/13],
    [ 3.,  1/26,  1/13],
    [ 0.,  79/26, 1/13],
    [ 0.,  53/26, 40/13],
    [ 0., -51/26, 40/13]
])
    for record_path in record_path_list:
        records = dumpRecords.read_records(record_path)
        memory = dumpRecords.extract_records(records)
        status_vector = dumpRecords.align_list(memory['x'])
        # print("1")
        # print(status_vector[:,-1,:])
        # if opt_value is None:
        #     opt_value = np.zeros((status_vector.shape[0], status_vector.shape[2]))
        #     opt_value= status_vector[:, -1, :]
        time = np.array(memory['time'][-1][:len(status_vector[0])])
        status_vectors.append(status_vector)
    
    status_vectors = np.array(status_vectors)
        
    plot_compare_errors_graph(time, status_vectors, figure_dir, var_name='y', file_name_prefix=f"compare_response", opt_value=opt_value, labels=["Exponential algorithm", "Fixed-time algorithm"])

if __name__ == "__main__":
    from config import config

    current_dir = os.path.dirname(os.path.realpath(__file__))
    record_root_path = f"{current_dir}/records"
    config_list = [ "r_2"]
    num_agents = 12
    for config_index in config_list:
        print(f"Running configuration: {config_index}")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        record_root_path = f"{current_dir}/records"
        dumpRecords = DumpRecords(config[config_index], config_index, record_root_path=record_root_path)
        dumpRecords.plot_graph()

    # config_list = ["finite_1", "finite_2", "finite_3", "finite_4",  "finite_5", "finite_6", "fixed_1", "fixed_2", "fixed_3", "fixed_4", "fixed_5", "fixed_6"]
    # dumpRecords = DumpRecords(config['fixed_1'], 'fixed_1', record_root_path)
    # dumpRecords.plot_compared_graph(config_list)


    plot_compare_response(["/app/reassmble/fixed-high/records/fixed_high_order/a_1", "/app/reassmble/fixed-high/records/fixed_high_order/r_1"], "/mnt/binghao/NESeeking/Nash-equilibrium/reassmble/fixed-high/records")