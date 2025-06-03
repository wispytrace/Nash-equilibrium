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

mpl.rcParams['figure.dpi'] = 600 
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

class DumpRecords:
    charater = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
    
    def __init__(self, config, index) -> None:
        self.config = config
        self.index = index
        self.agent_nums = self.config['agent_config']['model_config']['N']
    
    def read_records(self, model, index):
        record_path = f"/app/records/{model}/{index}"
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

    def plot_matrix_graph(self, figure_dir):
        from .matrix import Graph
        matrix = self.config['adjacency_matrix']
        graph = Graph.load_matrix(matrix)
        graph.draw_graph(figure_dir)
        
    def plot_status_graph(self, time, status_vector, opt_value, figure_dir, file_name=None, var_name='x', ylabel='Action'):
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())
        status_vector = np.array(status_vector)
        shape = status_vector.shape
        for i in range(shape[0]):
            for j in range(shape[2]):
                plt.plot(time, np.array(status_vector[i,:,j]), color=mcolors.TABLEAU_COLORS[colors[(i*shape[2]+j)%len(colors)]], label="${}{}{}$".format(var_name, self.charater[i+1], self.charater[j+1]))
            # plt.plot(time, opt_value[i], '--', color=mcolors.TABLEAU_COLORS[colors[2*i+1]], label=["$y{}_1$".format(self.charater[i+1]), "$y{}_2$".format(self.charater[i+1])])

        # plt.legend(loc='lower left', bbox_to_anchor=(0.85, 0))
        plt.xlim(left=0,right=3)
        plt.ylim(top=230)
        # plt.ylim(0, max(opt_value)+2)
        plt.legend(loc='upper right', ncol=2, fontsize=12)
        plt.xlabel('Time(s)', fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        if file_name is not None:
            plt.savefig(figure_dir + f"/{file_name}.png")
        else:
            plt.savefig(figure_dir + "/status.png")
        # print(figure_dir + f"/{file_name}.png")
    
    def plot_trajectory_graph(self, status_vector, figure_dir):
        plt.clf()
        colors = list(mcolors.TABLEAU_COLORS.keys())
        status_vector = np.array(status_vector)
        shape = status_vector.shape
        for i in range(shape[0]):
            x = status_vector[i,:,0]
            y = status_vector[i,:,1]
            plt.plot(x, y, 
             color=colors[i],
             linestyle='-',
             linewidth=1,
             alpha=0.7,
             label=f'Player {i+1}')
            
            # 标记起始点和终点
            plt.scatter(x[0], y[0], color=colors[i], marker='o', s=50, edgecolor='black')
            plt.scatter(x[-1], y[-1], color=colors[i], marker='s', s=50, edgecolor='black')
        plt.xlabel("$x_{i1}$(m)", fontsize=15)
        plt.ylabel("$x_{i2}$(m)", fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),  # 留出2%的边距
            frameon=True,
            framealpha=0.9,
            edgecolor='gray',
            facecolor='white',
            fontsize=11,
            borderpad=0.8,
            borderaxespad=0.5
        )
        # 自动调整坐标轴范围
        margin = 0.1  # 10%的边界留白
        x_min, x_max = np.min(status_vector[:, :, 0]), np.max(status_vector[:, :, 0])
        y_min, y_max = np.min(status_vector[:, :, 1]), np.max(status_vector[:, :, 1])
        plt.xlim(x_min - (x_max - x_min)*margin, x_max + (x_max - x_min)*5*margin)
        plt.ylim(y_min - (y_max - y_min)*margin, y_max + (y_max - y_min)*5*margin)

        # 显示图表
        plt.tight_layout()
        plt.savefig(figure_dir + "/trajectories.png")

    def plot_error_graph(self, time, status_vector, opt_value, figure_dir):

        agent_nums = len(status_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        estimation_error = copy.deepcopy(status_vector)
        for i in range(agent_nums):
            for j in range(len(status_vector[i])):
                estimation_error[i, j] = np.fabs(status_vector[i, j][0] - opt_value[i])

        for i in range(agent_nums):
            plt.clf()
            plt.plot(time, estimation_error[i], color=mcolors.TABLEAU_COLORS[colors[i]])
            plt.plot(time, [0]*len(time), '--', color="black") 
            plt.xlabel('time(sec)')
            plt.ylabel("|x{} - x{}*|".format(self.charater[i+1], self.charater[i+1]))
            plt.xlim(0, 2.5)
            plt.savefig(figure_dir + "/error_{}.jpeg".format(i+1))

    def plot_assemble_error_graph(self, time, status_vector, opt_value, figure_dir):
        plt.clf()
        agent_nums = len(status_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        estimation_error = copy.deepcopy(status_vector)
        for i in range(agent_nums):
            for j in range(len(status_vector[i])):
                estimation_error[i, j] = status_vector[i, j][0] - opt_value[i]

        for i in range(agent_nums):
            plt.plot(time, estimation_error[i], color=mcolors.TABLEAU_COLORS[colors[i]], label=("|x{} - x{}*|".format(self.charater[i+1], self.charater[i+1])))
            plt.legend(loc='lower left', bbox_to_anchor=(0.75, 0))
            plt.xlabel('time(sec)')
            plt.xlim(left=0, right=2.5)
            plt.ylabel("Error between Player's Action and Nash Equilibrium".format(self.charater[i+1], self.charater[i+1]))
            plt.savefig(figure_dir + "/assemble_error.jpeg")
        

    def plot_assembl_estimate_graph(self, time, status_vector, estimate_vector, opt_value, figure_dir):
        plt.clf()
        agent_nums = len(status_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        estimate_error = copy.deepcopy(estimate_vector)
        for i in range(agent_nums):
            for j in range(agent_nums):
                for k in range(len(time)):
                    estimate_error[j, k, i] = estimate_vector[j, k, i] - status_vector[i, k]

        for i in range(agent_nums):
            for j in range(agent_nums):
                plt.plot(time, estimate_error[j, :, i],
                    color=mcolors.TABLEAU_COLORS[colors[(i+j)%(len(colors))]], label="z{}{} - x{}".format(self.charater[j+1], self.charater[i+1],  self.charater[i+1]))
                plt.legend(loc='lower left', bbox_to_anchor=(0.51, 0), ncol=3, prop={'size': 7, 'weight': 'bold'})

        plt.xlabel('time(sec)')
        plt.xlim(0, 2.5)
        plt.ylabel("Error between Player's Action and Estimation from Others".format(self.charater[i+1]))
        plt.savefig(figure_dir+"/assemble_estimation.jpeg".format(i+1))

    def get_settle_result(self, time, optimal_value, status_vector, abs_error, result_dir):

        with open(result_dir + '/results.txt', 'w') as f:
            for i in range(len(status_vector)): 
                value = optimal_value[i]
                value_low = value * (1 - abs_error)
                value_high = value * (1 + abs_error)
                time_high = [0]
                time_low = [0]
            
                for j in range(len(status_vector[i])-1):
                    if (value_low - status_vector[i][j][0]) * (value_low - status_vector[i][j+1]) <= 0:
                        time_low.append(time[j])
                    if (value_high - status_vector[i][j][0]) * (value_high - status_vector[i][j+1]) <= 0:
                        time_high.append(time[j])

                f.write("player {}:\n".format(i))
                f.write("final results:" + str(status_vector[i][-1][0])+"\n")
                f.write("min_time_low: {}\n".format(min(time_low)))
                f.write("min_time_high: {}\n".format(min(time_high)))
                f.write("max_time_low: {}\n".format(max(time_low)))
                f.write("max_time_high: {}\n".format(max(time_high)))
                f.flush()
                
            f.write(self.get_fixed_upper_bound())
            f.flush()
    
    def get_estimate_difference_ns(self, src_vector, dst_vector):
        src_shape = src_vector.shape
        norms = []
        for i in range(src_shape[1]):
            dst_star = dst_vector[:,i,:]
            norm = []
            for j in range(src_shape[0]):
                src_star = src_vector[j,i,:,:]
                error = src_star.flatten()-dst_star.flatten()
                norm.append(np.linalg.norm(error))
            norm = np.array(norm)
            norms.append(np.linalg.norm(norm))
        norms = np.array(norms)
        return norms

    def get_estimate_difference(self, src_vector, dst_vector):
        # print(src_vector.shape, dst_vector.shape)
        if src_vector.shape != dst_vector.shape:
            norms = self.get_estimate_difference_ns(src_vector, dst_vector)
        else:
            difference = dst_vector-src_vector
            shape = difference.shape
            # difference = np.swapaxes(difference, 0, 1) 
            # difference.transpose(1, 0, 2, 3)
            # print(difference.shape)
            norms = np.zeros(shape[1])
            for i in range(shape[1]):
                norms[i] = np.linalg.norm(difference[:,i,:])

        return norms


    def plot_assemble_estimation_graph(self, times, src_vectors, dst_vectors, figure_dir, file_name, y_label='', right=4.5):
        nums = len(src_vectors)
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())

        for i in range(nums):
            norms = self.get_estimate_difference(src_vectors[i], dst_vectors[i])
            # print(norms[-5])
            # temp_index = i if i==0 else i+4
            temp_index = i

            plt.plot(times[i], norms, color=mcolors.TABLEAU_COLORS[colors[temp_index]], label=f"Set {temp_index+1}")

            # plt.annotate(str(opt_value[i]), xy=(1.5, opt_value[i]), xytext=(1.75, opt_value[i]+4),arrowprops=dict(arrowstyle="->", facecolor='blue', edgecolor='blue'))
        plt.legend(fontsize=15)
        plt.xlabel('Time(s)', fontsize=15)
        plt.ylim(bottom=0)
        plt.xlim(left=0, right=right)
        plt.ylabel(y_label, fontsize=15)
        
        plt.savefig(figure_dir+f"/{file_name}.jpeg")


    def plot_compared_graph(self, config_index_list):
        compared_dir = f"/app/records/compared/"
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
        # labels = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
        for index in config_index_list:
            print(f"extracting {index}.......")
            record_path = f"/app/records/{self.config['agent_config']['model']}/{index}"
            records = self.read_records(self.config['agent_config']['model'], index)
            memory = self.extract_records(records)
            status_vector = self.align_list(memory['x'])
            virtual_vector = self.align_list(memory['y'])
            estimate_vector = self.align_list(memory['z'])
            cost_estimate = self.align_list(memory['v'])
            partial_cost = self.align_list(memory['partial_cost'])
            if "oi" in memory.keys():
                oi = self.align_list(memory['oi'])
                zero_opt_value = np.zeros(oi.shape)
                ois.append(oi)
                zero_opt_values.append(zero_opt_value)
            opt_value_signle = status_vector[:,-1,:]
            opt_value = np.zeros(status_vector.shape)
            for j in range(status_vector.shape[1]):
                opt_value[:, j, :] = opt_value_signle
            time = np.array(memory['time'][-1][:len(status_vector[0])])

            status_vectors.append(status_vector)
            virtual_vectors.append(virtual_vector)
            estimate_vectors.append(estimate_vector)
            cost_estimates.append(cost_estimate)
            partial_costs.append(partial_cost)
            opt_values.append(opt_value)
            times.append(time)
        

        self.plot_assemble_estimation_graph(times, estimate_vectors, virtual_vectors, figure_dir, "virtual_status_estimate", "||z-$1_N \otimes y$||", right=0.2)
        self.plot_assemble_estimation_graph(times, cost_estimates, partial_costs, figure_dir, "partial_cost_estimate", "||v-$1_N \otimes F(y)$||", right=0.2)
        self.plot_assemble_estimation_graph(times, virtual_vectors, opt_values, figure_dir, "virtual_status_opt", "||y-$x^{\star}$||", right=3)
        self.plot_assemble_estimation_graph(times, status_vectors, virtual_vectors, figure_dir, "status_virtual", "||x-y||", right=3)
        self.plot_assemble_estimation_graph(times, status_vectors, opt_values, figure_dir, "status_opt", "||x-$x^{\star}$||", right=3)
        if "oi" in memory.keys():
            print(np.array(zero_opt_values).shape, np.array(ois).shape)
            self.plot_assemble_estimation_graph(np.array(times)[:,1:], np.array(ois), np.array(zero_opt_values), figure_dir, "hi_opt", "||$h$||", right=3)


    def plot_graph(self):
        record_path = f"/app/records/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(self.config['agent_config']['model'], self.index)
        memory = self.extract_records(records)
        
        
        
        status_vector = self.align_list(memory['x'])
        virtual_vector = self.align_list(memory['y'])
        estimate_vector = self.align_list(memory['z'])
        cost_estimate = self.align_list(memory['v'])
        partial_cost = self.align_list(memory['partial_cost'])
        opt_value = np.zeros(status_vector.shape)
        time = np.array(memory['time'][-1][:len(status_vector[0])])
        ui = self.align_list(memory['ui'])


        # oi = self.align_list(memory['oi'])
        # track_error = self.align_list(memory['track_error'])
        # dot_x = self.align_list(memory['dot_x'])
        # dot_y = self.align_list(memory['doty'])
        # dot_track_error = self.align_list(memory['dot_track_error'])
        # partial_cost = self.align_list(memory['partial_cost'])

        time = np.array(memory['time'][-1][:len(status_vector[0])])
                
        # config_index_list = ["0","0_5", "0_3", "0_4"]
        self.plot_matrix_graph(figure_dir)
        # self.plot_compared_graph(config_index_list,figure_dir)

        self.plot_status_graph(time, status_vector, virtual_vector, figure_dir)
        self.plot_status_graph(time, virtual_vector, virtual_vector,figure_dir, "virtual_status", 'y')
        time = np.array(memory['time'][-1][:len(ui[0])])
        self.plot_status_graph(time, ui, ui,figure_dir, "ui", "u", "Control torque vector")
        self.plot_trajectory_graph(status_vector, figure_dir)
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
    
    def get_subfigure(self, bbox, xlim, ylim, start, end):
        fig, ax = plt.subplots(1, 1)
        axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=bbox,
                   bbox_transform=ax.transAxes)

        x_ratio = 0.5 # x轴显示范围的扩展比例
        y_ratio = 0.5 # y轴显示范围的扩展比例
        axins.set_xlim(xlim[0], xlim[1])
        axins.set_ylim(ylim[0], ylim[1])
        ax.annotate('', xy=(start[0], start[1]), xytext=(end[0], end[1]),  
            arrowprops=dict(facecolor='black', arrowstyle='->'))
        return ax, axins
    
    def plot_compared_estimate1_error_graph(self, config):
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.4, 0.15, 1, 1),
                   bbox_transform=ax.transAxes)

        x_ratio = 0.5 # x轴显示范围的扩展比例
        y_ratio = 0.5 # y轴显示范围的扩展比例
        axins.set_xlim(0.25, 0.6)
        axins.set_ylim(0, 50)
        ax.annotate('', xy=(0.5, 5), xytext=(1.025, 55),  
            arrowprops=dict(facecolor='black', arrowstyle='->'))

        compared_dir = f"/app/records/compared/"
        labels = config['labels']
        timescale = config['timescale']
        font_size = config['font_size']
        box=config['box']
        colors = list(mcolors.TABLEAU_COLORS.keys())
        compared_name = ''
        count = 0
        for config_index in labels.keys():
            model, index = config_index.split('@')
            records = self.read_records(model, index)
            memory = self.extract_records(records)
            norm_error = copy.deepcopy(np.array(memory['time'][-1]))
            for i in range(len(norm_error)):
                norm_error[i] = 0
                for j in range(len(np.array(memory['z']))):
                    opt_value = np.array(memory['x'])[:,i,0]
                    norm_error[i] += np.linalg.norm(np.array(memory['z'])[j, i] - opt_value)
            ax.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            axins.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            compared_name += config_index
            count += 1
        compared_dir += compared_name
        os.makedirs(compared_dir, exist_ok=True)
        ax.set_xlim(timescale[0], timescale[1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel('time(sec)', fontsize=15)
        ax.set_ylabel("||z - $1_N \otimes x$||", fontsize=15)
        # plt.rcParams.update({'font.size':font_size}) 
        ax.legend(loc='lower left', bbox_to_anchor=(box[0], box[1]), fontsize=15)

        # mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec='k', lw=0.5)
        plt.savefig(compared_dir+"/compared_estimate.jpeg".format(str(labels)[:10]))

                
    def plot_compared_estimate2_error_graph(self, config):
        plt.clf()
        compared_dir = f"/app/records/compared/"
        labels = config['labels']
        timescale = config['timescale']
        font_size = config['font_size']
        box=config['box']
        ax, axins = self.get_subfigure((0.4, 0.14, 1, 1),(0.4, 0.7),(0, 100),(0.6, 30), (1.025,210))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        compared_name = ''
        count = 0
        for config_index in labels.keys():
            model, index = config_index.split('@')
            records = self.read_records(model, index)
            memory = self.extract_records(records)
            norm_error = copy.deepcopy(np.array(memory['time'][-1]))
            for i in range(len(norm_error)):
                norm_error[i] = 0
                for j in range(len(np.array(memory['v']))):
                    opt_value = np.array(memory['partial_cost'])[:,i]
                    norm_error[i] += np.linalg.norm(np.array(memory['v'])[j, i] - opt_value)
            ax.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            axins.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            compared_name += config_index
            count += 1
        compared_dir += compared_name
        os.makedirs(compared_dir, exist_ok=True)
        ax.set_xlim(timescale[0], timescale[1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel('time(sec)', fontsize=15)
        ax.set_ylabel("||v - $1_N \otimes F(x)$||", fontsize=15)
        # plt.rcParams.update({'font.size':font_size}) 
        ax.legend(loc='lower left', bbox_to_anchor=(box[0], box[1]), fontsize=15)
        plt.savefig(compared_dir+"/compared_estimate2.jpeg".format(str(labels)[:10]))

    # def plot_compared_graph(self, config):
    #     plt.clf()
    #     compared_dir = f"/app/records/compared/"
    #     labels = config['labels']
    #     timescale = config['timescale']
    #     font_size = config['font_size']
    #     box=config['box']
    #     opt_value = [26.0990, 31.0459, 36.0000, 40.9505, 45.9000]
    #     colors = list(mcolors.TABLEAU_COLORS.keys())
    #     compared_name = ''
    #     count = 0
    #     for config_index in labels.keys():
    #         model, index = config_index.split('@')
    #         records = self.read_records(model, index)
    #         memory = self.extract_records(records)
    #         norm_error = copy.deepcopy(np.array(memory['time'][-1]))
    #         for i in range(len(norm_error)):
    #             norm_error[i] = np.linalg.norm(np.array(memory['x'])[:,i])
    #         plt.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         compared_name += config_index
    #         count += 1
    #     compared_dir += compared_name
    #     os.makedirs(compared_dir, exist_ok=True)
    #     plt.xlim(timescale[0], timescale[1])
    #     plt.ylim(bottom=0)
    #     plt.xlabel('time(sec)', fontsize=15)
    #     plt.ylabel("||x - x*||", fontsize=15)
    #     # plt.rcParams.update({'font.size':font_size}) 
    #     plt.legend(loc='lower left', bbox_to_anchor=(box[0], box[1]), fontsize=12)
    #     plt.savefig(compared_dir+"/compared.png".format(str(labels)[:10]))



class FixedSettleTime:
    
    @staticmethod
    def get_settle_time2019(p, q, alpha, beta):

        k = 1
        mp = (1-k*p) / (q-p)
        mq = (k*q-1) / (q-p)

        gama = sp.gamma(mp)*sp.gamma(mq) / (np.power(alpha, k) * sp.gamma(k)* (q-p))

        gama = gama * np.power((alpha/beta), mp)

        return gama
    
    @staticmethod
    def get_settle_time2013(p, q, alpha, beta):
        
        settle_time = (1/(alpha*(1-p)) + 1/(beta*(q-1)))

        return settle_time
    
    @staticmethod
    def get_settle_time2021(p, q, c1, c2, c3):
        settle_time = np.pi / ((q-p)*np.power(c3, (p-1)/(q-p))*np.power(c1, (q-1)/(q-p))*np.power(c2, (1-p)/(q-p))*np.sin((1-p)/(q-p)*np.pi) )
        return settle_time
    
    @staticmethod
    def get_settle_time2020(p, q, c1, c2, c3):
        settle_time =  np.log(1+(c3/c1))/(c3*(1-p)) + np.log(1+(c3/c2))/(c3*(q-1))
        return settle_time
    
    
    @staticmethod
    def get_settle_time2024(p, q, alpha, beta):
        
        settle_time = (1/alpha)*(np.power(alpha/beta, (1-p)/(q-p)))*(1/(1-p) + 1/(q-1))

        return settle_time


    @staticmethod
    def get_equilibrium_settle_parameter(p, q, delta, eta, m=2):

        delta = 2**((p+1)/2)*delta*(m**p)
        eta = 2**((q+1)/2)*eta*(m**q)
        p_hat = (p+1)/2
        q_hat = (q+1)/2
                
        return delta, eta, p_hat, q_hat

    @staticmethod
    def get_directed_consensus_parameter(p, q, alpha, beta, matrix):
        
        N = len(matrix)
        graph = Graph.load_matrix(matrix)
        min_eigenvalue = min(graph.get_LM_eigenvalue_from_matrix())

        k2 = (alpha/(p+1))**(2*p/(p+1)) + (beta/(q+1))**(2*p/(p+1)) + (2**((q-1)/(q+1)))*((alpha/(p+1))**(2*q/(q+1)))+ (2**((q-1)/(q+1)))*((beta/(q+1))**(2*q/(q+1)))
        k3 = min((alpha**2)*((N*N)**(1-2*p))/k2, (beta**2)*((N*N)**(1-2*q))/k2)
        
        return 0.5*k3*min_eigenvalue, 0.5*k3*min_eigenvalue, 2*p/(p+1), 2*q/(q+1)


if __name__ == "__main__":
    a_matrix = np.array([[0, 1, 0, 0, 1],
                         [1, 0, 1, 0, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 0]])
    
    d_matrix = np.array([[2, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 2]])
    laplapian_matrix = d_matrix - a_matrix
    I_matrix = np.eye(5)
    M_matrix = np.zeros((25, 25))
    for i in range(5):
        for j in range(5):
            if a_matrix[i][j] > 0:
                    M_matrix[int(i*5+j)][int(i*5+j)] = 1
    L_otimics_I = np.kron(laplapian_matrix, I_matrix)
    P = L_otimics_I + M_matrix
    eigenvalue,_= np.linalg.eig((P))
    print(max(eigenvalue), min(eigenvalue))
    max_eigenvalue = max(eigenvalue)
    min_eigenvalue = min(eigenvalue)
    mu = 0.5
    nu = 1.5
    alhpa_min = 200
    beta_min = 200
    
    varphi_min = 200
    sigma_min = 200
    
    # 2.27 ==> 2.27
    # 3.40 ==> 3.5
    # 1.13 ==> 1.2
    
    c1 = np.power(2,(mu+1)/2) * alhpa_min * np.power(min_eigenvalue, mu+1)/np.power(max_eigenvalue, (mu+1)/2)
    c2 = np.power(2, (nu+1)/2) * beta_min * np.power(min_eigenvalue, nu+1)/np.power(max_eigenvalue, (nu+1)/2) / np.power(5, 0.5)
    c5 = np.power(2, 1) * alhpa_min * np.power(min_eigenvalue, 2)/np.power(max_eigenvalue, 1)
    
    c3 = np.power(2, (mu+1)/2) * varphi_min * np.power(min_eigenvalue, mu+1)/np.power(max_eigenvalue, (mu+1)/2)
    c4 = np.power(2, (nu+1)/2) * sigma_min * np.power(min_eigenvalue, nu+1)/np.power(max_eigenvalue, (nu+1)/2) / np.power(5, 0.5)
    c6 = np.power(2, 1) * alhpa_min * np.power(min_eigenvalue, 2)/np.power(max_eigenvalue, 1)
    
    # T1 = FixedSettleTime.get_settle_time2024((mu+1)/2, (nu+1)/2, c1, c2)
    # T2 = FixedSettleTime.get_settle_time2024((mu+1)/2, (nu+1)/2, c3, c4)
    # print(T1, T2)
    
    # T1 = FixedSettleTime.get_settle_time2013((mu+1)/2, (nu+1)/2, c1, c2)
    # T2 = FixedSettleTime.get_settle_time2013((mu+1)/2, (nu+1)/2, c3, c4)
    # print(T1, T2)
    
    # T3 = FixedSettleTime.get_settle_time2021((mu+1)/2, (nu+1)/2, c1, c2, c5)
    # T4 = FixedSettleTime.get_settle_time2021((mu+1)/2, (nu+1)/2, c1, c2, c6)
    
    # print(T3, T4, T3+T4)
    
    T3 = FixedSettleTime.get_settle_time2020((mu+1)/2, (nu+1)/2, c1, c2, c5)
    T4 = FixedSettleTime.get_settle_time2020((mu+1)/2, (nu+1)/2, c3, c4, c6)
    print(c5, c6)
    
    print(T3, T4, T3+T4)
    
    # print(T1, T2, T1+T2, c1 ,c2 ,c3 ,c4)