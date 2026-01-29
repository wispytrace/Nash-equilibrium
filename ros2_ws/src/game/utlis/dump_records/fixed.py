import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import copy
import json
import scipy.special as sp
from collections import defaultdict
from .matrix import Graph
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams['figure.dpi'] = 600 
plt.rcParams['lines.linewidth'] = 0.5

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
        matrix = self.config['adjacency_matrix']
        graph = Graph.load_matrix(matrix)
        graph.draw_graph(figure_dir)
        
    def plot_status_graph(self, time, status_vector, opt_value, figure_dir):
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(len(status_vector)):
            plt.plot(time, status_vector[i],
                        color=mcolors.TABLEAU_COLORS[colors[2*i]], label="x{}".format(self.charater[i+1]))
            plt.plot(time, [opt_value[i]]*len(time), '--', color=mcolors.TABLEAU_COLORS[colors[2*i+1]], label="x{}*".format(self.charater[i+1]))

        # plt.legend(loc='lower left', bbox_to_anchor=(0.85, 0))
        plt.xlim(0, 2.5)
        plt.ylim(0, max(opt_value)+4)
        plt.legend(loc='upper right', ncol=2, fontsize=12)
        plt.xlabel('Time(s)', fontsize=15)
        plt.ylabel('Action', fontsize=15)
        plt.savefig(figure_dir + "/status.png")
        print("save to", figure_dir + "/status.png")

    def plot_ui_graph(self, time, ui, figure_dir):
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(len(ui)):
            plt.plot(time, ui[i],
                        color=mcolors.TABLEAU_COLORS[colors[2*i]], label="u{}".format(self.charater[i+1]))

        # plt.legend(loc='lower left', bbox_to_anchor=(0.85, 0))
        plt.xlim(0, 2.5)
        # plt.ylim(bottom=0)
        plt.legend(loc='upper right', fontsize=12)
        plt.xlabel('Time(s)', fontsize=15)
        plt.ylabel('Control input', fontsize=15)
        plt.savefig(figure_dir + "/status_update.png")
        print("save to "+figure_dir + "/status_update.png")
        

    def plot_estimation_graph(self, time, status_vector, estimate_vector, opt_value, figure_dir):
        agent_nums = len(status_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(agent_nums):
            plt.clf()
            plt.plot(time, status_vector[i],
                        color=mcolors.TABLEAU_COLORS[colors[i]], label="x {}".format(self.charater[i+1]))
            for j in range(agent_nums):
                plt.plot(time, estimate_vector[j, : , i] - status_vector[i][0],
                    color=mcolors.TABLEAU_COLORS[colors[(i+j)%(len(colors))]], label="z{}{}".format(self.charater[j+1], self.charater[i+1]))
            plt.plot(time, [opt_value[i]]*len(time), '--', label="x{}*".format(self.charater[i+1]))
            plt.annotate(str(opt_value[i]), xy=(1.5, opt_value[i]), xytext=(1.75, opt_value[i]+4),arrowprops=dict(arrowstyle="->", facecolor='blue', edgecolor='blue'))
            plt.legend()
            plt.xlabel('Time(s)')
            plt.ylim(0, opt_value[i]+10)
            plt.ylabel("x{} and its estimates from other players".format(self.charater[i+1]))
            plt.savefig(figure_dir+"/estimation_{}.jpeg".format(i+1))

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
            plt.xlabel('Time(s)')
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
            plt.xlabel('Time(s)')
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

        plt.xlabel('Time(s)')
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
    
    def get_fixed_upper_bound(self):
        try:
            alpha = self.config['agent_config']['model_config']['share']['min_c1']
            beta = alpha
            delta = self.config['agent_config']['model_config']['share']['min_delta']
            eta = delta
            matrix = self.config['adjacency_matrix']
            p = self.config['agent_config']['model_config']['share']['p']
            q = self.config['agent_config']['model_config']['share']['q']
            
            c_1, c_2, p_hat, q_hat = FixedSettleTime.get_directed_consensus_parameter(p, q, alpha, beta, matrix)
            dircted_consensus_2013 = FixedSettleTime.get_settle_time2013(p_hat, q_hat, c_1, c_2)
            dircted_consensus_2019 = FixedSettleTime.get_settle_time_2019(p_hat, q_hat, c_1, c_2)
            
            c_1, c_2, p_hat, q_hat = FixedSettleTime.get_equilibrium_settle_parameter(p, q, delta, eta)
            equilibrium_consensus_2013 = FixedSettleTime.get_settle_time2013(p_hat, q_hat, c_1, c_2)
            equilibrium_consensus_2019 = FixedSettleTime.get_settle_time_2019(p_hat, q_hat, c_1, c_2)
            
            res = f'''dircted_consnsus_time(2013): {dircted_consensus_2013}
            dircted_consnsus_time(2013): {dircted_consensus_2019}
            equilibirum_consensus(2013): {equilibrium_consensus_2013}
            equilibirum_consensus(2019): {equilibrium_consensus_2019}
            total_time(2013): {dircted_consensus_2013 + equilibrium_consensus_2013}
            total_time(2019): {dircted_consensus_2019 + equilibrium_consensus_2019}
            '''
            return res
        except Exception as e:
            print(repr(e))
            return "None about fixed time"

    def plot_graph(self):
        record_path = f"/app/records/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(self.config['agent_config']['model'], self.index)
        memory = self.extract_records(records)
        
        time = np.array(memory['time'][-1])
        status_vector = np.array(memory['x'])
        estimate_vector = np.array(memory['z'])
        # ui = np.array(memory['update_value'])
        
        
        opt_value = [2.0475755648379073, 2.496178193752349, 2.9672215956150314, 3.421972769128372, 3.876074463373425]
        # abs_error = 0.02
        self.plot_matrix_graph(figure_dir)
        self.plot_status_graph(time, status_vector, opt_value, figure_dir)
        # self.plot_ui_graph(time[:-1], ui, figure_dir)
        # self.plot_estimation_graph(time, status_vector, estimate_vector, opt_value, figure_dir)
        self.plot_error_graph(time, status_vector, opt_value, figure_dir)
        self.plot_assemble_error_graph(time, status_vector, opt_value, figure_dir)
        # self.plot_assembl_estimate_graph(time, status_vector, estimate_vector, opt_value, figure_dir)
        # self.get_settle_result(time, opt_value, status_vector, abs_error, result_dir)
    
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
    
    # def plot_compared_estimate1_error_graph(self, config):
    #     plt.clf()
    #     fig, ax = plt.subplots(1, 1)
    #     axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
    #                bbox_to_anchor=(0.4, 0.15, 1, 1),
    #                bbox_transform=ax.transAxes)

    #     x_ratio = 0.5 # x轴显示范围的扩展比例
    #     y_ratio = 0.5 # y轴显示范围的扩展比例
    #     axins.set_xlim(0.25, 0.6)
    #     axins.set_ylim(0, 50)
    #     ax.annotate('', xy=(0.5, 5), xytext=(1.025, 55),  
    #         arrowprops=dict(facecolor='black', arrowstyle='->'))

    #     compared_dir = f"/app/records/compared/"
    #     labels = config['labels']
    #     timescale = config['timescale']
    #     font_size = config['font_size']
    #     box=config['box']
    #     colors = list(mcolors.TABLEAU_COLORS.keys())
    #     compared_name = ''
    #     count = 0
    #     for config_index in labels.keys():
    #         model, index = config_index.split('@')
    #         records = self.read_records(model, index)
    #         memory = self.extract_records(records)
    #         norm_error = copy.deepcopy(np.array(memory['time'][-1]))
    #         for i in range(len(norm_error)):
    #             norm_error[i] = 0
    #             for j in range(len(np.array(memory['z']))):
    #                 opt_value = np.array(memory['x'])[:,i,0]
    #                 norm_error[i] += np.linalg.norm(np.array(memory['z'])[j, i] - opt_value)
    #         ax.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         axins.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         compared_name += config_index
    #         count += 1
    #     compared_dir += compared_name
    #     os.makedirs(compared_dir, exist_ok=True)
    #     ax.set_xlim(timescale[0], timescale[1])
    #     ax.set_ylim(bottom=0)
    #     ax.set_xlabel('time(sec)', fontsize=15)
    #     ax.set_ylabel("||z - $1_N \otimes x$||", fontsize=15)
    #     # plt.rcParams.update({'font.size':font_size}) 
    #     ax.legend(loc='upper right', fontsize=15)

    #     # mark_inset(ax, axins, loc1=1, loc2=1, fc="none", ec='k', lw=0.5)
    #     plt.savefig(compared_dir+"/compared_estimate.jpeg".format(str(labels)[:10]))


    def plot_compared_estimate1_error_graph(self, config):
        plt.clf()
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
            plt.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            compared_name += config_index
            count += 1
        compared_dir += compared_name
        os.makedirs(compared_dir, exist_ok=True)
        plt.xlim(timescale[0], timescale[1])
        plt.ylim(bottom=0)
        plt.xlabel('Time(s)',  fontsize=15)
        plt.ylabel("||z - $1_N \otimes x$||",  fontsize=15)
        plt.rcParams.update({'font.size':font_size}) 
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig(compared_dir+"/compared_estimate.jpeg".format(str(labels)[:10]))

    def plot_compared_estimate2_error_graph(self, config):
        plt.clf()
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
                for j in range(len(np.array(memory['v']))):
                    opt_value = np.array(memory['partial_cost'])[:,i]
                    norm_error[i] += np.linalg.norm(np.array(memory['v'])[j, i] - opt_value)
            plt.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            compared_name += config_index
            count += 1
        compared_dir += compared_name
        os.makedirs(compared_dir, exist_ok=True)
        plt.xlim(timescale[0], timescale[1])
        plt.ylim(bottom=0)
        plt.xlabel('Time(s)',  fontsize=15)
        plt.ylabel("||v - $1_N \otimes F(x)$||",  fontsize=15)
        plt.rcParams.update({'font.size':font_size}) 
        plt.legend(loc='upper right', fontsize=15)
        plt.savefig(compared_dir+"/compared_estimate2.jpeg".format(str(labels)[:10]))
                
    # def plot_compared_estimate2_error_graph(self, config):
    #     plt.clf()
    #     compared_dir = f"/app/records/compared/"
    #     labels = config['labels']
    #     timescale = config['timescale']
    #     font_size = config['font_size']
    #     box=config['box']
    #     ax, axins = self.get_subfigure((0.4, 0.14, 1, 1),(0.4, 0.7),(0, 100),(0.6, 30), (1.025,210))
    #     colors = list(mcolors.TABLEAU_COLORS.keys())
    #     compared_name = ''
    #     count = 0
    #     for config_index in labels.keys():
    #         model, index = config_index.split('@')
    #         records = self.read_records(model, index)
    #         memory = self.extract_records(records)
    #         norm_error = copy.deepcopy(np.array(memory['time'][-1]))
    #         for i in range(len(norm_error)):
    #             norm_error[i] = 0
    #             for j in range(len(np.array(memory['v']))):
    #                 opt_value = np.array(memory['partial_cost'])[:,i]
    #                 norm_error[i] += np.linalg.norm(np.array(memory['v'])[j, i] - opt_value)
    #         ax.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         axins.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         compared_name += config_index
    #         count += 1
    #     compared_dir += compared_name
    #     os.makedirs(compared_dir, exist_ok=True)
    #     ax.set_xlim(timescale[0], timescale[1])
    #     ax.set_ylim(bottom=0)
    #     ax.set_xlabel('time(sec)', fontsize=15)
    #     ax.set_ylabel("||v - $1_N \otimes F(x)$||", fontsize=15)
    #     # plt.rcParams.update({'font.size':font_size}) 
    #     ax.legend(loc='upper right', fontsize=15)
    #     plt.savefig(compared_dir+"/compared_estimate2.jpeg".format(str(labels)[:10]))



    def plot_compared_graph(self, config):
        plt.clf()
        compared_dir = f"/app/records/compared/"
        labels = config['labels']
        timescale = config['timescale']
        font_size = config['font_size']
        box=config['box']
        opt_value = [2.0596, 2.5142, 2.9687, 3.4232, 3.8778]
        colors = list(mcolors.TABLEAU_COLORS.keys())
        compared_name = ''
        count = 0
        is_cite = {}
        for config_index in labels.keys():
            model, index = config_index.split('@')
            records = self.read_records(model, index)
            memory = self.extract_records(records)
            norm_error = copy.deepcopy(np.array(memory['time'][-1]))
            for i in range(len(norm_error)):
                norm_error[i] = np.linalg.norm(np.array(memory['x'])[:,i,0] - opt_value)

                if norm_error[i] < 5e-4:
                    if config_index  not in is_cite:
                        is_cite[config_index] = True
                    # print(i)
                        print(config_index, memory['time'][-1][i])
                else:
                    if config_index in is_cite:
                        del is_cite[config_index]
                norm_error[i] = np.log10(norm_error[i])
            plt.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
            compared_name += config_index
            count += 1
        compared_dir += compared_name
        os.makedirs(compared_dir, exist_ok=True)
        plt.xlim(timescale[0], timescale[1])
        # plt.ylim(bottom=0)
        plt.xlabel('Time(s)',  fontsize=15)
        plt.ylabel("$log_{10}(||x - x*||)$", fontsize=15)
        # plt.ylabel("||x - x*||", fontsize=15)
        # plt.rcParams.update({'font.size':font_size}) 
        plt.legend(loc='upper right', fontsize=10)
        plt.savefig(compared_dir+"/compared.png".format(str(labels)[:10]))
    
    # def plot_compared_graph(self, config):
    #     plt.clf()
    #     ax, axins = self.get_subfigure((0.4, 0.14, 1, 1),(0.75, 1.75),(0, 10),(0.9, 2), (1.025,8))
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
    #             norm_error[i] = np.linalg.norm(np.array(memory['x'])[:,i,0] - opt_value)
    #         ax.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         axins.plot(np.array(memory['time'][-1]), norm_error, color=mcolors.TABLEAU_COLORS[colors[count%(len(colors))]], label=labels[config_index])
    #         compared_name += config_index
    #         count += 1
    #     compared_dir += compared_name
    #     os.makedirs(compared_dir, exist_ok=True)
    #     ax.set_xlim(timescale[0], timescale[1])
    #     ax.set_ylim(bottom=0)
    #     ax.set_xlabel('time(sec)', fontsize=15)
    #     ax.set_ylabel("||x - x*||", fontsize=15)
    #     # plt.rcParams.update({'font.size':font_size}) 
    #     ax.legend(loc='lower left', bbox_to_anchor=(box[0], box[1]), fontsize=12)
    #     plt.savefig(compared_dir+"/compared.jpeg".format(str(labels)[:10]))
    
    

class FixedSettleTime:
    
    @staticmethod
    def get_settle_time_2019(p, q, alpha, beta):

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
  
