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
        
    def plot_status_graph(self, time, status_vector, opt_value, figure_dir, file_name=None):
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(len(status_vector)):
            plt.plot(time, np.array(status_vector[i]), color=mcolors.TABLEAU_COLORS[colors[i]], label="$x{}(t)$".format(self.charater[i+1]))
            # plt.plot(time, opt_value[i], '--', color=mcolors.TABLEAU_COLORS[colors[2*i+1]], label=["$y{}_1$".format(self.charater[i+1]), "$y{}_2$".format(self.charater[i+1])])

        # plt.legend(loc='lower left', bbox_to_anchor=(0.85, 0))
        plt.xlim(left=0)
        plt.ylim(top=10)
        # plt.ylim(0, max(opt_value)+10)
        plt.legend( ncol=2, fontsize=15)
        plt.xlabel('time', fontsize=15)
        plt.ylabel('Transient responses of Players 1-6', fontsize=15)
        if file_name is not None:
            plt.savefig(figure_dir + f"/{file_name}.png")
        else:
            plt.savefig(figure_dir + "/status.png")
        

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
            plt.xlabel('time(sec)')
            plt.ylim(0, opt_value[i]+10)
            plt.ylabel("x{} and its estimates from other players".format(self.charater[i+1]))
            plt.savefig(figure_dir+"/estimation_{}.jpeg".format(i+1))


    def get_interval_info(self, data):
        difference = [abs(data[i+1] - data[i]) for i in range(len(data)-1)]
        max_value = max(difference)
        average_value = sum(difference) / len(difference)
        counts = len(difference)

        return max_value, average_value ,counts



    def get_interval_detail(self, trigger_value, result_dir, time_delta, figure_dir):

        mark = ['o', 'x', '*', '+', 'h', 'd']
        trigger_times = []
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(len(trigger_value)):
            trigger_time = [[], [], [], [], [], []]
            for j in range(len(trigger_value[i])):
                for k in range(len(trigger_value[i][j])):
                    if trigger_value[i,j,k] > 0:
                        trigger_time[k].append(j*time_delta)
            trigger_times.append(trigger_time)
            # with open(f"{result_dir}/{i}_trigger.txt", 'w') as f:
            #     f.write(json.dumps(trigger_time))
            #     f.flush()

        
        fp = open(f"{result_dir}/summarize.txt", 'w')
        for i in range(len(trigger_times)):
            plt.clf()
            for j in range(len(trigger_times[i])):
                plt.scatter(trigger_times[i][j], [j+1]*len(trigger_times[i][j]), marker=mark[j],  color=mcolors.TABLEAU_COLORS[colors[(j)%(len(colors))]])
                max_value, average_value, counts = self.get_interval_info(trigger_times[i][j])
                fp.write(f"estimate {i}_{j} : max_value = {max_value}, average_value = {average_value}, counts = {counts} \n")
                fp.flush()

            plt.xlim(left=0)
            plt.savefig(figure_dir+"/evet_trigger_{}.jpeg".format(i+1))
        
        fp.close()




    def plot_estimation_graph(self, time, estimate_vector, opt_value, figure_dir):
        agent_nums = len(estimate_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(agent_nums):
            plt.clf()
            for j in range(agent_nums):
                plt.plot(time, estimate_vector[j, : , i],
                    color=mcolors.TABLEAU_COLORS[colors[(i+j)%(len(colors))]], label="z{}{}".format(self.charater[j+1], self.charater[i+1]))
            # plt.plot(time, [opt_value[i]]*len(time), '--', label="x{}*".format(self.charater[i+1]))
            # plt.annotate(str(opt_value[i]), xy=(1.5, opt_value[i]), xytext=(1.75, opt_value[i]+4),arrowprops=dict(arrowstyle="->", facecolor='blue', edgecolor='blue'))
            plt.legend()
            plt.xlabel('time(sec)')
            # plt.ylim(0, opt_value[i]+10)
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
            plt.xlabel('time(sec)')
            plt.ylabel("|x{} - x{}*|".format(self.charater[i+1], self.charater[i+1]))
            plt.xlim(0, 2.5)
            plt.savefig(figure_dir + "/error_{}.jpeg".format(i+1))


    def plot_assembl_estimate_graph(self, time, estimate_vector, opt_value, figure_dir):
        plt.clf()
        agent_nums = len(estimate_vector)
        colors = list(mcolors.TABLEAU_COLORS.keys())
        estimate_error = copy.deepcopy(estimate_vector)

        for i in range(agent_nums):
            for j in range(agent_nums):
                plt.plot(time, estimate_error[i, :, j],
                    color=mcolors.TABLEAU_COLORS[colors[(i)%(len(colors))]], label="z{}{}".format(self.charater[j+1], self.charater[i+1]))
                plt.plot(time, [opt_value[i]]*len(time), '--', color=mcolors.TABLEAU_COLORS[colors[(i)%(len(colors))]])
                # plt.legend(loc='lower left', bbox_to_anchor=(0.51, 0), ncol=3, prop={'size': 7, 'weight': 'bold'})

        plt.xlabel('time')
        plt.xlim(left=0)
        plt.ylabel("The estimates on $x_i$ of players")
        plt.savefig(figure_dir+"/assemble_estimation.jpeg".format(i+1))


    def plot_graph(self):
        record_path = f"/app/records/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(self.config['agent_config']['model'], self.index)
        memory = self.extract_records(records)
        
        status_vector = self.align_list(memory['x'])
        is_trigger_vector = self.align_list(memory['is_trigger'])
        estimate_vector = self.align_list(memory['z'])
        opt_value = [-3.007632442244374, -3.807408455069461, -2.6078307207927387, -2.207915437999935, -1.4074073115710062, 4.073985201697069]
        time = np.array(memory['time'][-1][:len(status_vector[0])])
                
        
        self.plot_matrix_graph(figure_dir)
        self.plot_status_graph(time, status_vector, status_vector, figure_dir, "status")
        self.get_interval_detail(is_trigger_vector, result_dir, 1e-4, figure_dir)
        self.plot_assembl_estimate_graph(time, estimate_vector, opt_value, figure_dir)