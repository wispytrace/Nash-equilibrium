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
# plt.style.use('classic')
plt.rcParams.update({
    'font.size': 16,              # 根据需要设置适当的大小
})
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
        
    def plot_status_graph(self, time, status_vector, opt_value, figure_dir, file_name=None, ylabel=None, xlabel='x', ytop=10):
        plt.clf()

        colors = list(mcolors.TABLEAU_COLORS.keys())
        for i in range(len(status_vector)):
            plt.plot(time, np.array(status_vector[i]), color=mcolors.TABLEAU_COLORS[colors[i]], label="${}{}(t)$".format(xlabel,self.charater[i+1]))

        plt.xlim(left=0, right=2)
        plt.ylim(top=ytop)
        # plt.ylim(0, max(opt_value)+10)
        plt.legend( ncol=3)
        plt.xlabel('time')
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel('Transient responses of Players 1-6')
        if file_name is not None:
            plt.savefig(figure_dir + f"/{file_name}.png")
            plt.savefig(figure_dir + f"/{file_name}.eps")
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
            plt.xlabel('time')
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
            plt.xlabel('time')
            plt.ylabel("|x{} - x{}*|".format(self.charater[i+1], self.charater[i+1]))
            plt.xlim(0, 2.5)
            plt.savefig(figure_dir + "/error_{}.jpeg".format(i+1))

    def plot_assembl_estimate_graph(self, time, status_vector, estimate_vector, opt_value, figure_dir, ytop):
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
                    color=mcolors.TABLEAU_COLORS[colors[(i*agent_nums+j)%(len(colors))]], label="z{}{} - x{}".format(self.charater[j+1], self.charater[i+1],  self.charater[i+1]))
                plt.legend(loc='lower left', bbox_to_anchor=(0.51, 0.5), ncol=3, prop={'size': 7, 'weight': 'bold'})

        plt.xlabel('time')
        plt.xlim(0, 2)
        if ytop is not None:
            plt.ylim(top=ytop)
        plt.ylabel("Error between Player's Action and Estimation from Others")
        plt.savefig(figure_dir+"/assemble_estimation.jpeg")

    # def plot_assembl_estimate_graph(self, time, estimate_vector, opt_value, figure_dir):
    #     plt.clf()
    #     agent_nums = len(estimate_vector)
    #     colors = list(mcolors.TABLEAU_COLORS.keys())
    #     estimate_error = copy.deepcopy(estimate_vector)

    #     for i in range(agent_nums):
    #         for j in range(agent_nums):
    #             plt.plot(time, estimate_error[i, :, j],
    #                 color=mcolors.TABLEAU_COLORS[colors[(i)%(len(colors))]], label="z{}{}".format(self.charater[j+1], self.charater[i+1]))
    #             plt.plot(time, [opt_value[i]]*len(time), '--', color=mcolors.TABLEAU_COLORS[colors[(i)%(len(colors))]])
    #             # plt.legend(loc='lower left', bbox_to_anchor=(0.51, 0), ncol=3, prop={'size': 7, 'weight': 'bold'})

    #     plt.xlabel('time')
    #     plt.xlim(left=0)
    #     plt.ylabel("The estimates on $x_i$ of players")
    #     plt.savefig(figure_dir+"/assemble_estimation.jpeg".format(i+1))


    def plot_graph(self):
        record_path = f"/app/records/{self.config['agent_config']['model']}/{self.index}"
        figure_dir = record_path + "/figure"
        result_dir = record_path + "/result"
        os.makedirs(figure_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        records = self.read_records(self.config['agent_config']['model'], self.index)
        memory = self.extract_records(records)
        
        status_vector = self.align_list(memory['x'])
        estimate_vector = self.align_list(memory['z'])
        cost = self.align_list(memory['cost'])
        for i in range(len(cost)):
            print(len(cost[i]))
        opt_value = [-3.007632442244374, -3.807408455069461, -2.6078307207927387, -2.207915437999935, -1.4074073115710062, 4.073985201697069]
        time = np.array(memory['time'][-1][:len(status_vector[0])])
                
        
        self.plot_matrix_graph(figure_dir)
        self.plot_status_graph(time, status_vector, status_vector, figure_dir, "Transient responses of Players 1-6", ytop=6)
        self.plot_assembl_estimate_graph(time, status_vector, estimate_vector, opt_value, figure_dir, 0.5)

        time = time[:len(cost[0])]
        self.plot_status_graph(time, cost, cost, figure_dir, "cost", "Cost Function",xlabel="f", ytop=400)
        # self.plot_assembl_estimate_graph(time, estimate_vector, opt_value, figure_dir)
        # self.plot_estimation_graph(time, estimate_vector, opt_value, figure_dir)