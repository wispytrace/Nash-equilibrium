from dump_records import *
import sys
sys.path.append('/app/ros2_ws/src/game/configs')
import configs

config_name = str(sys.argv[1])
config_index = str(sys.argv[2])
plot_model = str(sys.argv[3])


config = configs.CONFIG_MAP[config_name][config_index]
plot = Draw[plot_model](config, config_index)

plot.plot_graph()


# labels = configs.CONFIG_MAP["fixed"]['compared']['labels']
# plot.plot_compared_graph({
#         'box': [0.45, 0.82],
#         'timescale': [0, 10],
#         'font_size' : 8,
#         'labels' : {"asym2@asym_4": "Algorithm from [23]", "asym@exp_2": "Algorithm from [25]", "fixed@p1": "Algorithm from [31]", "fixed4@r_r1":"Algorithm (6)-(11)"}},
# )

# plot.plot_compared_graph({
#         'box': [0.45, 0.82],
#         'timescale': [0, 10],
#         'font_size' : 8,
#         'labels' : {"asym2@asym_4": "Lu, et al., 2018", "asym@exp_2": "Zou, eta al., 2021", "fixed@p1": "Sun, eta al., 2020",  "fixed4@r_r1":"Fixed-time algorithm (6)-(11)"}},
# )
# plot.plot_compared_graph({
#         'box': [0.25, 0.75],
#         'timescale': [0, 6],
#         'font_size' : 8,
#         'labels' : {"euler@2": "Asymptotic convergence algorithm", "euler_asym@1": "Fixed-time convergence algorithm"},
#     })

# plot.plot_compared_estimate1_error_graph({
#         'box': [0.68, 0.60],
#         'timescale': [0, 0.2],
#         'font_size' : 8,
#         'labels' : {"fixed4@r_0": "Set 1", "fixed4@r_2": "Set 2", "fixed4@r_3": "Set 3", "fixed4@r_4": "Set 4"},
#     })

# plot.plot_compared_estimate2_error_graph({
#         'box': [0.68, 0.60],
#         'timescale': [0, 0.2],
#         'font_size' : 8,
#         'labels' : {"fixed4@r_0": "Set 1", "fixed4@r_2": "Set 2", "fixed4@r_3": "Set 3", "fixed4@r_4": "Set 4"},
#     })


# plot.plot_compared_graph({
#         'box': [0.68, 0.60],
#         'timescale': [0, 5],
#         'font_size' : 8,
#         'labels' : {"fixed4@r_0": "Set 1", "fixed4@r_2": "Set 2", "fixed4@r_3": "Set 3", "fixed4@r_4": "Set 4"},
#     })

# plot.plot_compared_estimate1_error_graph({
#         'box':  [0.68, 0.60],
#         'timescale': [0, 2.5],
#         'font_size' : 8,
#         'labels' : {"fixed4@5": "Set 1", "fixed4@6": "Set 2", "fixed4@7": "Set 3", "fixed4@8": "Set 4"},
#     })

# plot.plot_compared_estimate2_error_graph({
#         'box':  [0.68, 0.60],
#         'timescale': [0, 2.5],
#         'font_size' : 8,
#         'labels' : {"fixed4@5": "Set 1", "fixed4@6": "Set 2", "fixed4@7": "Set 3", "fixed4@8": "Set 4"},
#     })


# plot.plot_compared_graph(["3", "3_1", "3_2", "3_3"])
# plot.plot_compared_graph(["3", "3_7"])
# plot.plot_compared_graph(["3", "3_8"])
# plot.plot_compared_graph(["3", "3_2"])
# plot.plot_compared_graph(["3", "3_10"])
# plot.plot_compared_graph(["3", "3_9"])
# plot.plot_compared_graph(["3", "3_5"])

# plot.plot_compared_graph(["3", "3_7", "3_8", "3_2", "3_10", "3_9"])

