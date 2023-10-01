
import json
import matplotlib.pyplot as plt
# imports framework
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
from scipy.stats import ttest_ind,wilcoxon

n_hidden_neurons = 10

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))



filename = 'final_boxplot.json'



with open(filename, 'r') as json_file:
    best_solution_dict = json.load(json_file)




#this loop takes the solution from the json file and then runs the game and forms another dictipnary to calculate the gain
enemy_dict = {}
for enemy_name,target in best_solution_dict.items():
    
    enemy_number = int(enemy_name[-1])
    
    algorithm_dict = {}
    for algorithm, solutions in target.items():

        
        solution_gains = []
        for solution in solutions:
            
            gains_one_solution = []
            for test_index in range(5):
                experiment_name = 'test'
                env = Environment(experiment_name=experiment_name,
                                  enemies=[enemy_number],
                                  playermode="ai",
                                  player_controller=player_controller(n_hidden_neurons),
                                  enemymode="static",
                                  level=2,
                                  speed="fastest",
                                  visuals=False)  # Set visuals to True to see the game
                
                # Play the game
                fitness, player_life, enemy_life, game_time = env.play(pcont=np.array(solution))
                gain = player_life-enemy_life
                print(player_life,enemy_life,'\n\n')
                gains_one_solution.append(gain)
            
            solution_gains.append(gains_one_solution)
        
        algorithm_dict[algorithm] = solution_gains
    
    enemy_dict[enemy_name] = algorithm_dict
        

        

# Compute averages
averages = {}
for enemy, algorithms in enemy_dict.items():
    averages[enemy] = {}
    for algorithm, results in algorithms.items():
        averages[enemy][algorithm] = [sum(res) / len(res) for res in results]



# Group data by enemies
grouped_data = {}
for enemy, algorithms in averages.items():
    grouped_data[enemy] = []
    for algorithm, avg_results in sorted(algorithms.items()):
        grouped_data[enemy].append(avg_results)

# Colors for 2-parent and 4-parent algorithms
colors = ['lightblue', 'lightgreen']

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Create boxplots with different colors
boxes = []
for i, (enemy, data) in enumerate(grouped_data.items()):
    bp = ax.boxplot(data, positions=[i*3, i*3+1], widths=0.6, patch_artist=True)
    boxes.append(bp)
    for j, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[j])

# Set x-ticks and labels
ax.set_xticks([i*3 + 0.5 for i in range(len(grouped_data))])
ax.set_xticklabels(list(grouped_data.keys()))

# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='b', lw=4, label='2-parent'),
                   Line2D([0], [0], color='g', lw=4, label='4-parent')]
ax.legend(handles=legend_elements, loc='upper right')

# Set title and y-label
ax.set_title('Average Gains for Different Enemies and Algorithms')
ax.set_ylabel('Average Gain')

plt.tight_layout()
plt.show()




# Perform t-test

enemy_1_parent_2 = averages['Enemy_1']['algorithm_nparent_2']
enemy_1_parent_4 = averages['Enemy_1']['algorithm_nparent_4']
enemy_7_parent_2 = averages['Enemy_7']['algorithm_nparent_2']
enemy_7_parent_4 = averages['Enemy_7']['algorithm_nparent_4']
enemy_8_parent_2 = averages['Enemy_8']['algorithm_nparent_2']
enemy_8_parent_4 = averages['Enemy_8']['algorithm_nparent_4']








t_stat, p_value = ttest_ind(enemy_1_parent_2, enemy_1_parent_4)
print(f' Enemy 1 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')


t_stat, p_value = ttest_ind(enemy_7_parent_2, enemy_7_parent_4)
print(f' Enemy 7 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')


t_stat, p_value = ttest_ind(enemy_8_parent_2, enemy_8_parent_4)
print(f' Enemy 8 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')




print('\nWilcoxin\n')

# t_stat, p_value = wilcoxon(enemy_1_parent_2, enemy_1_parent_4)
# print(f' Enemy 1 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')


stat, p_value = wilcoxon(enemy_7_parent_2, enemy_7_parent_4)
print(f' Enemy 7 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')


stat, p_value = wilcoxon(enemy_8_parent_2, enemy_8_parent_4)
print(f' Enemy 8 ------ p value = {round(p_value,3)} and t_stat = {round(t_stat,3)}')









