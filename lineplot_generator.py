import json
import matplotlib.pyplot as plt
import numpy as np


filename = 'final_lineplot.json'

with open(filename, 'r') as json_file:
    enemy_result_dictionary = json.load(json_file)

def confidence_interval(mean_values, std_dev_values, n=2):  # n=2 because you have 2 runs in this example
    z_value = 1.96
    margin_error = z_value * (std_dev_values / np.sqrt(n))
    return mean_values - margin_error, mean_values + margin_error


for enemy in enemy_result_dictionary.keys():

    best_values = enemy_result_dictionary[enemy]['best']
    mean_values = enemy_result_dictionary[enemy]['mean']
    std_values = enemy_result_dictionary[enemy]['std']
    

    avg_best_2_parent = np.mean(best_values['parent_2'],axis = 0)
    avg_best_4_parent = np.mean(best_values['parent_4'],axis = 0)
    avg_mean_2_parent = np.mean(mean_values['parent_2'],axis = 0)
    avg_mean_4_parent = np.mean(mean_values['parent_4'],axis = 0)
    avg_std_2_parent = np.mean(std_values['parent_2'],axis = 0)
    avg_std_4_parent = np.mean(std_values['parent_4'],axis = 0)
    
    
    
    
    
    
    x = np.arange(1, 41)  # Assuming 40 generations
    
    plt.figure(figsize=(4,3))
    
    # Plotting best fitness values 
    plt.plot(x, avg_best_2_parent, label='Best Fitness 2 Parent', color='blue')
    plt.fill_between(x, avg_best_2_parent - avg_std_2_parent, avg_best_2_parent + avg_std_2_parent, color='blue', alpha=0.2)
    
    plt.plot(x, avg_best_4_parent, label='Best Fitness 4 Parent', color='red')
    plt.fill_between(x, avg_best_4_parent - avg_std_4_parent, avg_best_4_parent + avg_std_4_parent, color='red', alpha=0.2)
    
    # Plotting mean fitness values
    plt.plot(x, avg_mean_2_parent, label='Mean Fitness 2 Parent', color='green')
    plt.fill_between(x, avg_mean_2_parent - avg_std_2_parent, avg_mean_2_parent + avg_std_2_parent, color='green', alpha=0.2)
    
    plt.plot(x, avg_mean_4_parent, label='Mean Fitness 4 Parent', color='purple')
    plt.fill_between(x, avg_mean_4_parent - avg_std_4_parent, avg_mean_4_parent + avg_std_4_parent, color='purple', alpha=0.2)
    
    plt.title(f'Mean and Best Fitness vs Generations 2-4 Parent Enemy {enemy[6]}')   
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
    
    #print the best values for algorithm enemy combination
    print(f'{enemy} 2 parent final best value:  {round(np.max(avg_best_2_parent),4)}')
    print(f'{enemy} 2 parent final mean value:  {round(np.max(avg_mean_2_parent),4)}')
    print(f'{enemy} 4 parent final best value:  {round(np.max(avg_best_4_parent),4)}')
    print(f'{enemy} 4 parent final mean value:  {round(np.max(avg_mean_4_parent),4)}')
    
    
    
    
    
    
    
    
    
    