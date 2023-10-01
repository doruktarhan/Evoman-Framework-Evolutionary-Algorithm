###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
from deap import base, creator, tools
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# Attribute generator: initialize each weight uniformly between BOUND_LOW and BOUND_UP
def uniform_bounds(low, up):
    return [np.random.uniform(l, u) for l, u in zip(low, up)]




#Wrapper function for 
def evaluate_individual(individual):
    return evaluate(env, [individual])[0],




def differential_mutation(population, F=0.5):
    mutants = []
    for i in range(len(population)):
        # Ensure we select three distinct individuals, excluding the current one
        choices = [j for j in range(len(population)) if j != i]
        indices = np.random.choice(choices, 3, replace=False)
        a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]
        
        # Compute the mutant vector
        mutant = [ci + F*(ai - bi) for ai, bi, ci in zip(a, b, c)]
        
        # Ensure boundaries (assuming -1 and 1 as in your previous code)
        mutant = [np.clip(val, -1, 1) for val in mutant]
        
        mutants.append(mutant)
    
    return mutants



#define the bounds for the NN weight generation
upper_bound= 1
lower_bound = -1
population_size = 50




#def main():
    # choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[2],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
print(n_vars)
# start writing your own code from here
print(env.get_num_sensors())


###############################################################################

#Define the individual and the fitness for toolbox
creator.create('FitnessMax', base.Fitness,weights = (1.0,))
creator.create('Individual',list,fitness = creator.FitnessMax)

#define the toolbox
toolbox = base.Toolbox()

#define the functions for generation weights of the network as individuals
toolbox.register('generator_ind',uniform_bounds,[upper_bound]*n_vars,[lower_bound]*n_vars)
toolbox.register('individual',tools.initIterate, creator.Individual, toolbox.generator_ind)
toolbox.register('population',tools.initRepeat,list,toolbox.individual)
toolbox.register('evaluate',evaluate_individual)
toolbox.register('mutate',differential_mutation,F=0.5)


#Basically we defined the form of the individual in generator_ind function
#The individual function calls the generator and forms an individual
#The population function calls the individual function n times to form a population

population = toolbox.population(100)






'''
if __name__ == '__main__':
    pop_to_inspect  = main()
'''
    