# **Project: EvoMan Framework - Evolutionary Algorithm Analysis**

This project utilizes the EvoMan framework to analyze the performance of evolutionary algorithms against different enemies in the game. The main objective is to compare the performance of algorithms using different numbers of parents (2-parent vs. 4-parent) across multiple generations.

## **Overview:**

1. **EvoMan Framework**: A popular framework used for benchmarking the performance of evolutionary algorithms in a game environment. The player's agent, controlled by an AI, battles against various enemies, and the performance is evaluated based on the agent's fitness.

2. **Problem Description**: The goal is to optimize the weights of a neural network controller that dictates the behavior of the player's agent in the game. The performance of the agent is measured in terms of its fitness, which is influenced by factors like the agent's remaining life, the enemy's remaining life, and the time taken to defeat the enemy.

## **Scripts:**

### 1. **plot_single_enemy_2_algorithms.py**
This script is responsible for running the evolutionary algorithms with different numbers of parents (2 and 4) against three different enemies. It performs multiple runs and records the best and mean fitness values for each generation. The results are saved in JSON format for further analysis.

### 2. **box_plot_generator.py**
This script reads the best solutions obtained from the previous script and simulates the game multiple times to calculate the average gain for each solution. It then generates box plots to visually compare the performance of the 2-parent and 4-parent algorithms against each enemy. Additionally, it performs a t-test to statistically verify the differences in the average gains between the two algorithms.

### 3. **lineplot_generator.py**
This script reads the fitness values recorded during the runs of the evolutionary algorithms and plots them against the number of generations. It provides a visual representation of how the fitness values evolve over time for both the 2-parent and 4-parent algorithms.

## **Data Files:**

1. **final_lineplot.json**: Contains the best and mean fitness values for each generation, recorded during the runs of the evolutionary algorithms.

2. **final_boxplot.json**: Contains the best solutions obtained from the runs of the evolutionary algorithms.

## **Usage:**

To run the scripts, ensure you have the EvoMan framework and all necessary dependencies installed. Execute each script in the order they are listed above. The plots will be displayed on the screen, and the results will be saved in the specified JSON files.

## **Dependencies:**

- EvoMan Framework
- Python 3.x
- DEAP library
- NumPy
- Matplotlib
- SciPy

## **Conclusions:**

The project provides insights into the performance of evolutionary algorithms using different numbers of parents. The visualizations and statistical tests offer a comprehensive understanding of how the choice of the number of parents can influence the algorithm's efficiency in finding optimal solutions in the EvoMan game environment.
