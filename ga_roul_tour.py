
import pandas as pd 
import numpy as np
import random
from genetic import *


df=pd.read_csv("sample.csv")


num_weights = 5

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 30
num_parents_mating = 10

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
#new_population = numpy.random.uniform(low=0, high=1.0, size=pop_size)
new_population = np.random.dirichlet(np.ones(num_weights),size=sol_per_pop)
new_population = 1000000*new_population

z_score = 1
num_generations = 50
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(df, z_score, new_population)
    
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool_tour(new_population, fitness, 
                                      num_parents_mating)
    alpha=0.4
    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,alpha,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
        
    offspring_new_pop = mutation(offspring_crossover,num_weights)
    
    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_new_pop
    
    print("gen",generation)
    print("Best solution fitness : ", fitness[best_match_idx])
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(df, z_score, new_population)


# Then return the index of that solution corresponding to the best fitness.
possible_best_match_idx= np.where(fitness == np.min(fitness))[0]

if len(possible_best_match_idx)>=2:
    best_match_idx=possible_best_match_idx[0]
else:
    best_match_idx = int(np.where(fitness == np.min(fitness))[0])

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
