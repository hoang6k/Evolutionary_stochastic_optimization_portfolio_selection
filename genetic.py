import numpy as np
import random


def var_fitness(df, exposures, z_score):
    percentage = df.pct_change()

    #Different Dollar Exposures into Portfolio
    value_ptf = percentage * exposures
    value_ptf['Value of Portfolio'] = value_ptf.sum(axis=1)

    ptf_percentage = value_ptf['Value of Portfolio']
    ptf_percentage = ptf_percentage.sort_values(axis=0, ascending=True)

    VaR =  np.percentile(ptf_percentage, z_score)
    
    return -VaR


def cal_pop_fitness(df, z_score, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = []
    for i in range(len(pop)):
        fitness.append(var_fitness(df, pop[i], z_score))
    return fitness


def select_mating_pool_tour(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    num_fighters = 5
    for parent_num in range(num_parents):
        fighter = []
        for i in range(num_fighters):
            fighter.append(random.randint(0, len(pop)-1))
        min_fitness_idx = fighter[0]
        min_fitness_value = fitness[fighter[0]]
        for i in fighter:
            if min_fitness_value > fitness[i]:
                min_fitness_idx = i
                min_fitness_value = fitness[i]
        #max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = -99999999999
    return parents


def select_mating_pool_roul(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))

    # fitness_rev=[-fit for fit in fitness]
    total_fitness = float(sum(fitness))
    rel_fitness = [(max(fitness)+min(fitness)-f)/total_fitness for f in fitness]
    # Generate probability intervals for each individual
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))]
    # Draw new population
    for parent_num in range(num_parents):
        r = random.random()
        for i in range(pop.shape[0]):
            if r <= probs[i]:
                parents[parent_num, :] = pop[i, :]
                break
    return parents

def crossover(parents,alpha, offspring_size):
    offspring = np.empty(offspring_size)
    
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        if (k//parents.shape[0])%2==0:
            offspring[k,:]=alpha*parents[parent1_idx,:]+(1-alpha)*parents[parent2_idx,:]
        else:
            offspring[k,:]=(1-alpha)*parents[parent1_idx,:]+alpha*parents[parent2_idx,:]
    return offspring

def mutation(offspring_crossover,num_weights,prob_mutation=0.1):
    # Mutation changes a single gene in each offspring randomly.
    mutation_ind=random.sample(range(num_weights),2)
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        prob_mutation=0.1
        num_prob=random.random()
        if num_prob<=prob_mutation:
           sum_two_mutation= offspring_crossover[idx, mutation_ind[0]]+offspring_crossover[idx, mutation_ind[1]]
           random_value = np.random.uniform(0, sum_two_mutation)
           offspring_crossover[idx, mutation_ind[0]]= random_value
        
           offspring_crossover[idx, mutation_ind[1]]=sum_two_mutation-random_value
    return offspring_crossover