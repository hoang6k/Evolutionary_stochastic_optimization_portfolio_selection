import numpy as np
import pandas as pd
import copy
from time import time

np.random.seed(0)


class Chromosome:
    def __init__(self, weight):
        if weight is None:
            weight = []
        self._weight = weight
        self._fitness = None
    
    def calculate_fitness(self, pct_change, z_score):
        print(pct_change)
        print(self._weight)
        value_ptf = pct_change * self._weight
        value_ptf['Value of Portfolio'] = value_ptf.sum(axis=1)
        ptf_percentage = value_ptf['Value of Portfolio']
        ptf_percentage = ptf_percentage.sort_values(axis=0, ascending=True)
        _VaR =  np.percentile(ptf_percentage, z_score)
        self._fitness = -_VaR
        return self._fitness



class Population:
    def __init__(self, pct_change, z_score: float = 1.0, first_population: list = None):
        if first_population is None:
            first_population = []
        self._pct_change = pct_change
        self._z_score = z_score
        self._population_size = len(first_population)
        self._generations = [first_population]
        _fitness = np.asarray([chromo.calculate_fitness(self._pct_change, self._z_score) for chromo in first_population])
        self._generations_fitness = [_fitness]
        self._generations_solution = first_population[np.argmin(_fitness)]
        self._print = False
    

    def crossover(self, father, mother, alpha=0.5):
        child_1 = (1 - alpha) * father + alpha * mother
        child_2 = alpha * father + (1 - alpha) * mother
        return child_1, child_2


    def roulette_wheel_selection(self, generation, k):
        # for i in range(int(self._offspring_number / 2)):
        #     if self._print:
        #         print('\t\t{}_th 2 childs'.format(i + 1))
        #     _crossover = bool(np.random.rand(1) <= self._crossover_probability)
        #     if _crossover:
        #         father = generation[self.roulette_wheel_selection(array=fitness)]
        #         mother = generation[self.roulette_wheel_selection(array=fitness)]
        #         child_1, child_2 = self.crossover(father, mother)
        #         children.append(child_1)
        #         children.append(child_2)
        # print('Time of cross-over: {} seconds'.format(time() - start_time))
        # children_fitness = np.max(np.asarray([chromo.calculate_fitness() for chromo in children]))
        # print('\tCROSS-OVER fitness: {}'.format(children_fitness))
        pass

    def tournament_selection(self, generation, k, alpha):
        start_time = time()
        fitness = np.asarray([chromo._fitness for chromo in generation])
        parents = []
        fitness_parents = []
        children = []

        # select mating pool
        for i in range(self._offspring_number):
            chosen_indexes = np.random.choice(self._population_size, size=k, replace=False)
            best_index = chosen_indexes[np.argmin(fitness[chosen_indexes])]
            parents.append(generation[best_index])
            fitness_parents.append(fitness_parents[best_index])
        
        # crossover mating pool
        for i in range(int(self._offspring_number / 2)):
            _crossover = bool(np.random.rand(1) <= self._crossover_probability)
            if _crossover:
                index = np.random.randint(len(parents))
                father = parents.pop(index)
                father_fitness = fitness_parents.pop(index)
                index = np.random.randint(len(parents))
                mother = parents.pop(index)
                mother_fitness = fitness_parents.pop(index)
                alpha = father_fitness / (father_fitness + mother_fitness)
                child_1, child_2 = self.crossover(father, mother, alpha)
                children.append(child_1)
                children.append(child_2)
        print('Time of cross-over: {} seconds'.format(time() - start_time))
        children_fitness = np.max(np.asarray([chromo.calculate_fitness() for chromo in children]))
        print('\tCROSS-OVER fitness: {}'.format(children_fitness))



        pass

    def rank_selection(self, generation, k):
        pass

    def boltzmann_selection(self, generation, k):
        pass
    
    def elitism_selection(self, generation, k):
        pass

    def generate_next_population(self):
        print('\nIteration {}'.format(len(self._generations_fitness)))
        current_generation = self._generations[-1]
        current_generation_fitness = self._generations_fitness[-1]
        fitness = np.asarray([chromo._fitness for chromo in current_generation])

        # cross-over phase
        start_time = time()
        if self._print:
            print('----CROSS-OVER PHASE')

        selection_switcher = {
            'roulette_wheel': self.roulette_wheel_selection,
            'tournament': self.tournament_selection,
            'rank': self.rank_selection,
            'boltzmann': self.boltzmann_selection,
            'elitism': self.elitism_selection
        }
        
        children = selection_switcher.get(self._selection_method['type'], \
                                        lambda: 'Invalid selection method')(\
                                            current_generation,\
                                            self._selection_method['k'],\
                                            self._selection_method['alpha'])
        if isinstance(children, str):
            pass
        

        pass


    def generate_populations(self, config: dict, _print=False):
        self._offspring_number = config['offspring_number']
        self._crossover_probability = config['crossover_probability']
        self._selection_method = config['selection_method']
        self._mutation_probability = config['mutation_probability']
        self._mutation_size = config['mutation_size']
        self._chromosomes_replace = config['chromosomes_replace']
        self._generations_number = config['generations_number']
        self._stop_criterion_depth = config['stop_criterion_depth']
        self._print = _print

        current_generation_fitness = self._generations_fitness[-1]
        current_solution = self._generations_solution[-1]
        print('Initial fitness: {}'.format(current_generation_fitness))
        
        depth = 0
        best_fitness = current_generation_fitness
        for epoch in range(self._generations_number):
            new_generation_fitness, new_generation_solution = self.generate_next_population()
            print('Generation {}: fitness {}'.format(epoch + 1, new_generation_fitness))
            if new_generation_fitness <= best_fitness:
                depth += 1
                print('\tFitness not increase for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                print('\tFitness increased')
                depth = 0
                best_fitness = new_generation_fitness


    @staticmethod
    def population_initialization(df, z_score: float = 1.0, population_size: int = 100, genes_number: int = None):
        df.drop(['DTYYYYMMDD'], axis=1, inplace=True)
        pct_change = df.pct_change()
        new_population = np.random.dirichlet(np.ones(genes_number), size=population_size)
        return Population(pct_change, z_score, [Chromosome(chromo) for chromo in new_population])




if __name__ == '__main__':
    config = {'population_size': 200, 'offspring_number': 50,
                'crossover_probability': 1.0,
                'selection_method': {'type': 'tournament', 'k': 10, 'alpha': 0.4},
                'mutation_probability': 0.66, 'mutation_size': 10,
                'chromosomes_replace': 50,
                'generations_number': 500, 'stop_criterion_depth': 50}
               
    config['offspring_number'] = int(config['population_size'] / 2)
    config['chromosomes_replace'] = int(config['population_size'] / 2)

    path = 'data/data_concat.csv'
    df = pd.read_csv(path)
    genes_number = len(df.columns) - 1
    z_score = 1.0
    print(genes_number)

    population = Population.population_initialization(df, z_score, population_size=config['population_size'], \
                                                        genes_number=genes_number)


    pass

