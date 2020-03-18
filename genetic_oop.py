import numpy as numpy
import pandas as pd
import copy

np.random.seed(0)


class Chromosome:
    def __init__(self, weight):
        if weight is None:
            weight = []
        self._fitness = None
    
    


class Population:
    def __init__(self, first_population: list = None):
        if first_population is None:
            first_population = []
        self._population_size = len(first_population)
        self._generations = [first_population]
        self._generations_fitness = []
        self._generations_solution = []
        self._print = False
    

    def generate_next_population(self):
        print('\nIteration {}'.format(len(self._generations_fitness)))
        current_generation = self._generations[-1]
        current_generation_fitness = self._generations_fitness[-1]
        fitness = []
        

        pass


    def generate_populations(self, config: dict, _print=False):
        self._offspring_number = generate_config['offspring_number']
        self._crossover_probability = generate_config['crossover_probability']
        self._crossover_probability = generate_config['selection_type']
        self._mutation_probability = generate_config['mutation_probability']
        self._mutation_size = generate_config['mutation_size']
        self._chromosomes_replace = generate_config['chromosomes_replace']
        self._generations_number = generate_config['generations_number']
        self._stop_criterion_depth = generate_config['stop_criterion_depth']
        self._print = _print

        current_generation_fitness = self._generations_fitness[-1]
        current_solution = self._generations_solution[-1]
        print('Initial fitness: {}'.format(current_generation_fitness))
        
        depth = 0
        best_fitness = current_generation_fitness
        for epoch in range(self._generations_number):
            new_generation_fitness, new_generation_solution = self.generate_next_population()
            print('Generation {}: fitness {}'.format(i + 1, new_generation_fitness))
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
    def population_initialization(population_size: int = 100, genes_number: int):
        new_population = np.random.dirichlet(np.ones(genes_number), size=population_size)
        return Population(new_population)




if __name__ == '__main__':
    config = {'population_size': 200, 'offspring_number': 50,
                'crossover_probability': 1.0, 'selection_type': 'roulette_wheel',
                'mutation_probability': 0.66, 'mutation_size': 10,
                'chromosomes_replace': 50,
                'generations_number': 500, 'stop_criterion_depth': 50}
               
    config['offspring_number'] = int(config['population_size'] / 2)
    config['chromosomes_replace'] = int(config['population_size'] / 2)

    path = 'data/data_concat.csv'
    df = pd.read_csv(path)
    genes_number = len(df.columns) - 1

    population = Population.population_initialization(population_size=config['population_size'], genes_number=genes_number)


    pass

