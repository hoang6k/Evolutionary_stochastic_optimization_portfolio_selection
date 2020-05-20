import numpy as np
import pandas as pd
import copy
from time import time

from genetic import Chromosome, Population

np.random.seed(0)


class Particle(Chromosome):
    def __init__(self, weight):
        super().__init__(weight)
        self._best_position = self._weight

    @staticmethod
    def calculate_sd_e(df, z_score, _lambda, optimize, inertia_factor, self_conf_factor, swarm_conf_factor):
        df.drop(['DTYYYYMMDD'], axis=1, inplace=True)
        Particle._method = optimize
        Particle._z_score = z_score
        Particle._lambda = _lambda
        Particle._pct_change = df.pct_change()
        Particle._annual_returns = Particle._pct_change.mean()
        Particle._annual_cov_matrix = Particle._pct_change.cov()
        Particle._inertia_factor = inertia_factor
        Particle._self_conf_factor = self_conf_factor
        Particle._swarm_conf_factor = swarm_conf_factor


class Swarm(Population):
    def __init__(self, first_population: list = None):
        super().__init__(first_population=first_population)
    
    def generate_next_population(self):
        if self.verbose > 0:
            print('\nIteration {}'.format(len(self._all_best_fitness)))
        generation = self._generations[-1]
        np.random.shuffle(generation)
        generation_fitness = np.asarray([chromo._fitness for chromo in generation])

        # selection phase
        selection_switcher = {
            'roulette_wheel': self.roulette_wheel_selection,
            'tournament': self.tournament_selection,
            'rank': self.rank_selection,
            'boltzmann': self.boltzmann_selection,
            'elitism': self.elitism_selection
        }
        parents = selection_switcher.get(
                    self._selection_method['type'], 
                    lambda: 'Invalid selection method')(
                        generation,
                        self._selection_method['k']
                    )

        # cross-over phase
        if self.verbose > 1:
            print('----CROSS-OVER PHASE')
            start_time = time()
        crossover_switcher = {
            '1point': self.crossover_1point,
            '2points': self.crossover_2points,
            'uniform': self.crossover_uniform
        }
        children = crossover_switcher.get(
                    self._crossover_method['type'],
                    lambda: 'Invalid crossover method')(
                        parents,
                        self._crossover_method['parameters']
                    )
        best_fitness = np.min([chromo._fitness for chromo in children])
        if self.verbose > 0:
            if self.verbose > 1:
                print('Time of cross-over: {} seconds'.format(time() - start_time))
            print('\tCROSS-OVER best fitness: {}'.format(best_fitness))

        # mutation phase
        if self.verbose > 1:
            print('****MUTATION PHASE')
            start_time = time()
        new_children = self.mutation(children)
        best_fitness = np.min(np.asarray([chromo._fitness for chromo in new_children]))
        if self.verbose > 0:
            if self.verbose > 1:
                print('Time of mutation: {} seconds'.format(time() - start_time))
            print('\tMUTATION best fitness: {}'.format(best_fitness))

        # replace worst chromosomes
        sorted_indexes = np.argsort(generation_fitness)
        worst_indexes = sorted_indexes[-self._chromosomes_replace:]
        worst_indexes.sort()
        worst_indexes = np.flip(worst_indexes)
        for idx in worst_indexes:
            generation.pop(idx)
        new_generation = generation + new_children
        new_generation_fitness = np.asarray([chromo._fitness for chromo in new_generation])
        self._generations.append(new_generation)
        self._all_best_fitness.append(np.min(new_generation_fitness))
        self._generations_solution.append(new_generation[np.argmin(new_generation_fitness)])
        # if self._all_best_fitness[-1] < self._best_fitness:
        #     self._best_solution = self._generations_solution[-1]
        #     self._best_fitness = self._all_best_fitness[-1]
        return self._all_best_fitness[-1]

    def print(self):
        print('Population size: ' + str(self._population_size))
        print('Offspring number: ' + str(self._offspring_number))
        print('Selection type: ' + self._selection_method['type'].capitalize())
        print('Crossover method: ' + self._crossover_method['type'].capitalize())
        print('Crossover probability: ' + str(self._crossover_probability))
        print('Mutation probability: ' + str(self._mutation_probability))
        print('Mutation size: ' + str(self._mutation_size))
        print('Max generations number: ' + str(self._generations_number))
        print('Stop criterion depth: ' + str(self._stop_criterion_depth), end='\n\n')

    def generate_populations(self, config: dict, verbose=False):
        self._offspring_number = int(self._population_size * config['offspring_ratio'])
        if self._offspring_number % 2 == 1:
            self._offspring_number += 1
        self._crossover_probability = config['crossover_probability']
        self._selection_method = config['selection_method']
        self._crossover_method = config['crossover_method']
        self._mutation_probability = config['mutation_probability']
        self._mutation_size = int(self._offspring_number * config['mutation_ratio'])
        self._chromosomes_replace = self._offspring_number
        self._generations_number = config['generations_number']
        self._stop_criterion_depth = config['stop_criterion_depth']
        self.verbose = verbose
        self.print()

        print('Initial fitness: {}'.format(self._best_fitness))        
        depth = 0
        for epoch in range(self._generations_number):
            new_best_fitness = self.generate_next_population()
            print('Generation {}: fitness {}'.format(epoch + 1, new_best_fitness))
            if new_best_fitness >= self._best_fitness:
                depth += 1
                if self.verbose > 0:
                    print('\tFitness not improved for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            elif self._best_fitness - new_best_fitness < 1e-5:
                self._best_solution = self._generations_solution[-1]
                self._best_fitness = self._all_best_fitness[-1]
                depth += 1
                if self.verbose > 0:
                    print('\tFitness improved a little for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                self._best_solution = self._generations_solution[-1]
                self._best_fitness = self._all_best_fitness[-1]
                depth = 0
                if self.verbose > 0:
                    print('\tFitness improved')
        return self._best_solution, self._best_fitness


    @staticmethod
    def population_initialization(df, z_score: float = 1.0, _lambda=0.4, optimize='VaR',
                                    population_size=100, genes_number: int = None,
                                    inertia_factor=0.5, self_conf_factor=1.5, swarm_conf_factor=1.5):
        Particle.calculate_sd_e(df, z_score, _lambda, optimize,
                                inertia_factor=inertia_factor,
                                self_conf_factor=self_conf_factor,
                                swarm_conf_factor=swarm_conf_factor)
        new_population = np.random.dirichlet(np.ones(genes_number), size=population_size)
        return Swarm([Particle(chromo) for chromo in new_population])


if __name__ == '__main__':
    #optimize function: VaR, VaRp, markovitz, markovitz_sqrt, sharp_coef, sharp_coef_sqrt
    config = {'optimize_function': 'VaR',
                'population_size': 500, 'offspring_ratio': 0.5,
                'crossover_probability': 1.0,
                'selection_method': {'type': 'roulette_wheel', 'k': 25},
                'crossover_method': {'type': 'uniform', 'parameters': None},
                'mutation_probability': 1.0, 'mutation_ratio': 0.1,
                'generations_number': 1000, 'stop_criterion_depth': 100,
                'inertia_factor': 0.5, 'self_conf_factor': 1.5, 'swarm_conf_factor': 1.5}

    # path = 'data/data_concat.csv'
    path = 'data/dulieudetai.csv'
    df = pd.read_csv(path)
    genes_number = len(df.columns) - 1
    z_score = 1.0
    _lambda = 0.4

    if config['optimize_function'] not in ['markovitz', 'markovitz_sqrt']:
        swarm = Swarm.population_initialization(df, z_score, _lambda,
                                                optimize=config['optimize_function'],
                                                population_size=config['population_size'],
                                                genes_number=genes_number,
                                                inertia_factor=0.5,
                                                self_conf_factor=1.5,
                                                swarm_conf_factor=1.5)
        solution, fitness = swarm.generate_populations(config=config, verbose=1)

        cs_type = 'rd' if config['crossover_method']['type'] == 'uniform' else '2p'
        print(solution._weight)
        print(fitness)
        if config['optimize_function'] in ['sharp_coef', 'sharp_coef_sqrt']:
            fitness = -fitness
        fitness = np.asarray([fitness])
        solution = np.reshape(solution._weight, (genes_number))
        data = np.reshape(np.concatenate([fitness, solution]), (1,-1))
        result = pd.DataFrame(data, columns=[config['optimize_function']] + list(df))
        result.to_csv('result/result_' + path[path.rfind('/') + 1:-4] + '_' + config['optimize_function'] + '_' +
                        cs_type + str(config['population_size']) + '.csv', index=False)
    else:
        pass