import numpy as np
import pandas as pd
import copy
from time import time
import matplotlib.pyplot as plt

from genetic import Chromosome, Population


class Particle(Chromosome):
    def __init__(self, weight, velocity):
        super().__init__(weight)
        self._best_position = self._weight
        self._velocity = velocity

    @classmethod
    def calculate_sd_e(cls, df, z_score, _lambda, optimize, inertia_factor, self_conf_factor, swarm_conf_factor):
        super().calculate_sd_e(df, z_score, _lambda, optimize)
        cls._inertia_factor = inertia_factor
        cls._self_conf_factor = self_conf_factor
        cls._swarm_conf_factor = swarm_conf_factor


class Swarm(Population):
    def __init__(self, first_population: list = None):
        super().__init__(first_population=first_population)
    
    def update_velocity_and_position(self, particle):
        veloc = particle._velocity
        pos = particle._weight
        rp, rg = np.random.uniform(size=2)
        new_veloc = Particle._inertia_factor * veloc + \
                    Particle._self_conf_factor * rp * (particle._best_position - pos) + \
                    Particle._swarm_conf_factor * rg * (self._best_solution._weight - pos)
        new_pos = pos + new_veloc
        particle._weight = new_pos
        particle._velocity = new_veloc
        if (new_pos > 0).all() and (new_pos <= 0.4).all():
            fitness = particle._fitness
            new_fitness = particle.calculate_fitness()
            if new_fitness < fitness:
                particle._best_position = new_pos

    def generate_next_population(self):
        if self.verbose > 0:
            print('\nIteration {}'.format(len(self._all_best_fitness)))
            # print('\nIteration {} - Record {}'.format(len(self._all_best_fitness), self._best_solution.calculate_fitness()))
        generation = self._generations[-1]

        # update velocity, position and best position
        for particle in generation:
            self.update_velocity_and_position(particle)

        # update best swarm position
        new_generation_fitness = np.asarray([ptc._fitness for ptc in generation])
        best_fitness = np.min(new_generation_fitness)
        if best_fitness < self._best_solution._fitness:
            self._best_solution = generation[np.argmin(new_generation_fitness)]

        # update
        self._generations.append(generation)
        self._all_best_fitness.append(np.min(new_generation_fitness))
        self._generations_solution.append(generation[np.argmin(new_generation_fitness)])
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
                self._best_fitness = self._all_best_fitness[-1]
                depth += 1
                if self.verbose > 0:
                    print('\tFitness improved a little for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                self._best_fitness = self._all_best_fitness[-1]
                depth = 0
                if self.verbose > 0:
                    print('\tFitness improved')
        return self._best_solution, self._best_fitness, self._all_best_fitness


    @classmethod
    def population_initialization(cls, df, z_score: float = 1.0, _lambda=0.4, optimize='VaR',
                                    population_size=100, genes_number: int = None,
                                    inertia_factor=0.5, self_conf_factor=1.5, swarm_conf_factor=1.5):
        Particle.calculate_sd_e(df, z_score, _lambda, optimize,
                                inertia_factor=inertia_factor,
                                self_conf_factor=self_conf_factor,
                                swarm_conf_factor=swarm_conf_factor)
        new_population = np.random.dirichlet(np.ones(genes_number), size=population_size)
        velocity = np.random.uniform(-1, 1, size=(population_size, genes_number))
        normal = np.ones(genes_number)
        normal_norm_square = np.sum(normal ** 2)
        proj_on_normal = np.dot(np.dot(velocity, normal).reshape((-1,1)) / normal_norm_square, normal.reshape((1,-1)))
        proj_on_plane = velocity - proj_on_normal
        return cls([Particle(ptc, np.asarray(veloc)) for (ptc, veloc) in zip(new_population, proj_on_plane.tolist())])


if __name__ == '__main__':
    np.random.seed(0)
    
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
                                                inertia_factor=config['inertia_factor'],
                                                self_conf_factor=config['self_conf_factor'],
                                                swarm_conf_factor=config['swarm_conf_factor'],)
        solution, fitness, all_best_fitness = swarm.generate_populations(config=config, verbose=1)

        # draw fitness improving line
        epochs = np.arange(len(all_best_fitness))
        plt.plot(epochs, all_best_fitness)
        plt.xlabel('Epoch')
        plt.ylabel('Best fitness')
        plt.title('Partical swarm optimization')
        plt.show()

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
                        'pso.csv', index=False)
    else:
        pass