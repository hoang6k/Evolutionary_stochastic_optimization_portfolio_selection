import numpy as np
import pandas as pd
import copy
from time import time

np.random.seed(0)


class Chromosome:
    _pct_change = None
    _z_score = 1.0

    def __init__(self, weight):
        if weight is None:
            weight = []
        self._weight = weight
        self._fitness = self.calculate_fitness()
    
    def calculate_fitness(self):
        value_ptf = Chromosome._pct_change * self._weight * 1e6
        value_ptf['Value of Portfolio'] = value_ptf.sum(axis=1)
        ptf_percentage = value_ptf['Value of Portfolio']
        ptf_percentage = ptf_percentage.sort_values(axis=0, ascending=True)
        _VaR =  np.percentile(ptf_percentage, Chromosome._z_score)
        self._fitness = -_VaR
        return self._fitness


class Population:
    def __init__(self, first_population: list = None):
        if first_population is None:
            first_population = []
        self._population_size = len(first_population)
        self._generations = [first_population]
        _fitness = [chromo._fitness for chromo in first_population]
        self._best_fitness = np.min(np.asarray(_fitness))
        self._all_best_fitness = [self._best_fitness]
        self._generations_solution = [first_population[np.argmin(_fitness)]]
        self._best_solution = self._generations_solution[-1]
        self.verbose = False
    
    def mutation(self, children):
        new_children = []
        chosen_indexes = np.random.choice(len(children), size=self._mutation_size, replace=False)
        for i in range(len(children)):
            if i not in chosen_indexes:
                new_children.append(copy.deepcopy(children[i]))
                continue
            chromosome = children[i]
            if self.verbose > 1:
                print('\t\tStarting mutation {}th child'.format(len(new_children) + 1))
            _mutation = bool(np.random.rand(1) <= self._mutation_probability)
            if _mutation:
                mutation_genes_indexes = np.random.choice(
                                            np.arange(len(chromosome._weight)), 
                                            size=np.random.randint(len(chromosome._weight)), 
                                            replace=False)
                sorted_indexes = np.sort(mutation_genes_indexes)
                new_weight = np.array(chromosome._weight)
                new_weight[sorted_indexes] = chromosome._weight[mutation_genes_indexes]
                new_children.append(Chromosome(new_weight))
        return new_children

    def crossover_2points(self, parents, alpha=None):
        genes_number = len(parents[0]._weight)
        children = []
        for i in range(int(self._offspring_number / 2)):
            if self.verbose > 1:
                print('\t\t{}_th 2 childs'.format(i + 1))
            _crossover = bool(np.random.rand(1) <= self._crossover_probability)
            if _crossover:
                index = np.random.randint(len(parents))
                father = parents.pop(index)._weight
                index = np.random.randint(len(parents))
                mother = parents.pop(index)._weight
                two_points = np.random.choice(np.arange(genes_number), size=2, replace=False)
                two_points.sort()
                _cs_genes_father = father[two_points[0]:two_points[1] + 1]
                _cs_genes_mother = mother[two_points[0]:two_points[1] + 1]
                cs_genes_father = _cs_genes_father * np.sum(_cs_genes_mother) / np.sum(_cs_genes_father)
                cs_genes_mother = _cs_genes_mother *  np.sum(_cs_genes_father) / np.sum(_cs_genes_mother)
                try:
                    weight_1 = np.concatenate((father[:two_points[0]], cs_genes_mother, father[two_points[1] + 1:]))
                    weight_2 = np.concatenate((mother[:two_points[0]], cs_genes_father, mother[two_points[1] + 1:]))
                except IndexError:
                    weight_1 = np.concatenate((father[:two_points[0]], cs_genes_mother))
                    weight_2 = np.concatenate((mother[:two_points[0]], cs_genes_father))
                children.append(Chromosome(weight_1))
                children.append(Chromosome(weight_2))
        return children

    def crossover_1point(self, parents, alpha=None):
        children = []
        for i in range(int(self._offspring_number / 2)):
            if self.verbose > 1:
                print('\t\t{}_th 2 childs'.format(i + 1))
            _crossover = bool(np.random.rand(1) <= self._crossover_probability)
            if _crossover:
                index = np.random.randint(len(parents))
                father = parents.pop(index)
                index = np.random.randint(len(parents))
                mother = parents.pop(index)
                if alpha is None:
                    alpha = father._fitness / (father._fitness + mother._fitness)
                child_1 = Chromosome((1 - alpha) * father._weight + alpha * mother._weight)
                child_2 = Chromosome(alpha * father._weight + (1 - alpha) * mother._weight)
                children.append(child_1)
                children.append(child_2)
        return children

    def roulette_wheel_selection(self, generation, k=5):
        fitness = np.asarray([chromo._fitness for chromo in generation])
        # change probability distribution from min to max
        fitness = 1 - fitness / np.sum(fitness)
        fitness /= np.sum(fitness)
        parents = []
        for i in range(self._offspring_number):
            chosen_indexes = np.random.choice(np.arange(len(fitness)), size=k, replace=False, p=fitness)
            best_index = chosen_indexes[np.argmax(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def tournament_selection(self, generation, k):
        fitness = np.asarray([chromo._fitness for chromo in generation])
        parents = []
        for i in range(self._offspring_number):
            chosen_indexes = np.random.choice(self._population_size, size=k, replace=False)
            best_index = chosen_indexes[np.argmin(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def rank_selection(self, generation, k):
        pass

    def boltzmann_selection(self, generation, k):
        pass
    
    def elitism_selection(self, generation, k):
        pass

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
            '2points': self.crossover_2points
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
        if self._all_best_fitness[-1] < self._all_best_fitness[-2]:
            self._best_solution = self._generations_solution[-1]
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
            elif self._best_fitness - new_best_fitness < 1e-6:
                depth += 1
                if self.verbose > 0:
                    print('\tFitness improved but less than 1e-6')
                if self.verbose > 0:
                    print('\tFitness improved a little for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                if self.verbose > 0:
                    print('\tFitness improved')
                depth = 0
                self._best_fitness = new_best_fitness
        return self._best_solution, self._best_fitness


    @staticmethod
    def population_initialization(df, z_score: float = 1.0, population_size: int = 100, genes_number: int = None):
        df.drop(['DTYYYYMMDD'], axis=1, inplace=True)
        Chromosome._pct_change = df.pct_change()
        Chromosome._z_score = z_score
        new_population = np.random.dirichlet(np.ones(genes_number), size=population_size)
        return Population([Chromosome(chromo) for chromo in new_population])


if __name__ == '__main__':
    config = {'population_size': 200, 'offspring_ratio': 0.5,
                'crossover_probability': 1.0,
                'selection_method': {'type': 'roulette_wheel', 'k': 10},
                'crossover_method': {'type': '2points', 'parameters': None},
                'mutation_probability': 1.0, 'mutation_ratio': 0.1,
                'generations_number': 500, 'stop_criterion_depth': 50}

    # path = 'data/data_concat.csv'
    path = 'data/dulieu2018.csv'
    df = pd.read_csv(path)
    genes_number = len(df.columns) - 1
    z_score = 1.0

    population = Population.population_initialization(df, z_score,
                                                        population_size=config['population_size'],
                                                        genes_number=genes_number)
    solution, fitness = population.generate_populations(config=config, verbose=1)

    print(solution._weight)
    print(fitness)
    solution = np.reshape(solution._weight, (1, genes_number))
    result = pd.DataFrame(solution, columns=list(df))
    # result.to_csv('result_vn30.csv')
    result.to_csv('result_' + path[path.rfind('/') + 1:-4])
