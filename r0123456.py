import Reporter
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# Modify the class name to match your student number.
class r0652717:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        self.iter = 1

        # Set Parameters
        self.init_population = self.distanceMatrix.shape[0] * 100
        self.mu = int(0.5 * self.init_population)

        self.island_one_part = 0.2
        self.island_two_part = 0.8

        # Island one: highly competitive
        self.lambdaa_one = 0.6
        self.alpha_one = 0.1
        self.beta_one = 0.01
        self.k = 10

        self.lambdaa_two = 0.6
        self.alpha_two = 0.1
        self.beta_two = 0.1
        self.k = 10

        self.offspring_one = int(self.lambdaa_one * self.mu/2) - int(self.lambdaa_one*self.mu/2) % 2
        self.offspring_two = int(self.lambdaa_two * self.mu) - int(self.lambdaa_two*self.mu) % 2



        # Initialize population
        self.island_one, self.island_two = self.initialize_islands(self.init_population, self.k, self.distanceMatrix)
        self.costs_one = self.calc_all_cost(self.island_one)
        self.costs_two = self.calc_all_cost(self.island_two)

        # initialize arrays which hold output
        best_one_res = []
        best_two_res = []
        mean_one_res = []
        mean_two_res = []

        # Your code here.

        while self.test_convergence:

            offspring_one = self.breed(self.island_one, self.costs_one, self.offspring_one, self.alpha_one)
            offspring_two = self.breed(self.island_two, self.costs_two, self.offspring_two, self.alpha_two)

            improved_one = self.local_search(self.island_one, self.costs_one, self.beta_one)
            improved_two = self.local_search(self.island_two, self.costs_two, self.beta_two)

            mutated_one = self.mutate_population(self.island_one, self.costs_one, self.alpha_two)
            mutated_two = self.mutate_population(self.island_two, self.costs_two, self.alpha_two)

            self.island_one = self.addition(self.island_one, offspring_one, improved_one, mutated_one)
            self.island_two = self.addition(self.island_two, offspring_two, improved_two, mutated_two)

            print(self.island_one.shape)

            self.island_one = self.island_one[self.select_good_individuals(self.mu, self.costs_one, replace=False)]
            self.island_two = self.island_two[self.select_good_individuals(self.mu, self.costs_one, replace=False)]

            print(self.island_two.shape)


            # measure performance
            best_one_res.append(np.min(self.costs_one))
            best_two_res.append(np.min(self.costs_two))

            mean_one_res.append(np.mean(self.costs_one))
            mean_two_res.append(np.mean(self.costs_two))

            index_one = np.argmin(self.costs_one)
            index_two = np.argmin(self.costs_two)

            plt.plot(np.arange(self.iter), best_one_res)
            plt.plot(np.arange(self.iter), best_two_res)
            plt.show()

            print(self.iter)
            print("Best one: ", best_one_res[-1])
            print("Best two: ", best_two_res[-1])




            self.iter += 1
            # timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            # if timeLeft < 0:
            #     break

        # Your code here.
        return 0

    def init_greedy_tournament(self, init_pop_size, k, distance_matrix):
        n_cities = distance_matrix.shape[0]
        routes = np.zeros((init_pop_size, n_cities), dtype=int)
        options = np.arange(1, n_cities)

        for i in tqdm(range(init_pop_size)):
            cur_options = np.copy(options)
            cur_city = 0
            for j in range(n_cities - 1):
                if cur_options.shape[0] < k:
                    cur_k = cur_options.shape[0]
                else:
                    cur_k = k
                possible_cities = np.random.choice(cur_options, cur_k, replace=False)
                selected = np.argmin(distance_matrix[cur_city, possible_cities])
                cur_city = possible_cities[selected]
                routes[i, j + 1] = cur_city
                cur_options = np.delete(cur_options, np.where(cur_options == cur_city))
        return routes

    def random_permutations(self, init_pop_size, cycle_length):
        shape_range = np.arange(1, cycle_length, dtype=int)
        individuals = np.tile(shape_range, (init_pop_size, 1))
        individuals = np.apply_along_axis(np.random.permutation, 1, individuals)
        individuals = np.hstack((np.zeros((init_pop_size, 1), dtype=int), individuals))
        return individuals

    def initialize_islands(self, init_pop_size, k, distance_matrix):
        island_size = int(init_pop_size / 2)
        island_one = self.init_greedy_tournament(island_size, k, distance_matrix)
        island_two = self.random_permutations(island_size, distance_matrix.shape[0])
        return island_one, island_two

    def convert_distance(self):
        self.distance_matrix[np.where(np.isinf(self.distance_matrix))] = -1
        self.distance_matrix[np.where(self.distance_matrix == -1)] = 10 * np.max(self.distance_matrix)

    def calc_all_cost(self, routes):
        return np.apply_along_axis(self.calc_cost, 1, routes)

    def scramble_batch(self, routes):
        return np.apply_along_axis(self.scramble_mutation, 1, routes)

    def mutate_population(self, population, costs, alpha):
        selected = self.select_bad_individuals(int(population.shape[0] * alpha), costs, replace=False)
        selected_ind = population[selected, :]
        mutated = np.apply_along_axis(self.scramble_mutation, 1, selected_ind)
        return mutated

    def calc_cost(self, route):
        cost = 0
        for i in range(route.shape[0] - 1):
            cost += self.distanceMatrix[route[i], route[i + 1]]
        cost += self.distanceMatrix[route[-1], route[0]]
        return cost

    def scramble_mutation(self, route):
        indices = np.random.choice(route.shape[0], 2, replace=False)
        randint_one = np.min(indices)
        randint_two = np.max(indices)
        mutated = np.array(route)
        mutated[randint_one: randint_two] = np.random.permutation(mutated[randint_one:randint_two])
        return mutated

    def nwox(self, x1, x2):
        n = x1.shape[0]

        a = np.random.randint(0, n - 1)
        if a == n:
            b = n
        else:
            b = np.random.randint(a, n - 1)

        y1 = np.array(x1)
        y2 = np.array(x2)

        y1[np.where(np.in1d(x1, x2[a:b + 1]))] = -1
        y2[np.where(np.in1d(x2, x1[a:b + 1]))] = -1

        part_one_y1 = y1[np.where(y1[0:b + 1] != -1)][0:a]
        part_two_y1 = y1[a:n][np.where(y1[a:n] != -1)][-(n - b - 1):]
        y1 = np.concatenate((part_one_y1, x2[a:b + 1], part_two_y1))

        part_one_y2 = y2[np.where(y2[0:b + 1] != -1)][0:a]
        part_two_y2 = y2[a:n][np.where(y2[a:n] != -1)][-(n - b - 1):]
        y2 = np.concatenate((part_one_y2, x1[a:b + 1], part_two_y2))

        return y1, y2

    def ranked_exp_decay(self, costs, best=True):
        """Best = True -> RANK 0 is LOWEST cost (best individual, highest prob)"""
        if best == False:
            costs = -costs
        temp = costs.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(costs))
        highest = costs[ranks[-1]]
        alpha = 0.99
        s = alpha ** (1000)
        a = np.log(s) / (costs.shape[0] - 1)
        exponent = a * ranks
        weights = np.exp(exponent)
        prob = weights / np.sum(weights)
        # plt.plot(ranks, weights, 'o')
        # plt.show()

        return prob

    def select_bad_individuals(self, n, costs, replace=True):
        prob = self.ranked_exp_decay(costs, best=False)
        selected = np.random.choice(costs.shape[0], size=n, p=prob, replace=replace)
        return selected

    def select_good_individuals(self, n, costs, replace=True):
        prob = self.ranked_exp_decay(costs, best=True)
        selected = np.random.choice(costs.shape[0], size=n, p=prob, replace=replace)
        return selected

    def breed(self, population, costs, lambdaa, alpha):
        parents = population[self.select_good_individuals(lambdaa, costs, replace=True), :]
        offspring = np.empty(parents.shape, dtype=int)
        for i in np.arange(0, parents.shape[0], step=2):
            offspring[i, :], offspring[i + 1, :] = self.nwox(parents[i, :], parents[i + 1, :])
        offspring_to_mutate = np.random.choice(offspring.shape[0], size=int(alpha*offspring.shape[0]), replace=False)
        offspring[offspring_to_mutate, :] = np.apply_along_axis(self.scramble_mutation, 1, offspring[offspring_to_mutate, :])
        offspring[offspring_to_mutate, :] = np.apply_along_axis(self.inversion_mut, 1, offspring[offspring_to_mutate, :])
        offspring = np.apply_along_axis(self.k_opt, 1, offspring)
        return offspring

    def eliminate(self, new_pop_size, population, costs):
        n_to_del = population.shape[0] - new_pop_size
        to_delete = self.select_bad_individuals(n_to_del, costs, replace=False)
        population = np.delete(population, to_delete, axis=0)
        costs = np.delete(costs, to_delete, axis=0)
        return population, costs

    def add_to_pop(self, routes, population, costs):
        new_costs = self.calc_all_cost(routes)
        population = np.concatenate((population, routes))
        costs = np.concatenate((costs, new_costs))
        return population, costs

    def inversion_mut(self, route):
        indices = np.random.choice(route.shape[0] - 1, 2, replace=False)
        low = np.min(indices) + 1
        high = np.max(indices) + 1

        flipped = np.array(route)
        flipped[low:high] = np.flip(route[low:high])
        return flipped

    def k_opt(self, route):
        k = self.k
        k_neighbours = np.empty((k, route.shape[0]), dtype=int)

        for i in range(k):
            k_neighbours[i, :] = self.inversion_mut(route)

        costs = self.calc_all_cost(k_neighbours)
        best = np.argmin(costs)
        return k_neighbours[best, :]

    def two_opt_swap(self, route, i, j):
        new_route = np.array(route)
        new_route[i:j + 1] = np.flip(route[i:j + 1])
        return new_route

    def local_search(self, population, costs, beta):
        selected = self.select_good_individuals(int(population.shape[0] * beta), costs, replace=False)
        selected_ind = population[selected, :]
        improved = np.apply_along_axis(self.k_opt, 1, selected_ind)
        return improved

    def two_opt(self, route, cost=None, inv_length=10):
        if cost is None:
            cost = self.calc_cost(route)
        i = np.random.randint(0, route.shape[0] - inv_length)
        if i + inv_length > route.shape[0]:
            range_end = route.shape[0]
        else:
            range_end = i + inv_length
        for j in range(i + 1, range_end):
            new_route = self.two_opt_swap(route, i, j)
            new_cost = self.calc_cost(new_route)
            if new_cost < cost:
                return self.two_opt(route, new_cost)
        return route

    def two_island_greedy(self, distance):
        not_visited = np.arange(distance.shape[0])
        current = 0
        not_visited = np.delete(not_visited, 0)
        for i in range(distance.shape[0] - 1):
            options = distance[not_visited, :]
            greedy = np.argmin(options)
            current = options[greedy]
            not_visited = np.delete(not_visited, current)

    def test_convergence(self):
        return True

    def addition(self, population, offspring, mutated, improved):
        population = np.vstack((population, offspring))
        population = np.vstack((population, mutated))
        population = np.vstack((population, improved))
        return population






def main():
    filename = "data/tour29.csv"
    optimize = r0652717()
    optimize.optimize(filename)


if __name__ == "__main__":
    main()
