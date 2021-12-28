import Reporter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Modify the class name to match your student number.
class r0652717:

    def __init__(self):

        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        self.distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.convert_distance()
        self.iter = 0

        # PARAMETERS
        init_size = 2000

        # Your code here.
        population = self.init_population(init_size, 200, self.distance_matrix)
        costs = self.calc_all_cost(population)

        test_convergence = True
        while test_convergence:
            mean_objective = 0.0
            best_objective = 0.0
            best_solution = np.array([1,2,3,4,5])

            # Your code here.

            parents = population[self.select_good_individuals(600, costs, replace=False), :]
            offspring = self.breed(parents)
            to_mutate = population[self.select_bad_individuals(300, costs, replace=False), :]
            mutated = self.scrample_batch(to_mutate)

            population, costs = self.add_to_pop(offspring, population, costs)
            population, costs = self.add_to_pop(mutated, population, costs)

            population, costs = self.eliminate(1000, population, costs)

            print(population.shape)
            print("Min: ", np.min(costs))
            print("Mean: ", np.mean(costs))

            print("LOOP", self.iter)
            self.iter += 1


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(mean_objective, best_objective, best_solution)
            if time_left < 0:
                break

        # Your code here.
        return 0

    def init_population(self, init_pop_size, k, distance_matrix):
        n_cities = distance_matrix.shape[0]
        routes = np.zeros((init_pop_size, n_cities), dtype=int)
        options = np.arange(1, n_cities)

        for i in tqdm(range(init_pop_size)):
            cur_options = np.copy(options)
            cur_city = 0
            for j in range(n_cities-1):
                if cur_options.shape[0] < k:
                    cur_k = cur_options.shape[0]
                else:
                    cur_k = k
                possible_cities = np.random.choice(cur_options, cur_k, replace=False)
                selected = np.argmin(distance_matrix[cur_city, possible_cities])
                cur_city = possible_cities[selected]
                routes[i, j+1] = cur_city
                cur_options = np.delete(cur_options, np.where(cur_options == cur_city))
        return routes

    def convert_distance(self):
        self.distance_matrix[np.where(np.isinf(self.distance_matrix))] = 99999999

    def calc_all_cost(self, routes):
        return np.apply_along_axis(self.calc_cost, 1, routes)

    def scrample_batch(self, routes):
        return np.apply_along_axis(self.scramble_mutation, 1, routes)

    def calc_cost(self, route):
        cost = 0
        for i in range(route.shape[0] - 1):
            cost += self.distance_matrix[route[i], route[i+1]]
        cost += self.distance_matrix[route[-1], route[0]]
        return cost

    def scramble_mutation(self, route):
        randint_one = np.random.randint(0, route.shape[0]-2)
        randint_two = np.random.randint(randint_one+1, route.shape[0] - 1)
        mutated = np.array(route)
        mutated[randint_one : randint_two] = np.random.permutation(mutated[randint_one:randint_two])
        return mutated

    def nwox(self, x1, x2):
        n = x1.shape[0]

        a = np.random.randint(0, n-1)
        if a == n:
            b = n
        else:
            b = np.random.randint(a, n-1)


        y1 = np.array(x1)
        y2 = np.array(x2)

        y1[np.where(np.in1d(x1, x2[a:b+1]))] = -1
        y2[np.where(np.in1d(x2, x1[a:b+1]))] = -1

        part_one_y1 = y1[np.where(y1[0:b+1] != -1)][0:a]
        part_two_y1 = y1[a:n][np.where(y1[a:n] != -1)][-(n-b-1):]
        y1 = np.concatenate((part_one_y1, x2[a:b+1], part_two_y1))

        part_one_y2 = y2[np.where(y2[0:b+1] != -1)][0:a]
        part_two_y2 = y2[a:n][np.where(y2[a:n] != -1)][-(n-b-1):]
        y2 = np.concatenate((part_one_y2, x1[a:b+1], part_two_y2))

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
        s = 0.001
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

    def breed(self, parents):
        offspring = np.empty(parents.shape, dtype=int)
        for i in np.arange(0, parents.shape[0], step=2):
            offspring[i, :], offspring[i+1, :] = self.nwox(parents[i, :], parents[i+1, :])
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




def main():
    filename = "data/tour250.csv"
    optimize = r0652717()
    optimize.optimize(filename)


if __name__ == "__main__":
    main()