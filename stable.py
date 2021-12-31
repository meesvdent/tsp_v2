import Reporter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Modify the class name to match your student number.
class r0652717:

    def __init__(self):

        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename, ):
        # Read distance matrix from file.
        file = open(filename)
        self.distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.convert_distance()
        self.iter = 1

        # PARAMETERS
        init_size = 1000
        mu = init_size
        alpha = int(mu * 0.1)
        beta = int(mu * 0.001)
        lambdaa = int(mu * 0.6) - int(mu*0.6)%2
        self.k = 10
        if alpha == 0:
            alpha = 1
        if beta == 0:
            beta = 4
        if lambdaa == 0:
            lambdaa = 2

        mean = np.array([])
        best = np.array([])


        # Your code here.
        population = self.init_population(init_size, int(0.7 * self.distance_matrix.shape[0]), self.distance_matrix)
        costs = self.calc_all_cost(population)

        test_convergence = True
        converged = False
        while test_convergence:


            mean_objective = np.mean(costs)
            best_objective = np.min(costs)
            best_solution = np.array(population[np.argmin(costs), :])

            if self.iter > 20 and np.allclose(np.array(mean[-10:], dtype=float), np.array(best[-10:], dtype=float)):
                converged = True
                print("Converged!")
                lambdaa = int(mu * 0)
                alpha = int(mu * 1)
                beta = 1
            else:
                converged = False
                lambdaa = int(mu * 0.6)
                alpha = int(mu * 0.01)
                beta = int(mu * 0.3)

            if mean_objective > 2 * best_objective:
                mean = np.concatenate((mean, np.array([None])))
            else:
                mean = np.concatenate((mean, np.array([mean_objective])))
            best = np.concatenate((best, np.array([best_objective])))
            if self.iter % 5 == 0:
                plt.plot(np.arange(self.iter), mean)
                plt.plot(np.arange(self.iter), best)
                plt.show()

            # Your code here.
            if converged == False:
                parents = population[self.select_good_individuals(lambdaa, costs, replace=True), :]
                offspring = self.breed(parents)
                #offspring_to_mutate = np.random.choice(offspring.shape[0], size=int(0.1*offspring.shape[0]), replace=False)
                #offspring[offspring_to_mutate, :] = np.apply_along_axis(self.scramble_mutation, 1, offspring[offspring_to_mutate, :])
                # offspring = np.apply_along_axis(self.k_opt, 1, offspring)

            to_mutate = population[self.select_bad_individuals(alpha, costs, replace=True), :]
            mutated = self.scramble_batch(to_mutate)
            if converged:
                mutated = np.apply_along_axis(self.k_opt, 1, mutated)

            to_local_search = self.select_good_individuals(beta, costs, replace=False)
            to_be_improved = population[to_local_search, :]
            improved = np.apply_along_axis(self.k_opt, 1, to_be_improved)

            if not converged:
                population, costs = self.add_to_pop(offspring, population, costs)
            population, costs = self.add_to_pop(mutated, population, costs)
            population, costs = self.add_to_pop(improved, population, costs)

            print("pre elim shape: ", population.shape)

            population, costs = self.eliminate(mu, population, costs)

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
            print("Time left: ", time_left)
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

    def scramble_batch(self, routes):
        return np.apply_along_axis(self.scramble_mutation, 1, routes)

    def calc_cost(self, route):
        cost = 0
        for i in range(route.shape[0] - 1):
            cost += self.distance_matrix[route[i], route[i+1]]
        cost += self.distance_matrix[route[-1], route[0]]
        return cost

    def scramble_mutation(self, route):
        randint_one = np.random.randint(1, route.shape[0]-2)
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

    def ranked_exp_decay(self, costs, selection_param, best=True):
        """Best = True -> RANK 0 is LOWEST cost (best individual, highest prob)"""
        if best == False:
            costs = -costs
        temp = costs.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(costs))
        highest = costs[ranks[-1]]
        alpha = 0.99
        s = alpha ** selection_param
        a = np.log(s) / (costs.shape[0] - 1)
        exponent = a * ranks
        weights = np.exp(exponent)
        prob = weights / np.sum(weights)
        # plt.plot(ranks, weights, 'o')
        # plt.show()

        return prob

    def select_bad_individuals(self, n, costs, replace=True):
        prob = self.ranked_exp_decay(costs, selection_param=500, best=False)
        selected = np.random.choice(costs.shape[0], size=n, p=prob, replace=replace)
        return selected

    def select_good_individuals(self, n, costs, replace=True):
        prob = self.ranked_exp_decay(costs, selection_param=200, best=True)
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

    def inversion_mut(self, route):
        indices = np.random.choice(route.shape[0]-1, 2, replace=False)
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


    def two_opt(self, route, cost=None, inv_length = 10):
        if cost is None:
            cost = self.calc_cost(route)
        i = np.random.randint(0, route.shape[0] - inv_length)
        if i + inv_length > route.shape[0]:
            range_end = route.shape[0]
        else:
            range_end = i + inv_length
        for j in range(i+1, range_end):
            new_route = self.two_opt_swap(route, i, j)
            new_cost = self.calc_cost(new_route)
            if new_cost < cost:
                return self.two_opt(route, new_cost)
        return route



def main():
    filename = "data/tour250.csv"
    optimize = r0652717()
    optimize.optimize(filename)


if __name__ == "__main__":
    main()

