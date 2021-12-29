import numpy as np

file = open("data/tour29.csv")
distance = np.loadtxt(file, delimiter=',')[:10, :10]
file.close()

route = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

def two_opt_swap(route, i, j):
    new_route = np.array(route)
    new_route[i:j+1] = np.flip(route[i:j+1])
    return new_route

def calc_cost(route, distance):
    cost = 0
    for i in range(route.shape[0] - 1):
        cost += distance[route[i], route[i+1]]
    cost += distance[route[-1], route[0]]
    return cost

def two_opt(route, distance, cost=None):
    if cost == None:
        cost = calc_cost(route, distance)
    for i in range(1, route.shape[0]-1):
        for j in range(2, route.shape[0]):
            new_route = two_opt_swap(route, i, j)
            new_cost = calc_cost(new_route, distance)
            if new_cost < cost:
                return two_opt(new_route, distance, new_cost)
    return route

print(two_opt(route, distance))


