from r0652717 import r0652717

tours = [
    "29",
    "250",
    "750",
    "1000"
]

tourlengths = [
    29,
    250,
    750,
    1000
]

pop_sizes = [
    100,
    500,
    1000,
    2000
]

mus = [
    0.5,
    1
]

alphas = [
    0.05,
    0.2,
    0.75,
    1
]

betas = [
    0.01,
    0.1,
    0.5,
    1
]

lambdaas = [
    0.3,
    0.6,
    0.8,
]

ks = [
    2,
    32,
    64,
]

init_params = [
    0.1,
    0.5,
    0.8,
    0.9,
    0.95,
]

def main():
    tournames = ["data/tour" + tour_number + ".csv" for tour_number in tours]
    for i in range(len(tournames)):
        tourname = tournames[i]
        print(tourname)
        tourlength = tourlengths[i]
        j = 0
        for pop_size in pop_sizes:
            print("pop size: ", pop_size)
            for mu in mus:
                mu = int(pop_size * mu)
                print("mu: ", mu)
                for alpha in alphas:
                    alpha = int(alpha * mu)
                    print("alpha: ", alpha)
                    for beta in betas:
                        beta = int(beta * mu)
                        print("beta: ", beta)
                        for lambdaa in lambdaas:
                            lambdaa = int(lambdaa * mu) - ((int(lambdaa * mu)) % 2)
                            print("lambda: ", lambdaa)
                            if lambdaa == 0:
                                lambdaa = 2
                            for k in ks:
                                print("k: ", k)
                                for init_param in init_params:
                                    init_param = int(init_param * tourlength)
                                    print("init param: ", init_param)
                                    population = r0652717(
                                        pop_size,
                                        mu, alpha,
                                        beta,
                                        lambdaa,
                                        k,
                                        init_param,
                                        str(tourlength) + "-" + str(j))
                                    population.optimize(tourname)
                                    j += 1


if __name__ == "__main__":
    main()
