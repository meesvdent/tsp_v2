import numpy as np

def nwox(x1, x2):
    n = x1.shape[0]

    a = np.random.randint(0, n+1)
    if a == n:
        b = n
    else:
        b = np.random.randint(a, n+1)

    a, b = 2, 4

    print("crossover points: ", a, b)

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




x1 = np.array(["A", "E", "B", "C", "G", "M", "D", "H", "O", "J", "K", "L", "F", "N", "I"])
x2 = np.array(["F", "D", "A", "N", "K", "H", "L", "M", "I", "G", "J", "E", "B", "C", "O"])

x3 = np.array([3, 5, 1, 4, 7, 6, 2, 8])
x4 = np.array([4, 6, 5, 1, 8, 3, 2, 7])

print(nwox(x3, x4))