import numpy as np


n = 100  # number of people in the population
m = 20  # number of tests
k = 2  # number of infected people
delta = 0.05  # crossover probability of BSC channel
q = k / n  # probability of an individual to be infected

#no_iter = 10  # number of BP iterations
#no_trials = 1024  # number of simulation trials
no_iter = 101  # number of BP iterations
no_trials = 120  # number of simulation trials

# d_v = 3 # number of 1's per column for regular sparse testing matrix
# H = Random_Regular_Sparse_Matrix(n,m,d_v)
nu = np.log(2)  # nu/k is the probability that an entry in Bernoulli testing matrix is 1