import numpy as np

class Uniform(object):

    def __init__(self, arms, T, K, S, J, delta, sigma, epsilon, oracle, utility, oracle_args=[]):
        self.arms = arms
        self.T = T
        self.K = K
        self.S = S
        self.J = J
        self.delta = delta
        self.sigma = sigma
        self.epsilon = epsilon
        self.oracle = oracle
        self.utility = utility
        self.oracle_args = oracle_args

        self.A = np.arange(len(arms))
        self.util = np.zeros(len(arms))
        self.cost = [0]

    def run_alg(self):
        for i in range(len(self.K)):
            # Pull each arm T[i]/jn number of times.
            for j in self.A:
                for x in range(int(self.T[i]/(self.A.size*self.J[i]))):
                    self.arms[j].pull_arm(self.S[i], self.J[i], i)
                    self.cost[-1] += self.J[i]
                self.util[j] = self.arms[j].get_util()
            # Get the reduced set for this round
            self.A = self.oracle(self.util, self.K[i], self.A, *self.oracle_args)
            self.cost.append(self.cost[-1])

        return self.A
