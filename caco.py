import numpy as np

class CACO(object):

    def __init__(self, arms, K, S, J, delta, sigma, epsilon, oracle, utility, oracle_args=[]):
        self.arms = arms
        self.K = K
        self.S = S
        self.J = J
        self.delta = delta
        self.sigma = sigma
        self.epsilon = epsilon
        self.oracle = oracle
        self.utility = utility
        self.oracle_args = oracle_args

        self.A = np.arange(len(self.arms))
        self.util = np.zeros(len(self.arms))
        self.rad = np.zeros(len(self.arms))
        self.cost = [0]

    def run_alg(self):
        for i in range(len(self.K)):
            for j in self.A:
                self.arms[j].pull_arm(self.S[i], self.J[i], i)
                self.util[j] = self.arms[j].get_util()

                # if save_data:
                #     data.num_pulls[j][J[i]] = 1
            self.cost[-1] += self.A.size*self.J[i]

            while True:

                A_new = self.oracle(self.util, self.K[i], self.A, *self.oracle_args)
                for a in self.A:
                    self.rad[a] = self.sigma*np.sqrt(2*np.log(4*self.A.size*(self.cost[-1]**3)/self.delta)/self.arms[a].get_gain())

                old_set = set(self.A)
                A_new_set = set(A_new)
                util_tilde = np.zeros(len(self.util))

                for a in old_set:
                    if a in A_new_set:
                        util_tilde[a] = self.util[a] - self.rad[a]
                    else:
                        util_tilde[a] = self.util[a] + self.rad[a]

                A_tilde = self.oracle(util_tilde, self.K[i], self.A, *self.oracle_args)
                closeness = abs(self.utility(self.util, A_new, *self.oracle_args) -
                                self.utility(self.util, A_tilde, *self.oracle_args))

                # if save_decisions:
                #     data.decisions[i]['A_new'].append(A_new)
                #     data.decisions[i]['A_tilde'].append(A_tilde)
                #     data.decisions[i]['util_tilde'].append(util_tilde)
                #     data.decisions[i]['closeness'].append(closeness)

                if closeness < self.epsilon:
                    self.cost.append(self.cost[-1])
                    self.A = A_new
                    break

                A_tilde_set = set(A_tilde)
                symmetric_difference = list((A_new_set - A_tilde_set) | (A_tilde_set - A_new_set))
                p = symmetric_difference[0]
                for a in symmetric_difference:
                    if self.rad[p] < self.rad[a]:
                        p = a

                self.arms[p].pull_arm(self.S[i], self.J[i], i)
                self.util[p] = self.arms[p].get_util()
                self.cost[-1] += self.J[i]

                # if save_data:
                #     data.num_pulls[p][J[i]] += 1
                #     if save_decisions:
                #         data.decisions[i]['symmetric_difference'].append(symmetric_difference)
                #         data.decisions[i]['arm_pulled'].append(p)
        self.cost.append(self.cost[-1])
        # if save_data:
        #     data.A_i.append(A_new)
        #     data.utils_i.append(util)
        #     data.rad_i.append(rad)
        #     data.cost_i.append(cost)
        return A_new
