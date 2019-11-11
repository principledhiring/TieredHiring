import utils
import numpy as np
import math

class BRUTaS(object):

    def __init__(self, arms, T, K, S, J, delta, sigma, epsilon, oracle, utility, oracle_args=[]):
        self.arms = arms
        self.T = T
        self.k = K[-1]
        self.K = utils.get_budget_Ks(self.arms,K)
        self.S = S
        self.J = J
        self.delta = delta
        self.sigma = sigma
        self.epsilon = epsilon
        self.oracle = oracle
        self.utility = utility
        self.oracle_args = oracle_args

        self.n = len(self.arms)
        self.active_arms = np.arange(self.n)
        self.util = np.zeros(self.n)
        self.cost = [0]
        self.A_i = []
        self.B_i = []

    def log_tilde(self, n):
        sm = 0
        for i in range(1, n+1):
            sm += 1/i
        return sm


    def run_alg(self):
        for i in range(len(self.K)):
            print('\tround {}'.format(i))
            T_tilde_i = [0]
            for t in range(self.K[i]):
                print('\t\tphase {}'.format(t))
                T_tilde_i.append(math.ceil(
                    ((self.T[i]*self.K[i]*self.J[i]) - self.K[i])/(self.log_tilde(self.K[i])*self.J[i]*(self.K[i]-t+1))))
                num_pulls = T_tilde_i[-1] - T_tilde_i[-2]
                for arm in self.active_arms:
                    for pull in range(num_pulls):
                        self.arms[arm].pull_arm(self.S[i], self.J[i], i)
                    self.util[arm] = self.arms[arm].get_util()
                    self.cost[-1] += self.J[i]*num_pulls
                    # if save_data:
                    #     if self.J[i] in data.num_pulls[arm]:
                    #         data.num_pulls[arm][J[i]] += num_pulls
                    #     else:
                    #         data.num_pulls[arm][J[i]] = num_pulls
                M_i = self.oracle(self.util, self.k, self.active_arms, self.A_i, *self.oracle_args)
                print("got M")

                if len(M_i) == 0:
                    # failure condition, if set returned is empty
                    return None
                M_i_tilde = {}
                cnt = 0
                for arm in self.active_arms:
                    if arm not in M_i:
                        # if arm is not in M_i, then we add it to accepted set
                        A_i_arm = self.A_i + [arm]
                    else:
                        # if arm is in M_i, we reject it
                        A_i_arm = self.A_i
                    # we remove arm from Active_arms either way
                    M_i_tilde[arm] = self.oracle(self.util, self.k,
                                                        self.active_arms[self.active_arms != arm],
                                                        A_i_arm, *self.oracle_args)
                    cnt += 1
                candidate = self.active_arms[0]
                util_M_i = self.utility(self.util, M_i, *self.oracle_args)
                candidate_value = util_M_i - self.utility(self.util, M_i_tilde[candidate],
                                                            *self.oracle_args)
                for arm in self.active_arms:
                    diff = util_M_i - self.utility(self.util, M_i_tilde[arm], *self.oracle_args)
                    if diff > candidate_value:
                        candidate = arm
                        candidate_value = diff


                if candidate in M_i:
                    self.A_i.append(candidate)
                    # if save_data:
                    #     data.decisions.append(1)
                    if len(self.A_i) == self.k and (t < self.K[i]-1 or i < len(self.K)-1):
                        print("Finished early in round {} of {}, phase {} of {}".format(i+1, len(self.K), t+1, self.K[i]))
                        # print(A_i)
                        # if save_data:
                        #     data.A_i = A_i
                        #     data.B_i = B_i
                        #     data.cost_i.append(cost)
                        #     data.T_tilde_i.append(T_tilde_i)
                        #     data.ended_early = (i+1, t+1)
                        #     data.M_i.append(M_i)
                        #     data.utils_i.append(util)
                        return self.A_i
                else:
                    self.B_i.append(candidate)
                    # if save_data:
                    #     data.decisions.append(0)
                # removing choosen arm from Active_arms and setting its util to 0
                self.active_arms = self.active_arms[self.active_arms != candidate]
            self.cost.append(self.cost[-1])
            # if save_data:
            #     data.cost_i.append(cost)
            #     data.T_tilde_i.append(T_tilde_i)
            #     data.M_i.append(M_i)
            #     data.utils_i.append(util)
        # if save_data:
        #     data.A_i = A_i
        #     data.B_i = B_i
        return self.A_i
