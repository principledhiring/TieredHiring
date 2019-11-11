import numpy as np
import utils
import oracles
from uniform import Uniform
from brutas import BRUTaS
from caco import CACO
import pickle
from arm import Normal_Arm
import os

class Experiment(object):
    def __init__(self, arms, sigma, save_directory='./data/',
                 q=2):
    # """Initialize a Experiment super object.
    # Args:
    #     arms: array of arms
    #     sigma: float sigma of arms
    #     q: int number of groups for submodular utility
    #     save_directory: directory to save experiments into"""
        # Set up groups
        # Set up arms
        self.sigma = sigma
        self.arms = arms
        self.n = len(arms)
        # Set up groups
        self.q = q
        self.groups = np.arange(self.q)
        self.arms_groups = np.random.randint(len(self.groups), size=self.n)
        self.oracle_args = [self.groups, self.arms_groups]
        # Save directory
        self.save_directory = save_directory
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        return

    def run_experiment(self, delta, epsilon, oracle_type='topK',
                       algorithm='uniform', T=None, K=[10,5],
                       S=[1,15], J=[1,6], seed=None, name=None):
    # """Run a single experiment.
    # Args:
    #     delta: float
    #     epsilon: float
    #     oracle_type: 'topK' or 'submod'
    #     algorithm: 'caco', 'brutas', 'uniform', or 'random'
    #     T: (int list) budgets for each round
    #     J: (int list) cost for each round
    #     S: (int list) information gain for each round
    #     K: (int list) The number of arms to move onto the next round.
    #     seed: int
    # """
        if seed is not None:
            np.random.seed(seed)
        # Reset arms
        for arm in self.arms:
            arm.reset()

        # Set up the oracle
        if oracle_type == 'submod':
            oracle = oracles.submodular_max_oracle
            oracle_utility = oracles.submodular_max_utility
            this_oracle_args = self.oracle_args
        elif oracle_type == 'topK':
            oracle = oracles.top_k_oracle
            oracle_utility = oracles.top_k_utility
            this_oracle_args = []
        else:
            raise ValueError('Unknown oracle type ' + oracle_type)

        # Set up the algorithm
        if algorithm == 'caco':
            mab_alg = CACO(self.arms, K, S, J, delta,
                           self.sigma, epsilon, oracle, oracle_utility,
                           this_oracle_args)
        elif algorithm == 'brutas':
            if oracle_type == 'submod':
                oracle = oracles.c_submod_oracle
            else:
                oracle = oracles.c_top_k_oracle
            mab_alg = BRUTaS(self.arms, T, K, S, J, delta,
                              self.sigma, epsilon, oracle, oracle_utility,
                              this_oracle_args)
        elif algorithm == 'uniform':
            mab_alg = Uniform(self.arms, T, K, S, J, delta,
                              self.sigma, epsilon, oracle, oracle_utility,
                              this_oracle_args)
        elif algorithm == 'random':
            mab_alg = None
        else:
            raise ValueError('Unknown algorithm ' + algorithm)

        # Run the algorithm
        A = mab_alg.run_alg();

        # Save all of the things!
        results_dict = {'A': A,
                        'mab_alg': mab_alg, #Contains all of the information
                        'arms': self.arms}
        directory = os.path.join(self.save_directory, oracle_type, algorithm)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if name is None:
            name = "delta_{}_sigma_{}_epsilon_{}.pkl".format(delta, self.sigma, epsilon);
        file_path = os.path.join(directory, name)
        with open(file_path, 'wb+') as f:
            pickle.dump(results_dict, f)

class Random_Experiment(Experiment):

    def __init__(self, arm_seed, n, sigma, dist_mean, *args, **kwargs):
    # """Initialize an Random_Experiment object.
    # this will make n arms with random true utility sampled from  a
    # normal with mean dist_mean and unit variance. Each arm's sigma will be
    # the argument sigma.
    # Args:
    #     arm_seed: the seed to build the arms
    #     n: int number of arms
    #     sigma: float sigma of arms
    #     dist_mean: float mean to pull arms around
        self.arm_seed = arm_seed
        np.random.seed(self.arm_seed)
        # Set up arms
        arms = [Normal_Arm(np.random.normal(dist_mean),
            {"sigma": sigma, "dist_mean": dist_mean}) for i in range(n)]
        super(Random_Experiment, self).__init__(arms, sigma, *args, **kwargs)


# ex = Random_Experiment(1, 50, 0.2, 10, save_directory="data/")
# ex.run_experiment(0.3, 0.3, T=[200,60], algorithm="uniform")
# ex.run_experiment(0.3, 0.3, T=[200,60], algorithm="brutas")
# ex.run_experiment(0.3, 0.3, algorithm="caco")
# ex.run_experiment(0.3, 0.3, T=[200,60], algorithm="uniform", oracle_type="submod")
# ex.run_experiment(0.3, 0.3, T=[200,60], algorithm="brutas", oracle_type="submod")
# ex.run_experiment(0.3, 0.3, algorithm="caco", oracle_type="submod")



#     def grid_test(delta, epsilon, oracle_type="topK", algorithm="uniform", budgets=[1],
#                   k_type="flipped_poly", js_type="linear", ms=[2],
#                   k_rates=[1,5,10], j_rates=[1], s_rates=[2]):
#     """Runs an algorithm over m_range, varying s's and j's as desired
#     Args:
#         delta: float
#         epsilon: float
#         oracle_type: "topK" or "Submod"
#         algorithm: "caco", "brutas", "uniform", "random"
#         budgets: int list
#         k_type: 'fraction' - removes half the arms each round. m=None
#                 'lin' - k_i's decrease linearly. m required
#                 'poly' - k_i's decrease polynomially. m, power required
#                         pass in power in (0, 1) for convex function from 0 to m
#                         pass in power in (1, inf) for concave function from 0 to m
#                 'flipped_poly' - k_i's as above but rotated diagonally over linear line
#         js_type: 'linear', 'poly', 'mult'
#         ms:
#         k_rates: lists of rates for k to grow
#         j_rates: list of rates for j to grow
#         s_rates: list of rates for s to grow


#     """
#     for m in ms: # Loop over all of the rounds to test
#         for k_rate in k_rates: # Test all of the k growing rates
#             for j_rate in j_rates: # Test all of the j growing rates
#                 for s_rate in s_rates: # Test all of the s growing rates
#                     # j_rate must be strictly lower than s_rate
#                     if j_rate >= s_rate:
#                         continue
#                     J, S = get_costs_gains(m, j_rate=j_rate, s_rate=s_rate, type=js_type)
#                     K = get_ks(k, len(arms), m=m, rate=k_rate, type=k_type)
#                     if alg != 'caco':
#                         # if it's a budgetted algorithm
#                         # change K[i] to number of decisions needed to get K[i]
#                         K = [len(arms) - k for k in K]
#                         for i in range(1, len(K)-1):
#                             K[i] -= K[:i]
#                         K[-1] = len(arms) - sum(K[:-1])

#                         b = [budgets]*m
#                         if set_Ts == None:
#                             set_Ts = itertools.product(*b)
#                         for T in set_Ts:
#                             # Set up the budget correctly
#                             T = list(T)
#                             T =
#                             for arm in arms:
#                                 arm.reset()
#                             np.random.seed(seed)
#                             if alg == "cutAR":
#                                 print("starting_{}".format(T))
#                                 final_A, data = cutAR(arms, T, K, S, J, k,  delta, sigma, epsilon, oracle, oracle_utility,
#                                                       oracle_args=oracle_args, save_data=True)
#                             elif alg == "random":
#                                 final_A, data = uniform_random(arms, T, K, S, J,  delta, sigma, epsilon, oracle,
#                                                                oracle_utility, oracle_args=oracle_args,
#                                                                save_decisions=save_decisions, uniform=False)
#                             elif alg == "uniform":
#                                 final_A, data = uniform_random(arms, T, K, S, J,  delta, sigma, epsilon, oracle,
#                                                                oracle_utility, oracle_args=oracle_args,
#                                                                save_decisions=save_decisions, uniform=True)
#                             data.seed = seed
#                             data.normal_sig = normal_sig

#                             # pickling
#                             file_name = "{}_n{}_m{}_k{}_krate{}_ktype_{}_srate{}_jrate{}_jstype_{}_eps{}_delt{}_sig{}_nsig{}"\
#                                 .format(alg, len(arms), m, k, k_rate, k_type, s_rate, j_rate, js_type,
#                                         epsilon, delta, sigma, normal_sig)
#                             i = 0
#                             file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                             while os.path.exists(file_path):
#                                 i += 1
#                                 file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                             with open(file_path, 'wb+') as f:
#                                 pickle.dump(data, f)

# def grid_test(arms, k, delta, sigma, epsilon, oracle, oracle_utility, oracle_args=[], alg="cut", budgets=[1],
#               m_range=(2, 6), k_type="flipped_poly", k_range=(1, 10, 1), js_type='linear',
#               j_range=(1, 1, 1), s_range=(2, 2, 1), save_dir='data/', save_decisions=False, seed=None, normal_sig=1, set_Ts=None):
#     """Runs alg over m_range, varying s's and j's as desired.

#     Args:
#         alg (string): cut, cutAR, uniform, random
#         m_range (int tuple): range of m to test, m > 1 (start, stop(-1), step)
#         j_range (int or float): rates that costs grow at (start, stop(-1), step)
#         s_range (int or float): rates that gains grow at (start, stop(-1), step)
#     """

#     j_rates = np.arange(j_range[0], j_range[1]+1, j_range[2])
#     s_rates = np.arange(s_range[0], s_range[1]+1, s_range[2])
#     k_rates = np.arange(k_range[0], k_range[1]+1, k_range[2])

#     script_path = os.path.abspath(__file__)
#     script_dir = os.path.split(script_path)[0]
#     abs_dir_path = os.path.join(script_dir, save_dir)

#     if not os.path.exists(abs_dir_path):
#         os.makedirs(abs_dir_path)

#     for m in range(m_range[0], m_range[1]+1):
#         for k_rate in k_rates:
#             for j_rate in j_rates:
#                 for s_rate in s_rates:
#                     if s_rate > j_rate:
#                         J, S = get_costs_gains(m, j_rate=j_rate, s_rate=s_rate, type=js_type)
#                         K = get_ks(k, len(arms), m=m, rate=k_rate, type=k_type)
#                         if alg != 'cut':
#                             # if it's a budgetted algorithm
#                             # change K[i] to number of decisions needed to get K[i]
#                             K = [len(arms) - k for k in K]
#                             for i in range(1, len(K)-1):
#                                 K[i] -= K[:i]
#                             K[-1] = len(arms) - sum(K[:-1])

#                             b = [budgets]*m
#                             if set_Ts == None:
#                                 set_Ts = itertools.product(*b)
#                             for T in set_Ts:
#                                 for arm in arms:
#                                     arm.reset()
#                                 np.random.seed(seed)
#                                 if alg == "cutAR":
#                                     print("starting_{}".format(T))
#                                     final_A, data = cutAR(arms, T, K, S, J, k,  delta, sigma, epsilon, oracle, oracle_utility,
#                                                           oracle_args=oracle_args, save_data=True)
#                                 elif alg == "random":
#                                     final_A, data = uniform_random(arms, T, K, S, J,  delta, sigma, epsilon, oracle,
#                                                                    oracle_utility, oracle_args=oracle_args,
#                                                                    save_decisions=save_decisions, uniform=False)
#                                 elif alg == "uniform":
#                                     final_A, data = uniform_random(arms, T, K, S, J,  delta, sigma, epsilon, oracle,
#                                                                    oracle_utility, oracle_args=oracle_args,
#                                                                    save_decisions=save_decisions, uniform=True)
#                                 data.seed = seed
#                                 data.normal_sig = normal_sig

#                                 # pickling
#                                 file_name = "{}_n{}_m{}_k{}_krate{}_ktype_{}_srate{}_jrate{}_jstype_{}_eps{}_delt{}_sig{}_nsig{}"\
#                                     .format(alg, len(arms), m, k, k_rate, k_type, s_rate, j_rate, js_type,
#                                             epsilon, delta, sigma, normal_sig)
#                                 i = 0
#                                 file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                                 while os.path.exists(file_path):
#                                     i += 1
#                                     file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                                 with open(file_path, 'wb+') as f:
#                                     pickle.dump(data, f)
#                         else:
#                             for arm in arms:
#                                 arm.reset()
#                             np.random.seed(seed)
#                             final_A, data = cut(delta, epsilon, sigma, arms, K, S, J, oracle, oracle_utility,
#                                                 oracle_args=oracle_args,
#                                                 save_decisions=save_decisions)
#                             data.seed = seed
#                             data.normal_sig = normal_sig

#                             # pickling
#                             file_name = "{}_n{}_m{}_k{}_krate{}_ktype_{}_srate{}_jrate{}_jstype_{}_eps{}_delt{}_sig{}_nsig{}"\
#                                 .format(alg, len(arms), m, k, k_rate, k_type, s_rate, j_rate, js_type,
#                                         epsilon, delta, sigma, normal_sig)
#                             i = 0
#                             file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                             while os.path.exists(file_path):
#                                 i += 1
#                                 file_path = os.path.join(abs_dir_path, file_name + "_%s.pkl" % i)
#                             with open(file_path, 'wb+') as f:
#                                 pickle.dump(data, f)
