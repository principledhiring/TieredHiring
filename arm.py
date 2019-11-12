import numpy as np
import random

class Arm(object):

    def __init__(self, true_utility, props):
        self.true_utility = true_utility
        self.props = props
        self.cost = 0
        self.gain = 0
        self.util = props["dist_mean"]

    def get_prop(self, key):
        return self.props[key]

    def get_util(self):
        return self.util

    def get_gain(self):
        return self.gain

    def get_cost(self):
        return self.cost

    def reset(self):
        self.cost = 0
        self.gain = 0
        self.util = self.props["dist_mean"]

class Set_Arm(Arm):

    class Real_Pull(object):
        def __init__(self,arr):
            self.remaining_pulls = arr
            random.shuffle(self.remaining_pulls)
            self.already_pulled  = []

        def sample(self):
            smpld = self.remaining_pulls.pop()
            self.already_pulled.append(smpld)
            return smpld

    def __init__(self, real_pulls, *args, **kwargs):
        # initialize a Set_Arm which has hard pull data
        # real_pulls: list of lists
        real_pulls = [Set_Arm.Real_Pull(x) for x in real_pulls]
        self.real_pulls = real_pulls
        super(Set_Arm, self).__init__(*args, **kwargs)
        return

    def pull_arm(self, s, j, stage):
        # real_pulls for stage
        rpfs = self.real_pulls[stage]
        if (len(rpfs.remaining_pulls)):
            reward = rpfs.sample()
        else:
            reward = 0
            for i in range(s):
                # sample from distribution
                sample = np.random.normal(self.true_utility, self.props["sigma"])
                reward += sample
        self.gain += s
        self.cost += j
        self.util = (self.util*(self.gain-s)*1.0+reward)/(self.gain)
        return reward

class Normal_Arm(Arm):

    def __init__(self, *args, **kwargs):
        super(Normal_Arm, self).__init__(*args, **kwargs)

    def get_true_utility(self):
        return self.true_utility

    def pull_arm(self, s, j, stage):
        reward = 0
        for i in range(s):
            # sample from distribution
            sample = np.random.normal(self.true_utility, self.props["sigma"])
            reward += sample
        self.gain += s
        self.cost += j
        self.util = (self.util*(self.gain-s)*1.0+reward)/(self.gain)
        return reward
