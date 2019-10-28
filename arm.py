class Arm(object):

    def __init__(self, true_utility, arm_pull_strategy, props):
        self.true_utility = abs(true_utility)
        self.arm_pull_strategy = arm_pull_strategy
        self.props = props
        self.cost = 0
        self.gain = 0
        self.util = props["dist_mean"]

    def get_true_utility(self):
        return self.true_utility

    def get_prop(self, key):
        return self.props[key]

    def get_util(self):
        return self.util

    def get_gain(self):
        return self.gain

    def get_cost(self):
        return self.cost

    def pull_arm(self, s, j):
        reward = 0
        for i in range(s):
            reward += self.arm_pull_strategy(self.true_utility, self.props["sigma"])
        self.gain += s
        self.cost += j
        self.util = (self.util*(self.gain-s)*1.0+reward)/(self.gain)

    def reset(self):
        self.cost = 0
        self.gain = 0
        self.util = self.props["dist_mean"]