import numpy as np


class Expert:
    def __init__(self, id):
        self.id = id


class RegretMatchingDecisionMaker:

    def __init__(self, experts):
        self.n = len(experts)
        # experts list
        self.experts = experts
        # cumulative expected reward for our Regret Matching algorithm
        self.expected_reward = 0.
        # cumulative expected rewards for experts
        self.experts_rewards = np.zeros(self.n)
        # cumulative regrets towards experts
        self.regrets = np.zeros(self.n)
        # probability disribution over experts to draw decision from
        self.p = np.full(self.n, 1. / self.n)

    def decision(self):
        expert = np.random.choice(self.experts, 1,  p=self.p)
        return expert[0]

    def update_rule(self, rewards_vector):
        self.expected_reward += np.dot(self.p, rewards_vector)
        self.experts_rewards += rewards_vector
        self.regrets = self.experts_rewards - self.expected_reward
        self._update_p()

    def _update_p(self):
        sum_w = np.sum([self._w(i) for i in np.arange(self.n)])
        if sum_w <= 0:
            self.p = np.full(self.n, 1. / self.n)
        else:
            self.p = np.asarray(
                [self._w(i) / sum_w for i in np.arange(self.n)]
            )

    def _w(self, i):
        return max(0, self.regrets[i])
