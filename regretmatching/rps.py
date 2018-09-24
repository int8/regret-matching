from regretmatching.model import RegretMatchingDecisionMaker, Expert
import numpy as np

ROCK = Expert('rock')
PAPER = Expert('paper')
SCISSORS = Expert('scissors')

RPS_EXPERTS = [ROCK, PAPER, SCISSORS]

RPS_REWARD_VECTORS = {
    ROCK:     np.asarray([0, 1, -1]),  # opponent playing ROCK
    PAPER:    np.asarray([-1, 0, 1]),  # opponent playing PAPER
    SCISSORS: np.asarray([1, -1, 0]),  # opponent playing PAPER
}


class RPSPlayer(RegretMatchingDecisionMaker):
    def __init__(self):
        super(RPSPlayer, self).__init__(RPS_EXPERTS)
        self.sum_p = np.full(3, 0.)
        self.games_played = 0

    def move(self):
        return self.decision()

    def learn_from(self, opponent_move):
        reward_vector = RPS_REWARD_VECTORS[opponent_move]
        self.update_rule(reward_vector)
        self.games_played += 1
        self.sum_p += self.p

    def current_best_response(self):
        return np.round(self.sum_p / self.games_played, 4)

    def eps(self):
        return np.max(self.regrets / self.games_played)
