from multiagent.scenarios.arch.commons import *
from multiagent.scenarios.simple_reference_3 import Scenario as S


class Scenario(S):

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = max_reward(world)
        rew += collision_reward(agent, world)
        return rew
