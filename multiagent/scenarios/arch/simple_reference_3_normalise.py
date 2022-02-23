from multiagent.scenarios.arch.commons import *
from multiagent.scenarios.simple_reference_3 import Scenario as S


class Scenario(S):
    def reset_world(self, world):
        super().reset_world(world)
        self.norm_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l, a in zip(world.landmarks, world.agents)]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l, a, nd in zip(world.landmarks, world.agents, self.norm_dist):
            dists = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
            rew -= dists / nd
        rew += collision_reward(agent, world)
        return rew
