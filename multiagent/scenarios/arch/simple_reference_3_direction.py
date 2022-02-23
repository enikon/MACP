from multiagent.scenarios.arch.commons import *
from multiagent.scenarios.simple_reference_3 import Scenario as S


def vec_len(x):
    return np.sqrt(np.sum(np.square(x)))


class Scenario(S):
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l, a in zip(world.landmarks, world.agents):
            dists = \
                (1 - np.abs(
                    np.dot(l.state.p_pos - a.state.p_pos, a.state.p_vel)\
                    /(vec_len(l.state.p_pos - a.state.p_pos)
                        *
                      vec_len(a.state.p_vel)
                    )
                ))/2
            rew -= (1+dists) * (vec_len(l.state.p_pos - a.state.p_pos ))
        rew += collision_reward(agent, world)
        return rew
