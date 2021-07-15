import random
from multiagent.scenarios.commons import *
from multiagent.scenarios.simple_spread_random_one import Scenario as S


class Scenario(S):

    def initialise(self, world):
        self.perm_oh = permutation_utils(world)

    def reset_world(self, world):
        super().reset_world(world)

        self.assignment = np.eye(len(world.agents))
        self.assignment_obs = []

        obs_seed = random.randint(0, 1) * 2 - 1
        for i, agent in enumerate(world.agents):
            self.assignment_obs.append(self.assignment * self.assignment[(agent.index + obs_seed) % len(world.agents)])

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = max_reward(world)
        rew += collision_reward(agent, world)
        return rew

    def observation(self, agent, world):

        entity_pos, other_pos = obs_relative(agent, world)

        landmarks_info = list(zip(entity_pos, self.assignment_obs[agent.index]))
        random.Random(self.wland[agent.index]).shuffle(landmarks_info)
        random_entity_pos, random_entity_assignment = list(zip(*landmarks_info))

        random_entity_pos = list(random_entity_pos)
        random_entity_assignment = list(random_entity_assignment)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + random_entity_pos + other_pos + random_entity_assignment)
