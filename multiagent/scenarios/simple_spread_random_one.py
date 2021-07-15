import random
from multiagent.scenarios.commons import *
from multiagent.scenarios.simple_spread import Scenario as S


class Scenario(S):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        self.perm_oh = permutation_utils(world)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        super().reset_world(world)
        self.wagents = [random.randint(*int_range) for _ in range(len(world.agents))]
        self.wland = [random.randint(*int_range) for _ in range(len(world.agents))]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = permutation_reward(world, self.perm_oh)
        rew += collision_reward(agent, world)
        return rew

    def observation(self, agent, world):
        entity_pos, other_pos = obs_relative(agent, world)

        # shuffle landmark order
        landmarks_info = list(entity_pos)
        random.Random(self.wland[agent.index]).shuffle(landmarks_info)
        random_entity_pos = list(landmarks_info)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + random_entity_pos + other_pos)
