import random
from multiagent.scenarios.commons import *
from multiagent.scenarios.simple_spread_random_one import Scenario as S


class Scenario(S):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 5
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        self.insert_shift = [
            np.insert(np.eye(num_landmarks-1), i, np.zeros((1, num_landmarks-1)), -1)
            for i in range(num_landmarks)]
        self.perm_oh = permutation_utils(world)
        self.index_oh = np.diag(np.arange(num_landmarks))

        auxiliary_assignments = 6
        self.adapter = np.ones(num_agents * (num_landmarks-1))
        self.adapter[:-auxiliary_assignments] = 0

        self.reset_world(world)
        return world

    def reset_world(self, world):
        super().reset_world(world)

        self.assignment_base = self.perm_oh[random.randint(0, len(self.perm_oh)-1)]
        np.random.shuffle(self.adapter)
        auxiliary_matrix = self.adapter.reshape((-1, len(world.landmarks)-1))
        auxiliary_shift = np.take(self.insert_shift, np.argmax(self.assignment_base, -1), axis=0)
        self.assignment = self.assignment_base + np.sum(auxiliary_shift * np.expand_dims(auxiliary_matrix, -1), 1)

        filter = np.array(1-np.sign(np.sum(self.perm_oh * (1-self.assignment), axis=(-1,-2))), dtype=bool)
        self.perm_reduced = self.perm_oh[filter]
        h = 0

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = permutation_reward(world, self.perm_reduced)
        rew += collision_reward(agent, world)
        return rew

    def observation(self, agent, world):

        entity_pos, other_pos = obs_relative(agent, world)

        # TODO better assigning, reintroduce shuffle
        landmarks_info = list(zip(entity_pos, self.assignment))
        #random.Random(self.wland[agent.index]).shuffle(landmarks_info)
        random_entity_pos, random_entity_assignment = list(zip(*landmarks_info))

        random_entity_pos = list(random_entity_pos)
        random_entity_assignment = list(random_entity_assignment)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + random_entity_pos + other_pos + random_entity_assignment)
