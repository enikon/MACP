from multiagent.scenarios.commons import *
from multiagent.scenarios.simple_reference_3 import Scenario as S


class Scenario(S):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        self.perms_oh = permutation_utils(world)
        self.reset_world(world)
        return world

    def completion(self, world):
        rew = permutation_reward(world, self.perms_oh)
        collisions = 0
        for a in world.agents:
            collisions -= collision_reward(a, world)
        collisions /= 2

        occupied_landmarks = 0
        #EDIT
        for l in world.landmarks:
            for a in world.agents:
                if is_collision(a, l):
                    occupied_landmarks += 1
                    break

        max_dists = -permutation_max_reward(world, self.perms_oh)
        return [rew, collisions, occupied_landmarks, max_dists, 1 if occupied_landmarks == len(world.landmarks) else 0]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = permutation_reward(world, self.perms_oh)
        rew += collision_reward(agent, world)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
