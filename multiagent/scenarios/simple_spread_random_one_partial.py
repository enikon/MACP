import random
import itertools

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from multiagent.scenarios.commons import *


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        self.reset_world(world)
        if num_agents > num_landmarks:
            raise Exception('Not enough landmarks for the agents')

        self.perms = np.fromiter(itertools.chain.from_iterable(itertools.permutations(range(num_landmarks), num_agents)), dtype=int).reshape(-1, num_agents)
        self.perms_oh = np.eye(num_landmarks)[self.perms]

        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        len_agent = len(world.agents)
        len_landmark = len(world.landmarks)
        len_al = (len_agent, len_landmark)

        self.wagents = [random.randint(*int_range) for _ in range(len_agent)]
        self.wland = [random.randint(*int_range) for _ in range(len_landmark)]

        sh_a = np.eye(len_agent)[np.random.choice(*len_al)]
        sh_l = np.transpose(np.eye(len_landmark)[np.random.choice(*len_al[::-1])])
        self.shutter = sh_a + sh_l - sh_a * sh_l

        # TODO shutter for agents ?

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        apos = np.array([a.state.p_pos for a in world.agents])
        lpos = np.array([l.state.p_pos for l in world.landmarks])
        dists = np.sqrt([np.sum(np.square(apos[:, None, :] - lpos[None, :, :]), axis=-1)])
        rew = -np.min(np.sum(dists * self.perms_oh, axis=(-1, -2)))

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        mask = self.shutter[:, [agent.index]]
        landmarks_info = list(zip(entity_pos * mask, entity_color * mask))

        # TODO wagent for coverage of other agents

        random.Random(self.wland[agent.index]).shuffle(landmarks_info)
        random_entity_pos, random_entity_color = list(zip(*landmarks_info))
        random_entity_pos = list(random_entity_pos)
        random_entity_color = list(random_entity_color)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + random_entity_pos + other_pos + comm)
