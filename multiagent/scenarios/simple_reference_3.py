from multiagent.scenarios.arch.commons import *
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.index = i
            landmark.collide = False
            landmark.movable = False

        self.reset_world(world)
        return world

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
                if is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def completion(self, world):
        rew = sum_reward(world)
        collisions = 0
        for a in world.agents:
            collisions -= collision_reward(a, world)
        collisions /= 2

        occupied_landmarks = 0
        for a, l in zip(world.agents, world.landmarks):
            if is_collision(a, l):
                occupied_landmarks += 1
        max_dists = -max_reward(world)
        return [rew, collisions, occupied_landmarks, max_dists, 1 if occupied_landmarks == len(world.landmarks) else 0]

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])
            elif i == 1:
                agent.color = np.array([0.35, 0.85, 0.35])
            elif i == 2:
                agent.color = np.array([0.85, 0.35, 0.35])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([0.25, 0.25, 0.75])
            elif i == 1:
                landmark.color = np.array([0.25, 0.75, 0.25])
            elif i == 2:
                landmark.color = np.array([0.75, 0.25, 0.25])

        # set random initial states
        for i, agent in enumerate(world.agents):
            n = True
            while n:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
                n = False
                for j in range(i):
                    n = n or point_dist(agent, world.agents[j]) < world.agents[0].size

        for i, landmark in enumerate(world.landmarks):
            n = True
            while n:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                n = False
                for j in range(i):
                    n = n or point_dist(landmark, world.landmarks[j]) < world.agents[0].size

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = sum_reward(world)
        rew += collision_reward(agent, world)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            if world.agents[i] is agent:
                #Pass 0,0
                entity_pos.append(agent.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        for other in world.agents:
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
