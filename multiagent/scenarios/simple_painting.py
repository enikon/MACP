from multiagent.scenarios.arch.commons import *
from multiagent.scenario import BaseScenario
import math


class StepWorld(World):
    def __init__ (self, step):
        super().__init__()
        self.__step = step

    def step(self):
        super().step()
        self.__step(self)


class Scenario(BaseScenario):
    def make_world(self):
        def step(self):
            for i, agent in enumerate(self.agents):
                for j, landmark in enumerate(self.landmarks):
                    if is_collision(agent, landmark):
                        self.vision[j] = 1.0
                        self.agents[i].vision[j] = 1.0
                        self.landmarks[j].color = self.agents[i].color

        world = StepWorld(step)

        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 49
        self.scenario_dim = math.isqrt(num_landmarks)
        self.scenario_grid_space = 2.0/self.scenario_dim
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.index = i
            agent.collide = True
            agent.silent = True
            agent.size = self.scenario_grid_space/2 * 0.8
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.index = i
            landmark.collide = False
            landmark.movable = False
            #landmark.size = 2.0/self.scenario_dim*0.8
        # make initial conditions

        self.reset_world(world)
        step(world)

        return world

    def benchmark_data(self, agent, world):
        # rew = 0
        # collisions = 0
        # occupied_landmarks = 0
        # min_dists = 0
        # for l in world.landmarks:
        #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        #     min_dists += min(dists)
        #     rew -= min(dists)
        #     if min(dists) < 0.1:
        #         occupied_landmarks += 1
        # if agent.collide:
        #     for a in world.agents:
        #         if is_collision(a, agent):
        #             rew -= 1
        #             collisions += 1
        return (0, 0, 0, 0) #(rew, collisions, min_dists, occupied_landmarks)

    def completion(self, world):
        # rew = sum_reward(world)
        # collisions = 0
        # for a in world.agents:
        #     collisions -= collision_reward(a, world)
        # collisions /= 2
        #
        # occupied_landmarks = 0
        # for a, l in zip(world.agents, world.landmarks):
        #     if is_collision(a, l):
        #         occupied_landmarks += 1
        # max_dists = -max_reward(world)
        return [0,0,0,0]
        #return [rew, collisions, occupied_landmarks, max_dists, 1 if occupied_landmarks == len(world.landmarks) else 0]

    def reset_world(self, world):

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])
            elif i == 1:
                agent.color = np.array([0.35, 0.85, 0.35])
            elif i == 2:
                agent.color = np.array([0.85, 0.35, 0.35])
            agent.vision = np.zeros(len(world.landmarks))-1
        world.vision = np.zeros(len(world.landmarks))-1

        self.lit = []
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            self.lit.append(-1.0)
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        lbord = -1+self.scenario_grid_space/2
        rbord = 1-self.scenario_grid_space/2
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.array([
                lbord + self.scenario_grid_space * (i % self.scenario_dim),
                lbord + self.scenario_grid_space * (i // self.scenario_dim)
            ])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        #rew = sum_reward(world)
        #rew += collision_reward(agent, world)
        return 0.0

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

        other_pos = []
        for other in world.agents:
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [agent.vision] + other_pos)
