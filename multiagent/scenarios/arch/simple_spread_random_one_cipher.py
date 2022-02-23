import random

from multiagent.scenario import BaseScenario
from multiagent.scenarios.arch.commons import *


class CipherWorld(World):
    def __init__(self, step_handler):
        super().__init__()
        self.step_handler = step_handler

    def step(self):
        super().step()
        self.step_handler(self)


class Scenario(BaseScenario):
    def make_world(self):
        world = CipherWorld(self.update_collision)
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        # make constants
        self.perms = np.fromiter(
            itertools.chain.from_iterable(itertools.permutations(range(num_landmarks), num_agents)), dtype=int).reshape(
            -1, num_agents)
        self.perms_oh = np.eye(num_landmarks)[self.perms]
        self.triangle = np.tril(np.ones((num_landmarks,num_landmarks)))
        self.last_eye = np.eye(num_landmarks) - np.diag(np.ones(num_landmarks-1), 1)

        # make initial conditions
        self.reset_world(world)
        if num_agents > num_landmarks:
            raise Exception('Not enough landmarks for the agents')

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

        self.wagents = [random.randint(*int_range) for _ in range(len(world.agents))]
        self.wland = [random.randint(*int_range) for _ in range(len(world.agents))]

        self.progress = np.zeros(len(world.landmarks))
        self.spot = np.zeros(len(world.landmarks))
        self.msga = []

        self.update_collision(world)

        # TODO ciphering the agents order not landmark order

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
        rew = 0

        nagents = (len(world.agents))

        apos = np.array([a.state.p_pos for a in world.agents])
        lpos = np.array([l.state.p_pos for l in world.landmarks])
        dists = np.sqrt([np.sum(np.square(apos[:, None, :] - lpos[None, :, :]), axis=-1)])
        rew = -np.min(np.sum(dists * self.perms_oh, axis=(-1, -2)))

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return (rew - (len(world.landmarks) - np.sum(self.progress))*5) / nagents

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        collision = 0
        active = 0

        entity_pos = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            if self.is_collision(entity, agent):
                collision = 1
                if i < np.sum(self.progress):
                    active = 1

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

        landmarks_info = list(zip(entity_pos, entity_color))
        random.Random(self.wland[agent.index]).shuffle(landmarks_info)
        random_entity_pos, random_entity_color = list(zip(*landmarks_info))
        random_entity_pos = list(random_entity_pos)
        random_entity_color = list(random_entity_color)

        comm[0][0] = collision
        comm[0][1] = active

        # if collision == 1:
        #     print("\t\t\t"+agent.name+" "+str(collision)+":"+str(active))

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + random_entity_pos + other_pos + comm)

    def update_collision(self, world):
        prev = self.spot
        self.collision_matrix = np.array([
            [1 if self.is_collision(a, l) else 0 for a in world.agents]
            for l in world.landmarks
        ])
        self.spot = 1-np.prod(1-self.collision_matrix, axis=-1)
        df = self.spot - prev
        change = np.sign(df - 1)+1

        self.progress = (self.progress + (
            (
                    (1 - np.abs(np.sign(np.sum(change) - 1)))
                    * np.prod(1 - (self.spot - self.progress - change))
                    * change
            )
        )) * (1 - self.position_mask(1 - self.spot))

        #self.___debug(df)

    def position_mask(self, x):
        return 1-np.prod(1 - self.triangle * x, axis=-1)

    # DEBUG STUFF

    # # NOT IMPLEMENTED DEBUG ONLY
    # # self.___progress_result([0,0,0],[0,0,0],[0,0,0])
    # # self.___progress_result([1,0,0],[1,1,1],[0,0,1])
    # def ___progress_result(self, progress, spot, change):
    #     progress = np.array(progress)
    #     spot = np.array(spot)
    #     change = np.array(change)
    #
    #     # Add only if it is a singular entry \
    #     # and if noone to the right is active
    #
    #     progress = (progress + (
    #         (
    #                 (1 - np.abs(np.sign(np.sum(change) - 1)))
    #                 * np.prod(1 - (spot - progress - change))
    #                 * change
    #         )
    #     )) * (1 - self.position_mask(1 - spot))
    #     return progress
    #
    def ___debug(self, df):
        msg = ""
        chng = False
        for i, d in enumerate(df):
            if d == 1:
                self.msga.append(chr(65 + i) + " ")
                chng = True
            elif d == -1:
                self.msga.append(chr(65 + i) + "\' ")
                chng = True

        for i in self.msga:
            msg += i

        if not msg == "" and chng:
            print(msg, end=' ')
            print("("+str(np.sum(self.progress))+")")