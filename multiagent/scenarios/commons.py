import numpy as np
import itertools
from multiagent.core import World, Agent, Landmark

int_range = (-2**31, 2**31-1)


def world_definition(world, num_agents, num_landmarks):
    # add agents
    world.agents = [Agent() for i in range(num_agents)]
    for i, agent in enumerate(world.agents):
        agent.name = 'agent %d' % i
        agent.index = i
        agent.collide = True
        agent.silent = True
        agent.size = 0.15
    # add landmarks
    world.landmarks = [Landmark() for i in range(num_landmarks)]
    for i, landmark in enumerate(world.landmarks):
        landmark.name = 'landmark %d' % i
        landmark.index = i
        landmark.collide = False
        landmark.movable = False
    # make initial conditions


def world_reset(world):
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
        n = True
        while n:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            n = False
            for j in range(i):
                n = n or point_dist(landmark, world.landmarks[j]) < world.agents[0].size


def point_dist(a1, a2):
    delta_pos = a1.state.p_pos - a2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    return dist

def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


def permutation_utils(world):
    num_agents, num_landmarks = len(world.agents), len(world.landmarks)
    perms = np.fromiter(itertools.chain.from_iterable(
        itertools.permutations(range(num_landmarks), num_agents)),
        dtype=int).reshape(-1, num_agents)
    perms_oh = np.eye(num_landmarks)[perms]
    return perms_oh


def permutation_reward(world, _permutation_utils):
    apos = np.array([a.state.p_pos for a in world.agents])
    lpos = np.array([l.state.p_pos for l in world.landmarks])
    dists = np.sqrt([np.sum(np.square(apos[:, None, :] - lpos[None, :, :]), axis=-1)])
    return -np.min(np.sum(dists * _permutation_utils, axis=(-1, -2)))


def permutation_max_reward(world, _permutation_utils):
    apos = np.array([a.state.p_pos for a in world.agents])
    lpos = np.array([l.state.p_pos for l in world.landmarks])
    dists = np.sqrt([np.sum(np.square(apos[:, None, :] - lpos[None, :, :]), axis=-1)])
    index = np.argmin(np.sum(dists * _permutation_utils, axis=(-1, -2)))
    return -np.max((dists * _permutation_utils)[index])


def collision_reward(agent, world):
    rew = 0
    if agent.collide:
        for a in world.agents:
            if is_collision(a, agent):
                rew -= 1
    return rew


def sum_reward(world):
    rew = 0
    for l, a in zip(world.landmarks, world.agents):
        dists = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
        rew -= dists
    return rew


def max_reward(world):
    dists = []
    for l, a in zip(world.landmarks, world.agents):
        dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))))
    return -np.max(dists)


def obs_relative(agent, world, skip_self=False):
    entity_pos = []
    for entity in world.landmarks:  # world.entities:
        entity_pos.append(entity.state.p_pos - agent.state.p_pos)

    other_pos = []
    for other in world.agents:
        if skip_self and other is agent:
            continue
        other_pos.append(other.state.p_pos - agent.state.p_pos)

    return entity_pos, other_pos