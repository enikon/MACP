import tensorflow as tf


class NoiserAL(object):
    def __init__(self, chance, intensity, size):
        self.chance = 1-chance
        self.intensity = intensity
        self.size = size

    def __call__(self, x=None):
        activator = (tf.sign(tf.random.uniform(self.size)-self.chance)+1)/2
        level = tf.random.uniform(self.size, minval=-1, maxval=1) * self.intensity + 1
        return activator * level


class NoiseType:
    def __init__(self, num_agents, num_positions):
        self.num_agents = num_agents
        self.num_positions = num_positions

        self.NoiseType_NONE = 0
        self.NoiseType_RECEIVE = 1 << 0
        self.NoiseType_SEND = 1 << 1

        self.NoiseType_ALL_AGENTS = (1 << 2 + self.num_agents) - (1 << 2)
        self.NoiseType_ALL_POSITIONS = (1 << 2 + self.num_agents + self.num_positions) - (1 << 2 + self.num_agents)

        self.size = 2 + num_agents + num_positions

    def NoiseType_RS(self, i):
        if i >= 2:
            raise Exception("NoiseType_RS "+str(i)+" while only "+str(2)+" defined")
        return 1 << i

    def NoiseType_AGENT(self, i):
        if i >= self.num_agents:
            raise Exception("NoiseType_AGENT "+str(i)+" while only "+str(self.num_agents)+" defined")
        return 1 << 2 + i

    def NoiseType_POSITION(self, i):
        if i >= self.num_agents:
            raise Exception("NoiseType_POSITION "+str(i)+" while only "+str(self.num_positions)+" defined")
        return 1 << 2 + self.num_agents + i


class NoiseManager(object):
    def __init__(self, noises_list, noise_type):
        self.noises_list = noises_list
        self.noise_type = noise_type

        self.noises = [[
                [x for x, t in filter(
                    lambda x:
                    x[1] & noise_type.NoiseType_AGENT(i_a)
                    and x[1] & noise_type.NoiseType_RS(i_rs)
                    , noises_list)
                ]
                for i_rs in range(2)]
            for i_a in range(noise_type.num_agents)]

        h=0

    def getNoiser(self, t):
        return [x for x, t in filter(lambda x: x[1] & t == t, self.noises_list)]

    # INPUT agent idx, position, receive vs send
    # Return appropriate Noiser with OrnUhl
    
#
#
# class NoiseGroup(object):
#     def __init__(self):
#         pass
