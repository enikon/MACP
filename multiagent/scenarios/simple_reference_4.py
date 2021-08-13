from multiagent.scenarios.commons import *
from multiagent.scenarios.simple_reference_3 import Scenario as S


class Scenario(S):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 4
        world.collaborative = True

        world_definition(world, num_agents, num_landmarks)

        self.reset_world(world)
        return world
