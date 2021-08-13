from multiagent.environment import MultiAgentEnv


def scenario_environment(scenario_name='simple_spread', benchmark=False):
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnvB(world, scenario.reset_world, scenario.reward, scenario.observation,
        None, scenario.completion)

    return env


class MultiAgentEnvB(MultiAgentEnv):
    def __init__(self, world, reset, reward, observation, benchmark, mybenchmark):
        super().__init__(world, reset, reward, observation, benchmark)
        self.my_benchmark = mybenchmark

    def _get_benchmark(self):
        return self.my_benchmark(self.world)




