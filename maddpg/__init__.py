class AgentTrainer(object):
    def __init__(self, name, model, obs_shape_n, act_space_n, args):
        raise NotImplemented()

    def action(self, *obs):
        raise NotImplemented()

    def process_experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    def update(self, agents, t):
        raise NotImplemented()
