import gymnasium as gym


class DiscreteProductEnv(gym.Env):

    def __init__(self, gw, orm):

        self.gw = gw
        self.orm = orm

        self.observation_space = gym.spaces.MultiDiscrete(gw.shape + orm.shape) 
        self.action_space = gym.spaces.MultiDiscrete((len(gw.action_dirs), orm.max_n_eps_actions))


    def reset(self, seed=None, options=None):
        self.gw.reset(seed=seed, options=options)
        self.orm.reset()

        return self.gw.s + (self.orm.q,), {}
    
    def step(self, action):
        gw_action, orm_eps_action = action

        label = self.gw.labels[self.gw.s]
        next_q, reward = self.orm.step(label, eps_action=orm_eps_action)
        next_s, info = self.gw.step(gw_action)

        return self.gw.s + (self.orm.q,), reward, False, False, info

