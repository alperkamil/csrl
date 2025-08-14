import gymnasium as gym
import numpy as np
from itertools import product

from .array_envs import ArrayEnv


class DiscreteProductEnv(gym.Env, ArrayEnv):

    def __init__(self, gw, orm, discounting='adaptive', **kwargs):

        self.gw = gw
        self.orm = orm

        super().__init__(discounting=discounting, **kwargs)

        self.observation_space = gym.spaces.MultiDiscrete(gw.shape + orm.shape) 
        self.action_space = gym.spaces.MultiDiscrete((len(gw.action_dirs), orm.max_eps_actions))


    def reset(self, seed=None, options=None):
        initial_gw_state, info = self.gw.reset(seed=seed, options=options)
        initial_orm_mode = self.orm.reset()

        return initial_gw_state + (initial_orm_mode,), info
    
    def step(self, action):
        gw_action, orm_eps_action = action

        label = self.gw.labels[self.gw.s]
        next_gw_state, info = self.gw.step(gw_action)
        next_orm_mode, reward = self.orm.step(label, eps_action=orm_eps_action)

        return next_gw_state + (next_orm_mode,), reward, False, False, info
    

    def get_transition_reward_arrays(self):
        transition_shape = tuple(self.observation_space.nvec) + tuple(self.action_space.nvec) + (self.gw.max_transitions, len(self.observation_space.nvec))
        transition_states = np.zeros(transition_shape, dtype=int)
        transition_probs = np.zeros(transition_shape[:-1], dtype=float)  # Ignore the last dimension, which is for destination product states
        rewards = np.zeros(transition_shape[:-2], dtype=float)  # Ignore the last two dimensions, which are for transition probabilities and destination product states

        for row, col, mode, action, eps_action, dst_id in product(*map(range, transition_probs.shape)):
            gw_cell = (row, col)
            gw_action_dir = self.gw.action_dirs[action]
            dst_gw_states, probs = self.gw.get_transition_probs(gw_cell, gw_action_dir)

            label = self.gw.labels[gw_cell]
            next_orm_modes, orm_rewards = self.orm.get_next_modes_rewards(mode, self.gw.labels[gw_cell])
            next_mode = next_orm_modes[eps_action]
            reward = orm_rewards[eps_action]

            transition_probs[row, col, mode, action, eps_action, dst_id] = probs[dst_id]
            transition_states[row, col, mode, action, eps_action, dst_id] = dst_gw_states[dst_id] + (next_mode,)
            rewards[row, col, mode, action, eps_action] = reward


        return transition_states, transition_probs, rewards



        
