
from .. import GridWorldEnv, OmegaRewardMachine, DiscreteProductEnv
from .test_omega_automata import ltl_dict
from .test_gridworld_envs import gridworld_dict
import numpy as np


def test_product_construction():


    for name, kwargs in gridworld_dict.items():
        gw = GridWorldEnv(**kwargs)
        for oa_type in ['dpa', 'ldba']:
            ltl = ltl_dict[name]['ltl']
            orm = OmegaRewardMachine(ltl=ltl, oa_type=oa_type)
            product_env = DiscreteProductEnv(gw, orm)
            transition_states, transition_probs, rewards = product_env.get_transition_reward_arrays()
