
from .. import GridWorldEnv, OmegaRewardMachine, DiscreteProductEnv, ValueIteration
from .test_omega_automata import ltl_dict
from .test_gridworld_envs import gridworld_dict
import numpy as np



def test_value_iteration():


    for name, kwargs in gridworld_dict.items():
        gw = GridWorldEnv(**kwargs)
        gw.rewards[-1, -1] = 1.0  # Set the reward for the goal state
        vi = ValueIteration(gw)
        vi.run()
        gw.plot(values=vi.values, policy=vi.get_greedy_policy()[..., 0])

 
        for oa_type in ['dpa', 'ldba']:
            ltl = ltl_dict[name]['ltl']
            orm = OmegaRewardMachine(rmax=0.0001, ltl=ltl, oa_type=oa_type)
            product_env = DiscreteProductEnv(gw, orm)
            transition_states, transition_probs, rewards = product_env.get_transition_reward_arrays()
            vi = ValueIteration(product_env)
            # vi.run(max_iterations=10_000_000, tolerance=1e-8)
            vi.run_numba(max_iterations=10_000_000, tolerance=1e-10)

            values = vi.values
            policy = vi.get_greedy_policy()[..., 0]
            values_list, policy_list = [], []
            for mode in np.ndindex(orm.shape):
                values_list.append(values[:, :, mode[0]])
                policy_list.append(policy[:, :, mode[0]])

            gw.plot_list(values_list, policy_list)

