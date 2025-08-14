
from .. import OmegaRewardMachine
from .. import OmegaAutomaton
from .test_omega_automata import ltl_dict
import numpy as np


def test_orm_construction():
    """Test the construction of OmegaRewardMachine objects from LTL formulas.
    This function iterates over a predefined dictionary of LTL formulas and their properties,
    creates OmegaAutomaton objects for both DPA and LDBA types, and constructs OmegaRewardMachine objects.
    It checks if the ORM is constructed correctly by verifying its attributes and properties.
    """

    for name, info in ltl_dict.items():
        for oa_type in ['dpa', 'ldba']:
            ltl = info['ltl']
            orm = OmegaRewardMachine(ltl=ltl, oa_type=oa_type)
            transition_modes, rewards = orm.get_vectorized_transitions_rewards()

def test_orm_acceptance():
    """Test the acceptance of OmegaRewardMachine objects based on LTL formulas.
    This function iterates over a predefined dictionary of LTL formulas and their properties,
    creates OmegaRewardMachine objects for DPA type, and tests the acceptance of paths based on rewards.
    It checks if the computed rewards fall within expected ranges for acceptance or rejection.
    """

    for name, info in ltl_dict.items():
        ltl = info['ltl']
        for accepting, path in info.get('paths', []):
            for oa_type in ['dpa', 'ldba']:
                orm = OmegaRewardMachine(min_discount=0.99, ltl=ltl, oa_type=oa_type)
                max_G = 0
                for i in range(1 if oa_type=='dpa' else 100):
                    orm.reset()
                    G = 0  # Return; sum of discounted rewards
                    total_discount = 1.0
                    for t, label in enumerate(path):
                        mode, reward = orm.step(label, np.random.randint(0, orm.max_eps_actions))
                        reward /= orm.reward_scale  # Scale down the reward

                        G += total_discount * max(0,reward)  # Nonnegative rewards only
                        # if t < 5:
                            # print(label, reward, orm.mode)

                        discount = 1.0 - np.abs(reward)  # Update the discount factor based on the reward
                        total_discount *= discount
                    max_G = max(max_G, G)

                # print(orm.__dict__)
                print(f"LTL: {ltl}, G: {max_G}, accepting: {accepting}")
                if 0 <= max_G <= 0.2:
                    accepts = False
                elif 0.8 <= max_G <= 1.0:
                    accepts = True
                else:
                    raise ValueError(f"Unexpected reward G={max_G} for path {path} in LTL {ltl}")

                assert accepting==accepts

            # TODO: Add more tests for LDBA reward machines





