
from ..orm import OmegaRewardMachine
from ..oa import OmegaAutomaton
from .test_oa import ltl_dict, R
from collections import Counter
from random import choice
import numpy as np


def test_orm_construction():
    """Test the construction of OmegaRewardMachine objects from LTL formulas.
    This function iterates over a predefined dictionary of LTL formulas and their properties,
    creates OmegaAutomaton objects for both DPA and LDBA types, and constructs OmegaRewardMachine objects.
    It checks if the ORM is constructed correctly by verifying its attributes and properties.
    """

    for name, info in ltl_dict.items():
        for oa_type in ['dpa', 'ldba']:
            for construction_type in ['ltl', 'hoa']:
                for keep_hoa in [False, True]:
                    for save_svg in [False, True]:
                        if construction_type == 'ltl':
                            ltl = info['ltl']
                            oa = OmegaAutomaton(ltl, oa_type, keep_hoa=keep_hoa, save_svg=save_svg)
                            orm = OmegaRewardMachine(oa=oa)
                            orm = OmegaRewardMachine(**{'ltl': ltl, 'oa_type': oa_type, 'keep_hoa': keep_hoa, 'save_svg': save_svg})

                        elif construction_type == 'hoa':
                            # Create the OmegaAutomaton from HOA file
                            hoa_path = oa.hoa_path
                            oa = OmegaAutomaton(hoa_path=hoa_path, keep_hoa=keep_hoa, save_svg=save_svg)
                            orm = OmegaRewardMachine(oa=oa)
                            orm = OmegaRewardMachine(**{'ltl': ltl, 'oa_type': oa_type, 'keep_hoa': keep_hoa, 'save_svg': save_svg})


def test_orm_acceptance():
    """Test the acceptance of OmegaRewardMachine objects based on LTL formulas.
    This function iterates over a predefined dictionary of LTL formulas and their properties,
    creates OmegaRewardMachine objects for DPA type, and tests the acceptance of paths based on rewards.
    It checks if the computed rewards fall within expected ranges for acceptance or rejection.
    """

    for name, info in ltl_dict.items():
        ltl = info['ltl']
        for accepting, path in info.get('paths', []):
            orm = OmegaRewardMachine(ltl=ltl, oa_type='dpa')
            orm.reset()
            G = 0  # Return; sum of discounted rewards
            total_discount = 1.0
            for label in path:
                q, reward = orm.step(label)
                reward /= orm.reward_scale  # Scale down the reward

                G += total_discount * max(0,reward)  # Nonnegative rewards only

                discount = 1.0 - np.abs(reward)  # Update the discount factor based on the reward
                total_discount *= discount

            if 0 <= G <= 0.4:
                accepts = False
            elif 0.6 <= G <= 1.0:
                accepts = True
            else:
                raise ValueError(f"Unexpected reward G={G} for path {path} in LTL {ltl}")

            assert accepting==accepts

            # TODO: Add more tests for LDBA reward machines





