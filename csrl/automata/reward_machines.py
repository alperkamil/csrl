from .omega_automata import OmegaAutomaton
import numpy as np
from itertools import product

class OmegaRewardMachine:
    """
    Transforms the LTL formula to an omega reward machine (ORM) and stores the specifications.

    Attributes
    ----------
    min_discount : float
        The minimum discount factor for the ORM. This value is used to compute the rewards in the ORM.

    reward_scale : float
        The scale factor for the rewards in the ORM.

    nonnegative_rewards : bool
        A boolean indicating whether the rewards in the ORM are clipped to be nonnegative.
    
    oa : OmegaAutomaton
        The OmegaAutomaton object used to construct the ORM.

    deterministic : bool
        A boolean indicating whether the ORM is deterministic or not.
    
    aps : list of str
        The atomic propositions (APs) of the ORM. These are the basic building blocks of the LTL formula.

    labels : list of str
        The labels of the ORM. A label is a set of APs, and the labels are the power set of the APs.
        
    m0 : int
        The initial mode of the ORM.

    delta : list of dicts
        A list representation of the transition function of the ORM.
        For DPAs, `delta[mode][label]` is the ORM mode that the ORM makes a transition to when the symbol `label` is consumed in given `mode`.
        For LDBAs, `delta[mode][label]` is the list of ORM modes that the ORM can make a nondeterministic transition to when the symbol `label` is consumed in given `mode`.

    shape : tuple
        The tuple of the number of ORM modes.; i.e., : `(n_ms,)`

    rewards : list of dicts
        A list of dictionaries representing the rewards for each mode in the ORM.
        For DPAs, `rewards[mode][label]` is the reward for consuming the symbol `label` in given `mode`.
        For LDBAs, `rewards[mode][label]` is a list of rewards for consuming the symbol `label` in given `mode`, corresponding to each nondeterministic transition.

    

    Parameters
    ----------
    min_discount: float, optional
        The minimum discount factor for the ORM. The default value is `0.9`. This value is used to compute the rewards in the ORM.

    reward_scale: float, optional
        The scale factor for the rewards in the ORM. The default value is `10.0`.

    nonnegative_rewards: bool, optional
        A boolean indicating whether the rewards in the ORM are clipped to be nonnegative. The default value is `False`.

    oa_kwargs: dict, optional
        The keyword arguments to be passed to the `OmegaAutomaton` constructor.

    
    """
    
    def __init__(self, rmax=0.1, oa=None, **oa_kwargs):

        self.rmax = rmax
        self.oa = oa if oa is not None else OmegaAutomaton(**oa_kwargs)
        self.deterministic = self.oa.spot_oa.is_deterministic()
        

        self.aps = self.oa.aps
        self.labels = self.oa.labels
        self.delta = self.oa.delta
        self.shape = (self.oa.shape[1],)  # Number of ORM modes (memory states)

        self.mode0 = self.oa.q0  # Initial mode / memory state
        
        self.rewards = [{} for _ in range(len(self.oa.acc))]
        self.max_eps_actions = 1
        for mode in range(len(self.oa.acc)):  # For all modes / memory states
            for label in self.oa.labels:
                if self.deterministic:
                    self.rewards[mode][label] = self.calculate_reward(self.oa.acc[mode][label])
                else:
                    self.rewards[mode][label] = [self.calculate_reward(color) for color in self.oa.acc[mode][label]]
                    self.max_eps_actions = max(self.max_eps_actions, len(self.rewards[mode][label]))

        self.mode = self.mode0

        try:
            import pydot
            dot = self.oa.spot_oa.to_str(format='dot')
            color_syms = ['⓿', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽', '❾', '❿', '⓫', '⓬', '⓭', '⓮', '⓯', '⓰', '⓱', '⓲', '⓳', '⓴']
            for color, color_sym in enumerate(color_syms):
                reward_str = '%+.1g' % self.calculate_reward(color)
                dot = dot.replace(color_sym, reward_str)
            self.svg = '<div>'+pydot.graph_from_dot_data(dot)[0].create_svg().decode('utf-8')+'</div>'
            if self.oa.shape[0] > 20:
                self.svg += '<div style="color: red; font-weight: bold;">Warning: The ORM has more than 20 colors. The visualization of the rewards may be incorrect.</div>'

        except:
            self.svg = '<div>pydot is not installed. The ORM will cannot be visualized</div>'



    def calculate_reward(self, color):
        """Calculates the reward for a given color in the ORM.

        Parameters
        ----------
        color: int
            The color for which the reward is to be calculated.

        Returns
        -------
        reward: float
            The reward for the given color.
        """
        reward_abs_value = self.rmax**(self.oa.shape[0]-color) 
        reward_sign = 2 * int(color%2==1) - 1
        reward = reward_sign * reward_abs_value
            
        return reward


    def reset(self):
        """Resets the ORM to its initial mode.

        Returns
        -------
        m0: int
            The initial mode of the ORM after reset.
        """
        self.mode = self.mode0
        return self.mode0


    def step(self, label, eps_action=0):
        """Takes a step in the ORM by consuming a label and returns the next mode and reward.

        Parameters
        ----------
        label: tuple
            The label to be consumed in the ORM.

        eps_action: int, optional
            The epsilon action to be taken in the ORM. This is only used if the ORM is nondeterministic. The default value is `0`.

        Returns
        -------
        next_mode: int
            The next mode of the ORM after consuming the label.
        
        reward: float or list of floats
            The reward associated with consuming the label in the current mode.
        """
        next_modes, rewards = self.get_next_modes_rewards(self.mode, label)
        next_mode = next_modes[eps_action]
        reward = rewards[eps_action]

        self.mode = next_mode

        return next_mode, reward
    

    def get_next_modes_rewards(self, mode, label):
        """Returns the next modes and rewards for a given mode and label in the ORM.

        Parameters
        ----------
        mode: int
            The current mode of the ORM.
        
        label: tuple
            The label for which the next modes and rewards are to be returned.

        Returns
        -------
        modes: list of int
            A list of integers representing the next modes for the given mode and label.
        
        rewards: list of floats
            A list of floats representing the rewards for the given mode and label.
        """
      
        if self.deterministic:
            modes = [self.delta[mode][label]]
            rewards = [self.rewards[mode][label]]
        else:
            modes = self.delta[mode][label]
            rewards = self.rewards[mode][label]

        modes_rep, rewards_rep = [], []
        for i in range(self.max_eps_actions - len(rewards)):
            modes_rep.append(modes[i % len(modes)])
            rewards_rep.append(rewards[i % len(rewards)])

        rewards += rewards_rep
        modes += modes_rep

        return modes, rewards

        
    
    def get_eps_actions(self, label):
        """Returns the number of epsilon actions for a given label in the ORM.

        Parameters
        ----------
        label: tuple
            The label for which the epsilon actions are to be returned.

        Returns
        -------
        eps_actions: list of int
            A list of integers representing the epsilon actions for the given label.
        """
        n_eps_actions = 1 if self.deterministic else len(self.delta[self.mode][label])
        return range(n_eps_actions)
        

    def _repr_html_(self):
        """
        Returns the string of the SVG representation of the OA within div tags for visualization in a Jupyter notebook.

        Returns
        -------
        svg: str
            The string of the SVG representation of the ORM within div tags.
        
        """
        return self.svg



    def get_vectorized_transitions_rewards(self):
        """
        Returns the vectorized transitions and rewards for the ORM.

        Returns
        -------
        transition_modes: np.ndarray
            A numpy array of shape `(n_ms, n_labels, max_eps_actions)` representing the next mode for each mode and label in the ORM.
        
        rewards: np.ndarray
            A numpy array of shape `(n_ms, n_labels, max_eps_actions)` representing the rewards for each mode and label in the ORM.
        """

        transition_shape = self.shape + (len(self.labels), self.max_eps_actions)  # Ignore + (len(self.shape),) as it is only one dimensional
        
        transition_modes = np.zeros(transition_shape, dtype=int)
        transition_rewards = np.zeros(transition_shape, dtype=float)

        for mode, label_id, in product(*map(range, transition_shape[:-1])):  # Drop the last dimension as it is implicit in assignments
            label = self.labels[label_id]
            next_modes, rewards  = self.get_next_modes_rewards(mode, label)
            transition_modes[mode, label_id] = next_modes
            transition_rewards[mode, label_id] = rewards

        
        return transition_modes, transition_rewards
