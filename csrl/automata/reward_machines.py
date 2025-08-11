from .omega_automata import OmegaAutomaton

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
        
    q0 : int
        The initial state of the ORM.

    delta : list of dicts
        A list representation of the transition function of the ORM.
        For DPAs, `delta[q][label]` is the ORM state that the ORM makes a transition to when the symbol `label` is consumed in the ORM state `q`.
        For LDBAs, `delta[q][label]` is the list of ORM states that the ORM can make a nondeterministic transition to when the symbol `label` is consumed in the ORM state `q`.

    shape : tuple
        The tuple of the number of ORM states.; i.e., : `(n_ms,)`

    rewards : list of dicts
        A list of dictionaries representing the rewards for each state in the ORM.
        For DPAs, `rewards[q][label]` is the reward for consuming the symbol `label` in the ORM state `q`.
        For LDBAs, `rewards[q][label]` is a list of rewards for consuming the symbol `label` in the ORM state `q`, corresponding to each nondeterministic transition.

    

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
    
    def __init__(self, min_discount=0.9, reward_scale=10.0, nonnegative_rewards=False, oa=None, **oa_kwargs):

        self.min_discount = min_discount
        self.reward_scale = reward_scale
        self.nonnegative_rewards = nonnegative_rewards
        self.oa = oa if oa is not None else OmegaAutomaton(**oa_kwargs)
        self.deterministic = self.oa.spot_oa.is_deterministic()
        

        self.aps = self.oa.aps
        self.labels = self.oa.labels
        self.q0 = self.oa.q0
        self.delta = self.oa.delta
        self.shape = (self.oa.shape[1],)
        
        self.rewards = [{} for _ in range(len(self.oa.acc))]
        self.max_n_eps_actions = 1
        for q in range(len(self.oa.acc)):
            for label in self.oa.labels:
                if self.deterministic:
                    self.rewards[q][label] = self.calculate_reward(self.oa.acc[q][label])
                else:
                    self.rewards[q][label] = [self.calculate_reward(color) for color in self.oa.acc[q][label]]
                    self.max_n_eps_actions = max(self.max_n_eps_actions, len(self.rewards[q][label]))

        self.q = self.q0

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
        """
        Calculates the reward for a given color in the ORM.

        Parameters
        ----------
        color: int
            The color for which the reward is to be calculated.

        Returns
        -------
        reward: float
            The reward for the given color.
        """
        rmax = (1 - self.min_discount)  # Maximum unscaled reward
        reward = self.reward_scale * rmax**(self.oa.shape[0]-color)
        if self.nonnegative_rewards:
            reward *= int(color%2==1)
        else:
            reward *= 2 * int(color%2==1) - 1
            
        return reward


    def reset(self):
        """
        Resets the ORM to its initial state.

        Returns
        -------
        q0: int
            The initial state of the ORM after reset.
        """
        self.q = self.q0
        return self.q0


    def step(self, label, eps_action=0):
        """
        Takes a step in the ORM by consuming a label and returns the next state and reward.

        Parameters
        ----------
        label: str
            The label to be consumed in the ORM.

        eps_action: int, optional
            The epsilon action to be taken in the ORM. This is only used if the ORM is nondeterministic. The default value is `0`.

        Returns
        -------
        next_q: int
            The next state of the ORM after consuming the label.
        
        reward: float or list of floats
            The reward associated with consuming the label in the current state.
        """
        if self.deterministic:
            next_q = self.delta[self.q][label]
            reward = self.rewards[self.q][label]
        else:
            next_q = self.delta[self.q][label][eps_action]
            reward = self.rewards[self.q][label][eps_action]
        
        self.q = next_q

        return next_q, reward
    

    def get_n_eps_actions(self, label):
        """
        Returns the number of epsilon actions for a given label in the ORM.

        Parameters
        ----------
        label: str
            The label for which the epsilon actions are to be returned.

        Returns
        -------
        n_eps_actions: list of int
            A list of integers representing the epsilon actions for the given label.
        """
        n_eps_actions = 1 if self.deterministic else len(self.delta[self.q][label])
        return n_eps_actions
        

    def _repr_html_(self):
        """
        Returns the string of the SVG representation of the OA within div tags for visualization in a Jupyter notebook.

        Returns
        -------
        svg: str
            The string of the SVG representation of the ORM within div tags.
        
        """
        return self.svg

