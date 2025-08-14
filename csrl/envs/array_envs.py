from abc import ABC, abstractmethod


class ArrayEnv(ABC):

    @abstractmethod
    def __init__(self, *args, discounting=0.99, **kwargs):
        self.discounting = discounting


    @abstractmethod
    def get_transition_reward_arrays(self):
        """Returns the transition states, transition probabilities, and rewards in a vectorized format."""
        pass
