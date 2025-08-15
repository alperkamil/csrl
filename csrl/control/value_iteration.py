import numpy as np
from numba import jit, prange

class ValueIteration:

    def __init__(self, array_env):
        
        transtion_states, transition_probs, rewards = array_env.get_transition_reward_arrays()
        self.transition_states = transtion_states
        self.transition_probs = transition_probs
        self.rewards = rewards

        # Store the environment's discount factor
        self.discounting = -1 if array_env.discounting == 'adaptive' else array_env.discounting
        self.n_transitions = self.transition_probs.shape[-1]

        # Construct the value arrays
        self.state_shape = tuple(array_env.observation_space.nvec)
        self.action_shape = tuple(array_env.action_space.nvec)

        self.values = np.zeros(self.state_shape)
        self.next_values = np.zeros_like(self.values)
        self.policy = np.zeros(self.state_shape + (len(self.action_shape),), dtype=int)
        self.tmp_action_values = np.zeros(self.action_shape)

    
    def reset(self):
        """Reset the value iteration process."""
        self.values.fill(0)
        self.next_values.fill(0)
        self.policy.fill(0)


    def compute_action_values(self, state):
        """Compute action values for a given state."""
        for action in np.ndindex(self.action_shape):
            self.tmp_action_values[action] = 0
            for dst in range(self.n_transitions):
                index = state + action + (dst,)
                next_state = tuple(self.transition_states[index])
                prob = self.transition_probs[index]
                reward = self.rewards[state + action]
                if self.discounting < 0:
                    discount = 1.0 - np.abs(reward)
                else:
                    discount = self.discounting
                self.tmp_action_values[action] += prob * (reward + discount * self.values[next_state])
        
        return self.tmp_action_values
            

    def run(self, max_iterations=10_000, tolerance=1e-7):
        for i in range(max_iterations):
            for state in np.ndindex(self.state_shape):
                action_values = self.compute_action_values(state)
                self.next_values[state] = np.max(action_values)
            if np.max(np.abs(self.next_values - self.values)) < tolerance:
                break
            self.values, self.next_values = self.next_values, self.values

        return self.values


    def get_greedy_policy(self):
        for state in np.ndindex(self.state_shape):
            action_values = self.compute_action_values(state)
            best_action = np.argmax(action_values)
            self.policy[state] = best_action

        return self.policy
    
    def run_numba(self, max_iterations=10_000, tolerance=1e-7):
        values = self.values.reshape(-1)
        next_values = self.next_values.reshape(-1)
        tmp_action_values = self.tmp_action_values.reshape(-1)
        transition_probs = self.transition_probs.reshape((values.shape[0], tmp_action_values.shape[0], self.transition_probs.shape[-1]))
        rewards = self.rewards.reshape((values.shape[0], tmp_action_values.shape[0]))

        transition_states = self.transition_states.reshape((-1, self.transition_states.shape[-1])).T
        transition_states = np.ravel_multi_index(transition_states, self.values.shape)
        transition_states = transition_states.reshape((values.shape[0], tmp_action_values.shape[0], self.transition_probs.shape[-1]))

        for i in range(max_iterations):
            iterate_numba(values, next_values, tmp_action_values, self.discounting, transition_states, transition_probs, rewards)
            values, next_values = next_values, values

            err = np.max(np.abs(next_values - values))
            if err < tolerance:
                break

        print(f"Value iteration ended after {i+1} iterations with max error {err:.5e}")


        return values
        

@jit(nopython=True, parallel=True, fastmath=True, cache=False)
def iterate_numba(values, next_values, tmp_action_values, discounting, transition_states, transition_probs, rewards):
    for state in prange(values.shape[0]):
        action_values = np.zeros_like(tmp_action_values)
        for action in prange(tmp_action_values.shape[0]):
            action_values[action] = 0
            for dst in range(transition_probs.shape[-1]):
                next_state = transition_states[state, action, dst]
                prob = transition_probs[state, action, dst]
                reward = rewards[state, action]
                value = values[next_state]
                if discounting < 0:
                    discount = 1.0 - np.abs(reward)
                else:
                    discount = discounting

                action_values[action] += prob * (reward + discount * value)

        next_values[state] = np.max(action_values)
    return next_values
