"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
from itertools import product
from .oa import OmegaAutomaton
import os
from multiprocessing import shared_memory, Pool, cpu_count

import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt

if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact

import cProfile, pstats, io
from pstats import SortKey

import random
import pickle
from numba import jit

# TODO: Add documentation for the attributes of this class and the methods
class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.

    Parameters
    ----------
    mdp : mdp.GridMDP
        The MDP that models the environment.

    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.

    discount : float
        The discount factor.

    discountB : float
        The discount factor to be applied to B states.
        
    discountB : float
        The discount factor to be applied to C states.

    """

    def __init__(self, mdp, oa=None, discount=0.99, discountB=0.95, discountC=0.9):
        # TODO: Seperate the initialization code into two parts for MDPs and SGs
        self.mdp = mdp
        self.oa = oa if oa else OmegaAutomaton(' | '.join([ap+' | !'+ap for ap in (mdp.AP+set(mdp.adversary))]))  # Get transitions for every atomic propositions used in the MDP
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.discountC = discountC  # Same
        # TODO: Replace the last element of the shape with (len(mdp.A),)
        self.shape = self.oa.shape + mdp.shape + (len(mdp.A)+self.oa.shape[1],)  # +self.oa.shape[1] is for epsilon transitions in LDBAs

        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=np.object)
        for i,q,r,c in self.states():  # Enrich the action set with epsilon-actions
            self.A[i,q,r,c] = list(range(len(mdp.A))) + [len(mdp.A)+e_a for e_a in self.oa.eps[q]]  # List of actions that can be taken in a particular state

        # Create the reward matrix
        if self.oa.oa_type == 'dpa':
            self.reward = np.zeros(self.shape[:-1])
            for i,q,r,c in self.states():
                self.reward[i,q,r,c] = 1-self.discount if (i>0 and i%2==0) else 0
        else:
            self.reward = np.zeros(self.shape[:-1])
            for i,q,r,c in self.states():
                acc_type = self.oa.acc[q][mdp.label[r,c]][i]
                self.reward[i,q,r,c] = 1-self.discountB if acc_type else (-1e-20 if acc_type is False else 0)  # The small negative reward is for identification of C states
            
        # Create the transition matrix
        if mdp.robust:  # This is for the scenario where the transition probabilities are determined by the joint actions of both players
            if self.oa.oa_type == 'dpa':
                self.transition_probs = np.empty(self.shape+(len(self.mdp.A),),dtype=np.object)  
                for i,q,r,c in self.states():
                    for action in range(len(self.mdp.A)):
                        for action_ in range(len(self.mdp.A)):
                            q_ = self.oa.delta[q][mdp.label[r,c]]  # OA transition
                            i_ = i
                            if i>0 and self.oa.acc[q][mdp.label[r,c]] > i:
                                i_= self.oa.acc[q][mdp.label[r,c]] # Priority Transition
                                
                            mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action],mdp.A[action_])  # MDP transition
                            self.transition_probs[i,q,r,c][action][action_] = [(i_,q_,)+s for s in mdp_states], probs
            else:
                self.transition_probs = np.empty(self.shape+(len(self.mdp.A),),dtype=np.object)  
                for i,q,r,c in self.states():
                    for action in range(len(self.mdp.A)):
                        for action_ in range(len(self.mdp.A)):
                            q_ = self.oa.delta[q][mdp.label[r,c]]  # OA transition
                            mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action],mdp.A[action_])  # MDP transition
                            self.transition_probs[i,q,r,c][action][action_] = [(i,q_,)+s for s in mdp_states], probs
                        
        
        elif mdp.secure:
            self.transition_probs = np.empty(self.oa.shape+mdp.shape+(len(self.mdp.A),len(self.mdp.A)),dtype=np.object)
            for i,q,r,c in self.states():
                for action in range(len(self.mdp.A)):
                    for action_ in range(len(self.mdp.A)):
                        mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action])
                        expected_state = mdp_states[np.argmax(probs)]
                        mdp_states_, probs_ = mdp.get_transition_prob((r,c),mdp.A[action_])
                        
                        states = []
                        label = mdp.label[r,c] + ('m',) if action != action_ else mdp.label[r,c]
                        for s in mdp_states_:
                            l = label + ('u',) if s != expected_state else label
                            q_ = self.oa.delta[q][l]
                            states.append((i,q_)+s)
                        self.transition_probs[i,q,r,c][action][action_] = states, probs_
            
        elif mdp.adversary is not None:  # This is for two independent players on the grid
            self.transition_probs = np.empty(mdp.shape+(len(self.mdp.A),),dtype=np.object)  # Standard transition probs
            for r,c in self.states(short=True):
                for action in range(len(self.mdp.A)):
                    self.transition_probs[r,c][action] = mdp.get_transition_prob((r,c),mdp.A[action])
        else:
            self.transition_probs = np.empty(self.shape,dtype=np.object)
            for i,q,r,c in self.states():
                for action in self.A[i,q,r,c]: 
                    if action < len(self.mdp.A):
                        q_ = self.oa.delta[q][mdp.label[r,c]]  # OA transition
                        mdp_states, probs = mdp.get_transition_prob((r,c),mdp.A[action])  # MDP transition
                        self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in mdp_states], probs
                    else:  # epsilon-actions
                        self.transition_probs[i,q,r,c][action] = ([(i,action-len(mdp.A),r,c)], [1.])
        self.k = 1

    def states(self,second=None,short=None):
        """State generator.

        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_pairs, n_qs, n_rows, n_cols, n_actions = self.shape
        if second:
            for i,q,r1,c1,r2,c2 in product(range(n_pairs),range(n_qs),range(n_rows),range(n_cols),range(n_rows),range(n_cols)):
                yield i,q,r1,c1,r2,c2
        elif short:
            for r,c in product(range(n_rows),range(n_cols)):
                yield r,c
        else:
            for i,q,r,c in product(range(n_pairs),range(n_qs),range(n_rows),range(n_cols)):
                yield i,q,r,c


    def random_state(self):
        """Generates a random state coordinate.

        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_pairs, n_qs, n_rows, n_cols, n_actions = self.shape
        mdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_pairs),np.random.randint(n_qs)) + mdp_state

    def q_learning(self,start=None,T=None,K=None):
        """Performs the Q-learning algorithm and returns the action values.

        Parameters
        ----------
        start : int
            The start state of the MDP.

        T : int
            The episode length.

        K : int
            The number of episodes.

        Returns
        -------
        Q: array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions) 
            The action values learned.
        """

        T = T if T else np.prod(self.shape[:-1])
        K = K if K else 100000

        Q = np.zeros(self.shape)

        for k in range(K):
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            alpha = np.max((1.0*(1 - 1.5*k/K),0.001))
            epsilon = np.max((1.0*(1 - 1.5*k/K),0.01))
            for t in range(T):

                reward = self.reward[state]
                gamma = self.discountB if reward else self.discount

                # Follow an epsilon-greedy policy
                if np.random.rand() < epsilon or np.max(Q[state])==0:
                    action = np.random.choice(self.A[state])  # Choose among the MDP and epsilon actions
                else:
                    action = np.argmax(Q[state])

                # Observe the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]

                # Q-update
                Q[state][action] += alpha * (reward + gamma*np.max(Q[next_state]) - Q[state][action])

                state = next_state

        return Q

    def greedy_policy(self, value):
        """Returns a greedy policy for the given value function.

        Parameters
        ----------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.

        Returns
        -------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.

        """
        policy = np.zeros((value.shape),dtype=np.int)
        for state in self.states():
            action_values = np.empty(len(self.A[state]))
            for i,action in enumerate(self.A[state]):
                action_values[i] = np.sum([value[s]*p for s,p in zip(*self.transition_probs[state][action])])
            policy[state] = self.A[state][np.argmax(action_values)]
        return policy

    def value_iteration(self, T=None, threshold=None):
        """Performs the value iteration algorithm and returns the value function. It requires at least one parameter.

        Parameters
        ----------
        T : int
            The number of iterations.

        threshold: float
            The threshold value to be used in the stopping condition.

        Returns
        -------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        """
        value = np.zeros(self.shape[:-1])
        old_value = np.copy(value)
        t = 0  # The time step
        d = np.inf  # The difference between the last two steps
        while (T and t<T) or (threshold and d>threshold):
            value, old_value = old_value, value
            for state in self.states():
                # Bellman operator
                action_values = np.empty(len(self.A[state]))
                for i,action in enumerate(self.A[state]):
                    action_values[i] = np.sum([old_value[s]*p for s,p in zip(*self.transition_probs[state][action])])
                gamma = self.discountB if self.reward[state]>0 else self.discount
                value[state] = self.reward[state] + gamma*np.max(action_values)
            t += 1
            d = np.nanmax(np.abs(old_value-value))

        return value

    def simulate(self, policy, policy_=None, value=None, start=None, start_=None, T=None, plot=True, animation=None):
        """Simulates the environment and returns a trajectory obtained under the given policy.

        Parameters
        ----------
        policy : array, size=(n_pairs,n_qs,n_rows,n_cols)
            The policy.

        start : int
            The start state of the MDP.

        T : int
            The episode length.

        plot : bool
            Plots the simulation if it is True.

        Returns
        -------
        episode: list
            A sequence of states
        """
        
        # TODO: Update the simulation code to handle two-player scenarios
        T = T if T else np.prod(self.shape[:-1])
        if policy_ is None:
            state = (self.shape[0]-1,self.oa.q0)+(start if start else self.mdp.random_state())
            episode = [state]
            for t in range(T):
                states, probs = self.transition_probs[state][policy[state]]
                state = states[np.random.choice(len(states),p=probs)]
                episode.append(state)

            if plot:
                def plot_agent(t):
                    val = value[episode[t][:2]] if value is not None else None
                    self.mdp.plot(value=val,policy=policy[episode[t][:2]],agent=episode[t][2:])
                t=IntSlider(value=0,min=0,max=T-1)
                interact(plot_agent,t=t)

            if animation:
                pad=5
                if not os.path.exists(animation):
                    os.makedirs(animation)
                for t in range(T):
                    self.mdp.plot(value=value[episode[t][:2]],policy=policy[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  LDBA State (Mode): '+str(episode[t][1]))
                    plt.close()
                os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')

            return episode

        else:
            if self.mdp.adversary:
                i,q = (0,self.oa.q0)
                s1 = start if start else self.mdp.random_state()
                s2 = start_ if start_ else self.mdp.random_state()

                label = self.mdp.label[s1]
                if s1 == s2:
                    label = self.mdp.adversary + label
                q = self.oa.delta[q][label]  # OA transition

                episode = [(i,q)+s1+s2]
                for t in range(T):
                    
                    a1 = policy[(i,q)+s1+s2]
                    if a1>= len(self.mdp.A):
                        i=a1-len(self.mdp.A)
                        a1 = policy[(i,q)+s1+s2]
                    states, probs = self.transition_probs[s1][a1]
                    s1 = random.choices(states,weights=probs)[0]
                    
                    a2 = policy_[(i,q)+s1+s2]
                    states, probs = self.transition_probs[s2][a2]
                    s2 = random.choices(states,weights=probs)[0]

                    label = self.mdp.label[s1]
                    if s1 == s2:
                        label = self.mdp.adversary + label
                    q = self.oa.delta[q][label]  # OA transition

                    episode.append((i,q)+s1+s2)

                if plot:
                    def plot_agent(t):
                        i,q,r1,c1,r2,c2 = episode[t]
                        val = value[i,q,:,:,r2,c2] if value is not None else None
                        self.mdp.plot(value=val,policy=policy[i,q,:,:,r2,c2],agent=(r1,c1),agent_=(r2,c2))
                    t=IntSlider(value=0,min=0,max=T-1)
                    interact(plot_agent,t=t)

                if animation:
                    pad=5
                    if not os.path.exists(animation):
                        os.makedirs(animation)
                    for t in range(T):
                        i,q,r1,c1,r2,c2 = episode[t]
                        self.mdp.plot(value=value[i,q,:,:,r2,c2],policy=policy[i,q,:,:,r2,c2],agent=(r1,c1),agent_=(r2,c2),save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  Accepting Pair: '+str(i+1)+',  DRA State (Mode): '+str(q))
                        plt.close()
                    os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')
                return episode
            
            
            if self.mdp.robust:
                
                if self.oa.oa_type == 'dpa':
                    e = self.k*(1-self.discount)
                    e_ = np.sqrt(e)

                    state = (0,self.oa.q0)+(start if start else self.mdp.random_state())
                    episode = [state]
                    tau = np.random.geometric(e_)
                    for t in range(T):
                        a1 = policy[state]
                        a2 = policy_[state]

                        states, probs = self.transition_probs[state][a1][a2]
                        next_state = states[np.random.choice(len(states),p=probs)]

                        if state[0]<next_state[0] and state[0]==1:
                            tau = np.random.geometric(e)
                        elif tau==0 and state[0]!=1:
                            next_state = (1,)+next_state[1:]
                        tau -= 1
                        state = next_state

                        episode.append(state)

                    if plot:
                        def plot_agent(t):
                            val = value[episode[t][:2]] if value is not None else None
                            self.mdp.plot(value=val,policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:])
                        t=IntSlider(value=0,min=0,max=T-1)
                        interact(plot_agent,t=t)

                    if animation:
                        pad=5
                        if not os.path.exists(animation):
                            os.makedirs(animation)
                        for t in range(T):
                            self.mdp.plot(value=value[episode[t][:2]],policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  Accepting Pair: '+str(episode[t][0]+1)+',  DRA State (Mode): '+str(episode[t][1]))
                            plt.close()
                        os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')

                    return episode
                
                else:
                    state = (0,self.oa.q0)+(start if start else self.mdp.random_state())
                    episode = [state]
                    for t in range(T):
                        i,q,r,c = state
                        a1 = policy[state]
                        if a1 >= len(self.mdp.A):
                            i=a1-len(self.mdp.A)
                            state = i,q,r,c
                            a1 = policy[state]
                        a2 = policy_[state]

                        states, probs = self.transition_probs[state][a1][a2]
                        state = states[np.random.choice(len(states),p=probs)]
                        episode.append((i,q,r,c))

                    if plot:
                        def plot_agent(t):
                            val = value[episode[t][:2]] if value is not None else None
                            self.mdp.plot(value=val,policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:])
                        t=IntSlider(value=0,min=0,max=T-1)
                        interact(plot_agent,t=t)

                    if animation:
                        pad=5
                        if not os.path.exists(animation):
                            os.makedirs(animation)
                        for t in range(T):
                            self.mdp.plot(value=value[episode[t][:2]],policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  Accepting Pair: '+str(episode[t][0]+1)+',  DRA State (Mode): '+str(episode[t][1]))
                            plt.close()
                        os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')

                    return episode
            
            if self.mdp.secure:
                state = (0,self.oa.q0)+(start if start else self.mdp.random_state())
                episode = [state]
                for t in range(T):
                    i,q,r,c = state
                    a1 = policy[state]
                    if a1 >= len(self.mdp.A):
                        i=a1-len(self.mdp.A)
                        state = i,q,r,c
                        a1 = policy[state]
                    a2 = policy_[state]
                    
                    states, probs = self.transition_probs[state][a1][a2]
                    state = states[np.random.choice(len(states),p=probs)]
                    episode.append((i,q,r,c))
                    
                if plot:
                    def plot_agent(t):
                        val = value[episode[t][:2]] if value is not None else None
                        self.mdp.plot(value=val,policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:])
                    t=IntSlider(value=0,min=0,max=T-1)
                    interact(plot_agent,t=t)

                if animation:
                    pad=5
                    if not os.path.exists(animation):
                        os.makedirs(animation)
                    for t in range(T):
                        self.mdp.plot(value=value[episode[t][:2]],policy=policy[episode[t][:2]],policy_=policy_[episode[t][:2]],agent=episode[t][2:],save=animation+os.sep+str(t).zfill(pad)+'.png',title='Time: '+str(t)+',  Accepting Pair: '+str(episode[t][0]+1)+',  DRA State (Mode): '+str(episode[t][1]))
                        plt.close()
                    os.system('ffmpeg -r 3 -i '+animation+os.sep+'%0'+str(pad)+'d.png -vcodec libx264 -y '+animation+'.mp4')

                return episode

    def plot(self, value=None, policy=None, policy_=None, iq=None, rc=None, rc_=None, **kwargs):
        """Plots the values of the states as a color matrix with two sliders.

        Parameters
        ----------
        value : array, shape=(n_pairs,n_qs,n_rows,n_cols)
            The value function.

        policy : array, shape=(n_pairs,n_qs,n_rows,n_cols)
            The policy to be visualized. It is optional.

        save : str
            The name of the file the image will be saved to. It is optional
        """
        if self.mdp.adversary is not None:
            if iq:
                if rc:
                    val = value[iq][rc] if value is not None else None
                    pol = policy_[iq][rc] if policy is not None else None
                    self.mdp.plot(val,pol,**kwargs)
                elif rc_:
                    val = value[iq][:,:,rc_[0],rc_[1]] if value is not None else None
                    pol = policy[iq][:,:,rc_[0],rc_[1]] if policy is not None else None
                    self.mdp.plot(val,pol,**kwargs)
                else:
                    # A helper function for the sliders
                    def plot_value(r1,c1,r2,c2):
                        val = value[iq][:,:,r2,c2] if value is not None else None
                        pol = policy[iq][:,:,r2,c2] if policy is not None else None
                        pol_ = policy_[iq][r1,c1,:,:] if policy_ is not None else None
                        self.mdp.plot(val,pol,pol_,**kwargs)
                    r1 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                    c1 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                    r2 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                    c2 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                    interact(plot_value,r1=r1,c1=c1,r2=r2,c2=c2)
            else:
                # A helper function for the sliders
                def plot_value(i,q,r1,c1,r2,c2):
                    val = value[i,q,:,:,r2,c2] if value is not None else None
                    pol = policy[i,q,:,:,r2,c2] if policy is not None else None
                    pol_ = policy_[i,q,r1,c1,:,:] if policy_ is not None else None
                    self.mdp.plot(val,pol,pol_,**kwargs)
                i = IntSlider(value=0,min=0,max=self.shape[0]-1)
                q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
                r1 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                c1 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                r2 = IntSlider(value=0,min=0,max=self.mdp.shape[0]-1)
                c2 = IntSlider(value=0,min=0,max=self.mdp.shape[1]-1)
                interact(plot_value,i=i,q=q,r1=r1,c1=c1,r2=r2,c2=c2)
        elif iq:
            val = value[iq] if value is not None else None
            pol = policy[iq] if policy is not None else None
            pol_ = policy_[iq] if policy_ is not None else None
            self.mdp.plot(val,pol,pol_,**kwargs)
        else:
            # A helper function for the sliders
            def plot_value(i,q):
                val = value[i,q] if value is not None else None
                pol = policy[i,q] if policy is not None else None
                pol_ = policy_[i,q] if policy_ is not None else None
                self.mdp.plot(val,pol,pol_,**kwargs)
            i = IntSlider(value=0,min=0,max=self.shape[0]-1)
            q = IntSlider(value=self.oa.q0,min=0,max=self.shape[1]-1)
            interact(plot_value,i=i,q=q)

    def shapley(self, T=None, name='', tt=2**15):
        """Performs the Shapley's algorithm and returns the value function. It requires at least one parameter.

        Parameters
        ----------
        T : int
            The number of iterations.

        Returns
        -------
        value: array, size=(n_pairs,n_qs,n_rows,n_cols)
            The value function.
        """
        n_actions = len(self.mdp.A)
        n_pairs = self.oa.shape[0]
            
        if self.mdp.adversary:
            suffix = str((T-1).bit_length())
            with open('shapley_adversary_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                pickle.dump(self,f)

            states = list(self.states(second=True))
            shape = self.oa.shape+self.mdp.shape+self.mdp.shape+(n_actions,)
            Q = np.zeros(shape)
            Q_ = np.zeros(shape)
            
            t = 0
            while t < T:
                value = np.max(Q,axis=-1)
                
                for state in states:
                    i,q,r1,c1,r2,c2 = state
                    s1 = r1,c1
                    s2 = r2,c2
                    for k in range(n_actions):
                        val=0
                        for s,p in zip(*self.transition_probs[s2][k]):
                            val += value[i,q][s1][s]*p
                        Q_[state][k] = val
                
                value_ = np.min(Q_,axis=-1)
                
                for state in states:
                    i,q,r1,c1,r2,c2 = state
                    s1 = r1,c1
                    s2 = r2,c2
                    
                    label = self.mdp.label[s1]
                    if s1 == s2:
                        label = self.mdp.adversary + label
                    q_ = self.oa.delta[q][label]  # OA transition
                    
                    acc_type = self.oa.acc[q][label][i]
                    reward = 0
                    gamma = self.discount
                    if acc_type is True:
                        reward = 1-self.discountB
                        gamma = self.discountB
                    elif acc_type is False:
                        gamma = self.discountC
                    
                    for j in range(n_actions):
                        val=0
                        for s,p in zip(*self.transition_probs[s1][j]):
                            val += value_[i,q_][s][s2]*p
                        Q[state][j] = reward + gamma*val
                
                eps_Q = self.discountC*np.max(Q,axis=0)
                for i in range(n_pairs):
                    Q[i] = np.maximum(Q[i],eps_Q)
                
                t+=1
                
                if t>=tt-1 and t.bit_length() != (t+1).bit_length():
                    suffix = str(t.bit_length())
                    np.save('shapley_adversary_'+name+'_Q-Q_-'+suffix,(Q,Q_))
                
            return Q,Q_

        
        elif self.mdp.robust:
            if self.oa.oa_type == 'dpa':
                return shapley_q_dpa(self,T,name,tt)
                
            else:
                suffix = str((T-1).bit_length())
                with open('shapley_robust_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                    pickle.dump(self,f)

                states = list(self.states(second=False))
                shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
                Q = np.zeros(shape)
                discount = np.copy(self.reward)
                for state in states:
                    discount[state] = self.discountB if self.reward[state]>0 else (self.discountC if self.reward[state]<0 else self.discount)

                t = 0
                while t < T:
                    value = np.max(np.min(Q,axis=-1),axis=-1)
                    for state in states:
                        gamma, reward, probs = discount[state], self.reward[state], self.transition_probs[state]
                        for j in range(n_actions):
                            for k in range(n_actions):
                                val = 0
                                for s,p in zip(*probs[j][k]):
                                    val += value[s]*p
                                Q[state][j][k] = reward + gamma*val

                    eps_Q = self.discountC*np.max(Q,axis=0)
                    for i in range(n_pairs):
                        Q[i] = np.maximum(Q[i],eps_Q)

                    t+=1

                    if t>=tt-1 and t.bit_length() != (t+1).bit_length():
                        suffix = str(t.bit_length())
                        np.save('shapley_robust_'+name+'_Q-'+suffix,Q)

                return Q
        
        
        elif self.mdp.secure:
            suffix = str((T-1).bit_length())
            with open('shapley_secure_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                pickle.dump(self,f)
                
            states = list(self.states(second=False))
            shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
            Q = np.zeros(shape)
            discount = np.copy(self.reward)
            for state in states:
                discount[state] = self.discountB if self.reward[state]>0 else (self.discountC if self.reward[state]<0 else self.discount)
        
            t = 0
            while t < T:
                value = np.max(np.min(Q,axis=-1),axis=-1)
                for state in states:
                    gamma, reward, probs = discount[state], self.reward[state], self.transition_probs[state]
                    for j in range(n_actions):
                        for k in range(n_actions):
                            val = 0
                            for s,p in zip(*probs[j][k]):
                                val += value[s]*p
                            Q[state][j][k] = reward + gamma*val
                            
                eps_Q = self.discountC*np.max(Q,axis=0)
                for i in range(n_pairs):
                    Q[i] = np.maximum(Q[i],eps_Q)
                
                t+=1
                
                if t>=tt-1 and t.bit_length() != (t+1).bit_length():
                    suffix = str(t.bit_length())
                    np.save('shapley_secure_'+name+'_Q-'+suffix,Q)
            
            return Q
    
    def minimax_q(self,start=None,start_=None,T=None,K=None,name='',tt=2**10):
        n_actions = len(self.mdp.A)
        n_pairs = self.oa.shape[0]
        init = list(zip(*np.where(self.mdp.structure == 'E')))
        not_init = set(zip(*np.where(self.mdp.structure != 'E')))
        dC, dB, d = self.discountC, self.discountB, self.discount
        
        if self.mdp.adversary:
            suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())
            with open('minimax_q_adversary_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                pickle.dump(self,f)
            
            shape = self.oa.shape + self.mdp.shape + self.mdp.shape + (n_actions,)
            Q,Q_ = np.zeros(shape),np.zeros(shape)
                
            for _ in range(K):
                epsilon = max(0.5-_/tt,0.05)
                alpha = max(0.5-_/tt,0.05)
                i = 0 if start else random.randrange(self.oa.shape[0])
                q = self.oa.q0 if start else random.randrange(self.oa.shape[1])

                s1, s2 = random.sample(init,k=2)

                label = self.mdp.label[s1]
                if s1 == s2:
                    label = self.mdp.adversary + label
                acc_type = self.oa.acc[q][label][i]
                q = self.oa.delta[q][label]  # OA transition
                reward = 0
                gamma = d
                if acc_type is True:
                    reward = 1-dB
                    gamma = dB
                elif acc_type is False:
                    gamma = dC
                max_action = random.randrange(n_actions)
                max_q = 0
                next_i = 0
                for t in range(T):
                    
                    if max_q==0 or random.random() < epsilon:
                        max_action = random.randrange(n_actions)
                        i = random.randrange(n_pairs)
                        max_q = Q[i,q][s1][s2][max_action]

                    states, probs = self.transition_probs[s1][max_action]
                    next_s1 = random.choices(states,weights=probs)[0]
                   
                    min_q, min_action, max_q_ = 1, 0, 0
                    for action_ in range(n_actions):
                        max_q_ = max(Q_[i,q][next_s1][s2][action_],max_q_)
                        if Q_[i,q][next_s1][s2][action_] < min_q:
                            min_action = action_
                            min_q = Q_[i,q][next_s1][s2][action_] 

                    Q[i,q][s1][s2][max_action] = max_q + alpha * (reward + gamma*min_q - max_q)

                    if max_q_==0 or random.random() < epsilon:
                        min_action = random.randrange(n_actions)
                        min_q = Q_[i,q][next_s1][s2][min_action]

                    states, probs = self.transition_probs[s2][min_action]
                    next_s2 = random.choices(states,weights=probs)[0]

                    label = self.mdp.label[next_s1]
                    if next_s1 == next_s2:
                        label = self.mdp.adversary + label
                    next_q = self.oa.delta[q][label]  # OA transition

                    acc_type = self.oa.acc[q][label][next_i]
                    reward = 0
                    gamma = d
                    if acc_type is True:
                        reward = 1-dB
                        gamma = dB
                    elif acc_type is False:
                        gamma = dC
                    
                    max_q, max_action = 0, 0
                    next_i = i
                    if _>tt:
                        for i_ in range(n_pairs):
                            g = 1 if i==i_ else min(2*_/K,dC)   
                            for action in range(n_actions):
                                if g*Q[next_i,next_q][next_s1][next_s2][action] > max_q:
                                    next_i = i_
                                    max_action = action
                                    max_q = g*Q[next_i,next_q][next_s1][next_s2][action]
                    else:
                        for action in range(n_actions):
                            if Q[i,next_q][next_s1][next_s2][action] > max_q:
                                max_action = action
                                max_q = Q[i,next_q][next_s1][next_s2][action]



                    Q_[i,q][next_s1][s2][min_action] = min_q + alpha * (max_q - min_q)

                    q,s1,s2,i = next_q, next_s1, next_s2, next_i
                    
                    if s1 in not_init:
                        break
                        
                if _>=tt-1 and _.bit_length() != (_+1).bit_length():
                    suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())+'_'+str(_.bit_length())
                    np.save('minimax_q_adversary_'+name+'_Q-Q_-'+suffix,(Q,Q_))

            return Q, Q_
        
        
        elif self.mdp.robust:
            if self.oa.oa_type == 'dpa':
                return minimax_q_dpa(self,start,T,K,name,tt)
            else:
                suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())
                with open('minimax_q_robust_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                    pickle.dump(self,f)

                shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
                Q = np.zeros(shape)

                discount = np.copy(self.reward)
                for state in self.states(second=False):
                    discount[state] = dB if self.reward[state]>0 else (dC if self.reward[state]<0 else d)

                for _ in range(K):
                    epsilon = max(0.5-_/tt,0.05)
                    alpha = max(0.5-_/tt,0.05)

                    i = 0 if start else random.randrange(n_pairs)
                    q = self.oa.q0 if start else random.randrange(self.oa.shape[1]-1)
                    state = (i,q)+(start if start else random.choice(init))

                    next_i = 0
                    max_action, min_action, max_q = 0, 0, 0
                    for t in range(T):
                        # Follow an epsilon-greedy policy
                        if max_q==0 or random.random() < epsilon:
                            max_action = random.randrange(n_actions)
                            min_action = random.randrange(n_actions)
                            state = (random.randrange(n_pairs),) + state[1:]
                            max_q = Q[state][max_action][min_action]

                        # Observe the next state
                        states, probs = self.transition_probs[state][max_action][min_action]
                        next_state = random.choices(states,weights=probs)[0]

                        next_max_action, next_min_action, next_max_q = 0, 0, 0
                        next_i = state[0]
                        if _>tt:
                            for i in range(n_pairs):
                                g = 1 if next_state[0]==i else min(2*_/K,dC)
                                s = (i,) + next_state[1:]
                                for j in range(n_actions):
                                    action_, min_q = 0, 1
                                    for k in range(n_actions):
                                        if g*Q[s][j][k] < min_q:
                                            action_ = k
                                            min_q = g*Q[s][j][k]
                                    if min_q > next_max_q:
                                        next_i = i
                                        next_max_action = j
                                        next_min_action = action_
                                        next_max_q = min_q
                        else:
                            for j in range(n_actions):
                                action_, min_q = 0, 1
                                for k in range(n_actions):
                                    if Q[next_state][j][k] < min_q:
                                        action_ = k
                                        min_q = Q[next_state][j][k]
                                if min_q > next_max_q:
                                    next_max_action = j
                                    next_min_action = action_
                                    next_max_q = min_q

                        reward = self.reward[state]
                        gamma = discount[state]

                        # Q-update
                        Q[state][max_action][min_action] = max_q + alpha * (reward + gamma*next_max_q - max_q)

                        state, max_action, min_action, max_q = (next_i,)+next_state[1:], next_max_action, next_min_action, next_max_q


                    if _>=tt-1 and _.bit_length() != (_+1).bit_length():
                        suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())+'_'+str(_.bit_length())
                        np.save('minimax_q_robust_'+name+'_Q-'+suffix,Q)

                return Q
        
        elif self.mdp.secure:
            suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())
            with open('minimax_q_secure_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
                pickle.dump(self,f)
            
            shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
            Q = np.zeros(shape)
            
            discount = np.copy(self.reward)
            for state in self.states(second=False):
                discount[state] = dB if self.reward[state]>0 else (dC if self.reward[state]<0 else d)
                
            for _ in range(K):
                epsilon = max(0.5-_/tt,0.05)
                alpha = max(0.5-_/tt,0.05)
                
                i = 0 if start else random.randrange(n_pairs)
                q = self.oa.q0 if start else random.randrange(self.oa.shape[1]-1)
                state = (i,q)+(start if start else random.choice(init))
                
                next_i = 0
                max_action, min_action, max_q = 0, 0, 0
                for t in range(T):
                    # Follow an epsilon-greedy policy
                    if max_q==0 or random.random() < epsilon:
                        max_action = random.randrange(n_actions)
                        min_action = random.randrange(n_actions)
                        state = (random.randrange(n_pairs),) + state[1:]
                        max_q = Q[state][max_action][min_action]
                
                    # Observe the next state
                    states, probs = self.transition_probs[state][max_action][min_action]
                    next_state = random.choices(states,weights=probs)[0]

                    next_max_action, next_min_action, next_max_q = 0, 0, 0
                    next_i = state[0]
                    if _>tt:
                        for i in range(n_pairs):
                            g = 1 if next_state[0]==i else min(2*_/K,dC)
                            s = (i,) + next_state[1:]
                            for j in range(n_actions):
                                action_, min_q = 0, 1
                                for k in range(n_actions):
                                    if g*Q[s][j][k] < min_q:
                                        action_ = k
                                        min_q = g*Q[s][j][k]
                                if min_q > next_max_q:
                                    next_i = i
                                    next_max_action = j
                                    next_min_action = action_
                                    next_max_q = min_q
                    else:
                        for j in range(n_actions):
                            action_, min_q = 0, 1
                            for k in range(n_actions):
                                if Q[next_state][j][k] < min_q:
                                    action_ = k
                                    min_q = Q[next_state][j][k]
                            if min_q > next_max_q:
                                next_max_action = j
                                next_min_action = action_
                                next_max_q = min_q

                    reward = self.reward[state]
                    gamma = discount[state]
                    
                    # Q-update
                    Q[state][max_action][min_action] = max_q + alpha * (reward + gamma*next_max_q - max_q)

                    state, max_action, min_action, max_q = (next_i,)+next_state[1:], next_max_action, next_min_action, next_max_q
                    
                        
                if _>=tt-1 and _.bit_length() != (_+1).bit_length():
                    suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())+'_'+str(_.bit_length())
                    np.save('minimax_q_secure_'+name+'_Q-'+suffix,Q)
                
            return Q

def minimax_q_dpa(self,start,T=None,K=None,name='',tt=2**15):
        
        suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())
        self.oa.spot_oa = None
        with open('minimax_q_robust_dpa_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
            pickle.dump(self,f)
        
        q0 = self.oa.q0
        reward = self.reward
        e = self.k*(1-self.discount)
        
        n_actions = len(self.mdp.A)
        shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
        Q = np.zeros(shape)
        
        transition_probs = np.zeros(shape+(3,))
        transition_states = np.zeros(shape+(3,4),dtype=np.int)
        for state in self.states():
            for action in range(n_actions):
                for action_ in range(n_actions):
                    for i,(s,p) in enumerate(zip(*self.transition_probs[state][action][action_])):
                        transition_probs[state][action][action_][i] = p
                        transition_states[state][action][action_][i] = s
        
        
        @jit(nopython=True)
        def numba_minimax(shape,q0,start,e,k,transition_probs,transition_states,reward,Q,T,K,tt):
            
            gamma = 1-e/k
            e_ = np.sqrt(e)
            n_actions = shape[-1]
            state0 = (0,q0,start[0],start[1])
            
            for _ in range(K):
                epsilon = max(0.5*tt/(_+tt),0.05)
                alpha = max(0.5*tt/(_+tt),0.05)
                state = state0
                
                if _ < 0.95*K:
                    state_ = (np.random.randint(shape[0]),np.random.randint(shape[1]),np.random.randint(shape[2]),np.random.randint(shape[3]))
                    max_val=0
                    for action in range(n_actions):
                        min_val = np.min(Q[state][action])
                        if min_val > max_val:
                            max_val = min_val
                    if max_val>0:
                        state=state_

                tau = np.random.geometric(e_) if state[0]==0 else np.random.geometric(e)
                max_action, min_action, max_q = 0, 0, 0
                for t in range(T):
                    # Follow an epsilon-greedy policy
                    if max_q==0 or random.random() < epsilon:
                        max_action = np.random.randint(n_actions)
                        min_action = np.random.randint(n_actions)
                        max_q = Q[state][max_action][min_action]

                    # Observe the next state
                    p = transition_probs[state][max_action][min_action]
                    cumsum = np.cumsum(p)
                    i=np.searchsorted(cumsum,random.random())
                    s = transition_states[state][max_action][min_action][i]
                    
                    next_state = (s[0],s[1],s[2],s[3])

                    if state[0]<next_state[0] and state[0]==1:
                        tau = np.random.geometric(e)
                    elif tau==0 and state[0]!=1:
                        next_state = (1,s[1],s[2],s[3])
                    tau -= 1

                    next_max_action, next_min_action, next_max_q = 0, 0, 0
                    for j in range(n_actions):
                        action_, min_q = 0, 1
                        for k in range(n_actions):
                            if Q[next_state][j][k] < min_q:
                                action_ = k
                                min_q = Q[next_state][j][k]
                        if min_q > next_max_q:
                            next_max_action = j
                            next_min_action = action_
                            next_max_q = min_q

                    # Q-update
                    Q[state][max_action][min_action] = max_q + alpha * (reward[state] + gamma*next_max_q - max_q)

                    state, max_action, min_action, max_q = next_state, next_max_action, next_min_action, next_max_q

            return Q
        
        Q=numba_minimax(shape,q0,start,e,self.k,transition_probs,transition_states,reward,Q,T,K,tt)
        suffix = str((T-1).bit_length())+'_'+str((K-1).bit_length())
        np.save('minimax_q_robust_dpa_'+name+'_Q-'+suffix,Q)
        
        return Q
    
    
    
    
def shapley_q_dpa(self,T=None,name='',tt=2**15):
        
        suffix = str((T-1).bit_length())
        self.oa.spot_oa = None
        with open('shapley_q_robust_dpa_'+name+'_csrl-'+suffix+'.pkl','wb') as f:
            pickle.dump(self,f)
        
        states = list(self.states(second=False))
        reward = self.reward
        e = self.k*(1-self.discount)
        
        n_actions = len(self.mdp.A)
        shape = self.oa.shape+self.mdp.shape+(n_actions,n_actions)
        Q = np.zeros(shape)
        value = np.zeros(self.oa.shape+self.mdp.shape)
        transition_probs = np.zeros(shape+(3,))
        transition_states = np.zeros(shape+(3,4),dtype=np.int)
        for state in self.states():
            for action in range(n_actions):
                for action_ in range(n_actions):
                    for i,(s,p) in enumerate(zip(*self.transition_probs[state][action][action_])):
                        transition_probs[state][action][action_][i] = p
                        transition_states[state][action][action_][i] = s
        
        
        @jit(nopython=True)
        def numba_shapley(shape,e,k,states,transition_probs,transition_states,reward,Q,value,T):
            
            gamma = 1-e/k
            e_ = np.sqrt(e)
            n_actions = shape[-1]

            t = 0
            while t < T:
                for state in states:
                    maximin_val = 0
                    for j in range(n_actions):
                        min_val = np.min(Q[state][j])
                        if min_val>maximin_val:
                            maximin_val=min_val
                    value[state] = maximin_val
                
                for state in states:
                    r = reward[state]
                    for j in range(n_actions):
                        for k in range(n_actions):
                            val=0
                            for l in range(3):
                                p=transition_probs[state][j][k][l]
                                s=transition_states[state][j][k][l]
                                val = val + value[s[0],s[1],s[2],s[3]]*p
                            Q[state][j][k] = r + gamma*val

                Q[0] = (1-e_)*Q[0] + e_*Q[1] 
                for i in range(2,shape[0]):
                    Q[i] = (1-e)*Q[i] + e*Q[1]

                t+=1

            return Q
            
        Q=numba_shapley(shape,e,self.k,states,transition_probs,transition_states,reward,Q,value,T)
        np.save('shapley_q_robust_dpa_'+name+'_Q-'+suffix,Q)
        
        return Q