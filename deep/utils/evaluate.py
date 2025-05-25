import torch
import torch.nn as nn
import gym
from . import nnetwork as dqn
from .atari_wrapper import make_atari, wrap_deepmind

def act_for_one_episode(net,environment,epsilon=5,render=False):
    ep_rews = []          # for measuring episode returns
    # Start episode.    
    s = environment.reset()      # first obs comes from starting distribution
    finished = False             # signal from environment that episode is over
    if render:
        environment.render()
    ## experience transitions associated with the policy.
    while not finished:
        a = dqn.get_action(dqn.to_tensor([s]),epsilon,net,environment)
        s_next, reward, finished, _ = environment.step(a)
        ep_rews.append(reward)
        if render:
            environment.render()    
        # move to the next transition
        s = s_next
    return sum(ep_rews)

def plot(y,plt_name,plt_dir):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(y)
    fig.savefig(plt_dir + f'{plt_name}.png')