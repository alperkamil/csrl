from random import randint
import time
from collections import namedtuple
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
from datetime import datetime
import utils.nnetwork as dqn
from utils.nnetwork import get_q, device, to_tensor
from utils.replay_memory import ReplayMemory
from utils.atari_wrapper import make_atari,wrap_deepmind
from utils import logger
from utils.evaluate import act_for_one_episode,plot

# Set up Environment
env_name = 'BreakoutNoFrameskip-v4'
exp_name = datetime.now().strftime("%m_%d_%-H_%-M_") + env_name
env = make_atari(env_name)
env = wrap_deepmind(env,
                    episode_life=False, 
                    clip_rewards = False,
                    frame_stack=True, 
                    scale=True)
n_actions = env.action_space.n
obs_shape = env.observation_space.shape
cwd = os.getcwd()

def train(num_frames, batch_size, gamma, replay_memory_size, 
		  replay_start_size,target_update_frequency,
		  ep_start, ep_end,eps_decay,train_freq=1,logger_kwargs=dict(),
		  starting_model=None,model_save_frequency=None,noop=False,
          n_episodes_plot=None,plot_frequency=None,act_epsilon=None):
    # Set up log
    log = logger.EpochLogger(**logger_kwargs)
    log.save_config(locals())
    
    # Set up neural nets
    policy_net = dqn.CNN(obs_shape[:2],n_actions).to(device)
    policy_net.apply(dqn.init_weights) # initialize weights (will be overwritten if starting model)
    lag_net = dqn.CNN(obs_shape[:2],n_actions).to(device)

    if starting_model:
        model_path = f'/data/{starting_model}/pyt_save/model.pt'
        policy_net.load_state_dict(torch.load(cwd+model_path,map_location=device))

    lag_net.load_state_dict(policy_net.state_dict())
    # optimizer = RMSprop(policy_net.parameters(), lr=0.000025,momentum=.95,eps=.01)
    optimizer = Adam(policy_net.parameters(), lr=0.000025)
    # Set up Replay Buffer memory
    Transition = namedtuple('Transition',('index','state', 'action', 'next_state', 'reward','done'))
    mem = ReplayMemory(replay_memory_size)
    
    # Prepare for main loop
    ep_rews = [] 			# for measuring episode returns
    done = False 			# signal from environment that episode is over
    updated = False			# signal that the nets have been updated at least once
    i_ep, i_frames, noop_frames, noop_len = 0, 0, 0, randint(0,30)
    train_freq = 4 			 # update at most every four frames
    epsilon = ep_start
    q_ep = 0.				 # for debugging
    state = env.reset() 	 # first obs comes from starting distribution
    ep_time = time.time()
    plt_rews = []           # for plotting episodes with epsilon=0 

    # Main loop
    while i_frames < num_frames: # experience transitions associated with the policy.  
        if mem.size < replay_start_size: # initialize replay buffer
            act = env.action_space.sample() #random action
        else:
            epsilon = max(ep_end , ep_start - i_frames*(ep_start - ep_end)/eps_decay)
            act = dqn.get_action(to_tensor([state]),epsilon,policy_net,env)
            if noop:
                if noop_frames < noop_len:
                    act = 0
                    noop_frames+=1
                else:
                    epsilon = max(ep_end , ep_start - i_frames*(ep_start - ep_end)/eps_decay)
                    act = dqn.get_action(to_tensor([state]),epsilon,policy_net,env)
        i_frames+=1
        q_ep += float(get_q(policy_net,to_tensor([state]))[act])
        next_state, rew, done, _ = env.step(act) 
        mem.append(Transition(i_frames,state,act,next_state,rew,done))
        ep_rews.append(rew)
        ok_to_update = (mem.size >= batch_size 
                          and mem.size >= replay_start_size 
                          and i_frames % train_freq == 0)
        if ok_to_update: # update the Q-function estimator policy_net
            batch = mem.sample(batch_size) # sample a mini-batch
            batch_loss = dqn.train_one_epoch(batch,
                                        policy_net =policy_net,
                                        target_net=lag_net,
                                        optimizer=optimizer,
                                        gamma = gamma)
            updated = True
            # refresh the target net every C frames.
            if i_frames > 0 and i_frames % target_update_frequency == 0:
                lag_net.load_state_dict(policy_net.state_dict())
        if done:
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            # Logging
            print(f'Episode finished with length {ep_len}, rew {ep_ret} and total Q estimate {q_ep}. Updated is {updated}.')
            log.log_tabular('Episode',i_ep)
            log.log_tabular('Frame',i_frames)
            log.log_tabular('EpLen',ep_len)
            log.log_tabular('EpRet',ep_ret)
            log.log_tabular('Q_ep',"%8.3f"%q_ep)
            log.log_tabular('Epsilon',"%8.3f"%epsilon)
            log.log_tabular('EpTime', "%8.3f"%(time.time()-ep_time))
            log.log_tabular('MemSize', mem.getSize())
            log.dump_tabular()
            # Reset
            i_ep +=1
            ep_time = time.time()
            q_ep, ep_rews = 0., []
            noop_frames, noop_len = 0,randint(0,30)
            state = env.reset() # env.reset moves to next life when episode_life is True
            if n_episodes_plot and plot_frequency: # for progress plot
                if i_ep % n_episodes_plot==0 and epsilon > act_epsilon: # act with epsilon = act_epsilon for one episode  
                    env_act = wrap_deepmind(make_atari(env_name),episode_life=False, 
                        clip_rewards = False,frame_stack=True, scale=True)
                    act_ret = act_for_one_episode(net=policy_net,environment=env_act, epsilon=act_epsilon,render=False)
                    plt_rews.append(act_ret)
                else: plt_rews.append(ep_ret)
        else:
            state = next_state
        
        if model_save_frequency and i_frames > 0 and i_frames % model_save_frequency==0 : 
            print('saving model...')
            log.pytorch_save(policy_net)
        
        if n_episodes_plot and plot_frequency:
            if n_episodes_plot and i_frames>0 and i_frames % plot_frequency==0:
                plt_path = f'/data/{exp_name}/'
                plot(plt_rews,"Returns vs Episode",cwd+plt_path)
        
    # after training loop
    return policy_net

## Main ##
# starting_model='06_10_19_17_BreakoutNoFrameskip-v4'
logger_kwargs = logger.setup_logger_kwargs(exp_name=exp_name) 
log = logger.EpochLogger(**logger_kwargs)
trained_net = train(num_frames = int(150000), batch_size = 32, gamma = .99, replay_memory_size = int(2e5),
                    replay_start_size = 50000, target_update_frequency=10000,ep_start=100,ep_end=10, 
                    eps_decay= int(1e6), train_freq=4, logger_kwargs=logger_kwargs,model_save_frequency=int(1e5),
                    noop=False,starting_model=None,n_episodes_plot=10,plot_frequency=1000,act_epsilon=5,)
log.pytorch_save(trained_net)