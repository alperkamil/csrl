"""
Pre-processing for atari game state:
1) skip every fourth frame.
2) downsize (1/2) and grayscale resulting frame sequence
3) In resulting frame sequence, max pool two frames into one (and sum reward) 
4) One state is 4 consecutive "frames" in the remaining frame-sequence

Based on by Jordi Torres's post:
https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
"""

import collections
import gym
import numpy as np

class MaxAndSkipEnv(gym.Wrapper):
    """obs becomes max pool of two frames, reward is sum of rewards"""
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    def step(self, action):
            total_reward = 0.0
            done = None
            for _ in range(self._skip):
               obs, reward, done, info = self.env.step(action)
               self._obs_buffer.append(obs)
               total_reward += reward
               if done:
                   break
            max_frame = np.max(np.stack(self._obs_buffer), axis=0)
            return max_frame, total_reward, done, info
    def reset(self):
           self._obs_buffer.clear()
           obs = self.env.reset()
           self._obs_buffer.append(obs)
           return obs

class BufferWrapper(gym.ObservationWrapper):
    """obs becomes a a sequence of n_steps frames"""
    def __init__(self, env, n_steps, dtype=np.uint8):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = \
                 gym.spaces.Box(old_space.low.repeat(n_steps, 
                 axis=0),old_space.high.repeat(n_steps, axis=0),     
                 dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low,
        dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = observation
            return self.buffer

class FireResetEnv(gym.Wrapper):
    """for games that require FIRE to start"""
    def __init__(self, env=None):
       super(FireResetEnv, self).__init__(env)
       assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
       assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def step(self, action):
           return self.env.step(action)
    def reset(self):
           self.env.reset()
           obs, _, done, _ = self.env.step(1)
           if done:
              self.env.reset()
           obs, _, done, _ = self.env.step(2)
           if done:
              self.env.reset()
           return obs

class ProcessFrame(gym.ObservationWrapper):
    """image downsize and grayscale"""
    def __init__(self, env=None):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                               shape=(105, 80, 1), dtype=np.uint8)
    def observation(self,obs):
        return ProcessFrame.preprocess(obs)
    
    def preprocess(img,rescale=2): 
        
        def to_grayscale(img):
            return np.mean(img, axis=2).astype(np.uint8)

        def downsample(img,rescale): # TODO use a gaussian kernel with bilinear interpolation.
            return img[::rescale, ::rescale]

        return to_grayscale(downsample(img,rescale))

class ImageToPyTorch(gym.ObservationWrapper):
    """reshape observations to channel x rows x cols"""
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,            
                                shape=(old_shape[-1], 
                                old_shape[0], old_shape[1]),
                                dtype=np.uint8)
    def observation(self, observation):
        return observation
    
    def reset(self):
        return self.observation(self.env.reset())

def make_atari_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame(env)
    env = ImageToPyTorch(env) 
    env = BufferWrapper(env, 4)
    return env